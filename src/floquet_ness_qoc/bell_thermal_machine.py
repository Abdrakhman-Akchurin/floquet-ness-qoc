"""Two-qubit Bell-state Floquet NESS / thermal-machine model.

This module is the cleaned version of the Bell-state notebook. It keeps the
scientific structure visible:

1. Build a Liouvillian for two coupled qubits connected to different baths.
2. Parameterize periodic detuning and dissipation/coupling profiles by Fourier
   coefficients.
3. Build the Floquet map over one period.
4. Solve for the stroboscopic NESS.
5. Optimize Bell-state fidelity with scipy.optimize.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time

import numpy as np
from numpy import kron
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.optimize import minimize

from .controls import bell_delta_from_theta, bell_gamma_from_theta, midpoint_grid
from .linalg_utils import (
    concurrence,
    fixed_point_from_map,
    left_right_super,
    partial_trace_two_qubit,
    purity,
    random_density,
    vec,
    mat,
)

Array = np.ndarray

# Single-qubit operators. Basis convention is |0>, |1>.
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
sp = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |1><0|
sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |0><1|
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# Two-qubit lifted operators.
I4 = kron(I2, I2)
sp1 = kron(sp, I2)
sm1 = kron(sm, I2)
sp2 = kron(I2, sp)
sm2 = kron(I2, sm)

n1_op = sp1 @ sm1
n2_op = sp2 @ sm2
p0_1 = sm1 @ sp1
p0_2 = sm2 @ sp2

sx1 = kron(sx, I2)
sy1 = kron(sy, I2)
sz1 = kron(sz, I2)
sx2 = kron(I2, sx)
sy2 = kron(I2, sy)
sz2 = kron(I2, sz)


@dataclass(frozen=True)
class BellMachineParams:
    """Physical parameters for the two-qubit thermal-machine model."""

    g: float = 0.01
    alpha: float = 1.2
    n1: float = 1.0
    n2: float = 0.0
    q: float = 1.0


@dataclass(frozen=True)
class BellOptimizationConfig:
    """Settings for the Bell-state Floquet optimization."""

    target: str = "psi_minus"
    m_order: int = 3
    n_slices: int = 48
    t_bounds: tuple[float, float] = (50.0, 2500.0)
    delta_coeff_bound: float = 0.2
    gamma_coeff_bound: float = 0.6
    seed: int = 1
    n_restarts: int = 3
    maxiter: int = 60
    t_penalty: float = 0.0


def nh_comm_super(h_eff: Array) -> Array:
    """Superoperator for -i(H_eff rho - rho H_eff†)."""
    d = h_eff.shape[0]
    ident = np.eye(d, dtype=complex)
    return -1j * (kron(ident, h_eff) - kron(h_eff.conj(), ident))


def bell_state(which: str = "psi_minus") -> Array:
    """Return |Psi-> or |Psi+> in the basis |00>, |01>, |10>, |11>."""
    e01 = np.array([0, 1, 0, 0], dtype=complex)
    e10 = np.array([0, 0, 1, 0], dtype=complex)
    key = which.lower().replace(" ", "_")
    if key in {"psi_minus", "psiminus", "-", "singlet"}:
        return (e01 - e10) / np.sqrt(2)
    if key in {"psi_plus", "psiplus", "+", "triplet"}:
        return (e01 + e10) / np.sqrt(2)
    raise ValueError("target must be 'psi_minus' or 'psi_plus'.")


def fidelity_pure(psi: Array, rho: Array) -> float:
    """Fidelity <psi|rho|psi> for a pure target state."""
    return float(np.real(np.vdot(psi, rho @ psi)))


def bloch_single_qubit(rho_one_qubit: Array) -> Array:
    """Bloch vector of a one-qubit reduced state."""
    return np.array(
        [
            np.real(np.trace(rho_one_qubit @ sx)),
            np.real(np.trace(rho_one_qubit @ sy)),
            np.real(np.trace(rho_one_qubit @ sz)),
        ],
        dtype=float,
    )


def precompute_liouvillian_parts(params: BellMachineParams) -> tuple[Array, Array, Array]:
    """Precompute L_const, L_delta, and L_gamma.

    The model separates the Liouvillian into

        L(t) = L_const + delta(t) L_delta + gamma(t) L_gamma.

    This is the key reason the optimization loop can evaluate many controls
    without rebuilding the full Liouvillian algebra every time.
    """
    g = params.g
    alpha = params.alpha
    n1 = params.n1
    n2 = params.n2
    q = params.q

    h_g = g * (sp1 @ sm2 + sm1 @ sp2)
    h_delta = 0.5 * (n2_op - n1_op)

    a_nh = (1.0 - n1) * n1_op + n1 * p0_1 + alpha * ((1.0 - n2) * n2_op + n2 * p0_2)
    h_gamma = -0.5j * a_nh

    l_const = nh_comm_super(h_g)
    l_delta = nh_comm_super(h_delta)
    l_gamma_h = nh_comm_super(h_gamma)

    # Jump superoperators per unit gamma.
    jp1 = left_right_super(sp1, sm1)
    jm1 = left_right_super(sm1, sp1)
    jp2 = left_right_super(sp2, sm2)
    jm2 = left_right_super(sm2, sp2)

    g1p = n1
    g1m = 1.0 - n1
    g2p = alpha * n2
    g2m = alpha * (1.0 - n2)

    l_gamma_jump = q * (g1p * jp1 + g1m * jm1 + g2p * jp2 + g2m * jm2)
    l_gamma = l_gamma_h + l_gamma_jump
    return l_const, l_delta, l_gamma


def build_propagators(
    theta_delta: Array,
    theta_gamma: Array,
    period: float,
    n_slices: int,
    parts: tuple[Array, Array, Array],
) -> tuple[Array, Array, Array, list[Array]]:
    """Build one-period step maps from Fourier controls."""
    l_const, l_delta, l_gamma = parts
    s_nodes = midpoint_grid(n_slices)
    delta_s = bell_delta_from_theta(theta_delta, s_nodes)
    gamma_s = bell_gamma_from_theta(theta_gamma, s_nodes)
    dt = float(period) / int(n_slices)

    steps = []
    for delta_value, gamma_value in zip(delta_s, gamma_s):
        liouvillian = l_const + delta_value * l_delta + gamma_value * l_gamma
        steps.append(expm(dt * liouvillian))
    return s_nodes, delta_s, gamma_s, steps


def floquet_map(step_maps: list[Array]) -> Array:
    """Compose Phi = U_N ... U_1 from chronological step maps."""
    phi = np.eye(step_maps[0].shape[0], dtype=complex)
    for step in step_maps:
        phi = step @ phi
    return phi


def unpack_theta(theta: Array, m_order: int) -> tuple[Array, Array, float]:
    """theta = [delta coefficients, gamma coefficients, logT]."""
    theta = np.asarray(theta, dtype=float)
    theta_delta = theta[:m_order]
    theta_gamma = theta[m_order : 2 * m_order]
    log_t = float(theta[-1])
    return theta_delta, theta_gamma, log_t


def evaluate_bell_controls(
    theta: Array,
    params: BellMachineParams,
    config: BellOptimizationConfig,
    parts: tuple[Array, Array, Array] | None = None,
) -> tuple[float, dict[str, object]]:
    """Evaluate the negative Bell fidelity objective.

    The optimizer minimizes this value, so the objective is -F plus an optional
    soft period penalty.
    """
    if parts is None:
        parts = precompute_liouvillian_parts(params)

    theta_delta, theta_gamma, log_t = unpack_theta(theta, config.m_order)
    period = float(np.exp(log_t))
    period = min(max(period, config.t_bounds[0]), config.t_bounds[1])

    s_nodes, delta_s, gamma_s, steps = build_propagators(
        theta_delta, theta_gamma, period, config.n_slices, parts
    )
    phi = floquet_map(steps)
    r_star, rho_star = fixed_point_from_map(phi, d=4)

    psi = bell_state(config.target)
    fidelity = fidelity_pure(psi, rho_star)
    objective = -(fidelity - config.t_penalty * (period / config.t_bounds[1]))

    diagnostics = {
        "T": period,
        "F": fidelity,
        "rho_star": rho_star,
        "r_star": r_star,
        "Phi": phi,
        "step_maps": steps,
        "s_nodes": s_nodes,
        "delta_s": delta_s,
        "gamma_s": gamma_s,
        "theta_delta": theta_delta,
        "theta_gamma": theta_gamma,
    }
    return float(objective), diagnostics


def paper_like_initial_guess(config: BellOptimizationConfig) -> Array:
    """Initial guess matching the simple loop used in the exploratory notebook."""
    theta_delta = np.zeros(config.m_order)
    theta_gamma = np.zeros(config.m_order)
    theta_delta[0] = -0.04
    theta_gamma[0] = np.sqrt(0.008)
    log_t = np.log(config.t_bounds[1])
    return np.concatenate([theta_delta, theta_gamma, [log_t]])


def optimize_bell_thermal_machine(
    params: BellMachineParams = BellMachineParams(),
    config: BellOptimizationConfig = BellOptimizationConfig(),
    output_dir: str | Path | None = "results/runs",
) -> dict[str, object]:
    """Optimize Bell-state fidelity over delta(s), gamma(s), and period T."""
    rng = np.random.default_rng(config.seed)
    parts = precompute_liouvillian_parts(params)
    bounds = (
        [(-config.delta_coeff_bound, config.delta_coeff_bound)] * config.m_order
        + [(-config.gamma_coeff_bound, config.gamma_coeff_bound)] * config.m_order
        + [(np.log(config.t_bounds[0]), np.log(config.t_bounds[1]))]
    )

    def objective(theta: Array) -> float:
        obj, _ = evaluate_bell_controls(theta, params, config, parts)
        return obj

    starts = [paper_like_initial_guess(config)]
    for _ in range(config.n_restarts - 1):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float))

    best = None
    for i, theta0 in enumerate(starts):
        history = {"objective": [], "F": [], "T": []}

        def callback(theta: Array) -> None:
            obj, diag = evaluate_bell_controls(theta, params, config, parts)
            history["objective"].append(float(obj))
            history["F"].append(float(diag["F"]))
            history["T"].append(float(diag["T"]))

        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.maxiter, "ftol": 1e-10},
            callback=callback,
        )
        obj, diagnostics = evaluate_bell_controls(result.x, params, config, parts)
        record = {
            "restart": i,
            "success": bool(result.success),
            "message": str(result.message),
            "n_iterations": int(result.nit),
            "n_function_evals": int(result.nfev),
            "theta": result.x,
            "objective": float(obj),
            "F": float(diagnostics["F"]),
            "T": float(diagnostics["T"]),
            "history": history,
            "diagnostics": diagnostics,
        }
        if best is None or record["objective"] < best["objective"]:
            best = record

    assert best is not None
    if output_dir is not None:
        save_bell_run(best, params, config, output_dir)
    return best


def micromotion_and_convergence(
    diagnostics: dict[str, object],
    target: str = "psi_minus",
    n_cycles_convergence: int = 50,
    seed: int = 0,
) -> dict[str, Array]:
    """Generate diagnostic arrays from an optimized Bell NESS."""
    step_maps = diagnostics["step_maps"]
    phi = diagnostics["Phi"]
    rho_star = diagnostics["rho_star"]
    psi = bell_state(target)

    r = vec(rho_star)
    f_t: list[float] = []
    pur_t: list[float] = []
    conc_t: list[float] = []
    bloch_1: list[Array] = []
    bloch_2: list[Array] = []
    pur1: list[float] = []
    pur2: list[float] = []

    for j in range(len(step_maps) + 1):
        rho = mat(r, d=4)
        f_t.append(fidelity_pure(psi, rho))
        pur_t.append(purity(rho))
        conc_t.append(concurrence(rho))
        rho1 = partial_trace_two_qubit(rho, keep=1)
        rho2 = partial_trace_two_qubit(rho, keep=2)
        bloch_1.append(bloch_single_qubit(rho1))
        bloch_2.append(bloch_single_qubit(rho2))
        pur1.append(purity(rho1))
        pur2.append(purity(rho2))
        if j < len(step_maps):
            r = step_maps[j] @ r

    rng = np.random.default_rng(seed)
    r_conv = vec(random_density(4, rng))
    convergence = []
    for _ in range(n_cycles_convergence + 1):
        rho = mat(r_conv, d=4)
        rho = 0.5 * (rho + rho.conj().T)
        rho = rho / np.trace(rho)
        convergence.append(float(norm(rho - rho_star)))
        r_conv = phi @ r_conv

    b1 = np.vstack(bloch_1)
    b2 = np.vstack(bloch_2)
    return {
        "F_t": np.array(f_t),
        "purity_t": np.array(pur_t),
        "concurrence_t": np.array(conc_t),
        "bloch_q1": b1,
        "bloch_q2": b2,
        "purity_q1": np.array(pur1),
        "purity_q2": np.array(pur2),
        "convergence_distance": np.array(convergence),
    }


def save_bell_run(
    best: dict[str, object],
    params: BellMachineParams,
    config: BellOptimizationConfig,
    output_dir: str | Path,
) -> Path:
    """Save a compact NPZ file for an optimized Bell run."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    diag = best["diagnostics"]
    micro = micromotion_and_convergence(diag, target=config.target, seed=config.seed)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    tag = f"{stamp}_bell_thermal_machine_{config.target}_N{config.n_slices}_M{config.m_order}.npz"
    path = outdir / tag
    np.savez(
        path,
        params=asdict(params),
        config=asdict(config),
        theta=best["theta"],
        theta_delta=diag["theta_delta"],
        theta_gamma=diag["theta_gamma"],
        T=diag["T"],
        F_star=diag["F"],
        rho_star=diag["rho_star"],
        purity_star=purity(diag["rho_star"]),
        concurrence_star=concurrence(diag["rho_star"]),
        s_nodes=diag["s_nodes"],
        delta_s=diag["delta_s"],
        gamma_s=diag["gamma_s"],
        history_F=np.array(best["history"]["F"], dtype=float),
        history_T=np.array(best["history"]["T"], dtype=float),
        **micro,
    )
    return path
