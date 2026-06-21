"""Single-qubit Floquet NESS optimization models.

The implementation here is intentionally clean and explicit. It uses a finite
Fourier ansatz for the periodic controls and SciPy's optimizers for the Fourier
coefficients and period. The one-period map is built from piecewise-constant
Liouvillian slices using matrix exponentials.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from typing import Callable

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .controls import fourier_interleaved, midpoint_grid
from .linalg_utils import (
    commutator_super,
    dissipator_super,
    expectation,
    fixed_point_from_map,
    purity,
    random_density,
    vec,
)

Array = np.ndarray

I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


@dataclass(frozen=True)
class SingleQubitParams:
    """Physical rates for the dissipative qubit."""

    gamma_down: float = 4.7
    gamma_phi: float = 0.3


@dataclass(frozen=True)
class FourierOptimizationConfig:
    """Common settings for Fourier-control optimization."""

    m_order: int = 3
    n_slices: int = 48
    coefficient_bound: float = 50.0
    t_bounds: tuple[float, float] = (0.05, 2.0)
    seed: int = 1
    n_restarts: int = 6
    maxiter: int = 250
    fd_eps: float = 1e-6


def make_liouvillian_parts(params: SingleQubitParams) -> dict[str, Array]:
    """Precompute constant Liouville-space pieces for the single-qubit model."""
    l_diss = np.zeros((4, 4), dtype=complex)
    if params.gamma_down:
        l_diss += params.gamma_down * dissipator_super(sigma_minus)
    if params.gamma_phi:
        l_diss += params.gamma_phi * dissipator_super(sz)

    return {
        "diss": l_diss,
        "x": commutator_super(0.5 * sx),
        "y": commutator_super(0.5 * sy),
        "z": commutator_super(0.5 * sz),
    }


def _compose_period_map(liouvillians: list[Array], dt: float) -> tuple[Array, list[Array]]:
    """Return Phi = U_N ... U_1 and the list of step maps in chronological order."""
    phi = np.eye(liouvillians[0].shape[0], dtype=complex)
    step_maps = []
    for l_slice in liouvillians:
        step = expm(dt * l_slice)
        step_maps.append(step)
        phi = step @ phi
    return phi, step_maps


def floquet_map_jx_fourier(
    theta_jx: Array,
    period: float,
    n_slices: int,
    parts: dict[str, Array],
) -> tuple[Array, dict[str, Array]]:
    """One-period map for H(t)=Jx(t) sigma_x/2."""
    s_nodes = midpoint_grid(n_slices)
    jx = fourier_interleaved(theta_jx, s_nodes)
    dt = float(period) / int(n_slices)
    liouvillians = [parts["diss"] + j * parts["x"] for j in jx]
    phi, step_maps = _compose_period_map(liouvillians, dt)
    return phi, {"s_nodes": s_nodes, "Jx": jx, "step_maps": step_maps}


def floquet_map_jx_delta_fourier(
    theta_jx: Array,
    theta_delta: Array,
    period: float,
    n_slices: int,
    parts: dict[str, Array],
) -> tuple[Array, dict[str, Array]]:
    """One-period map for H(t)=[Jx(t) sigma_x + Delta(t) sigma_z]/2."""
    s_nodes = midpoint_grid(n_slices)
    jx = fourier_interleaved(theta_jx, s_nodes)
    delta = fourier_interleaved(theta_delta, s_nodes)
    dt = float(period) / int(n_slices)
    liouvillians = [parts["diss"] + j * parts["x"] + d * parts["z"] for j, d in zip(jx, delta)]
    phi, step_maps = _compose_period_map(liouvillians, dt)
    return phi, {"s_nodes": s_nodes, "Jx": jx, "Delta": delta, "step_maps": step_maps}


def floquet_map_jx_piecewise(
    jx_bins: Array,
    period: float,
    parts: dict[str, Array],
) -> tuple[Array, dict[str, Array]]:
    """One-period map for piecewise-constant Jx values."""
    jx_bins = np.asarray(jx_bins, dtype=float)
    dt = float(period) / len(jx_bins)
    liouvillians = [parts["diss"] + j * parts["x"] for j in jx_bins]
    phi, step_maps = _compose_period_map(liouvillians, dt)
    return phi, {"Jx": jx_bins, "step_maps": step_maps}


def single_qubit_observables(rho: Array) -> dict[str, float]:
    """Bloch components, purity, and excited-state population."""
    z = expectation(rho, sz)
    return {
        "sx": expectation(rho, sx),
        "sy": expectation(rho, sy),
        "sz": z,
        "purity": purity(rho),
        "p_excited": 0.5 * (1.0 - z),
    }


def evaluate_sigma_z_jx(
    theta_jx: Array,
    period: float,
    n_slices: int,
    parts: dict[str, Array],
) -> tuple[float, dict[str, object]]:
    """Return <sigma_z>_NESS for a Jx-only Fourier control."""
    phi, aux = floquet_map_jx_fourier(theta_jx, period, n_slices, parts)
    r_star, rho_star = fixed_point_from_map(phi, d=2)
    obs = single_qubit_observables(rho_star)
    return obs["sz"], {"phi": phi, "r_star": r_star, "rho_star": rho_star, "observables": obs, **aux}


def evaluate_sigma_x_jx_delta(
    theta_jx: Array,
    theta_delta: Array,
    period: float,
    n_slices: int,
    parts: dict[str, Array],
) -> tuple[float, dict[str, object]]:
    """Return <sigma_x>_NESS for Jx and Delta Fourier controls."""
    phi, aux = floquet_map_jx_delta_fourier(theta_jx, theta_delta, period, n_slices, parts)
    r_star, rho_star = fixed_point_from_map(phi, d=2)
    obs = single_qubit_observables(rho_star)
    return obs["sx"], {"phi": phi, "r_star": r_star, "rho_star": rho_star, "observables": obs, **aux}


def _save_json(path: Path, payload: dict) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(payload), indent=2))


def optimize_sigma_z_jx_fourier(
    params: SingleQubitParams = SingleQubitParams(),
    config: FourierOptimizationConfig = FourierOptimizationConfig(),
    output_dir: str | Path | None = "results/runs",
) -> dict[str, object]:
    """Minimize <sigma_z> by optimizing Fourier coefficients of Jx(t) and T."""
    rng = np.random.default_rng(config.seed)
    parts = make_liouvillian_parts(params)
    n_coeff = 2 * config.m_order
    bounds = [(-config.coefficient_bound, config.coefficient_bound)] * n_coeff + [config.t_bounds]

    def objective(x: Array) -> float:
        val, _ = evaluate_sigma_z_jx(x[:n_coeff], float(x[-1]), config.n_slices, parts)
        return val

    best = None
    starts = []
    for _ in range(config.n_restarts):
        theta0 = 0.2 * config.coefficient_bound * rng.standard_normal(n_coeff)
        t0 = rng.uniform(*config.t_bounds)
        starts.append(np.concatenate([theta0, [t0]]))

    for i, x0 in enumerate(starts):
        history = {"objective": [], "T": []}

        def callback(xk: Array) -> None:
            history["objective"].append(float(objective(xk)))
            history["T"].append(float(xk[-1]))

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.maxiter, "ftol": 1e-12, "eps": config.fd_eps},
            callback=callback,
        )
        value, diagnostics = evaluate_sigma_z_jx(result.x[:n_coeff], float(result.x[-1]), config.n_slices, parts)
        record = {
            "restart": i,
            "success": bool(result.success),
            "message": str(result.message),
            "n_iterations": int(result.nit),
            "n_function_evals": int(result.nfev),
            "x": result.x,
            "theta_jx": result.x[:n_coeff],
            "T": float(result.x[-1]),
            "objective": float(value),
            "history": history,
            "diagnostics": diagnostics,
        }
        if best is None or record["objective"] < best["objective"]:
            best = record

    assert best is not None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        tag = f"{stamp}_single_sigma_z_jx_fourier_N{config.n_slices}_M{config.m_order}"
        meta = {"params": asdict(params), "config": asdict(config), "tag": tag}
        _save_json(outdir / f"{tag}.json", {**meta, **{k: v for k, v in best.items() if k != "diagnostics"}})
        np.savez(
            outdir / f"{tag}.npz",
            theta_jx=best["theta_jx"],
            T=best["T"],
            objective=best["objective"],
            rho_star=best["diagnostics"]["rho_star"],
            **best["diagnostics"]["observables"],
        )
    return best


def optimize_sigma_x_jx_delta_fourier(
    params: SingleQubitParams = SingleQubitParams(),
    config: FourierOptimizationConfig = FourierOptimizationConfig(),
    output_dir: str | Path | None = "results/runs",
) -> dict[str, object]:
    """Maximize <sigma_x> by minimizing -<sigma_x> over Jx(t), Delta(t), and T."""
    rng = np.random.default_rng(config.seed)
    parts = make_liouvillian_parts(params)
    n_coeff = 2 * config.m_order
    bounds = [(-config.coefficient_bound, config.coefficient_bound)] * (2 * n_coeff) + [config.t_bounds]

    def objective(x: Array) -> float:
        theta_jx = x[:n_coeff]
        theta_delta = x[n_coeff : 2 * n_coeff]
        sx_val, _ = evaluate_sigma_x_jx_delta(theta_jx, theta_delta, float(x[-1]), config.n_slices, parts)
        return -sx_val

    best = None
    for i in range(config.n_restarts):
        theta0 = 0.2 * config.coefficient_bound * rng.standard_normal(2 * n_coeff)
        t0 = rng.uniform(*config.t_bounds)
        x0 = np.concatenate([theta0, [t0]])
        history = {"objective": [], "sigma_x": [], "T": []}

        def callback(xk: Array) -> None:
            theta_jx = xk[:n_coeff]
            theta_delta = xk[n_coeff : 2 * n_coeff]
            sx_val, _ = evaluate_sigma_x_jx_delta(theta_jx, theta_delta, float(xk[-1]), config.n_slices, parts)
            history["objective"].append(float(-sx_val))
            history["sigma_x"].append(float(sx_val))
            history["T"].append(float(xk[-1]))

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.maxiter, "ftol": 1e-12, "eps": config.fd_eps},
            callback=callback,
        )
        theta_jx = result.x[:n_coeff]
        theta_delta = result.x[n_coeff : 2 * n_coeff]
        sx_val, diagnostics = evaluate_sigma_x_jx_delta(theta_jx, theta_delta, float(result.x[-1]), config.n_slices, parts)
        record = {
            "restart": i,
            "success": bool(result.success),
            "message": str(result.message),
            "n_iterations": int(result.nit),
            "n_function_evals": int(result.nfev),
            "theta_jx": theta_jx,
            "theta_delta": theta_delta,
            "T": float(result.x[-1]),
            "objective": float(-sx_val),
            "sigma_x": float(sx_val),
            "history": history,
            "diagnostics": diagnostics,
        }
        if best is None or record["objective"] < best["objective"]:
            best = record

    assert best is not None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        tag = f"{stamp}_single_sigma_x_jx_delta_fourier_N{config.n_slices}_M{config.m_order}"
        meta = {"params": asdict(params), "config": asdict(config), "tag": tag}
        _save_json(outdir / f"{tag}.json", {**meta, **{k: v for k, v in best.items() if k != "diagnostics"}})
        np.savez(
            outdir / f"{tag}.npz",
            theta_jx=best["theta_jx"],
            theta_delta=best["theta_delta"],
            T=best["T"],
            objective=best["objective"],
            sigma_x=best["sigma_x"],
            rho_star=best["diagnostics"]["rho_star"],
            **best["diagnostics"]["observables"],
        )
    return best


def optimize_sigma_z_jx_piecewise(
    n_bins: int = 24,
    j_bound: float = 70.0,
    t_bounds: tuple[float, float] = (0.02, 2.0),
    params: SingleQubitParams = SingleQubitParams(),
    seed: int = 1,
    n_restarts: int = 3,
    maxiter: int = 100,
    output_dir: str | Path | None = "results/runs",
) -> dict[str, object]:
    """Benchmark: optimize piecewise-constant Jx bins and T."""
    rng = np.random.default_rng(seed)
    parts = make_liouvillian_parts(params)
    bounds = [(-j_bound, j_bound)] * n_bins + [t_bounds]

    def evaluate(x: Array) -> tuple[float, dict[str, object]]:
        phi, aux = floquet_map_jx_piecewise(x[:n_bins], float(x[-1]), parts)
        _, rho_star = fixed_point_from_map(phi, d=2)
        obs = single_qubit_observables(rho_star)
        return obs["sz"], {"rho_star": rho_star, "observables": obs, **aux}

    def objective(x: Array) -> float:
        return evaluate(x)[0]

    best = None
    for i in range(n_restarts):
        j0 = 0.2 * j_bound * rng.standard_normal(n_bins)
        t0 = rng.uniform(*t_bounds)
        result = minimize(
            objective,
            np.concatenate([j0, [t0]]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        value, diagnostics = evaluate(result.x)
        record = {
            "restart": i,
            "success": bool(result.success),
            "j_bins": result.x[:n_bins],
            "T": float(result.x[-1]),
            "objective": float(value),
            "diagnostics": diagnostics,
        }
        if best is None or record["objective"] < best["objective"]:
            best = record

    assert best is not None
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        tag = f"{stamp}_single_sigma_z_jx_piecewise_K{n_bins}"
        np.savez(
            outdir / f"{tag}.npz",
            j_bins=best["j_bins"],
            T=best["T"],
            objective=best["objective"],
            rho_star=best["diagnostics"]["rho_star"],
            **best["diagnostics"]["observables"],
        )
    return best


def micromotion_from_step_maps(rho_star: Array, step_maps: list[Array], n_periods: int = 4) -> dict[str, Array]:
    """Propagate the NESS through intra-period micromotion over several periods."""
    r = vec(rho_star)
    bx: list[float] = []
    by: list[float] = []
    bz: list[float] = []
    purities: list[float] = []

    for _ in range(n_periods):
        for step in step_maps:
            rho = np.asarray(r).reshape((2, 2), order="F")
            bx.append(expectation(rho, sx))
            by.append(expectation(rho, sy))
            bz.append(expectation(rho, sz))
            purities.append(purity(rho))
            r = step @ r
    rho = np.asarray(r).reshape((2, 2), order="F")
    bx.append(expectation(rho, sx))
    by.append(expectation(rho, sy))
    bz.append(expectation(rho, sz))
    purities.append(purity(rho))

    return {
        "sx": np.array(bx),
        "sy": np.array(by),
        "sz": np.array(bz),
        "purity": np.array(purities),
    }
