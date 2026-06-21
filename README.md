# Floquet NESS Engineering with Quantum Optimal Control

This repository contains cleaned code for engineering nonequilibrium steady states (NESSs) in periodically driven open quantum systems.

The core idea is simple:

> Parameterize a periodic control field with a finite Fourier series, build the one-period Floquet map of the open quantum system, solve for the stroboscopic NESS, and use `scipy.optimize` to optimize the Fourier coefficients and the period.

The project has two main parts:

1. **Single-qubit Floquet NESS engineering**  
   A dissipative two-level system is driven by optimized periodic controls. The code searches for steady states with large excited-state population or large transverse Bloch polarization.

2. **Two-qubit Bell-state thermal machine**  
   Two coupled qubits interact with different baths. Periodic modulation is used to steer the thermal-machine steady state toward a high-fidelity Bell-state NESS.

## What the code actually optimizes

For the single-qubit Fourier cases, the control is written as

$$
J_x(s)=\sum_{m=1}^{M}\left[a_m\cos(2\pi m s)+b_m\sin(2\pi m s)\right],
\qquad s=t/T.
$$

The optimization variables are

$$
\theta = (a_1,b_1,a_2,b_2,\ldots,a_M,b_M,T).
$$

For the two-qubit Bell-state case, the code optimizes Fourier-like controls for detuning and dissipation/coupling strength:

$$
\delta(s)=\sum_{m=1}^{M} a_m \sin(2\pi m s),
$$

$$
\gamma(s)=\sin^2(\pi s)\left[b_0+\sum_{m=1}^{M-1}b_m\cos(2\pi m s)\right]^2.
$$

The squared form keeps $\gamma(s)\ge 0$, which is important because it represents a physical rate/coupling scale.

## Numerical workflow

Each optimization step does the following:

1. Read the current Fourier coefficients and period $T$.
2. Evaluate the periodic control fields on a time grid.
3. Build the time-dependent Liouvillian slices.
4. Compose the one-period Floquet map $\Phi$.
5. Solve the fixed-point equation $\rho_* = \Phi\rho_*$ with the trace constraint $\mathrm{Tr}(\rho_*)=1$.
6. Evaluate the objective, such as $\langle\sigma_z\rangle$, $\langle\sigma_x\rangle$, or Bell-state fidelity.
7. Use `scipy.optimize.minimize` with `L-BFGS-B` to update the Fourier coefficients and period.

The cleaned implementation uses direct matrix exponentials for the time slices. The original notebooks also contain QuTiP-based exploratory versions.

## Why the Bell-state case is a thermal machine

The two-qubit system is interesting because it is not just a generic two-qubit control problem. It acts as a small quantum thermal machine: the qubits are coupled to different effective baths, so the steady state is shaped by energy exchange, dissipation, and nonequilibrium flow.

Optimization is needed because the desired Bell-state NESS is not obtained just by turning on dissipation. The final state depends nonlinearly on:

- the full one-period Floquet map, not only the instantaneous Hamiltonian;
- the path traced by $\delta(t)$ and $\gamma(t)$;
- the period $T$, which controls how adiabatic or nonadiabatic the modulation is;
- the physical constraint $\gamma(t)\ge 0$;
- the competition between bath-induced mixing, coherent coupling, and convergence to the NESS.

So the optimization problem is to find a periodic path through parameter space that makes the dissipative thermal-machine dynamics stabilize the target Bell state stroboscopically.

## Representative results from the archived runs

| Case | Objective / metric | Representative value |
|---|---:|---:|
| Single qubit, Fourier $J_x(t)$ | optimized $\langle\sigma_z\rangle$ | `-0.791211` |
| Single qubit, Fourier $J_x(t)$ | excited-state population $(1-\langle\sigma_z\rangle)/2$ | `0.895606` |
| Single qubit, Fourier $J_x(t),\Delta(t)$ | optimized $\langle\sigma_x\rangle$ | `0.936456` |
| Single qubit, piecewise $J_x(t)$ | excited-state population | `0.938389` |
| Two-qubit thermal machine | Bell-state fidelity | `0.997945` |
| Two-qubit thermal machine | concurrence | `0.995930` |

These files are stored in `results/reference_runs/`.

## Repository structure

```text
floquet-ness-qoc-clean/
├── src/floquet_ness_qoc/
│   ├── linalg_utils.py              # vec/mat, Liouvillian superoperators, fixed-point solve
│   ├── controls.py                  # Fourier control parameterizations
│   ├── single_qubit.py              # single-qubit models and optimizers
│   └── bell_thermal_machine.py      # two-qubit Bell thermal-machine optimizer
├── scripts/
│   ├── run_single_sigma_z_fourier.py
│   ├── run_single_sigma_x_delta_fourier.py
│   ├── run_single_sigma_z_piecewise.py
│   ├── run_bell_thermal_machine.py
│   └── plot_reference_results.py
├── results/
│   ├── reference_runs/              # selected NPZ files from the original runs
│   └── reference_figures/           # selected presentation-ready figures
├── legacy_notebooks/                # original notebooks, kept for traceability
├── docs/                            # abstract, reports, and APS presentation
├── requirements.txt
└── pyproject.toml
```

## How to run

Create an environment and install the package locally:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -e .
```

Then run one of the scripts:

```bash
python scripts/run_single_sigma_z_fourier.py
python scripts/run_single_sigma_x_delta_fourier.py
python scripts/run_single_sigma_z_piecewise.py
python scripts/run_bell_thermal_machine.py
```

The scripts save compact `.npz` result files into `results/runs/`.

## Future work

A natural next step is experimental implementation in a circuit-QED platform. Circuit QED is a good candidate because superconducting qubits allow tunable drives, engineered dissipation, and reservoir-engineering protocols. In that setting, the optimized Fourier controls would need to be translated into experimentally feasible microwave/control pulses with bandwidth, amplitude, calibration, and noise constraints.

Other next steps:

- verify convergence with respect to the number of time slices $N$;
- compare Fourier controls against piecewise-constant controls more systematically;
- add analytic or adjoint gradients for faster optimization;
- include robustness tests against parameter noise and pulse distortion;
- connect the Bell-state thermal-machine optimization more directly to heat currents and thermodynamic performance.
