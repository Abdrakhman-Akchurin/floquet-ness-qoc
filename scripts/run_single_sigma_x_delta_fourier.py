"""Optimize a single dissipative qubit for transverse steady-state polarization.

Objective: maximize <sigma_x>_NESS using Fourier Jx(t), Delta(t), and optimized period T.
"""
from floquet_ness_qoc.single_qubit import (
    FourierOptimizationConfig,
    SingleQubitParams,
    optimize_sigma_x_jx_delta_fourier,
)


if __name__ == "__main__":
    params = SingleQubitParams(gamma_down=4.7, gamma_phi=0.3)
    config = FourierOptimizationConfig(
        m_order=3,
        n_slices=48,
        coefficient_bound=50.0,
        t_bounds=(0.05, 2.0),
        seed=1,
        n_restarts=6,
        maxiter=250,
    )
    best = optimize_sigma_x_jx_delta_fourier(params=params, config=config)
    obs = best["diagnostics"]["observables"]
    print("Best single-qubit sigma_x run")
    print(f"  T              = {best['T']:.6f}")
    print(f"  <sigma_x>_NESS = {obs['sx']:.6f}")
    print(f"  <sigma_z>_NESS = {obs['sz']:.6f}")
