"""Optimize a single dissipative qubit for high excited-state population.

Objective: minimize <sigma_z>_NESS using a Fourier Jx(t) drive and optimized period T.
"""
from floquet_ness_qoc.single_qubit import (
    FourierOptimizationConfig,
    SingleQubitParams,
    optimize_sigma_z_jx_fourier,
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
    best = optimize_sigma_z_jx_fourier(params=params, config=config)
    obs = best["diagnostics"]["observables"]
    print("Best single-qubit sigma_z run")
    print(f"  T                  = {best['T']:.6f}")
    print(f"  <sigma_z>_NESS     = {obs['sz']:.6f}")
    print(f"  excited population = {obs['p_excited']:.6f}")
