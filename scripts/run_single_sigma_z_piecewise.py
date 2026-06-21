"""Benchmark with piecewise-constant Jx(t) instead of a Fourier ansatz."""
from floquet_ness_qoc.single_qubit import SingleQubitParams, optimize_sigma_z_jx_piecewise


if __name__ == "__main__":
    best = optimize_sigma_z_jx_piecewise(
        n_bins=24,
        j_bound=70.0,
        t_bounds=(0.02, 2.0),
        params=SingleQubitParams(gamma_down=4.7, gamma_phi=0.3),
        seed=1,
        n_restarts=3,
        maxiter=100,
    )
    obs = best["diagnostics"]["observables"]
    print("Best piecewise-constant sigma_z run")
    print(f"  T                  = {best['T']:.6f}")
    print(f"  <sigma_z>_NESS     = {obs['sz']:.6f}")
    print(f"  excited population = {obs['p_excited']:.6f}")
