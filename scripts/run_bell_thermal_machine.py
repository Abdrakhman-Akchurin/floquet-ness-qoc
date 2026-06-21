"""Optimize Bell-state fidelity in the two-qubit thermal-machine model."""
from floquet_ness_qoc.bell_thermal_machine import (
    BellMachineParams,
    BellOptimizationConfig,
    optimize_bell_thermal_machine,
)
from floquet_ness_qoc.linalg_utils import concurrence, purity


if __name__ == "__main__":
    params = BellMachineParams(g=0.01, alpha=1.2, n1=1.0, n2=0.0, q=1.0)
    config = BellOptimizationConfig(
        target="psi_minus",
        m_order=3,
        n_slices=48,
        t_bounds=(50.0, 2500.0),
        delta_coeff_bound=0.2,
        gamma_coeff_bound=0.6,
        seed=1,
        n_restarts=3,
        maxiter=60,
    )
    best = optimize_bell_thermal_machine(params=params, config=config)
    rho = best["diagnostics"]["rho_star"]
    print("Best Bell thermal-machine run")
    print(f"  T             = {best['T']:.6f}")
    print(f"  Bell fidelity = {best['F']:.6f}")
    print(f"  purity        = {purity(rho):.6f}")
    print(f"  concurrence   = {concurrence(rho):.6f}")
