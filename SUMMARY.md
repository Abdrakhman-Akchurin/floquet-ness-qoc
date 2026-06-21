# Project Summary

## One-sentence summary

This project uses Fourier-parameterized periodic controls and `scipy.optimize` to engineer stroboscopic nonequilibrium steady states of dissipative quantum systems, including single-qubit target states and high-fidelity Bell-state NESSs in a two-qubit thermal machine.

## Scientific motivation

Open quantum systems normally lose coherence because they exchange energy and information with their environment. This project takes the opposite perspective: dissipation can be used as a resource. By periodically driving the system, the dissipative dynamics can converge to a useful stroboscopic steady state.

A Floquet NESS satisfies

$$
\rho_* = \Phi(T)\rho_*,
$$

where $\Phi(T)$ is the one-period open-system propagator. The state may move during the period, but it returns to the same density matrix at stroboscopic times $t=nT$.

## Actual implementation

The current code does not optimize arbitrary waveforms directly. It reduces each waveform to a small number of Fourier coefficients. Then it optimizes those coefficients and the period.

The computational loop is:

```text
Fourier coefficients + period
        ↓
periodic controls Jx(t), Δ(t), δ(t), γ(t)
        ↓
time-dependent Liouvillian L(t)
        ↓
one-period Floquet map Φ
        ↓
fixed-point solve for ρ*
        ↓
objective value
        ↓
scipy.optimize update
```

## Single-qubit part

The single-qubit system is a driven dissipative two-level system with relaxation and dephasing. The main examples are:

1. Optimize $J_x(t)$ to minimize $\langle\sigma_z\rangle$, which maximizes excited-state population.
2. Optimize $J_x(t)$ and $\Delta(t)$ to maximize $\langle\sigma_x\rangle$.
3. Compare smooth Fourier controls with piecewise-constant controls.

Representative results:

- Fourier $J_x(t)$: $\langle\sigma_z\rangle\approx -0.791$, excited-state population $\approx 0.896$.
- Fourier $J_x(t),\Delta(t)$: $\langle\sigma_x\rangle\approx 0.936$.
- Piecewise $J_x(t)$: excited-state population $\approx 0.938$.

## Two-qubit Bell-state thermal machine

The two-qubit case is more than a larger Hilbert space. It is a thermal-machine problem. The qubits are coupled to different baths, producing nonequilibrium dissipative dynamics. The goal is to use periodic modulation to stabilize an entangled Bell-state NESS.

The code optimizes:

- detuning path $\delta(t)$,
- dissipation/coupling path $\gamma(t)$,
- Floquet period $T$.

Optimization is needed because the target state is determined by the full period-averaged dissipative dynamics. A hand-designed loop in parameter space may produce entanglement, but the best Bell fidelity depends sensitively on the waveform, the period, and physical constraints such as $\gamma(t)\ge 0$.

Representative result:

- Bell-state fidelity $\approx 0.998$.
- Concurrence $\approx 0.996$.

## Future work

The most important future direction is experimental implementation using circuit QED. The theoretical optimized controls would need to be converted into experimentally feasible microwave/control waveforms. The next stage should include hardware constraints such as finite bandwidth, maximum drive amplitude, calibration error, dissipation-rate tunability, and noise robustness.
