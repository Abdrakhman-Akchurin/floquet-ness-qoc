# Code Notes

### 1. Core math utilities

File:

```text
src/floquet_ness_qoc/linalg_utils.py
```

Contains:

- vectorization / matrix reconstruction;
- commutator and Lindblad dissipator superoperators;
- fixed-point NESS solve;
- expectations, purity, concurrence, partial trace.

### 2. Control parameterizations

File:

```text
src/floquet_ness_qoc/controls.py
```

Contains:

- single-qubit Fourier controls;
- Bell-machine detuning control;
- Bell-machine nonnegative gamma control.

### 3. Physical systems

Files:

```text
src/floquet_ness_qoc/single_qubit.py
src/floquet_ness_qoc/bell_thermal_machine.py
```

These contain the actual physics and optimization routines.

### 4. Short runnable scripts

Files:

```text
scripts/run_single_sigma_z_fourier.py
scripts/run_single_sigma_x_delta_fourier.py
scripts/run_single_sigma_z_piecewise.py
scripts/run_bell_thermal_machine.py
```

Each script answers one question and prints the main result.

