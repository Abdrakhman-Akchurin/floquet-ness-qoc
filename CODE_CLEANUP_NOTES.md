# Code Cleanup Notes

## Main problem in the original files

The original project files were scientifically useful but hard to read because the same notebooks mixed:

- physics definitions;
- helper functions;
- optimization loops;
- result saving;
- diagnostic calculations;
- APS-ready plotting;
- exploratory conceptual figures.

That structure makes it hard to answer basic questions like “where is the model?”, “where is the objective?”, and “what do I run?”.

## New organization

The cleaned version separates the project into four levels.

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

## Important scientific cleanup

### Fourier coefficient ordering

The cleaned code uses one consistent ordering:

```text
[a1, b1, a2, b2, ..., aM, bM]
```

This avoids ambiguity between the optimizer and plotting code.

### The Bell folder name

The original folder was named `bell_floquent_final`. The cleaned code uses the clearer name:

```text
bell_thermal_machine.py
```

The word should be **Floquet**, not “Floquent.” The thermal-machine name is also more descriptive of the physics.

### Optimization method

The code summary should say:

```text
The implementation uses scipy.optimize.minimize, mainly L-BFGS-B, to optimize Fourier coefficients and the period.
```

Do **not** claim that the cleaned code implements analytic adjoint gradients. The reports discuss that as an algorithmic direction, but the practical implementation here is based on SciPy optimization of waveform parameters.

## What to keep out of the main code

These things should not be mixed into optimization scripts:

- presentation-only plot styling;
- synthetic explanatory figures;
- old parameter scans;
- failed restarts;
- duplicated result files;
- notebook checkpoint folders.

They can be kept in `legacy_notebooks/` or `docs/`, but they should not be the main GitHub entry point.
