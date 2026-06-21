"""Control parameterizations used by the optimization scripts."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def midpoint_grid(n_slices: int) -> Array:
    """Midpoints s_j=(j+1/2)/N in normalized time s=t/T."""
    return (np.arange(n_slices, dtype=float) + 0.5) / float(n_slices)


def fourier_interleaved(theta: Array, s: Array, include_dc: bool = False) -> Array:
    """Evaluate an interleaved Fourier series on normalized time points.

    If ``include_dc=False`` then ``theta`` is

        [a1, b1, a2, b2, ..., aM, bM]

    and the returned signal is

        sum_m a_m cos(2 pi m s) + b_m sin(2 pi m s).

    If ``include_dc=True`` then ``theta`` is

        [a0, a1, b1, a2, b2, ..., aM, bM].
    """
    theta = np.asarray(theta, dtype=float)
    s = np.asarray(s, dtype=float)

    offset = 1 if include_dc else 0
    if (len(theta) - offset) % 2 != 0:
        raise ValueError("Fourier coefficient vector has the wrong length.")

    out = np.zeros_like(s, dtype=float)
    if include_dc:
        out += theta[0]

    m_count = (len(theta) - offset) // 2
    for m in range(1, m_count + 1):
        a = theta[offset + 2 * (m - 1)]
        b = theta[offset + 2 * (m - 1) + 1]
        out += a * np.cos(2 * np.pi * m * s) + b * np.sin(2 * np.pi * m * s)
    return out


def fourier_design_interleaved(s: Array, m_order: int, include_dc: bool = False) -> Array:
    """Design matrix consistent with :func:`fourier_interleaved`."""
    s = np.asarray(s, dtype=float)
    cols = []
    if include_dc:
        cols.append(np.ones_like(s))
    for m in range(1, m_order + 1):
        cols.append(np.cos(2 * np.pi * m * s))
        cols.append(np.sin(2 * np.pi * m * s))
    return np.column_stack(cols)


def bell_delta_from_theta(theta_delta: Array, s: Array) -> Array:
    """Bell-machine detuning δ(s)=sum_m a_m sin(2πms)."""
    theta_delta = np.asarray(theta_delta, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s, dtype=float)
    for m, a_m in enumerate(theta_delta, start=1):
        out += a_m * np.sin(2 * np.pi * m * s)
    return out


def bell_gamma_from_theta(theta_gamma: Array, s: Array) -> Array:
    """Nonnegative Bell-machine coupling/rate profile.

    gamma(s) = sin^2(pi s) * (b0 + sum_{m=1}^{M-1} b_m cos(2πms))^2.

    This enforces gamma(s)>=0 and gamma(0)=gamma(1)=0.
    """
    theta_gamma = np.asarray(theta_gamma, dtype=float)
    s = np.asarray(s, dtype=float)
    envelope = np.sin(np.pi * s) ** 2
    f = theta_gamma[0] * np.ones_like(s, dtype=float)
    for m, b_m in enumerate(theta_gamma[1:], start=1):
        f += b_m * np.cos(2 * np.pi * m * s)
    return envelope * f**2
