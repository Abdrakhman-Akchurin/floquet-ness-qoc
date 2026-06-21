r"""Linear-algebra utilities for Liouville-space open-system simulations.

Conventions
-----------
Density matrices are vectorized by column-stacking/Fortran order:

    vec([[a, b], [c, d]]) = [a, c, b, d]^T.

With this convention, vec(A rho B) = (B^T \otimes A) vec(rho).
"""
from __future__ import annotations

import numpy as np
from numpy import kron


Array = np.ndarray


def vec(matrix: Array) -> Array:
    """Column-stack a square matrix into a vector."""
    return np.asarray(matrix, dtype=complex).reshape(-1, order="F")


def mat(vector: Array, d: int) -> Array:
    """Inverse of :func:`vec` for a d x d density matrix."""
    return np.asarray(vector, dtype=complex).reshape((d, d), order="F")


def trace_vector(d: int) -> Array:
    """Vector c such that c.T @ vec(rho) = Tr(rho)."""
    return vec(np.eye(d, dtype=complex))


def left_right_super(left: Array, right: Array) -> Array:
    """Superoperator for rho -> left @ rho @ right."""
    return kron(np.asarray(right).T, np.asarray(left))


def commutator_super(hamiltonian: Array) -> Array:
    """Liouville-space matrix for -i[H, rho]."""
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    d = hamiltonian.shape[0]
    ident = np.eye(d, dtype=complex)
    return -1j * (kron(ident, hamiltonian) - kron(hamiltonian.T, ident))


def dissipator_super(jump: Array) -> Array:
    """Lindblad dissipator D[L] rho = L rho L† - 1/2{L†L, rho}."""
    jump = np.asarray(jump, dtype=complex)
    d = jump.shape[0]
    ident = np.eye(d, dtype=complex)
    ldag_l = jump.conj().T @ jump
    return (
        kron(jump.conj(), jump)
        - 0.5 * kron(ident, ldag_l)
        - 0.5 * kron(ldag_l.T, ident)
    )


def fixed_point_from_map(phi: Array, d: int, symmetrize: bool = True) -> tuple[Array, Array]:
    """Return the normalized fixed point of a one-period map.

    Solves rho_* = Phi rho_* with Tr(rho_*) = 1 using the augmented system

        [I - Phi, c] [r_*]   [0]
        [c^T,     0] [lambda] = [1].

    Returns
    -------
    r_star:
        Vectorized fixed-point density matrix.
    rho_star:
        d x d density matrix.
    """
    n = phi.shape[0]
    c = trace_vector(d).reshape(-1, 1)
    lhs = np.block(
        [
            [np.eye(n, dtype=complex) - phi, c],
            [c.T, np.zeros((1, 1), dtype=complex)],
        ]
    )
    rhs = np.zeros(n + 1, dtype=complex)
    rhs[-1] = 1.0
    sol = np.linalg.solve(lhs, rhs)
    r_star = sol[:n]
    rho_star = mat(r_star, d=d)
    if symmetrize:
        rho_star = 0.5 * (rho_star + rho_star.conj().T)
    rho_star = rho_star / np.trace(rho_star)
    return vec(rho_star), rho_star


def expectation(rho: Array, observable: Array) -> float:
    """Real expectation value Tr(observable rho)."""
    return float(np.real(np.trace(np.asarray(observable, dtype=complex) @ rho)))


def purity(rho: Array) -> float:
    """Return Tr(rho^2)."""
    return float(np.real(np.trace(rho @ rho)))


def random_density(d: int, rng: np.random.Generator) -> Array:
    """Random physical density matrix from the Hilbert-Schmidt ensemble."""
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def partial_trace_two_qubit(rho: Array, keep: int) -> Array:
    """Partial trace of a two-qubit density matrix.

    Parameters
    ----------
    keep:
        ``1`` returns rho_1; ``2`` returns rho_2.
    """
    rho = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    if keep == 1:
        return np.einsum("abcb->ac", rho)
    if keep == 2:
        return np.einsum("abac->bc", rho)
    raise ValueError("keep must be 1 or 2")


def concurrence(rho: Array) -> float:
    """Wootters concurrence for a two-qubit density matrix."""
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    yy = kron(sy, sy)
    rho_tilde = yy @ rho.conj() @ yy
    evals = np.real(np.linalg.eigvals(rho @ rho_tilde))
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    roots = np.sqrt(evals)
    return max(0.0, float(roots[0] - roots[1] - roots[2] - roots[3]))
