"""Optional numpy-accelerated backends. Used automatically when numpy is available."""

from __future__ import annotations

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def graph_laplacian_np(
    adjacency,
    normalized: bool = True,
):
    """Numpy-accelerated graph Laplacian.

    Returns None if numpy is not available.
    """
    if not HAS_NUMPY:
        return None
    A = np.array(adjacency, dtype=np.float64)
    degrees = A.sum(axis=1)
    n = A.shape[0]

    if not normalized:
        L = np.diag(degrees) - A
        return L.tolist()

    # Normalized: L = I - D^{-1/2} A D^{-1/2}
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(np.maximum(degrees, 1e-15)), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L_norm.tolist()


def top_k_eigenvalues_np(
    matrix,
    k: int = 3,
):
    """Numpy-accelerated eigendecomposition.

    Returns (eigenvalues, eigenvectors) as Python lists, or (None, None)
    if numpy is not available.

    eigenvectors is a list of k vectors (each a list of floats),
    sorted by eigenvalue ascending — matching the pure-Python API.
    """
    if not HAS_NUMPY:
        return None, None
    M = np.array(matrix, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort ascending, take first k
    idx = np.argsort(eigenvalues)[:k]
    evals = eigenvalues[idx].tolist()
    # Each eigenvector is a column → transpose to get rows
    evecs = eigenvectors[:, idx].T.tolist()
    return evals, evecs
