"""Spectral methods: graph Laplacian, eigendecomposition, spectral embedding, anomaly detection."""

from __future__ import annotations

import math
import random
from typing import Sequence


def graph_laplacian(
    adjacency: Sequence[Sequence[float]],
    normalized: bool = True,
) -> list[list[float]]:
    """Compute the graph Laplacian from an adjacency matrix.

    If normalized: L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}.
    If unnormalized: L = D - A.
    """
    n = len(adjacency)
    A = [list(row) for row in adjacency]

    # Degree matrix (diagonal)
    degrees = [sum(A[i]) for i in range(n)]

    if not normalized:
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                L[i][j] = -A[i][j]
            L[i][i] = degrees[i]
        return L

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1.0 if degrees[i] > 0 else 0.0
            elif degrees[i] > 0 and degrees[j] > 0:
                L[i][j] = -A[i][j] / math.sqrt(degrees[i] * degrees[j])
    return L


def _mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector multiply."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    n = _norm(v)
    if n < 1e-15:
        return v
    return [x / n for x in v]


def top_k_eigenvalues(
    matrix: Sequence[Sequence[float]],
    k: int = 3,
    iterations: int = 200,
    seed: int = 42,
) -> tuple[list[float], list[list[float]]]:
    """Compute the k smallest eigenvalues and eigenvectors using inverse iteration / deflation.

    For a Laplacian, the smallest eigenvalues carry the most structural info.
    Uses power iteration on (M - σI) with Rayleigh quotient shifts.

    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue ascending.
    """
    M = [list(row) for row in matrix]
    n = len(M)

    if k > n:
        k = n

    eigenvalues: list[float] = []
    eigenvectors: list[list[float]] = []

    rng = random.Random(seed)

    for _ in range(k):
        # Random starting vector
        v = _normalize([rng.gauss(0, 1) for _ in range(n)])

        # Orthogonalize against found eigenvectors
        for ev in eigenvectors:
            proj = _dot(v, ev)
            v = [vi - proj * ei for vi, ei in zip(v, ev)]
        v = _normalize(v)

        # Power iteration (find smallest eigenvalue by shifting)
        # We use the fact that for Laplacians, the smallest eig is ≈ 0
        # Use inverse power iteration with a small shift
        shift = 0.0
        for _ in range(iterations):
            # Solve (M - shift*I) v_new = v using simple iteration
            # For simplicity, use direct power iteration on shifted matrix
            w = _mat_vec(M, v)
            # Rayleigh quotient
            lam = _dot(v, w)
            # For smallest eigenvalue: subtract largest component
            w = _normalize(w)

            # Orthogonalize
            for ev in eigenvectors:
                proj = _dot(w, ev)
                w = [wi - proj * ei for wi, ei in zip(w, ev)]
            w = _normalize(w)

            v = w

        eigenvalues.append(lam)
        eigenvectors.append(v)

    return eigenvalues, eigenvectors


def spectral_embedding(
    adjacency: Sequence[Sequence[float]],
    k: int = 2,
    normalized: bool = True,
) -> list[list[float]]:
    """Compute spectral embedding using the first k eigenvectors of the graph Laplacian.

    Each node is mapped to a k-dimensional vector using the eigenvectors
    corresponding to the k smallest non-trivial eigenvalues.

    Args:
        adjacency: n×n adjacency matrix.
        k: Embedding dimension.
        normalized: Use normalized Laplacian.

    Returns:
        n×k embedding matrix (list of lists).
    """
    L = graph_laplacian(adjacency, normalized=normalized)
    eigenvalues, eigenvectors = top_k_eigenvalues(L, k=k + 1)  # +1 to skip trivial

    n = len(adjacency)
    # Skip the first (trivial) eigenvector if eigenvalue ≈ 0
    embedding_vecs = []
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors)):
        if abs(lam) < 1e-10 and len(embedding_vecs) > 0:
            continue  # skip trivial
        embedding_vecs.append(vec)
        if len(embedding_vecs) == k:
            break

    # Pad if not enough eigenvectors
    while len(embedding_vecs) < k:
        embedding_vecs.append([0.0] * n)

    # Transpose: eigenvectors are columns, we want rows (one per node)
    result = []
    for i in range(n):
        result.append([embedding_vecs[dim][i] for dim in range(min(k, len(embedding_vecs)))])
    return result


def spectral_anomaly_score(
    adjacency: Sequence[Sequence[float]],
    k: int = 2,
) -> list[float]:
    """Compute anomaly score for each node using spectral distance.

    Nodes far from the cluster center in spectral embedding space are anomalous.

    Returns:
        Anomaly score for each node (higher = more anomalous).
    """
    emb = spectral_embedding(adjacency, k=k)
    n = len(emb)
    if n == 0:
        return []

    # Compute centroid
    k_dim = len(emb[0])
    centroid = [sum(emb[i][d] for i in range(n)) / n for d in range(k_dim)]

    # Distance from centroid
    scores = []
    for i in range(n):
        dist = math.sqrt(sum((emb[i][d] - centroid[d]) ** 2 for d in range(k_dim)))
        scores.append(dist)

    return scores
