"""Optimal transport: discrete measures, Wasserstein-1D, and Sinkhorn algorithm."""

from __future__ import annotations

import math
from typing import Sequence


class DiscreteMeasure:
    """A discrete probability measure with weights and support points."""

    def __init__(self, weights: Sequence[float], support: Sequence[float]):
        if len(weights) != len(support):
            raise ValueError("weights and support must have the same length")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.weights = list(w / total for w in weights)
        self.support = list(support)
        self.n = len(weights)

    def __repr__(self) -> str:
        return f"DiscreteMeasure(n={self.n})"

    def mean(self) -> float:
        return sum(w * x for w, x in zip(self.weights, self.support))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        return sum(w for w, s in zip(self.weights, self.support) if s <= x)


def wasserstein_1d(a: DiscreteMeasure, b: DiscreteMeasure) -> float:
    """Wasserstein-1 distance between two 1-D discrete measures.

    Uses the closed-form solution:
        W₁(a, b) = ∫ |F_a(x) - F_b(x)| dx
    computed via the sorted quantile formulation.
    """
    # Expand into "empirical samples" approach via quantile function
    # W1 = sum of |quantile_a(t) - quantile_b(t)| * dt over uniform t
    # Equivalently: sort all atoms, integrate |CDF diff|

    # Merge sorted support points
    points = sorted(set(a.support) | set(b.support))
    if len(points) < 2:
        return 0.0

    # Add sentinel points
    lo = points[0] - 1.0
    hi = points[-1] + 1.0
    all_pts = [lo] + points + [hi]

    total = 0.0
    for i in range(len(all_pts) - 1):
        x_left = all_pts[i]
        x_right = all_pts[i + 1]
        mid = (x_left + x_right) / 2.0
        diff = abs(a.cdf(mid) - b.cdf(mid))
        total += diff * (x_right - x_left)

    return total


def sinkhorn(
    a_weights: Sequence[float],
    b_weights: Sequence[float],
    cost_matrix: Sequence[Sequence[float]],
    reg: float = 1.0,
    iterations: int = 100,
) -> tuple[list[list[float]], float]:
    """Sinkhorn algorithm for regularised optimal transport.

    Uses log-domain stabilisation for numerical stability.

    Args:
        a_weights: Source distribution weights (sum to 1).
        b_weights: Target distribution weights (sum to 1).
        cost_matrix: C[i][j] = cost of moving mass from i to j.
        reg: Entropic regularisation parameter (λ).
        iterations: Number of Sinkhorn iterations.

    Returns:
        (transport_plan, distance): The optimal transport plan T and the
        regularised transport cost Σ T_ij C_ij.
    """
    n = len(a_weights)
    m = len(b_weights)
    a = list(a_weights)
    b = list(b_weights)
    C = [list(row) for row in cost_matrix]

    if len(C) != n or any(len(row) != m for row in C):
        raise ValueError("cost_matrix must be n×m matching weight lengths")

    # Log-domain stabilised Sinkhorn
    # K_ij = exp(-C_ij / reg)
    # Work in log space: f_i, g_j dual potentials
    f = [0.0] * n
    g = [0.0] * m

    for _ in range(iterations):
        # Update f: f_i = -reg * logsumexp_j((-C_ij + g_j) / reg) + reg * log(a_i)
        for i in range(n):
            log_sum = _logsumexp([(-C[i][j] + g[j]) / reg for j in range(m)])
            f[i] = reg * math.log(max(a[i], 1e-300)) - reg * log_sum

        # Update g: g_j = -reg * logsumexp_i((-C_ij + f_i) / reg) + reg * log(b_j)
        for j in range(m):
            log_sum = _logsumexp([(-C[i][j] + f[i]) / reg for i in range(n)])
            g[j] = reg * math.log(max(b[j], 1e-300)) - reg * log_sum

    # Compute transport plan in log domain
    T = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            T[i][j] = math.exp((f[i] + g[j] - C[i][j]) / reg)

    # Distance = sum T_ij * C_ij
    distance = sum(T[i][j] * C[i][j] for i in range(n) for j in range(m))

    return T, distance


def _logsumexp(xs: list[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(x - m) for x in xs))
