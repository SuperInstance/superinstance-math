"""Information geometry: statistical manifolds, Fisher-Rao geometry, and related divergences."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence


class StatisticalManifold(ABC):
    """Base class for a parametric statistical manifold.

    Subclasses represent a family of probability distributions parameterised
    by a finite-dimensional coordinate vector.  The manifold is equipped with
    the Fisher information metric (the unique Riemannian metric invariant under
    sufficient statistics — Chentsov's theorem).
    """

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Evaluate the probability density at *x*."""

    @abstractmethod
    def score(self, x: float) -> list[float]:
        """Score function ∂ log p(x;θ) / ∂θ evaluated at *x*.

        Returns a list whose length equals the manifold dimension.
        """

    @abstractmethod
    def fisher_information(self) -> list[list[float]]:
        """Fisher information matrix I(θ) = E[score scoreᵀ]."""

    def kl_divergence_to(self, other: "StatisticalManifold", n_samples: int = 2000) -> float:
        """KL(self ‖ other) estimated by Monte Carlo over a representative range."""
        return kl_divergence(self, other, n_samples)

    def cross_entropy_to(self, other: "StatisticalManifold", n_samples: int = 2000) -> float:
        """H(self, other) = -E_self[log other]."""
        return cross_entropy(self, other, n_samples)


# ---------------------------------------------------------------------------
# Normal manifold — the workhorse of information geometry
# ---------------------------------------------------------------------------

class NormalManifold(StatisticalManifold):
    """Univariate normal family N(μ, σ²) as a 2-D statistical manifold.

    Parameters are (μ, σ) with σ > 0.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        if std <= 0:
            raise ValueError("std must be positive")
        self.mean = mean
        self.std = std

    @property
    def variance(self) -> float:
        return self.std ** 2

    # -- core density -------------------------------------------------------

    def pdf(self, x: float) -> float:
        z = (x - self.mean) / self.std
        return math.exp(-0.5 * z * z) / (self.std * math.sqrt(2 * math.pi))

    def log_pdf(self, x: float) -> float:
        z = (x - self.mean) / self.std
        return -0.5 * math.log(2 * math.pi) - math.log(self.std) - 0.5 * z * z

    # -- score function (partial derivatives of log-likelihood) ------------

    def score(self, x: float) -> list[float]:
        d_mu = (x - self.mean) / self.variance
        d_sigma = ((x - self.mean) ** 2 - self.variance) / (self.variance * self.std)
        return [d_mu, d_sigma]

    # -- Fisher information matrix -----------------------------------------

    def fisher_information(self) -> list[list[float]]:
        # I_μμ = 1/σ²,  I_μσ = 0,  I_σσ = 2/σ²
        s2 = self.variance
        return [
            [1.0 / s2, 0.0],
            [0.0, 2.0 / s2],
        ]

    # -- Fisher-Rao distance (closed-form for normals) --------------------

    def fisher_rao_distance(self, other: "NormalManifold") -> float:
        """Geodesic distance under the Fisher information metric.

        Uses the exact closed-form via the hyperbolic space representation.
        The normal manifold with Fisher metric is isometric to the hyperbolic
        plane, giving:
          d = 2·arccosh( 1 + (Δμ)²/(2σ₁σ₂) + ½(σ₁/σ₂ + σ₂/σ₁ - 2) )
        which simplifies when σ₁ = σ₂ to |Δμ|/(σ√2).
        """
        if self.std == other.std:
            return abs(self.mean - other.mean) / (self.std * math.sqrt(2))

        d_mu2 = (self.mean - other.mean) ** 2
        r = self.std / other.std + other.std / self.std  # σ₁/σ₂ + σ₂/σ₁
        arg = 1.0 + d_mu2 / (2.0 * self.std * other.std) + 0.5 * (r - 2.0)
        return 2.0 * math.acosh(max(arg, 1.0))

    # -- alpha-connections --------------------------------------------------

    def alpha_connection(self, alpha: float = 0.0) -> list[list[list[float]]]:
        """Christoffel symbols of the α-connection for the normal manifold.

        Returns Γ^k_ij as a 2×2×2 array (indexed [k][i][j]).
        For α = 0 this is the Levi-Civita (metric) connection.
        For α = ±1 these are the e-/m-connections (flat).
        """
        s = self.std
        s2 = s * s
        s3 = s2 * s
        # Γ^μ_μμ = 0, Γ^μ_μσ = 0, Γ^μ_σσ = 0
        # Γ^σ_μμ = -σ/2, Γ^σ_μσ = -1/(2σ)·(1+α), Γ^σ_σσ = -α/σ
        # (These are the α-connection Christoffel symbols for the (μ,σ) coordinates)
        G = [
            [[0.0, 0.0], [0.0, 0.0]],        # Γ^0 (k=μ)
            [[-s / 2.0, -(1 + alpha) / (2.0 * s)], [-(1 + alpha) / (2.0 * s), -alpha / s]],  # Γ^1 (k=σ)
        ]
        return G

    # -- natural gradient ---------------------------------------------------

    def natural_gradient(self, loss_grad: Sequence[float]) -> list[float]:
        """Compute the natural gradient: G⁻¹ ∇L, where G is the Fisher matrix."""
        # Inverse of diagonal Fisher: [σ², σ²/2]
        fim_inv_diag = [self.variance, self.variance / 2.0]
        return [g * s for g, s in zip(loss_grad, fim_inv_diag)]


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def kl_divergence(p: StatisticalManifold, q: StatisticalManifold, n_samples: int = 2000) -> float:
    """KL(p ‖ q) via numerical integration over a representative range."""
    # Works well for normals; for general manifolds, MC integration
    if isinstance(p, NormalManifold) and isinstance(q, NormalManifold):
        # Closed-form for normals
        return (
            math.log(q.std / p.std)
            + (p.variance + (p.mean - q.mean) ** 2) / (2.0 * q.variance)
            - 0.5
        )
    # Generic MC estimation
    return cross_entropy(p, q, n_samples) - cross_entropy(p, p, n_samples)


def cross_entropy(p: StatisticalManifold, q: StatisticalManifold, n_samples: int = 2000) -> float:
    """H(p, q) = -E_p[log q(x)]."""
    if isinstance(p, NormalManifold):
        lo, hi = p.mean - 6 * p.std, p.mean + 6 * p.std
    else:
        lo, hi = -10.0, 10.0
    n = max(n_samples, 100)
    dx = (hi - lo) / n
    total = 0.0
    for i in range(n):
        x = lo + (i + 0.5) * dx
        total += -q.log_pdf(x) * p.pdf(x) * dx
    return total


def chentsov_theorem() -> str:
    """State Chentsov's theorem: the Fisher information metric is the UNIQUE
    Riemannian metric (up to scaling) on a statistical manifold that is
    invariant under sufficient statistics transformations."""
    return (
        "Chentsov's theorem (1982): The Fisher information metric is the unique "
        "(up to a constant factor) Riemannian metric on the space of probability "
        "distributions that is invariant under Markov morphisms (i.e., sufficient "
        "statistics transformations). This provides the fundamental justification "
        "for using the Fisher metric in information geometry."
    )
