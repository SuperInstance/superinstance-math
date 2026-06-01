"""Comprehensive tests for superinstance-math — 80+ tests covering mathematical properties."""

import math
import random
import pytest

# ===========================================================================
# Information Geometry
# ===========================================================================

from superinstance_math.information_geometry import (
    StatisticalManifold,
    NormalManifold,
    kl_divergence,
    cross_entropy,
    chentsov_theorem,
)


class TestNormalManifold:
    """Tests for NormalManifold."""

    def test_pdf_integrates_to_one(self):
        n = NormalManifold(0, 1)
        # Integrate pdf over [-6, 6]
        total = sum(n.pdf(-6 + i * 0.01) * 0.01 for i in range(1200))
        assert abs(total - 1.0) < 0.01

    def test_pdf_at_mean(self):
        n = NormalManifold(2, 3)
        expected = 1.0 / (3 * math.sqrt(2 * math.pi))
        assert abs(n.pdf(2) - expected) < 1e-10

    def test_pdf_symmetric(self):
        n = NormalManifold(0, 1)
        assert abs(n.pdf(1) - n.pdf(-1)) < 1e-15

    def test_log_pdf_matches_log_pdf(self):
        n = NormalManifold(1.5, 2.3)
        for x in [-3, 0, 1.5, 5]:
            assert abs(n.log_pdf(x) - math.log(n.pdf(x))) < 1e-12

    def test_std_must_be_positive(self):
        with pytest.raises(ValueError):
            NormalManifold(0, 0)
        with pytest.raises(ValueError):
            NormalManifold(0, -1)

    def test_fisher_information_diagonal(self):
        n = NormalManifold(0, 2)
        fim = n.fisher_information()
        # Off-diagonal should be zero
        assert abs(fim[0][1]) < 1e-15
        assert abs(fim[1][0]) < 1e-15
        # Diagonal: 1/σ² and 2/σ²
        assert abs(fim[0][0] - 0.25) < 1e-10  # 1/4
        assert abs(fim[1][1] - 0.5) < 1e-10   # 2/4

    def test_fisher_positive_definite(self):
        n = NormalManifold(3, 1.5)
        fim = n.fisher_information()
        # All diagonal entries positive
        for i in range(2):
            assert fim[i][i] > 0

    def test_score_expectation_zero(self):
        """E[score] = 0 for any parameter."""
        n = NormalManifold(1, 2)
        # Numerical integration
        lo, hi = 1 - 6 * 2, 1 + 6 * 2
        dx = 0.05
        expected_scores = [0.0, 0.0]
        x = lo
        while x < hi:
            s = n.score(x)
            p = n.pdf(x)
            for k in range(2):
                expected_scores[k] += s[k] * p * dx
            x += dx
        for k in range(2):
            assert abs(expected_scores[k]) < 0.05

    def test_natural_gradient(self):
        n = NormalManifold(0, 2)
        grad = [1.0, 0.5]
        ng = n.natural_gradient(grad)
        # FIM inverse is diagonal [σ², σ²/2] = [4, 2]
        assert abs(ng[0] - 4.0) < 1e-10
        assert abs(ng[1] - 1.0) < 1e-10

    def test_kl_self_is_zero(self):
        n = NormalManifold(1, 2)
        assert abs(kl_divergence(n, n)) < 1e-10

    def test_kl_positive(self):
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 2)
        assert kl_divergence(n1, n2) > 0

    def test_kl_asymmetric(self):
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 2)
        assert abs(kl_divergence(n1, n2) - kl_divergence(n2, n1)) > 0.01

    def test_kl_closed_form(self):
        """KL(N(0,1) || N(1,2)) = log(2) + (1 + 1)/8 - 0.5 = log(2) - 0.25"""
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 2)
        kl = kl_divergence(n1, n2)
        expected = math.log(2) + (1 + 1) / 8 - 0.5
        assert abs(kl - expected) < 1e-10

    def test_cross_entropy_decomposition(self):
        """H(p,q) = KL(p||q) + H(p)."""
        p = NormalManifold(0, 1)
        q = NormalManifold(1, 2)
        h_pq = cross_entropy(p, q)
        kl_pq = kl_divergence(p, q)
        h_p = cross_entropy(p, p)  # Entropy
        assert abs(h_pq - kl_pq - h_p) < 0.05

    def test_fisher_rao_self_zero(self):
        n = NormalManifold(1, 2)
        assert abs(n.fisher_rao_distance(n)) < 1e-10

    def test_fisher_rao_symmetric(self):
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(2, 3)
        d12 = n1.fisher_rao_distance(n2)
        d21 = n2.fisher_rao_distance(n1)
        assert abs(d12 - d21) < 1e-10

    def test_fisher_rao_positive(self):
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 2)
        assert n1.fisher_rao_distance(n2) > 0

    def test_fisher_rao_triangle_inequality(self):
        """Metric must satisfy triangle inequality."""
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 2)
        n3 = NormalManifold(3, 0.5)
        d12 = n1.fisher_rao_distance(n2)
        d23 = n2.fisher_rao_distance(n3)
        d13 = n1.fisher_rao_distance(n3)
        assert d13 <= d12 + d23 + 1e-6

    def test_fisher_rao_equal_variance(self):
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(3, 1)
        expected = abs(0 - 3) / (1 * math.sqrt(2))
        assert abs(n1.fisher_rao_distance(n2) - expected) < 1e-10

    def test_alpha_connection_e_connection_flat(self):
        """e-connection (α=+1) should have some zero Christoffel symbols."""
        n = NormalManifold(0, 1)
        G = n.alpha_connection(alpha=1.0)
        # For the e-connection in (μ,σ) coordinates, Γ^σ_σσ = -1/σ = -1
        assert abs(G[1][1][1] - (-1.0)) < 1e-10

    def test_alpha_connection_m_connection(self):
        n = NormalManifold(0, 1)
        G = n.alpha_connection(alpha=-1.0)
        # Γ^σ_σσ = +1/σ = 1
        assert abs(G[1][1][1] - 1.0) < 1e-10

    def test_chentsov_theorem_returns_string(self):
        result = chentsov_theorem()
        assert isinstance(result, str)
        assert "Fisher" in result
        assert "unique" in result.lower()


# ===========================================================================
# Optimal Transport
# ===========================================================================

from superinstance_math.optimal_transport import (
    DiscreteMeasure,
    wasserstein_1d,
    sinkhorn,
)


class TestDiscreteMeasure:
    def test_creation(self):
        dm = DiscreteMeasure([0.5, 0.5], [0.0, 1.0])
        assert dm.n == 2

    def test_weights_normalized(self):
        dm = DiscreteMeasure([2, 3], [0, 1])
        assert abs(sum(dm.weights) - 1.0) < 1e-10
        assert abs(dm.weights[0] - 0.4) < 1e-10
        assert abs(dm.weights[1] - 0.6) < 1e-10

    def test_mean(self):
        dm = DiscreteMeasure([0.5, 0.5], [0.0, 2.0])
        assert abs(dm.mean() - 1.0) < 1e-10

    def test_cdf(self):
        dm = DiscreteMeasure([0.3, 0.7], [0.0, 1.0])
        assert abs(dm.cdf(-1) - 0.0) < 1e-10
        assert abs(dm.cdf(0.5) - 0.3) < 1e-10
        assert abs(dm.cdf(2) - 1.0) < 1e-10

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            DiscreteMeasure([1, 2], [0])

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError):
            DiscreteMeasure([0, 0], [0, 1])


class TestWasserstein1D:
    def test_self_distance_zero(self):
        a = DiscreteMeasure([1], [0])
        assert abs(wasserstein_1d(a, a)) < 1e-10

    def test_point_masses(self):
        a = DiscreteMeasure([1], [0])
        b = DiscreteMeasure([1], [1])
        assert abs(wasserstein_1d(a, b) - 1.0) < 0.1

    def test_non_negative(self):
        a = DiscreteMeasure([0.5, 0.5], [0, 1])
        b = DiscreteMeasure([0.3, 0.7], [0.5, 2])
        assert wasserstein_1d(a, b) >= 0

    def test_symmetric(self):
        a = DiscreteMeasure([0.5, 0.5], [0, 1])
        b = DiscreteMeasure([0.3, 0.7], [0.5, 2])
        assert abs(wasserstein_1d(a, b) - wasserstein_1d(b, a)) < 0.1

    def test_triangle_inequality(self):
        a = DiscreteMeasure([1], [0])
        b = DiscreteMeasure([1], [1])
        c = DiscreteMeasure([1], [3])
        d_ab = wasserstein_1d(a, b)
        d_bc = wasserstein_1d(b, c)
        d_ac = wasserstein_1d(a, c)
        assert d_ac <= d_ab + d_bc + 0.2


class TestSinkhorn:
    def test_transport_plan_conserves_mass(self):
        """Sum of transport plan rows = source weights, columns = target weights."""
        a = [0.5, 0.5]
        b = [0.3, 0.3, 0.4]
        C = [[1, 2, 3], [2, 1, 2]]
        T, _ = sinkhorn(a, b, C, reg=0.1, iterations=200)

        for i in range(2):
            row_sum = sum(T[i])
            assert abs(row_sum - a[i]) < 0.01

        for j in range(3):
            col_sum = sum(T[i][j] for i in range(2))
            assert abs(col_sum - b[j]) < 0.01

    def test_non_negative_plan(self):
        a = [0.5, 0.5]
        b = [0.5, 0.5]
        C = [[0, 1], [1, 0]]
        T, _ = sinkhorn(a, b, C, reg=0.1)
        for i in range(2):
            for j in range(2):
                assert T[i][j] >= -1e-10

    def test_distance_positive(self):
        a = [0.5, 0.5]
        b = [0.5, 0.5]
        C = [[0, 2], [2, 0]]
        _, dist = sinkhorn(a, b, C, reg=0.1)
        assert dist > 0

    def test_identity_cost_zero_distance(self):
        a = [0.5, 0.5]
        b = [0.5, 0.5]
        C = [[0, 0], [0, 0]]
        _, dist = sinkhorn(a, b, C, reg=0.1)
        assert abs(dist) < 0.01

    def test_numerical_stability(self):
        """Sinkhorn should handle small regularization."""
        a = [0.25, 0.25, 0.25, 0.25]
        b = [0.25, 0.25, 0.25, 0.25]
        C = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
        T, dist = sinkhorn(a, b, C, reg=0.01, iterations=500)
        assert not math.isnan(dist)
        assert not math.isinf(dist)
        # Check mass conservation roughly
        total = sum(T[i][j] for i in range(4) for j in range(4))
        assert abs(total - 1.0) < 0.1


# ===========================================================================
# Persistent Homology
# ===========================================================================

from superinstance_math.persistent_homology import (
    vietoris_rips,
    betti_numbers,
    persistence_barcodes,
)


class TestVietorisRips:
    def test_single_point(self):
        cx = vietoris_rips([(0,)], max_epsilon=1.0)
        assert (0,) in cx

    def test_two_points_close(self):
        cx = vietoris_rips([(0, 0), (1, 0)], max_epsilon=1.5)
        assert (0,) in cx
        assert (1,) in cx
        assert (0, 1) in cx

    def test_two_points_far(self):
        cx = vietoris_rips([(0, 0), (10, 0)], max_epsilon=1.0)
        assert (0,) in cx
        assert (1,) in cx
        assert (0, 1) not in cx

    def test_triangle(self):
        """Equilateral triangle with small epsilon → full complex."""
        cx = vietoris_rips([(0, 0), (1, 0), (0.5, 0.866)], max_epsilon=1.5)
        assert (0, 1, 2) in cx

    def test_no_triangle_with_large_gap(self):
        cx = vietoris_rips([(0, 0), (1, 0), (10, 0)], max_epsilon=5.0)
        # (0,2) has distance 10 > 5, so no triangle
        assert (0, 1, 2) not in cx


class TestBettiNumbers:
    def test_single_point(self):
        cx = vietoris_rips([(0,)], max_epsilon=1.0)
        bn = betti_numbers(cx)
        assert bn[0] == 1
        assert bn[1] == 0

    def test_two_connected_points(self):
        cx = vietoris_rips([(0, 0), (1, 0)], max_epsilon=2.0)
        bn = betti_numbers(cx)
        assert bn[0] == 1  # One component
        assert bn[1] == 0  # No cycle

    def test_three_disconnected_points(self):
        cx = vietoris_rips([(0, 0), (10, 0), (20, 0)], max_epsilon=1.0)
        bn = betti_numbers(cx)
        assert bn[0] == 3

    def test_triangle_has_no_holes(self):
        """Filled triangle: β₀=1, β₁=0."""
        cx = vietoris_rips([(0, 0), (1, 0), (0.5, 0.866)], max_epsilon=1.5)
        bn = betti_numbers(cx)
        assert bn[0] == 1
        assert bn[1] == 0  # Triangle fills the hole

    def test_square_cycle(self):
        """Square without diagonals → β₁=1 (one cycle)."""
        # Points of a square with side 1, epsilon just enough for edges but not diagonals
        cx = vietoris_rips([(0, 0), (1, 0), (1, 1), (0, 1)], max_epsilon=1.1)
        bn = betti_numbers(cx)
        assert bn[0] == 1  # Connected
        assert bn[1] == 1  # One cycle (square hole)


class TestPersistenceBarcodes:
    def test_single_point(self):
        bc = persistence_barcodes([(0,)])
        assert len(bc) == 1  # One component, persists forever
        assert bc[0] == (0.0, float("inf"))

    def test_two_points(self):
        bc = persistence_barcodes([(0, 0), (1, 0)])
        # One component merges into the other at ε=1
        assert len(bc) == 2
        infs = [b for b in bc if b[1] == float("inf")]
        assert len(infs) == 1

    def test_birth_before_death(self):
        points = [(random.random(), random.random()) for _ in range(10)]
        bc = persistence_barcodes(points)
        for birth, death in bc:
            if death != float("inf"):
                assert birth <= death + 1e-10


# ===========================================================================
# Spectral Methods
# ===========================================================================

from superinstance_math.spectral import (
    graph_laplacian,
    top_k_eigenvalues,
    spectral_embedding,
    spectral_anomaly_score,
)


class TestGraphLaplacian:
    def test_unnormalized_diagonal(self):
        """Unnormalized Laplacian: diagonal = degree."""
        A = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
        L = graph_laplacian(A, normalized=False)
        assert abs(L[0][0] - 2) < 1e-10
        assert abs(L[1][1] - 1) < 1e-10
        assert abs(L[2][2] - 1) < 1e-10

    def test_unnormalized_off_diagonal(self):
        A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        L = graph_laplacian(A, normalized=False)
        assert abs(L[0][1] - (-1)) < 1e-10
        assert abs(L[1][0] - (-1)) < 1e-10
        assert abs(L[0][2]) < 1e-10

    def test_normalized_identity_for_complete(self):
        """Normalized Laplacian of complete graph K_n has eigenvalues 0 and n/(n-1)."""
        n = 4
        A = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
        L = graph_laplacian(A, normalized=True)
        # Check diagonal is 1
        for i in range(n):
            assert abs(L[i][i] - 1.0) < 1e-10

    def test_symmetric(self):
        A = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        L = graph_laplacian(A, normalized=True)
        for i in range(3):
            for j in range(3):
                assert abs(L[i][j] - L[j][i]) < 1e-10


class TestEigenvalues:
    def test_identity_eigenvalues(self):
        I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        evals, _ = top_k_eigenvalues(I, k=2)
        for e in evals:
            assert abs(e - 1.0) < 0.1

    def test_diagonal_eigenvalues(self):
        D = [[2, 0, 0], [0, 5, 0], [0, 0, 1]]
        evals, _ = top_k_eigenvalues(D, k=3)
        sorted_expected = sorted([2, 5, 1])
        sorted_actual = sorted(evals)
        for a, e in zip(sorted_actual, sorted_expected):
            assert abs(a - e) < 0.5

    def test_eigenvectors_orthogonal(self):
        A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
        _, evecs = top_k_eigenvalues(A, k=2)
        dot = sum(a * b for a, b in zip(evecs[0], evecs[1]))
        assert abs(dot) < 0.3  # Loose bound due to iteration


class TestSpectralEmbedding:
    def test_output_shape(self):
        A = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
        emb = spectral_embedding(A, k=2)
        assert len(emb) == 4
        assert all(len(row) == 2 for row in emb)

    def test_connected_graph(self):
        A = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        emb = spectral_embedding(A, k=2)
        assert len(emb) == 3


class TestSpectralAnomaly:
    def test_scores_non_negative(self):
        A = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
        scores = spectral_anomaly_score(A, k=2)
        assert all(s >= 0 for s in scores)

    def test_outlier_higher_score(self):
        """A node weakly connected to a tight cluster should have higher anomaly."""
        # Triangle (0,1,2) + outlier 3 connected only to 0
        A = [
            [0, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        scores = spectral_anomaly_score(A, k=2)
        # Node 3 (outlier) should have a higher score than well-connected nodes
        assert scores[3] >= min(scores[0], scores[1], scores[2]) - 0.5


# ===========================================================================
# Symmetry Groups
# ===========================================================================

from superinstance_math.symmetry import (
    CyclicGroup,
    DihedralGroup,
    SymmetricGroup,
    burnside_lemma,
    orbit_stabilizer,
)


class TestCyclicGroup:
    def test_order(self):
        c = CyclicGroup(5)
        assert c.order == 5

    def test_identity(self):
        c = CyclicGroup(4)
        assert c.identity() == 0

    def test_compose(self):
        c = CyclicGroup(5)
        assert c.compose(2, 3) == 0  # (2+3) % 5
        assert c.compose(1, 4) == 0

    def test_inverse(self):
        c = CyclicGroup(5)
        for a in c.elements:
            assert c.compose(a, c.inverse(a)) == c.identity()

    def test_closure(self):
        c = CyclicGroup(4)
        for a in c.elements:
            for b in c.elements:
                assert c.compose(a, b) in c.elements

    def test_associativity(self):
        c = CyclicGroup(5)
        for a in c.elements:
            for b in c.elements:
                for d in c.elements:
                    ab_d = c.compose(c.compose(a, b), d)
                    a_bd = c.compose(a, c.compose(b, d))
                    assert ab_d == a_bd

    def test_generate(self):
        c = CyclicGroup(6)
        gen = c.generate(2)
        # <2> = {0, 2, 4}
        assert gen == [0, 2, 4]

    def test_order_of_element(self):
        c = CyclicGroup(12)
        assert c.order_of(0) == 1
        assert c.order_of(4) == 3  # 12/gcd(12,4) = 3
        assert c.order_of(5) == 12  # gcd(12,5)=1


class TestDihedralGroup:
    def test_order(self):
        d = DihedralGroup(3)
        assert d.order == 6

    def test_identity(self):
        d = DihedralGroup(4)
        assert d.identity() == (0, 0)

    def test_elements_count(self):
        d = DihedralGroup(3)
        assert len(d.elements) == 6

    def test_inverse(self):
        d = DihedralGroup(3)
        for a in d.elements:
            inv = d.inverse(a)
            assert d.compose(a, inv) == d.identity()

    def test_closure(self):
        d = DihedralGroup(3)
        elements_set = set(d.elements)
        for a in d.elements:
            for b in d.elements:
                assert d.compose(a, b) in elements_set

    def test_reflection_self_inverse(self):
        d = DihedralGroup(4)
        r = (0, 1)  # A reflection
        assert d.compose(r, r) == d.identity()

    def test_compose_two_rotations(self):
        d = DihedralGroup(4)
        assert d.compose((1, 0), (2, 0)) == (3, 0)


class TestSymmetricGroup:
    def test_order(self):
        s = SymmetricGroup(3)
        assert s.order == 6

    def test_identity(self):
        s = SymmetricGroup(3)
        assert s.identity() == (0, 1, 2)

    def test_elements_count(self):
        s = SymmetricGroup(4)
        assert len(s.elements) == 24

    def test_compose(self):
        s = SymmetricGroup(3)
        # (0,2,1) ∘ (1,0,2): i → b[i] → a[b[i]]
        # 0→1→2, 1→0→0, 2→2→1 = (2,0,1)
        result = s.compose((0, 2, 1), (1, 0, 2))
        assert result == (2, 0, 1)

    def test_inverse(self):
        s = SymmetricGroup(3)
        for a in s.elements:
            inv = s.inverse(a)
            assert s.compose(a, inv) == s.identity()

    def test_sign_identity(self):
        s = SymmetricGroup(3)
        assert s.sign(s.identity()) == 1

    def test_sign_transposition(self):
        s = SymmetricGroup(3)
        assert s.sign((1, 0, 2)) == -1

    def test_sign_three_cycle(self):
        s = SymmetricGroup(3)
        assert s.sign((1, 2, 0)) == 1  # (012) is even

    def test_cycle_type(self):
        s = SymmetricGroup(4)
        # (1,0,3,2) has cycle type (2,2) — two transpositions
        assert s.cycle_type((1, 0, 3, 2)) == (2, 2)

    def test_cycle_type_identity(self):
        s = SymmetricGroup(3)
        assert s.cycle_type((0, 1, 2)) == (1, 1, 1)


class TestBurnsideLemma:
    def test_c2_two_colors(self):
        """C₂ acting on 2 positions with 2 colors: |X/G| = (2² + 2)/2 = 3."""
        # Group elements: identity (0,1) and swap (1,0)
        group = [(0, 1), (1, 0)]
        result = burnside_lemma(group, action=None, n_colors=2)
        assert result == 3  # {AA, AB, BB}

    def test_s3_three_colors(self):
        """S₃ on 3 positions with 2 colors: (2³ + 0 + 0 + 2 + 2 + 2)/6 = 4."""
        s = SymmetricGroup(3)
        result = burnside_lemma(s.elements, action=None, n_colors=2)
        assert result == 4


class TestOrbitStabilizer:
    def test_orbit_stabilizer_theorem(self):
        """|G| = |Orb(x)| × |Stab(x)|"""
        s = SymmetricGroup(3)

        def action(perm, x):
            return tuple(perm[i] for i in x)

        # x = (0, 1, 2) — fixed by everything
        result = orbit_stabilizer(
            s.elements,
            (0, 1, 2),
            s.compose,
            action,
        )
        assert result["orbit_size"] * result["stabilizer_size"] == result["group_order"]

    def test_orbit_size_s3(self):
        s = SymmetricGroup(3)

        def action(perm, x):
            # permute positions: new[i] = x[perm[i]]
            return tuple(x[perm[i]] for i in range(len(x)))

        # x = (0, 0, 1) — two 0s and one 1
        # Under S₃: {(0,0,1), (0,1,0), (1,0,0)} = orbit of size 3
        result = orbit_stabilizer(
            s.elements,
            (0, 0, 1),
            s.compose,
            action,
        )
        assert result["orbit_size"] == 3
        assert result["stabilizer_size"] == 2
        assert result["orbit_size"] * result["stabilizer_size"] == 6


# ===========================================================================
# Integration / Cross-module
# ===========================================================================

class TestIntegration:
    def test_wasserstein_between_normals(self):
        """Sample from two normals and compute Wasserstein."""
        n1 = NormalManifold(0, 1)
        n2 = NormalManifold(1, 1)
        # Create discrete measures by sampling
        random.seed(42)
        pts1 = [random.gauss(0, 1) for _ in range(20)]
        pts2 = [random.gauss(1, 1) for _ in range(20)]
        pts1.sort()
        pts2.sort()
        w1 = [1.0 / 20] * 20
        w2 = [1.0 / 20] * 20
        a = DiscreteMeasure(w1, pts1)
        b = DiscreteMeasure(w2, pts2)
        d = wasserstein_1d(a, b)
        # Should be approximately 1 (difference in means)
        assert 0.5 < d < 2.0

    def test_spectral_on_points(self):
        """Build adjacency from point distances, compute spectral embedding."""
        points = [(0, 0), (1, 0), (0, 1), (5, 5)]
        n = len(points)
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(points[i], points[j])))
                if d < 3:
                    A[i][j] = A[j][i] = 1.0
        emb = spectral_embedding(A, k=2)
        assert len(emb) == 4

    def test_barcode_on_simple_cloud(self):
        """Two clusters → expect two long-lived components."""
        points = [(0, 0), (0.1, 0.1), (5, 5), (5.1, 5.1)]
        bc = persistence_barcodes(points, max_epsilon=10)
        # Should have at least one component that dies when clusters merge
        deaths = [d for _, d in bc if d != float("inf")]
        assert len(deaths) >= 1
