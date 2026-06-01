# superinstance-math

> Five branches of advanced mathematics, pure Python, zero dependencies.

## What This Does

Information geometry, optimal transport, persistent homology, spectral graph theory, and group-theoretic symmetry — all in one package, all from scratch. No NumPy, no SciPy, no external dependencies. Just Python and mathematics.

Built for ML engineers who need the real math (not approximations), data scientists who want to understand what their tools are actually doing, and anyone who believes beautiful algorithms should be readable.

## The Key Idea

Most math libraries are thin wrappers around C/Fortran. This one is different — every algorithm is implemented in pure Python so you can read it, understand it, and modify it. The Sinkhorn algorithm uses log-domain stabilization because that's the numerically correct way. The Betti number computation does actual Z/2Z boundary rank reduction. The Fisher-Rao distance uses the exact arccosh closed form for normal distributions.

This is math you can learn from, not just call.

## Install

```bash
pip install superinstance-math
```

Zero dependencies. Works on Python 3.8+.

## Quick Start

### Information Geometry — Natural Gradient Descent

```python
from superinstance_math.information_geometry import NormalManifold

# Two normal distributions
manifold = NormalManifold()

# Fisher-Rao distance (the geodesic distance on the statistical manifold)
d = manifold.fisher_rao_distance(
    mu1=0.0, sigma1=1.0,
    mu2=1.0, sigma2=2.0
)
print(f"Fisher-Rao distance: {d:.4f}")

# Natural gradient — adjusts for curvature of parameter space
# Much better than ordinary gradient for optimization on manifolds
grad = [0.5, 0.3]  # ordinary gradient [∂L/∂μ, ∂L/∂σ]
fisher = manifold.fisher_information(mu=0.0, sigma=1.0)
nat_grad = manifold.natural_gradient(grad, fisher)
print(f"Natural gradient: {nat_grad}")
```

### Optimal Transport — Sinkhorn with Log Stabilization

```python
from superinstance_math.optimal_transport import DiscreteMeasure, wasserstein_1d, sinkhorn

# Two distributions
a = DiscreteMeasure([0.3, 0.4, 0.3], [0.0, 1.0, 2.0])
b = DiscreteMeasure([0.2, 0.5, 0.3], [0.5, 1.5, 3.0])

# 1D Wasserstein distance (closed form)
w1 = wasserstein_1d(a, b)
print(f"W₁ distance: {w1:.4f}")

# General Sinkhorn optimal transport
cost = [[abs(i - j) for j in [0.5, 1.5, 3.0]] for i in [0.0, 1.0, 2.0]]
plan, distance = sinkhorn(
    a_weights=[0.3, 0.4, 0.3],
    b_weights=[0.2, 0.5, 0.3],
    cost_matrix=cost,
    reg=0.1,  # entropic regularization
    iterations=200
)
print(f"Transport distance: {distance:.4f}")
print(f"Transport plan row sums: {[sum(row) for row in plan]}")  # ≈ source weights
```

### Persistent Homology — Shape of Data

```python
from superinstance_math.persistent_homology import vietoris_rips, betti_numbers, persistence_barcodes

# Points arranged in a circle — expect Betti numbers (1, 1) for H₀ and H₁
import math
points = [(math.cos(2*math.pi*i/8), math.sin(2*math.pi*i/8)) for i in range(8)]

# Build Vietoris-Rips complex up to distance 2.0
complex_ = vietoris_rips(points, max_epsilon=2.0, max_dim=2)

# Compute Betti numbers
bn = betti_numbers(complex_)
print(f"Betti numbers: {bn}")  # {0: 1, 1: 1} = one component, one hole

# Persistence barcodes — when features are born and die
bars = persistence_barcodes(points, max_epsilon=3.0, steps=30)
print(f"H₀ features: {len([b for b in bars if b['dim'] == 0])}")  # connected components
print(f"H₁ features: {len([b for b in bars if b['dim'] == 1])}")  # loops
```

### Spectral Analysis — Graph Anomaly Detection

```python
from superinstance_math.spectral import graph_laplacian, top_k_eigenvalues, spectral_embedding, spectral_anomaly_score

# Adjacency matrix of a graph
adj = [
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
]

# Graph Laplacian (normalized)
L = graph_laplacian(adj, normalized=True)

# Spectral embedding — project nodes into eigenvector space
eigvals, eigvecs = top_k_eigenvalues(L, k=3)
embedding = spectral_embedding(L, k=3)

# Anomaly scores — nodes far from the cluster in spectral space
scores = spectral_anomaly_score(adj)
print(f"Most anomalous node: {scores.index(max(scores))}")
```

### Symmetry Groups — Burnside's Lemma

```python
from superinstance_math.symmetry import CyclicGroup, DihedralGroup, burnside_lemma

# Dihedral group D4 — symmetries of a square
d4 = DihedralGroup(4)
print(f"Order of D4: {d4.order}")  # 8 (4 rotations + 4 reflections)

# How many distinct colorings of 4 beads with 3 colors,
# up to rotation and reflection?
def colorings_fixed(g_element, n_positions):
    # For Burnside: count colorings fixed by this group element
    cycles = d4.cycle_decomposition(g_element)
    return 3 ** len(cycles)

fixed_counts = [colorings_fixed(g, 4) for g in d4.elements]
distinct = burnside_lemma(fixed_counts, d4.order)
print(f"Distinct colorings: {distinct}")  # much fewer than 3^4 = 81

# Orbit-stabilizer theorem
orbit = d4.orbit((1, 0))  # orbit of rotation by 90°
stab = d4.stabilizer((1, 0))
print(f"|G| = |orbit| × |stabilizer|: {d4.order} = {len(orbit)} × {len(stab)}")
```

## API Reference

### information_geometry.py

| Type | Description |
|------|-------------|
| `StatisticalManifold` | Base class for parametric families |
| `NormalManifold` | Normal distribution manifold (μ, σ) |
| `NormalManifold.fisher_information(mu, sigma)` | 2×2 Fisher information matrix |
| `NormalManifold.fisher_rao_distance(mu1, σ1, mu2, σ2)` | Geodesic distance between distributions |
| `NormalManifold.alpha_connections(mu, sigma, alpha)` | Amari's α-connection coefficients |
| `NormalManifold.natural_gradient(grad, fisher)` | Fisher-corrected gradient for Riemannian optimization |
| `NormalManifold.kl_divergence(mu1, σ1, mu2, σ2)` | KL(p‖q) |
| `NormalManifold.chentsov_theorem(n)` | Verify Fisher metric uniqueness |

### optimal_transport.py

| Type | Description |
|------|-------------|
| `DiscreteMeasure(weights, support)` | Discrete probability distribution |
| `DiscreteMeasure.mean()` | Expected value |
| `DiscreteMeasure.cdf(x)` | Cumulative distribution |
| `wasserstein_1d(a, b)` | Closed-form W₁ for 1D measures |
| `sinkhorn(a_w, b_w, cost, reg, iters)` | Log-stabilized Sinkhorn → (plan, distance) |

### persistent_homology.py

| Function | Description |
|----------|-------------|
| `vietoris_rips(points, max_epsilon, max_dim)` | Build VR complex from point cloud |
| `betti_numbers(complex)` | B₀ (components), B₁ (loops), B₂ (voids) |
| `persistence_barcodes(points, max_epsilon, steps)` | Birth/death of topological features |

### spectral.py

| Function | Description |
|----------|-------------|
| `graph_laplacian(adj, normalized)` | L = D - A (or D^{-1/2}LD^{-1/2}) |
| `top_k_eigenvalues(matrix, k)` | Power iteration for largest-k eigenvalues |
| `spectral_embedding(L, k)` | Project into k-dimensional eigenvector space |
| `spectral_anomaly_score(adj)` | Per-node anomaly score via spectral distance |

### symmetry.py

| Type | Description |
|------|-------------|
| `CyclicGroup(n)` | Cₙ — rotations of regular n-gon |
| `DihedralGroup(n)` | Dₙ — rotations + reflections |
| `SymmetricGroup(n)` | Sₙ — all permutations |
| `burnside_lemma(fixed_counts, group_order)` | Count distinct objects under symmetry |
| `orbit_stabilizer(group, element)` | Orbit and stabilizer of an element |

## How It Works

**Sinkhorn** uses log-domain dual potentials (f, g) instead of the naive kernel approach. This prevents numerical underflow when the regularization parameter λ is small — critical for getting accurate transport plans.

**Betti numbers** compute the rank of boundary matrices over Z/2Z. H₀ = rank(∂₁) free part tells you connected components. H₁ = ker(∂₁)/im(∂₂) tells you independent loops.

**Fisher-Rao distance** between N(μ₁,σ₁) and N(μ₂,σ₂) uses the exact formula: `d = √2 · arccosh(√((σ₁²+σ₂²+(μ₁-μ₂)²)/(2σ₁σ₂)))`. This is a geodesic on the hyperbolic half-plane.

**Power iteration** finds the top-k eigenvalues by repeatedly multiplying a random vector by the matrix and orthogonalizing. Converges to the dominant eigenspace.

## Testing

94 tests covering:
- Metric axioms (non-negativity, symmetry, triangle inequality for Fisher-Rao and W₁)
- Conservation laws (Sinkhorn mass conservation, transport plan marginals)
- Group axioms (closure, associativity, identity, inverse for all group types)
- Topological invariants (Euler characteristic, Betti number properties)
- KL divergence positivity
- Orbit-stabilizer theorem verification

## License

MIT
