# superinstance-math

> Pure-Python mathematical foundations for ML: information geometry, optimal transport, persistent homology, spectral analysis, and symmetry groups

## Install

```bash
pip install superinstance-math

# Optional: numpy for 100-1000x faster spectral methods
pip install numpy
```

### Optional Dependencies

- **numpy** — When installed, `graph_laplacian()` and `top_k_eigenvalues()` automatically use numpy-accelerated backends, giving 100–1000× speedups on large matrices. The library works perfectly without numpy (pure-Python fallback), but installing it is strongly recommended for production use.

## Quick Start

```python
from superinstance_math import (
    # Information geometry
    NormalManifold, StatisticalManifold, kl_divergence, cross_entropy, chentsov_theorem,
    # Optimal transport
    DiscreteMeasure, wasserstein_1d, sinkhorn,
    # Persistent homology
    vietoris_rips, betti_numbers, persistence_barcodes,
    # Spectral methods
    graph_laplacian, top_k_eigenvalues, spectral_embedding, spectral_anomaly_score,
    # Symmetry groups
    CyclicGroup, DihedralGroup, SymmetricGroup, burnside_lemma, orbit_stabilizer,
)
```

### Information Geometry

```python
from superinstance_math import NormalManifold, kl_divergence, chentsov_theorem

# Normal distributions as points on a Riemannian manifold
p = NormalManifold(mean=0.0, std=1.0)
q = NormalManifold(mean=1.0, std=2.0)

# Fisher-Rao geodesic distance (closed-form for normals)
d = p.fisher_rao_distance(q)
print(f"Fisher-Rao distance: {d:.4f}")

# KL divergence (closed-form for normals)
kl = kl_divergence(p, q)
print(f"KL(p || q): {kl:.4f}")

# Cross-entropy H(p, q)
h = cross_entropy(p, q)
print(f"Cross-entropy: {h:.4f}")

# Fisher information matrix
fim = p.fisher_information()  # [[1/σ², 0], [0, 2/σ²]]

# Natural gradient from Euclidean gradient
ng = p.natural_gradient([0.5, -0.3])

# Alpha-connections (α=0 is Levi-Civita; α=±1 are e-/m-connections)
christoffel = p.alpha_connection(alpha=0.0)

# Chentsov's theorem statement
print(chentsov_theorem())
```

### Optimal Transport

```python
from superinstance_math import DiscreteMeasure, wasserstein_1d, sinkhorn

# Discrete probability measures
a = DiscreteMeasure(weights=[0.5, 0.5], support=[0.0, 1.0])
b = DiscreteMeasure(weights=[0.5, 0.5], support=[2.0, 3.0])

# Wasserstein-1 distance (closed-form for 1-D)
w = wasserstein_1d(a, b)
print(f"W1 distance: {w:.4f}")

# Sinkhorn algorithm for regularised OT
import math
cost = [[abs(i - j) for j in range(3)] for i in range(3)]
plan, dist = sinkhorn(
    a_weights=[0.5, 0.3, 0.2],
    b_weights=[0.4, 0.4, 0.2],
    cost_matrix=cost,
    reg=0.1,
    iterations=100,
)
print(f"Transport cost: {dist:.4f}")
```

### Persistent Homology

```python
from superinstance_math import vietoris_rips, betti_numbers, persistence_barcodes

# Build Vietoris-Rips complex
points = [(0, 0), (1, 0), (0.5, 0.87), (0.5, 0.5)]
simplices = vietoris_rips(points, max_epsilon=1.5)

# Compute Betti numbers (β₀=components, β₁=cycles, β₂=voids)
betti = betti_numbers(simplices)
print(f"Betti numbers: {betti}")  # e.g. {0: 1, 1: 0, 2: 0}

# Persistence barcodes for H₀
barcodes = persistence_barcodes(points, max_epsilon=2.0, steps=50)
print(f"Barcodes: {barcodes}")
```

### Spectral Methods

```python
from superinstance_math import graph_laplacian, top_k_eigenvalues, spectral_embedding, spectral_anomaly_score

# Adjacency matrix of a small graph
adj = [
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
]

# Graph Laplacian (normalized or unnormalized)
L = graph_laplacian(adj, normalized=True)

# Top-k eigenvalues and eigenvectors
eigenvalues, eigenvectors = top_k_eigenvalues(L, k=3)
print(f"Eigenvalues: {eigenvalues}")

# Spectral embedding (Laplacian eigenmaps)
emb = spectral_embedding(adj, k=2)

# Anomaly scores (distance from centroid in embedding space)
scores = spectral_anomaly_score(adj, k=2)
print(f"Anomaly scores: {[f'{s:.3f}' for s in scores]}")
```

### Symmetry Groups

```python
from superinstance_math import CyclicGroup, DihedralGroup, SymmetricGroup, burnside_lemma, orbit_stabilizer

# Cyclic group C_n (rotations of regular n-gon)
C = CyclicGroup(4)
print(C.elements)          # [0, 1, 2, 3]
print(C.compose(1, 3))     # 0
print(C.inverse(1))         # 3
print(C.generate(2))        # [0, 2] (subgroup)

# Dihedral group D_n (rotations + reflections)
D = DihedralGroup(3)
print(D.order)              # 6
print(D.elements)           # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Symmetric group S_n (all permutations)
S = SymmetricGroup(3)
perm = (1, 2, 0)
print(S.sign(perm))         # +1 (even)
print(S.cycle_type(perm))   # (3,)
print(S.compose(perm, (2, 0, 1)))  # (0, 1, 2)

# Burnside's lemma: count distinct colorings
# For S_3 acting on 3 positions with 2 colors:
colorings = burnside_lemma(S.elements, action=None, n_colors=2)
print(f"Distinct colorings: {colorings}")

# Orbit-stabilizer theorem
result = orbit_stabilizer(
    group_elements=S.elements,
    element=(0, 1, 2),
    compose=S.compose,
    action=S.compose,
)
print(f"Orbit size: {result['orbit_size']}, Stabilizer size: {result['stabilizer_size']}")
print(f"|G| = {result['group_order']} = {result['orbit_size']} × {result['stabilizer_size']}")
```

## API Reference

### `information_geometry`

| Symbol | Kind | Description |
|--------|------|-------------|
| `StatisticalManifold` | abstract class | Base class for parametric statistical manifolds. Provides `pdf()`, `score()`, `fisher_information()`, `kl_divergence_to()`, `cross_entropy_to()`. |
| `NormalManifold(mean, std)` | class | Univariate normal family N(μ, σ²) as a 2-D manifold. Adds `fisher_rao_distance()`, `alpha_connection()`, `natural_gradient()`, `log_pdf()`. |
| `kl_divergence(p, q)` | function | KL(p ‖ q) — closed-form for normals, Monte Carlo otherwise. |
| `cross_entropy(p, q)` | function | H(p, q) = −E_p[log q]. |
| `chentsov_theorem()` | function | Returns a statement of Chentsov's theorem on uniqueness of the Fisher metric. |

### `optimal_transport`

| Symbol | Kind | Description |
|--------|------|-------------|
| `DiscreteMeasure(weights, support)` | class | Discrete probability measure with `mean()`, `cdf()`. |
| `wasserstein_1d(a, b)` | function | W₁ distance between 1-D discrete measures via CDF integration. |
| `sinkhorn(a_weights, b_weights, cost_matrix, reg, iterations)` | function | Log-domain stabilised Sinkhorn. Returns `(transport_plan, cost)`. |

### `persistent_homology`

| Symbol | Kind | Description |
|--------|------|-------------|
| `vietoris_rips(points, max_epsilon)` | function | Build Vietoris-Rips complex (0/1/2-simplices). |
| `betti_numbers(complex)` | function | Compute β₀, β₁, β₂ via union-find and mod-2 boundary rank. |
| `persistence_barcodes(points, max_epsilon, steps)` | function | H₀ persistence barcodes as `(birth, death)` pairs. |

### `spectral`

| Symbol | Kind | Description |
|--------|------|-------------|
| `graph_laplacian(adjacency, normalized)` | function | Normalised or unnormalised graph Laplacian. Numpy-accelerated when available. |
| `top_k_eigenvalues(matrix, k, iterations, seed)` | function | k smallest eigenvalues/vectors via inverse iteration with deflation. Numpy-accelerated when available. |
| `spectral_embedding(adjacency, k, normalized)` | function | Laplacian eigenmap embedding into k dimensions. |
| `spectral_anomaly_score(adjacency, k)` | function | Per-node anomaly score via spectral distance from centroid. |

### `symmetry`

| Symbol | Kind | Description |
|--------|------|-------------|
| `CyclicGroup(n)` | class | C_n: rotations of a regular n-gon. Elements are ints 0…n−1. |
| `DihedralGroup(n)` | class | D_n: rotations + reflections. Elements are `(rotation, reflection)` tuples. |
| `SymmetricGroup(n)` | class | S_n: all permutations of {0,…,n−1}. Provides `sign()`, `cycle_type()`. |
| `burnside_lemma(group_elements, action, n_colors)` | function | Count distinct colorings via Burnside's lemma. |
| `orbit_stabilizer(group_elements, element, compose, action)` | function | Compute orbit, stabilizer, and verify orbit-stabilizer theorem. |

## Design Principles

- **Zero hard dependencies** — works with just the Python standard library.
- **Numpy-accelerated when available** — spectral methods automatically use numpy for 100–1000× speedups.
- **Pure-Python fallbacks** — every function works without numpy, just slower.
- **Educational clarity** — implementations are readable and well-documented, not obfuscated for performance.

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT
