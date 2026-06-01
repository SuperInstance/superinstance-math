# superinstance-math

> Mathematical foundations for spectral AI: information geometry, optimal transport, persistent homology, spectral analysis, and symmetry detection

## What This Does

superinstance-math is a Python library providing five interlocking mathematical modules used by the SuperInstance framework. It covers information geometry (Fisher-Rao metric, natural gradient), optimal transport (Sinkhorn, Wasserstein distances), topological data analysis (persistent homology), spectral methods (eigenvalue decomposition, spectral clustering), and symmetry detection (automorphism finding in point clouds and graphs).

## The Key Idea

Each module treats data as living on a geometric object — a statistical manifold, a probability simplex, a topological space, or a graph. By exploiting the geometry of these spaces (curvature, distances, holes, eigenvalues), you get better algorithms for learning, comparison, and structure discovery than treating everything as flat Euclidean vectors.

## Install

```bash
pip install superinstance-math
```

## Quick Start

```python
from superinstance_math.information_geometry import FisherRaoMetric, CategoricalManifold
from superinstance_math.optimal_transport import SinkhornSolver
from superinstance_math.persistent_homology import PersistenceCalculator
from superinstance_math.spectral import SpectralDecomposer
from superinstance_math.symmetry import SymmetryDetector

import numpy as np

# --- Information Geometry ---
manifold = CategoricalManifold(n_categories=3)
p = np.array([0.2, 0.3, 0.5])
q = np.array([0.4, 0.3, 0.3])
fisher = FisherRaoMetric(manifold)
distance = fisher.distance(p, q)
print(f"Fisher-Rao distance: {distance:.4f}")

# --- Optimal Transport ---
source = np.array([[0.0, 0.0], [1.0, 0.0]])
target = np.array([[0.0, 1.0], [1.0, 1.0]])
solver = SinkhornSolver(reg=0.01)
plan, cost = solver.solve(source, target)
print(f"OT cost: {cost:.4f}")

# --- Persistent Homology ---
points = np.random.randn(50, 3)
ph = PersistenceCalculator()
diagrams = ph.compute(points, max_dim=2)
print(f"Betti numbers: {[len(d) for d in diagrams]}")

# --- Spectral Analysis ---
A = np.random.randn(10, 10)
A = (A + A.T) / 2  # symmetric
decomposer = SpectralDecomposer()
eigenvalues, eigenvectors = decomposer.decompose(A, k=3)
print(f"Top 3 eigenvalues: {eigenvalues}")

# --- Symmetry Detection ---
detector = SymmetryDetector()
symmetries = detector.detect(points, tolerance=0.1)
print(f"Found {len(symmetries)} symmetry operations")
```

## API Reference

### `information_geometry`

| Type | Description |
|------|-------------|
| `CategoricalManifold(n)` | Probability simplex with n categories. |
| `FisherRaoMetric(manifold)` | Computes Fisher-Rao geodesic distances on the manifold. |
| `NaturalGradient(manifold)` | Computes natural gradient from Euclidean gradient and Fisher information matrix. |
| `AmariAlpha(alpha)` | Alpha-connection for information geometry (α=0 is Levi-Civita). |

### `optimal_transport`

| Type | Description |
|------|-------------|
| `SinkhornSolver(reg=0.01)` | Entropy-regularized OT via Sinkhorn iterations. |
| `WassersteinDistance(p=1)` | Exact p-Wasserstein distance computation. |
| `Barycenter(weights)` | Wasserstein barycenter of multiple distributions. |

### `persistent_homology`

| Type | Description |
|------|-------------|
| `PersistenceCalculator()` | Vietoris-Rips persistent homology. |
| `compute(points, max_dim)` | Returns persistence diagrams for dims 0..max_dim. |
| `PersistenceDiagram(dim)` | Birth/death pairs for one homology dimension. |
| `betti_numbers(threshold)` | Betti numbers at a given filtration value. |

### `spectral`

| Type | Description |
|------|-------------|
| `SpectralDecomposer()` | Eigenvalue decomposition and spectral clustering. |
| `decompose(A, k)` | Top-k eigendecomposition. |
| `cluster(A, k)` | Spectral clustering into k groups. |
| `SpectralEmbedding(d)` | Laplacian eigenmap embedding into d dimensions. |

### `symmetry`

| Type | Description |
|------|-------------|
| `SymmetryDetector()` | Finds rotational and reflective symmetries. |
| `detect(points, tolerance)` | Returns list of symmetry operations (rotation matrices, reflections). |
| `AutomorphismFinder()` | Graph automorphism detection via spectral signatures. |

## How It Works

- **Information Geometry**: Treats probability distributions as points on a Riemannian manifold. The Fisher information matrix defines the metric tensor, giving geodesic distances that respect the geometry of probability (unlike Euclidean distance on parameter vectors).

- **Optimal Transport**: Computes the minimum-cost coupling between two point clouds using the Sinkhorn algorithm with entropy regularization. The transport plan is a doubly-stochastic matrix mapping source to target.

- **Persistent Homology**: Builds a Vietoris-Rips complex from point clouds at increasing distance scales. Tracks when topological features (connected components, loops, voids) appear and disappear, giving a multi-scale topological signature.

- **Spectral Methods**: Exploits the eigenstructure of matrices (adjacency, Laplacian, covariance) to reveal clusters, communities, and low-dimensional structure.

- **Symmetry Detection**: Uses spectral hashing and geometric hashing to efficiently find approximate rotational and reflective symmetries in point sets.

## Testing

94 tests covering:
- Fisher-Rao metric computation and geodesic distances
- Natural gradient correctness vs. Euclidean gradient
- Sinkhorn convergence and transport plan properties
- Wasserstein distance triangle inequality
- Persistence diagram correctness on known topologies (circle, torus)
- Betti number verification
- Eigenvalue decomposition accuracy
- Spectral clustering on known community graphs
- Symmetry detection on symmetric point configurations

## License

MIT
