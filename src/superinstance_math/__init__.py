"""superinstance-math — Pure-Python math for ML engineers."""

from superinstance_math.information_geometry import (
    StatisticalManifold,
    NormalManifold,
    kl_divergence,
    cross_entropy,
    chentsov_theorem,
)
from superinstance_math.optimal_transport import (
    DiscreteMeasure,
    wasserstein_1d,
    sinkhorn,
)
from superinstance_math.persistent_homology import (
    vietoris_rips,
    betti_numbers,
    persistence_barcodes,
)
from superinstance_math.spectral import (
    graph_laplacian,
    top_k_eigenvalues,
    spectral_embedding,
    spectral_anomaly_score,
)
from superinstance_math.symmetry import (
    CyclicGroup,
    DihedralGroup,
    SymmetricGroup,
    burnside_lemma,
    orbit_stabilizer,
)

__version__ = "0.1.0"

__all__ = [
    "StatisticalManifold",
    "NormalManifold",
    "kl_divergence",
    "cross_entropy",
    "chentsov_theorem",
    "DiscreteMeasure",
    "wasserstein_1d",
    "sinkhorn",
    "vietoris_rips",
    "betti_numbers",
    "persistence_barcodes",
    "graph_laplacian",
    "top_k_eigenvalues",
    "spectral_embedding",
    "spectral_anomaly_score",
    "CyclicGroup",
    "DihedralGroup",
    "SymmetricGroup",
    "burnside_lemma",
    "orbit_stabilizer",
]
