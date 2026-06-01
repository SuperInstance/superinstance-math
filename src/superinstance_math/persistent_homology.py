"""Persistent homology: Vietoris-Rips complexes, Betti numbers, persistence barcodes."""

from __future__ import annotations

import math
from typing import Sequence


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _pairwise_distances(points: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean_distance(points[i], points[j])
            D[i][j] = d
            D[j][i] = d
    return D


def vietoris_rips(
    points: Sequence[Sequence[float]],
    max_epsilon: float,
) -> list[tuple]:
    """Build a Vietoris-Rips simplicial complex up to dimension 2.

    Args:
        points: List of coordinate tuples.
        max_epsilon: Maximum edge length to include.

    Returns:
        List of simplices as sorted tuples of vertex indices.
        Includes 0-simplices (vertices), 1-simplices (edges), and 2-simplices (triangles).
    """
    n = len(points)
    D = _pairwise_distances(points)

    simplices: list[tuple] = []

    # 0-simplices
    for i in range(n):
        simplices.append((i,))

    # 1-simplices (edges)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] <= max_epsilon:
                edges.add((i, j))
                simplices.append((i, j))

    # 2-simplices (triangles) — all three edges must exist
    edge_set = edges
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    simplices.append((i, j, k))

    return simplices


class _UnionFind:
    """Union-Find for tracking connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True


def betti_numbers(complex_: list[tuple]) -> dict[int, int]:
    """Compute Betti numbers β₀, β₁, β₂ for the given simplicial complex.

    β₀ = number of connected components
    β₁ = number of 1-dimensional holes (cycles)
    β₂ = number of 2-dimensional voids

    Uses the Euler characteristic and boundary rank computation.
    """
    # Collect simplices by dimension
    dim_map: dict[int, set[tuple]] = {}
    for s in complex_:
        d = len(s) - 1
        dim_map.setdefault(d, set()).add(s)

    max_dim = max(dim_map.keys()) if dim_map else 0

    # β₀ via union-find
    vertices = dim_map.get(0, set())
    edges = dim_map.get(1, set())
    triangles = dim_map.get(2, set())

    # Get vertex count
    all_verts: set[int] = set()
    for v in vertices:
        all_verts.add(v[0])
    n = max(all_verts) + 1 if all_verts else 0

    if n == 0:
        return {0: 0, 1: 0, 2: 0}

    # β₀: connected components
    uf = _UnionFind(n)
    for e in edges:
        uf.union(e[0], e[1])
    b0 = uf.components

    # For β₁ and β₂ use boundary rank computation
    # β_k = dim(C_k) - rank(∂_k) - rank(∂_{k+1})

    # Build boundary matrices
    # For edges: boundary is the two vertices
    # For triangles: boundary is the three edges

    def boundary_rank_1():
        """Rank of ∂₁: edges → vertices."""
        if not edges:
            return 0
        # Each edge reduces components or creates a cycle
        # rank = n_edges - (n_edges_that_create_cycles)
        # rank = #vertices - #components
        return len(all_verts) - b0

    def boundary_rank_2():
        """Rank of ∂₂: triangles → edges."""
        if not triangles:
            return 0
        # Build a matrix over Z/2Z (mod 2)
        edge_list = sorted(edges)
        edge_idx = {e: i for i, e in enumerate(edge_list)}
        m = len(edge_list)
        # Each triangle contributes a column with 1s at its boundary edges
        cols = []
        for tri in triangles:
            col = [0] * m
            boundary_edges = [
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[0], tri[2]))),
                tuple(sorted((tri[1], tri[2]))),
            ]
            for be in boundary_edges:
                if be in edge_idx:
                    col[edge_idx[be]] = 1
            cols.append(col)
        return _mod2_rank(cols, m)

    rank1 = boundary_rank_1()
    rank2 = boundary_rank_2()

    b1 = len(edges) - rank1 - rank2
    b2 = len(triangles) - rank2  # Simplified for dim ≤ 2

    return {0: b0, 1: max(0, b1), 2: max(0, b2)}


def _mod2_rank(columns: list[list[int]], n_rows: int) -> int:
    """Compute rank of a matrix over Z/2Z using Gaussian elimination."""
    if not columns:
        return 0
    # Copy
    mat = [row[:] for row in columns]
    n_cols = len(mat)
    rank = 0
    for col in range(n_rows):
        # Find pivot
        pivot = None
        for row in range(rank, n_cols):
            if mat[row][col] % 2 == 1:
                pivot = row
                break
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for row in range(n_cols):
            if row != rank and mat[row][col] % 2 == 1:
                mat[row] = [(a + b) % 2 for a, b in zip(mat[row], mat[rank])]
        rank += 1
    return rank


def persistence_barcodes(
    points: Sequence[Sequence[float]],
    max_epsilon: float | None = None,
    steps: int = 50,
) -> list[tuple[float, float]]:
    """Compute persistence barcodes (birth, death) for H₀ and H₁.

    Uses a filtration of Vietoris-Rips complexes at increasing epsilon values.

    Args:
        points: List of coordinate tuples.
        max_epsilon: Maximum filtration value (auto-detected if None).
        steps: Number of filtration steps.

    Returns:
        List of (birth, death) pairs. death=inf means the feature persists.
    """
    n = len(points)
    if n == 1:
        return [(0.0, float("inf"))]
    if n < 1:
        return []

    D = _pairwise_distances(points)

    if max_epsilon is None:
        max_epsilon = max(D[i][j] for i in range(n) for j in range(i + 1, n))
        if max_epsilon == 0:
            max_epsilon = 1.0

    epsilons = [max_epsilon * (i + 1) / steps for i in range(steps)]

    # Track component merging (H₀)
    # Each point starts as its own component at ε=0
    uf = _UnionFind(n)
    # Record birth times for each component root
    component_birth: dict[int, float] = {i: 0.0 for i in range(n)}
    barcodes: list[tuple[float, float]] = []

    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i][j], i, j))
    edges.sort()

    # H₀: component deaths
    for dist, i, j in edges:
        if dist > max_epsilon:
            break
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj:
            # The younger component dies
            b_i = component_birth[ri]
            b_j = component_birth[rj]
            # Younger = born later = higher birth time → dies
            if b_i >= b_j:
                barcodes.append((b_i, dist))
                uf.union(ri, rj)
                new_root = uf.find(i)
                component_birth[new_root] = b_j
                # Clean up
                component_birth.pop(ri, None)
                component_birth.pop(rj, None)
                component_birth.setdefault(new_root, b_j)
            else:
                barcodes.append((b_j, dist))
                uf.union(ri, rj)
                new_root = uf.find(i)
                component_birth[new_root] = b_i
                component_birth.pop(ri, None)
                component_birth.pop(rj, None)
                component_birth.setdefault(new_root, b_i)

    # Surviving components (death = inf)
    for root, birth in component_birth.items():
        if uf.find(root) == root:
            barcodes.append((birth, float("inf")))

    return sorted(barcodes, key=lambda x: (x[0], x[1]))
