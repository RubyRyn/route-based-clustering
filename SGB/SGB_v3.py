"""
Seed-Grow-Balance Clustering (v3)

Phase 1 (Seed)  — Furthest-first anchor selection
Phase 2 (Grow)  — Assignment with balance penalty (v1 style)
Phase 3 (Swap)  — Territory-based violation fixing using convex hulls + road distance

A "violation" is a client that:
  1. Sits inside another cluster's convex hull (Euclidean territory check)
  2. Is closer to that cluster's members by road distance (road reality check)

Swaps are executed most-severe-first, respecting ±20% size guardrails.
Runs in passes until violations stop decreasing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from typing import List, Tuple, Optional
from location import Location


class SeedGrowBalance:
    """
    Constrained partitioning of clients into K balanced,
    geographically coherent groups using road distances.
    """

    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 n_clusters: int = 14,
                 balance_weight: float = 0.4,
                 size_tolerance: float = 0.20,
                 max_grow_iterations: int = 100,
                 max_swap_passes: int = 50,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office at index 0, then clients).
            locations: List of Location objects (office first, then clients).
            n_clusters: Number of groups (employees).
            balance_weight: Balance penalty during Grow phase (0.0-1.0).
            size_tolerance: Allowed deviation from average (0.20 = ±20%).
            max_grow_iterations: Max iterations for grow phase refinement.
            max_swap_passes: Max passes for territory-based swapping.
            random_seed: For reproducibility.
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.balance_weight = balance_weight
        self.size_tolerance = size_tolerance
        self.max_grow_iterations = max_grow_iterations
        self.max_swap_passes = max_swap_passes

        np.random.seed(random_seed)

        # Client-only distance matrix (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.n_clients = len(self.client_matrix)

        # Distance from office to each client
        self.office_distances = road_matrix[0, 1:]

        # Size guardrails
        self.target_size = self.n_clients / n_clusters
        self.min_size = max(1, int(np.floor(self.target_size * (1 - size_tolerance))))
        self.max_size = int(np.ceil(self.target_size * (1 + size_tolerance)))

        # Client coordinates for visualization
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        self.office_coords = (locations[0].lat, locations[0].lon)

        # Results
        self.labels = None
        self.seeds = None
        self.labels_after_grow = None
        self.history = []

        print(f"SeedGrowBalance v3 initialized:")
        print(f"  Clients: {self.n_clients}")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Target size: {self.target_size:.1f}")
        print(f"  Allowed range: [{self.min_size}, {self.max_size}]")
        print(f"  Balance weight: {self.balance_weight}")

    # ══════════════════════════════════════════════
    # PHASE 1: SEED
    # ══════════════════════════════════════════════

    def _seed(self) -> np.ndarray:
        """
        Pick K seed clients using furthest-first traversal.
        First seed = most central client (lowest avg road distance to all others).
        """
        seeds = []

        # First seed: most central client
        avg_distances = np.mean(self.client_matrix, axis=1)
        first_seed = np.argmin(avg_distances)
        seeds.append(first_seed)

        # Track min distance from each client to nearest seed
        min_dist_to_seeds = self.client_matrix[first_seed].copy()

        for k in range(1, self.n_clusters):
            candidates = min_dist_to_seeds.copy()
            for s in seeds:
                candidates[s] = -1
            next_seed = np.argmax(candidates)
            seeds.append(next_seed)
            min_dist_to_seeds = np.minimum(
                min_dist_to_seeds, self.client_matrix[next_seed]
            )

        seeds = np.array(seeds)
        print(f"\nPhase 1 (Seed): Selected {len(seeds)} anchors")
        return seeds

    # ══════════════════════════════════════════════
    # PHASE 2: GROW (v1 style with balance penalty)
    # ══════════════════════════════════════════════

    def _compute_assignment_cost(self, client_idx: int, cluster_id: int,
                                  labels: np.ndarray) -> float:
        """
        Cost of assigning a client to a cluster.
        Blends geographic proximity with balance penalty.
        """
        cluster_members = np.where(labels == cluster_id)[0]

        # Geographic cost
        if len(cluster_members) == 0:
            geo_cost = self.office_distances[client_idx]
        else:
            geo_cost = np.mean(self.client_matrix[client_idx, cluster_members])

        # Balance penalty
        current_size = len(cluster_members)
        overshoot = (current_size - self.target_size) / self.target_size
        balance_cost = max(0, overshoot) * geo_cost

        total_cost = (1 - self.balance_weight) * geo_cost + \
                     self.balance_weight * balance_cost * np.mean(self.office_distances)

        return total_cost

    def _grow(self, seeds: np.ndarray) -> np.ndarray:
        """
        Assign all clients to clusters with balance penalty.
        Hardest clients (furthest from any seed) assigned first.
        """
        labels = np.full(self.n_clients, -1, dtype=int)

        # Assign seeds
        for cluster_id, seed_idx in enumerate(seeds):
            labels[seed_idx] = cluster_id

        # Sort unassigned: furthest from any seed first
        unassigned = np.where(labels == -1)[0]
        seed_distances = self.client_matrix[np.ix_(unassigned, seeds)]
        min_dist = np.min(seed_distances, axis=1)
        order = np.argsort(-min_dist)
        sorted_unassigned = unassigned[order]

        # Assign each client to best cluster
        for client_idx in sorted_unassigned:
            best_cluster = None
            best_cost = np.inf
            for cluster_id in range(self.n_clusters):
                cost = self._compute_assignment_cost(client_idx, cluster_id, labels)
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster_id
            labels[client_idx] = best_cluster

        sizes = self._get_cluster_sizes(labels)
        print(f"\nPhase 2 (Grow): Assignment with balance penalty")
        print(f"  Sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}], Std: {sizes.std():.1f}")
        return labels

    # ══════════════════════════════════════════════
    # PHASE 3: SWAP (territory-based violation fixing)
    # ══════════════════════════════════════════════

    def _compute_hulls(self, labels: np.ndarray) -> dict:
        """
        Compute convex hull for each cluster.
        Returns dict of {cluster_id: (ConvexHull, points_array, member_indices)}.
        Clusters with < 3 clients are skipped.
        """
        lons = self.client_coords[:, 1]
        lats = self.client_coords[:, 0]
        hulls = {}

        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() < 3:
                continue
            points = np.column_stack([lons[mask], lats[mask]])
            try:
                hull = ConvexHull(points)
                hulls[cluster_id] = (hull, points, np.where(mask)[0])
            except Exception:
                pass

        return hulls

    def _point_in_hull(self, point: np.ndarray, hull: ConvexHull) -> bool:
        """Check if a 2D point is inside a convex hull."""
        return np.all(
            hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-10
        )

    def _find_violations(self, labels: np.ndarray,
                          hulls: dict) -> List[Tuple[int, int, int, float]]:
        """
        Find all territorial violations.

        A violation = client inside another cluster's convex hull
                      AND closer to that cluster by road distance.

        Returns list of (client_idx, current_cluster, violating_cluster, severity)
        sorted most severe first.
        """
        lons = self.client_coords[:, 1]
        lats = self.client_coords[:, 0]
        violations = []

        for client_idx in range(self.n_clients):
            client_cluster = labels[client_idx]
            client_point = np.array([lons[client_idx], lats[client_idx]])

            # Average road distance to own cluster
            own_members = np.where(labels == client_cluster)[0]
            own_members = own_members[own_members != client_idx]
            if len(own_members) == 0:
                own_avg_road = self.office_distances[client_idx]
            else:
                own_avg_road = np.mean(
                    self.client_matrix[client_idx, own_members])

            # Check against each other cluster's hull
            for cluster_id, (hull, points, member_indices) in hulls.items():
                if cluster_id == client_cluster:
                    continue

                # Check 1: Inside convex hull? (Euclidean)
                if not self._point_in_hull(client_point, hull):
                    continue

                # Check 2: Closer by road distance?
                alt_members = np.where(labels == cluster_id)[0]
                alt_avg_road = np.mean(
                    self.client_matrix[client_idx, alt_members])

                if alt_avg_road < own_avg_road:
                    severity = own_avg_road - alt_avg_road
                    violations.append(
                        (client_idx, client_cluster, cluster_id, severity))
                    break  # Only report first violation per client

        # Sort: most severe first
        violations.sort(key=lambda x: -x[3])
        return violations

    def _swap(self, labels: np.ndarray) -> np.ndarray:
        """
        Territory-based swapping.
        Iteratively fix violations until they stop decreasing.
        """
        print(f"\nPhase 3 (Swap): Territory-based violation fixing")
        print(f"  Size guardrails: [{self.min_size}, {self.max_size}]")

        prev_violation_count = float('inf')

        for pass_num in range(self.max_swap_passes):
            # Compute hulls
            hulls = self._compute_hulls(labels)

            # Find violations
            violations = self._find_violations(labels, hulls)
            current_count = len(violations)

            # Check stopping condition
            if current_count == 0:
                print(f"  Pass {pass_num + 1}: 0 violations — clean territories!")
                break

            if current_count >= prev_violation_count:
                print(f"  Pass {pass_num + 1}: {current_count} violations "
                      f"(not improving, stopping)")
                break

            # Execute swaps (most severe first)
            sizes = self._get_cluster_sizes(labels)
            swaps_made = 0
            swaps_skipped = 0

            for client_idx, src_cluster, tgt_cluster, severity in violations:
                # Check size guardrails
                src_size_after = sizes[src_cluster] - 1
                tgt_size_after = sizes[tgt_cluster] + 1

                if src_size_after < self.min_size:
                    swaps_skipped += 1
                    continue
                if tgt_size_after > self.max_size:
                    swaps_skipped += 1
                    continue

                # Execute swap
                labels[client_idx] = tgt_cluster
                sizes[src_cluster] -= 1
                sizes[tgt_cluster] += 1
                swaps_made += 1

            self.history.append(labels.copy())

            sizes = self._get_cluster_sizes(labels)
            print(f"  Pass {pass_num + 1}: {current_count} violations, "
                  f"{swaps_made} swapped, {swaps_skipped} skipped (size limit), "
                  f"sizes=[{sizes.min()}-{sizes.max()}]")

            prev_violation_count = current_count

        return labels

    # ══════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════

    def _get_cluster_sizes(self, labels: np.ndarray) -> np.ndarray:
        sizes = np.zeros(self.n_clusters, dtype=int)
        for cluster_id in range(self.n_clusters):
            sizes[cluster_id] = np.sum(labels == cluster_id)
        return sizes

    def get_cluster_stats(self) -> dict:
        """Detailed stats about the final clustering."""
        if self.labels is None:
            return {}

        stats = {'clusters': []}
        for cluster_id in range(self.n_clusters):
            members = np.where(self.labels == cluster_id)[0]
            if len(members) == 0:
                continue

            office_dists = self.office_distances[members]

            if len(members) > 1:
                pairwise = self.client_matrix[np.ix_(members, members)]
                intra_avg = np.mean(pairwise[np.triu_indices(len(members), k=1)])
            else:
                intra_avg = 0

            stats['clusters'].append({
                'id': cluster_id,
                'size': len(members),
                'avg_office_dist': np.mean(office_dists),
                'max_office_dist': np.max(office_dists),
                'avg_intra_dist': intra_avg,
            })

        sizes = [c['size'] for c in stats['clusters']]
        stats['size_std'] = np.std(sizes)
        stats['size_range'] = (min(sizes), max(sizes))
        return stats

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def fit(self, visualize: bool = True) -> np.ndarray:
        """Run the full Seed → Grow → Swap pipeline."""
        print(f"\n{'=' * 60}")
        print(f"SEED-GROW-BALANCE v3 (Territory Swapping)")
        print(f"{'=' * 60}")

        # Phase 1
        self.seeds = self._seed()

        # Phase 2
        self.labels = self._grow(self.seeds)
        self.labels_after_grow = self.labels.copy()
        self.history.append(self.labels.copy())

        # Phase 3
        self.labels = self._swap(self.labels)

        # Final report
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Cluster sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}]")
        print(f"  Std: {sizes.std():.2f}")

        hulls = self._compute_hulls(self.labels)
        remaining = self._find_violations(self.labels, hulls)
        print(f"  Remaining violations: {len(remaining)}")

        stats = self.get_cluster_stats()
        intra_dists = [c['avg_intra_dist'] for c in stats['clusters']]
        print(f"  Avg intra-cluster distance: {np.mean(intra_dists):.1f}")
        print(f"{'=' * 60}")

        if visualize:
            self.visualize_territories()
            self.visualize_comparison()

        return self.labels

    # ══════════════════════════════════════════════
    # Visualization with convex hull territories
    # ══════════════════════════════════════════════

    def _draw_territories(self, ax, labels, colors, show_violations=True):
        """Draw convex hull territories on an axis. Returns violation count."""
        lons = self.client_coords[:, 1]
        lats = self.client_coords[:, 0]

        hulls = self._compute_hulls(labels)

        # Draw hull polygons
        for cluster_id, (hull, points, member_indices) in hulls.items():
            hull_pts = points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            polygon = MplPolygon(
                hull_pts, closed=True,
                facecolor=colors[cluster_id],
                edgecolor=colors[cluster_id],
                alpha=0.15, linewidth=0
            )
            ax.add_patch(polygon)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=colors[cluster_id], alpha=0.6, linewidth=2)

        # Find violations
        violations = self._find_violations(labels, hulls) if show_violations else []

        # Plot client dots
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() > 0:
                ax.scatter(lons[mask], lats[mask],
                           c=[colors[cluster_id]], s=40, alpha=0.8,
                           edgecolors='white', linewidths=0.5, zorder=3)

        # Highlight violations
        if violations:
            v_idx = [v[0] for v in violations]
            ax.scatter(lons[v_idx], lats[v_idx],
                       c='none', s=150, edgecolors='red',
                       linewidths=2.5, zorder=4)

        # Office
        ax.scatter(self.office_coords[1], self.office_coords[0],
                   c='red', marker='s', s=200, zorder=5,
                   edgecolors='black', linewidths=1.5)

        return len(violations)

    def visualize_territories(self, filename: str = "Output/sgb_v3_territories.png"):
        """Visualize final result with convex hull territories."""
        if self.labels is None:
            print("Run fit() first.")
            return

        fig, ax = plt.subplots(figsize=(14, 10))

        if self.n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:self.n_clusters]
        else:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.n_clusters))

        n_violations = self._draw_territories(ax, self.labels, colors)

        sizes = self._get_cluster_sizes(self.labels)
        ax.set_title(
            f"Seed-Grow-Balance v3 | K={self.n_clusters}\n"
            f"Sizes: [{sizes.min()}-{sizes.max()}] | "
            f"Std: {sizes.std():.1f} | "
            f"Violations: {n_violations}",
            fontsize=12
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    def visualize_comparison(self, filename: str = "Output/sgb_v3_comparison.png"):
        """Side-by-side: after Grow vs after Swap."""
        if self.labels_after_grow is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        if self.n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:self.n_clusters]
        else:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.n_clusters))

        for ax, lbls, title in zip(
            axes,
            [self.labels_after_grow, self.labels],
            ["After Grow (before swapping)", "After Swap (final)"]
        ):
            n_v = self._draw_territories(ax, lbls, colors)
            sizes = self._get_cluster_sizes(lbls)
            ax.set_title(
                f"{title}\nSizes: [{sizes.min()}-{sizes.max()}] | "
                f"Std: {sizes.std():.1f} | Violations: {n_v}",
                fontsize=11
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    # ══════════════════════════════════════════════
    # New client assignment
    # ══════════════════════════════════════════════

    def assign_new_client(self, new_client_distances: np.ndarray) -> int:
        """Assign a single new client to best existing cluster."""
        if self.labels is None:
            raise ValueError("Run fit() first.")

        sizes = self._get_cluster_sizes(self.labels)
        best_cluster = None
        best_score = np.inf

        for cluster_id in range(self.n_clusters):
            members = np.where(self.labels == cluster_id)[0]
            if len(members) == 0:
                continue

            geo_cost = np.mean(new_client_distances[members])

            size_factor = 1.0
            if sizes[cluster_id] >= self.max_size:
                size_factor = 2.0
            elif sizes[cluster_id] > self.target_size:
                size_factor = 1.2

            score = geo_cost * size_factor
            if score < best_score:
                best_score = score
                best_cluster = cluster_id

        return best_cluster

    def get_labels(self) -> np.ndarray:
        return self.labels

    def get_history(self) -> list:
        return self.history