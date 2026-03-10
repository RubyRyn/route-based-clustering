"""
Seed-Grow-Balance Clustering (v2)

Geography-first approach: 
  Phase 1 (Seed)  — Pick K well-spread anchors
  Phase 2 (Grow)  — Assign every client to nearest cluster (pure geography)
  Phase 3 (Balance) — Shave edges of oversized clusters to fix imbalance

Key insight: balance is achieved by moving BOUNDARY clients only,
never by forcing geographically distant clients into wrong clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from location import Location


class SeedGrowBalance:
    """
    Constrained partitioning of clients into K balanced,
    geographically coherent groups using road distances.
    """

    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 n_clusters: int = 14,
                 max_imbalance: float = 0.25,
                 max_refine_iterations: int = 100,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office at index 0, then clients).
            locations: List of Location objects (office first, then clients).
            n_clusters: Number of groups (employees).
            max_imbalance: How much size deviation to tolerate as a fraction of target.
                           0.25 means clusters can be 25% above/below target size.
                           Lower = more balanced but may sacrifice some geography.
            max_refine_iterations: Max passes for the balance phase.
            random_seed: For reproducibility.
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.max_imbalance = max_imbalance
        self.max_refine_iterations = max_refine_iterations

        np.random.seed(random_seed)

        # Client-only distance matrix (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.n_clients = len(self.client_matrix)

        # Distance from office to each client
        self.office_distances = road_matrix[0, 1:]

        # Target size
        self.target_size = self.n_clients / n_clusters
        self.min_size = max(1, int(np.floor(self.target_size * (1 - max_imbalance))))
        self.max_size = int(np.ceil(self.target_size * (1 + max_imbalance)))

        # Client coordinates for visualization
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        self.office_coords = (locations[0].lat, locations[0].lon)

        # Results
        self.labels = None
        self.seeds = None
        self.history = []

        print(f"SeedGrowBalance v2 initialized:")
        print(f"  Clients: {self.n_clients}")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Target size: {self.target_size:.1f}")
        print(f"  Allowed range: [{self.min_size}, {self.max_size}]")

    # ──────────────────────────────────────────────
    # PHASE 1: SEED
    # ──────────────────────────────────────────────

    def _seed(self) -> np.ndarray:
        """
        Pick K seed clients using furthest-first traversal.
        First seed = most central client (lowest average road distance to others).
        Then repeatedly pick the client furthest from all existing seeds.
        """
        seeds = []

        # First seed: most central client
        avg_distances = np.mean(self.client_matrix, axis=1)
        first_seed = np.argmin(avg_distances)
        seeds.append(first_seed)

        # Track min distance from each client to nearest seed
        min_dist_to_seeds = self.client_matrix[first_seed].copy()

        for k in range(1, self.n_clusters):
            # Exclude existing seeds
            candidates = min_dist_to_seeds.copy()
            for s in seeds:
                candidates[s] = -1

            next_seed = np.argmax(candidates)
            seeds.append(next_seed)

            # Update min distances
            min_dist_to_seeds = np.minimum(
                min_dist_to_seeds, self.client_matrix[next_seed]
            )

        seeds = np.array(seeds)
        print(f"\nPhase 1 (Seed): Selected {len(seeds)} anchors")
        return seeds

    # ──────────────────────────────────────────────
    # PHASE 2: GROW — Pure geography, no balance
    # ──────────────────────────────────────────────

    def _grow(self, seeds: np.ndarray) -> np.ndarray:
        """
        Assign every client to the cluster whose seed is closest (road distance).
        Pure nearest-seed assignment. No balance consideration.
        """
        # Distance from each client to each seed: shape (n_clients, n_clusters)
        dist_to_seeds = self.client_matrix[:, seeds]

        # Assign each client to nearest seed's cluster
        labels = np.argmin(dist_to_seeds, axis=1)

        sizes = self._get_cluster_sizes(labels)
        print(f"\nPhase 2 (Grow): Pure geographic assignment")
        print(f"  Sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}]")
        print(f"  Std: {sizes.std():.1f}")

        return labels

    # ──────────────────────────────────────────────
    # PHASE 3: BALANCE — Edge-shaving only
    # ──────────────────────────────────────────────

    def _get_boundary_clients_of_cluster(self, cluster_id: int,
                                          labels: np.ndarray) -> List[Tuple[int, float, int]]:
        """
        Find clients on the edge of a cluster — those closest to another cluster.
        
        Returns list of (client_idx, gap, best_neighbor_cluster_id)
        where gap = avg_dist_to_own - avg_dist_to_best_alt.
        Positive gap means the client is actually closer to the alternative.
        Sorted by gap descending (most transferable first).
        """
        members = np.where(labels == cluster_id)[0]
        if len(members) <= 1:
            return []

        boundary = []
        for client_idx in members:
            # Distance to own cluster's other members
            own_members = members[members != client_idx]
            own_avg = np.mean(self.client_matrix[client_idx, own_members])

            # Find best alternative cluster
            best_alt_cluster = None
            best_alt_avg = np.inf

            for alt_id in range(self.n_clusters):
                if alt_id == cluster_id:
                    continue
                alt_members = np.where(labels == alt_id)[0]
                if len(alt_members) == 0:
                    continue
                alt_avg = np.mean(self.client_matrix[client_idx, alt_members])
                if alt_avg < best_alt_avg:
                    best_alt_avg = alt_avg
                    best_alt_cluster = alt_id

            if best_alt_cluster is not None:
                gap = own_avg - best_alt_avg
                boundary.append((client_idx, gap, best_alt_cluster))

        # Most transferable first
        boundary.sort(key=lambda x: -x[1])
        return boundary

    def _balance(self, labels: np.ndarray) -> np.ndarray:
        """
        Balance cluster sizes by moving boundary clients from oversized 
        clusters to undersized neighbor clusters.
        
        Only moves clients where the geographic cost is acceptable:
        - gap > 0: client is CLOSER to the target cluster (always good)
        - gap > -threshold: client is slightly further but it's worth it for balance
        """
        print(f"\nPhase 3 (Balance): Edge-shaving refinement")

        # Compute a threshold: we allow moving a client even if it's slightly
        # further from the target cluster, as long as the penalty is small
        # relative to typical distances in the data.
        nonzero_dists = self.client_matrix[self.client_matrix > 0]
        geo_tolerance = np.percentile(nonzero_dists, 10) * 0.3
        print(f"  Geographic tolerance: {geo_tolerance:.2f}")

        for iteration in range(self.max_refine_iterations):
            sizes = self._get_cluster_sizes(labels)
            moves_made = 0

            # Strategy: move clients from the largest cluster to smaller ones
            # Repeat until sizes are within tolerance
            sort_order = np.argsort(-sizes)  # Largest first

            for big_cluster in sort_order:
                if sizes[big_cluster] <= np.ceil(self.target_size):
                    continue  # This cluster doesn't need to shrink

                boundary = self._get_boundary_clients_of_cluster(
                    big_cluster, labels)

                for client_idx, gap, target_cluster in boundary:
                    # Only move if:
                    # 1. Target isn't already bigger than source - 1
                    # 2. Geographic cost is acceptable
                    if (sizes[target_cluster] < sizes[big_cluster] - 1 and
                            gap > -geo_tolerance):

                        labels[client_idx] = target_cluster
                        sizes[big_cluster] -= 1
                        sizes[target_cluster] += 1
                        moves_made += 1

                        # Stop if this cluster is now at or below target
                        if sizes[big_cluster] <= np.ceil(self.target_size):
                            break

            self.history.append(labels.copy())
            sizes = self._get_cluster_sizes(labels)
            print(f"  Iteration {iteration + 1}: {moves_made} moves, "
                  f"sizes=[{sizes.min()}-{sizes.max()}], std={sizes.std():.1f}")

            if moves_made == 0:
                print(f"  No more beneficial moves found.")
                break

        return labels

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def _get_cluster_sizes(self, labels: np.ndarray) -> np.ndarray:
        sizes = np.zeros(self.n_clusters, dtype=int)
        for cluster_id in range(self.n_clusters):
            sizes[cluster_id] = np.sum(labels == cluster_id)
        return sizes

    def get_cluster_stats(self) -> dict:
        """Get detailed statistics about the clustering."""
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

    # ──────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────

    def fit(self, visualize: bool = True) -> np.ndarray:
        """Run the full Seed → Grow → Balance pipeline."""
        print(f"\n{'=' * 60}")
        print(f"SEED-GROW-BALANCE CLUSTERING (v2 — Geography First)")
        print(f"{'=' * 60}")

        # Phase 1
        self.seeds = self._seed()

        # Phase 2 — pure geography
        self.labels = self._grow(self.seeds)
        self.history.append(self.labels.copy())

        # Phase 3 — edge-shaving for balance
        self.labels = self._balance(self.labels)

        # Final report
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Cluster sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}]")
        print(f"  Std: {sizes.std():.2f}")
        
        stats = self.get_cluster_stats()
        intra_dists = [c['avg_intra_dist'] for c in stats['clusters']]
        print(f"  Avg intra-cluster distance: {np.mean(intra_dists):.1f}")
        print(f"  Max intra-cluster distance: {np.max(intra_dists):.1f}")
        print(f"{'=' * 60}")

        if visualize:
            self.visualize_final()
            self.visualize_comparison()

        return self.labels

    # ──────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────

    def visualize_final(self):
        """Visualize final clustering result."""
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))

        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            if mask.sum() > 0:
                ax.scatter(
                    self.client_coords[mask, 1],
                    self.client_coords[mask, 0],
                    c=[colors[cluster_id]],
                    s=50, alpha=0.7,
                    label=f"C{cluster_id + 1} ({mask.sum()})"
                )

        # Seeds
        if self.seeds is not None:
            seed_coords = self.client_coords[self.seeds]
            ax.scatter(seed_coords[:, 1], seed_coords[:, 0],
                       c='black', marker='*', s=200, zorder=5, label='Seeds')

        # Office
        ax.scatter(self.office_coords[1], self.office_coords[0],
                   c='red', marker='s', s=200, zorder=5, label='Farm')

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        sizes = self._get_cluster_sizes(self.labels)
        ax.set_title(
            f"Seed-Grow-Balance v2 | K={self.n_clusters} | "
            f"Sizes: [{sizes.min()}-{sizes.max()}] | Std: {sizes.std():.1f}"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        plt.savefig("Output/sgb_v2_final.png", dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_comparison(self):
        """
        Show Phase 2 (pure geography) vs Phase 3 (after balancing) side by side.
        """
        if len(self.history) < 2:
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))

        for ax_idx, (ax, lbls, title) in enumerate(zip(
            axes,
            [self.history[0], self.labels],
            ["Phase 2: Pure Geography", "Phase 3: After Balancing"]
        )):
            for cluster_id in range(self.n_clusters):
                mask = lbls == cluster_id
                if mask.sum() > 0:
                    ax.scatter(
                        self.client_coords[mask, 1],
                        self.client_coords[mask, 0],
                        c=[colors[cluster_id]],
                        s=40, alpha=0.7
                    )

            ax.scatter(self.office_coords[1], self.office_coords[0],
                       c='red', marker='s', s=150, zorder=5)

            sizes = self._get_cluster_sizes(lbls)
            ax.set_title(f"{title}\nSizes: [{sizes.min()}-{sizes.max()}] "
                         f"Std: {sizes.std():.1f}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig("Output/sgb_v2_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

    # ──────────────────────────────────────────────
    # New client assignment
    # ──────────────────────────────────────────────

    def assign_new_client(self, new_client_distances: np.ndarray) -> int:
        """
        Assign a single new client to the best existing cluster.
        Prefers geographic fit, with a soft nudge toward smaller clusters.
        """
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
