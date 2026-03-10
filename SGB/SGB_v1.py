"""
Seed-Grow-Balance Clustering

A constrained partitioning approach for delivery client assignment.

Instead of discovering clusters (bottom-up), this assigns clients to 
K employee groups (top-down) using a three-phase approach:

Phase 1 - SEED:   Pick K geographically spread anchor clients (furthest-first)
Phase 2 - GROW:   Assign remaining clients greedily (hardest-first), 
                   balancing geographic fit vs workload balance
Phase 3 - BALANCE: Local swap refinement to improve balance without 
                    destroying geographic coherence

Uses actual road distance matrix — no Euclidean assumptions.
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
                 balance_weight: float = None,
                 max_refine_iterations: int = 50,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office at index 0, then clients) in km.
            locations: List of Location objects (office first, then clients).
            n_clusters: Number of groups (employees).
            balance_weight: How aggressively to penalize imbalanced groups.
                           0.0 = pure geography (ignore balance)
                           1.0 = pure balance (ignore geography)
                           0.3-0.5 is usually the sweet spot.
            max_refine_iterations: Max passes for the balance/refine phase.
            random_seed: For reproducibility (only affects tie-breaking).
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.balance_weight = balance_weight
        self.max_refine_iterations = max_refine_iterations

        np.random.seed(random_seed)

        # Client-only distance matrix (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.n_clients = len(self.client_matrix)

        # Distance from office to each client
        self.office_distances = road_matrix[0, 1:]

        # Target size per cluster
        self.target_size = self.n_clients / n_clusters

        # Client coordinates for visualization
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        self.office_coords = (locations[0].lat, locations[0].lon)

        # Results
        self.labels = None
        self.seeds = None
        self.history = []

        print(f"SeedGrowBalance initialized:")
        print(f"  Clients: {self.n_clients}")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Target size per cluster: {self.target_size:.1f}")
        print(f"  Balance weight: {self.balance_weight}")

    # ──────────────────────────────────────────────
    # PHASE 1: SEED — Pick K spread-out anchor clients
    # ──────────────────────────────────────────────

    def _seed(self) -> np.ndarray:
        """
        Pick K seed clients using furthest-first traversal on road distances.
        
        Start from the client closest to the office (natural starting point),
        then repeatedly pick the client that is furthest from all existing seeds.
        This guarantees good geographic spread.
        
        Returns:
            Array of K client indices (0-based into client_matrix).
        """
        seeds = []

        # First seed: client closest to office
        first_seed = np.argmin(self.office_distances)
        seeds.append(first_seed)

        # Track minimum distance from each client to any seed
        min_dist_to_seeds = self.client_matrix[first_seed].copy()

        for _ in range(1, self.n_clusters):
            # Pick client with maximum min-distance to existing seeds
            # (exclude clients already chosen as seeds)
            candidates = min_dist_to_seeds.copy()
            for s in seeds:
                candidates[s] = -1  # Exclude existing seeds

            next_seed = np.argmax(candidates)
            seeds.append(next_seed)

            # Update min distances
            new_distances = self.client_matrix[next_seed]
            min_dist_to_seeds = np.minimum(min_dist_to_seeds, new_distances)

        seeds = np.array(seeds)
        print(f"\nPhase 1 (Seed): Selected {len(seeds)} anchor clients")
        return seeds

    # ──────────────────────────────────────────────
    # PHASE 2: GROW — Assign remaining clients greedily
    # ──────────────────────────────────────────────

    def _compute_assignment_cost(self, client_idx: int, cluster_id: int,
                                  labels: np.ndarray) -> float:
        """
        Compute cost of assigning a client to a cluster.
        
        Blends geographic cost (average road distance to cluster members)
        with balance cost (how far the cluster is from target size).
        
        Lower cost = better assignment.
        """
        cluster_members = np.where(labels == cluster_id)[0]

        # --- Geographic cost ---
        if len(cluster_members) == 0:
            # Empty cluster: cost is distance from office to this client
            # (the seed should be the only member, but handle edge case)
            geo_cost = self.office_distances[client_idx]
        else:
            # Average road distance to existing cluster members
            distances_to_members = self.client_matrix[client_idx, cluster_members]
            geo_cost = np.mean(distances_to_members)

        # --- Balance cost ---
        current_size = len(cluster_members)
        # How much bigger than target this cluster already is
        # Positive = over target, negative = under target
        overshoot = (current_size - self.target_size) / self.target_size
        # Penalize clusters that are already above target
        balance_cost = max(0, overshoot) * geo_cost

        # --- Combined cost ---
        total_cost = (1 - self.balance_weight) * geo_cost + \
                     self.balance_weight * balance_cost * np.mean(self.office_distances)

        return total_cost

    def _grow(self, seeds: np.ndarray) -> np.ndarray:
        """
        Assign all non-seed clients to clusters, hardest clients first.
        
        "Hardest" = furthest from any seed (these have the least obvious 
        assignment, so we give them first pick).
        
        Returns:
            Labels array (cluster ID for each client).
        """
        labels = np.full(self.n_clients, -1, dtype=int)

        # Assign seeds to their clusters
        for cluster_id, seed_idx in enumerate(seeds):
            labels[seed_idx] = cluster_id

        # Compute min distance from each non-seed client to nearest seed
        unassigned = np.where(labels == -1)[0]
        seed_distances = self.client_matrix[np.ix_(unassigned, seeds)]
        min_dist_to_any_seed = np.min(seed_distances, axis=1)

        # Sort: hardest (furthest) first
        order = np.argsort(-min_dist_to_any_seed)
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
        print(f"\nPhase 2 (Grow): Assigned all clients")
        print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
              f"std={sizes.std():.1f}")
        return labels

    # ──────────────────────────────────────────────
    # PHASE 3: BALANCE — Local swap refinement
    # ──────────────────────────────────────────────

    def _compute_cluster_cost(self, labels: np.ndarray) -> float:
        """
        Compute overall cost of current assignment.
        Sum of (average intra-cluster road distance) for each cluster,
        plus a penalty for size imbalance.
        """
        total_geo_cost = 0.0
        sizes = []

        for cluster_id in range(self.n_clusters):
            members = np.where(labels == cluster_id)[0]
            if len(members) <= 1:
                sizes.append(len(members))
                continue

            # Average pairwise distance within cluster
            pairwise = self.client_matrix[np.ix_(members, members)]
            avg_dist = np.mean(pairwise[np.triu_indices(len(members), k=1)])
            total_geo_cost += avg_dist * len(members)
            sizes.append(len(members))

        sizes = np.array(sizes)
        size_penalty = np.std(sizes) * np.mean(self.office_distances)

        return total_geo_cost + self.balance_weight * size_penalty * self.n_clients

    def _find_boundary_clients(self, labels: np.ndarray) -> List[int]:
        """
        Find clients that are on the boundary between clusters.
        A client is a boundary client if its 2nd-best cluster assignment 
        is within 30% of its current cluster's cost.
        """
        boundary_clients = []

        for client_idx in range(self.n_clients):
            current_cluster = labels[client_idx]

            # Cost in current cluster
            current_members = np.where(labels == current_cluster)[0]
            current_members = current_members[current_members != client_idx]
            if len(current_members) == 0:
                current_cost = self.office_distances[client_idx]
            else:
                current_cost = np.mean(
                    self.client_matrix[client_idx, current_members])

            # Best alternative cluster cost
            best_alt_cost = np.inf
            for cluster_id in range(self.n_clusters):
                if cluster_id == current_cluster:
                    continue
                members = np.where(labels == cluster_id)[0]
                if len(members) == 0:
                    alt_cost = self.office_distances[client_idx]
                else:
                    alt_cost = np.mean(self.client_matrix[client_idx, members])
                if alt_cost < best_alt_cost:
                    best_alt_cost = alt_cost

            # If alternative is within 30% of current, it's a boundary client
            if best_alt_cost <= current_cost * 1.3:
                boundary_clients.append(client_idx)

        return boundary_clients

    def _refine(self, labels: np.ndarray) -> np.ndarray:
        """
        Iteratively try swapping boundary clients between clusters 
        to improve overall cost (geography + balance).
        """
        current_cost = self._compute_cluster_cost(labels)
        print(f"\nPhase 3 (Refine): Starting cost = {current_cost:.1f}")

        for iteration in range(self.max_refine_iterations):
            improved = False
            boundary = self._find_boundary_clients(labels)

            # Shuffle to avoid systematic bias
            np.random.shuffle(boundary)

            for client_idx in boundary:
                current_cluster = labels[client_idx]
                best_new_cluster = current_cluster
                best_new_cost = current_cost

                # Try moving to each other cluster
                for cluster_id in range(self.n_clusters):
                    if cluster_id == current_cluster:
                        continue

                    # Temporarily move
                    labels[client_idx] = cluster_id
                    new_cost = self._compute_cluster_cost(labels)

                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_new_cluster = cluster_id

                    # Undo
                    labels[client_idx] = current_cluster

                # Apply best move if it improved
                if best_new_cluster != current_cluster:
                    labels[client_idx] = best_new_cluster
                    current_cost = best_new_cost
                    improved = True

            self.history.append(labels.copy())

            sizes = self._get_cluster_sizes(labels)
            print(f"  Refine iteration {iteration + 1}: cost={current_cost:.1f}, "
                  f"boundary={len(boundary)}, sizes=[{sizes.min()}-{sizes.max()}]")

            if not improved:
                print(f"  Converged after {iteration + 1} refinement iterations.")
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

    # ──────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────

    def fit(self, visualize: bool = True) -> np.ndarray:
        """
        Run the full Seed → Grow → Balance pipeline.
        
        Returns:
            Labels array (cluster ID for each client, 0-based).
        """
        print(f"\n{'=' * 60}")
        print(f"SEED-GROW-BALANCE CLUSTERING")
        print(f"{'=' * 60}")

        # Phase 1: Seed
        self.seeds = self._seed()

        # Phase 2: Grow
        self.labels = self._grow(self.seeds)
        self.history.append(self.labels.copy())

        # Phase 3: Refine
        self.labels = self._refine(self.labels)

        # Final stats
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Cluster sizes: {sorted(sizes)}")
        print(f"  Size range: {sizes.min()} - {sizes.max()}")
        print(f"  Size std: {sizes.std():.2f}")
        print(f"{'=' * 60}")

        if visualize:
            self.visualize_final()

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
                    self.client_coords[mask, 1],  # lon
                    self.client_coords[mask, 0],  # lat
                    c=[colors[cluster_id]],
                    s=50,
                    alpha=0.7,
                    label=f"Cluster {cluster_id + 1} ({mask.sum()})"
                )

        # Plot seeds
        if self.seeds is not None:
            seed_coords = self.client_coords[self.seeds]
            ax.scatter(
                seed_coords[:, 1], seed_coords[:, 0],
                c='black', marker='*', s=200, zorder=5,
                label='Seeds'
            )

        # Plot office
        ax.scatter(
            self.office_coords[1], self.office_coords[0],
            c='red', marker='s', s=200, zorder=5,
            label='Office/Farm'
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        sizes = self._get_cluster_sizes(self.labels)
        ax.set_title(
            f"Seed-Grow-Balance | K={self.n_clusters} | "
            f"Sizes: {sizes.min()}-{sizes.max()} | "
            f"Std: {sizes.std():.1f}"
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig("Output/sgb_final.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: Output/sgb_final.png")

    def visualize_phases(self):
        """Visualize the progression through phases."""
        if len(self.history) == 0:
            print("No history to visualize. Run fit() first.")
            return

        n_plots = min(len(self.history), 8)
        indices = np.linspace(0, len(self.history) - 1, n_plots, dtype=int)

        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))

        for plot_idx, hist_idx in enumerate(indices):
            ax = axes[plot_idx]
            lbls = self.history[hist_idx]

            for cluster_id in range(self.n_clusters):
                mask = lbls == cluster_id
                if mask.sum() > 0:
                    ax.scatter(
                        self.client_coords[mask, 1],
                        self.client_coords[mask, 0],
                        c=[colors[cluster_id]],
                        s=20, alpha=0.7
                    )

            sizes = self._get_cluster_sizes(lbls)
            phase = "Grow" if hist_idx == 0 else f"Refine {hist_idx}"
            ax.set_title(f"{phase}\n[{sizes.min()}-{sizes.max()}]")

        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("Output/sgb_phases.png", dpi=150, bbox_inches='tight')
        plt.show()

    # ──────────────────────────────────────────────
    # For adding new clients later
    # ──────────────────────────────────────────────

    def assign_new_client(self, new_client_distances: np.ndarray) -> int:
        """
        Assign a single new client to the best existing cluster.
        
        Args:
            new_client_distances: Array of road distances from new client 
                                  to all existing clients (length = n_clients).
        
        Returns:
            Best cluster ID for the new client.
        """
        if self.labels is None:
            raise ValueError("Must run fit() before assigning new clients.")

        best_cluster = None
        best_cost = np.inf

        for cluster_id in range(self.n_clusters):
            members = np.where(self.labels == cluster_id)[0]
            if len(members) == 0:
                continue

            # Average distance to cluster members
            geo_cost = np.mean(new_client_distances[members])

            # Balance penalty
            overshoot = (len(members) - self.target_size) / self.target_size
            balance_penalty = max(0, overshoot) * geo_cost

            cost = (1 - self.balance_weight) * geo_cost + \
                   self.balance_weight * balance_penalty * np.mean(self.office_distances)

            if cost < best_cost:
                best_cost = cost
                best_cluster = cluster_id

        return best_cluster

    # ──────────────────────────────────────────────
    # Interface compatibility
    # ──────────────────────────────────────────────

    def get_labels(self) -> np.ndarray:
        """Get cluster labels for each client."""
        return self.labels

    def get_history(self) -> list:
        """Get labels at each phase/iteration."""
        return self.history