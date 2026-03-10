"""
Petal Clustering (v2)

Divides delivery area into petal-shaped slices radiating from the office.

Phase 1 (Slice)  — Sort clients by bearing from office.
                    Find the K largest angular gaps between consecutive clients.
                    Cut at those gaps so each petal is a full radial wedge
                    (near + far clients together, never split by distance).
Phase 2 (Swap)   — Fix road-distance violations between adjacent petals.
                    Uses median road distance (no K parameter needed).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from location import Location


class PetalClustering:
    """
    Petal-based partitioning: angular slices from the office,
    cut at natural angular gaps, refined by road distance.
    """

    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 n_clusters: int = 14,
                 size_tolerance: float = 0.20,
                 max_swap_passes: int = 50,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office at index 0, then clients).
            locations: List of Location objects (office first, then clients).
            n_clusters: Number of groups (employees / petals).
            size_tolerance: Allowed deviation from average (0.20 = ±20%).
            max_swap_passes: Max passes for road-distance swap refinement.
            random_seed: For reproducibility.
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.size_tolerance = size_tolerance
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

        # Client coordinates
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        self.office_coords = (locations[0].lat, locations[0].lon)

        # Compute bearings from office to each client
        self.bearings = self._compute_bearings()

        # Results
        self.labels = None
        self.labels_after_slice = None
        self.cut_bearings = None  # The angular boundaries where we cut
        self.history = []

        print(f"PetalClustering v2 initialized:")
        print(f"  Clients: {self.n_clients}")
        print(f"  Petals: {self.n_clusters}")
        print(f"  Target size: {self.target_size:.1f}")
        print(f"  Allowed range: [{self.min_size}, {self.max_size}]")

    # ══════════════════════════════════════════════
    # BEARING COMPUTATION
    # ══════════════════════════════════════════════

    def _compute_bearings(self) -> np.ndarray:
        """
        Compute bearing (angle in degrees, 0-360) from office to each client.
        0° = North, 90° = East, 180° = South, 270° = West.
        """
        office_lat = np.radians(self.office_coords[0])
        office_lon = np.radians(self.office_coords[1])

        client_lats = np.radians(self.client_coords[:, 0])
        client_lons = np.radians(self.client_coords[:, 1])

        dlon = client_lons - office_lon

        x = np.sin(dlon) * np.cos(client_lats)
        y = (np.cos(office_lat) * np.sin(client_lats) -
             np.sin(office_lat) * np.cos(client_lats) * np.cos(dlon))

        bearings = np.degrees(np.arctan2(x, y))
        bearings = (bearings + 360) % 360

        return bearings

    # ══════════════════════════════════════════════
    # PHASE 1: SLICE — Gap-based angular cutting
    # ══════════════════════════════════════════════

    def _compute_angular_gap(self, angle1: float, angle2: float) -> float:
        """Compute the angular gap between two bearings (handles wraparound)."""
        diff = angle2 - angle1
        if diff < 0:
            diff += 360
        return diff

    def _slice(self) -> np.ndarray:
        """
        Sort clients by bearing, find K largest angular gaps,
        cut at those gaps.

        This ensures each petal is a full radial wedge — clients at the
        same bearing but different distances always stay together.
        """
        # Sort client indices by bearing
        sorted_indices = np.argsort(self.bearings)
        sorted_bearings = self.bearings[sorted_indices]
        n = len(sorted_indices)

        # Compute angular gap between each consecutive pair
        # (including wrap-around from last to first)
        gaps = []
        for i in range(n):
            next_i = (i + 1) % n
            gap = self._compute_angular_gap(
                sorted_bearings[i], sorted_bearings[next_i])
            gaps.append((gap, i))  # (gap_size, index_after_which_to_cut)

        # We need K cuts. But we can't just take the K largest gaps blindly
        # because that might create very unbalanced petals.
        # Strategy: score each gap by combining gap size with how close
        # the resulting petal sizes would be to the target.

        # First, find candidate cut positions: gaps that are large enough
        # to be natural boundaries
        gaps.sort(key=lambda x: -x[0])  # Largest gaps first

        # Take top K * 3 candidates (more than we need for flexibility)
        n_candidates = min(len(gaps), self.n_clusters * 3)
        candidate_gaps = sorted(gaps[:n_candidates], key=lambda x: x[1])

        # Now select K cuts from candidates that give best size balance
        # Use greedy selection: start with the largest gap, then add gaps
        # that best balance the resulting segment sizes
        selected_cuts = self._select_balanced_cuts(
            sorted_indices, sorted_bearings, gaps)

        # Assign labels based on selected cuts
        labels = np.full(self.n_clients, -1, dtype=int)

        # Sort cut positions in order around the circle
        cut_positions = sorted(selected_cuts)
        self.cut_bearings = []

        for k in range(self.n_clusters):
            start = cut_positions[k] + 1
            if k < self.n_clusters - 1:
                end = cut_positions[k + 1]
            else:
                end = cut_positions[0] + n  # Wrap around

            # Assign clients in this segment
            for i in range(start, end + 1):
                idx = sorted_indices[i % n]
                labels[idx] = k

            # Record cut bearing
            self.cut_bearings.append(sorted_bearings[cut_positions[k] % n])

        sizes = self._get_cluster_sizes(labels)
        print(f"\nPhase 1 (Slice): Gap-based angular cutting")
        print(f"  Sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}], Std: {sizes.std():.1f}")

        return labels

    def _select_balanced_cuts(self, sorted_indices: np.ndarray,
                               sorted_bearings: np.ndarray,
                               gaps: List[Tuple[float, int]]) -> List[int]:
        """
        Select K gap positions that give the best balance between
        angular gap size and resulting segment sizes.

        Uses a greedy approach:
        1. Always include the largest gap (most natural boundary)
        2. For each remaining cut, pick the gap that best splits
           the largest existing segment
        """
        n = len(sorted_indices)

        # Start with the largest gap
        gaps_by_size = sorted(gaps, key=lambda x: -x[0])
        selected = [gaps_by_size[0][1]]  # Position of largest gap

        for _ in range(1, self.n_clusters):
            # Compute current segment sizes
            cuts_sorted = sorted(selected)
            segments = []
            for i in range(len(cuts_sorted)):
                start = cuts_sorted[i] + 1
                if i < len(cuts_sorted) - 1:
                    end = cuts_sorted[i + 1]
                    size = end - start + 1
                else:
                    end = cuts_sorted[0] + n
                    size = end - start + 1
                segments.append((size, start, end, i))

            # Find the largest segment — it needs to be split
            segments.sort(key=lambda x: -x[0])
            largest_size, seg_start, seg_end, seg_idx = segments[0]

            # Find the best gap within this segment
            # "Best" = largest angular gap that falls within the segment
            best_gap_pos = None
            best_gap_size = -1

            for gap_size, gap_pos in gaps_by_size:
                if gap_pos in selected:
                    continue

                # Check if this gap is within the largest segment
                # Need to handle wrap-around
                pos_in_segment = False
                if seg_end >= seg_start:
                    pos_in_segment = (seg_start <= gap_pos <= seg_end)
                else:
                    pos_in_segment = (gap_pos >= seg_start or gap_pos <= seg_end % n)

                if pos_in_segment:
                    # Also check that splitting here doesn't create
                    # a segment smaller than min_size
                    left_size = gap_pos - seg_start + 1
                    right_size = largest_size - left_size
                    if seg_end < seg_start:
                        # Wrap around case
                        if gap_pos >= seg_start:
                            left_size = gap_pos - seg_start + 1
                        else:
                            left_size = (n - seg_start) + gap_pos + 1
                        right_size = largest_size - left_size

                    if left_size >= self.min_size and right_size >= self.min_size:
                        if gap_size > best_gap_size:
                            best_gap_size = gap_size
                            best_gap_pos = gap_pos

            # If no valid gap in the largest segment, try splitting
            # at the midpoint of the segment
            if best_gap_pos is None:
                # Find any gap near the midpoint
                mid = (seg_start + largest_size // 2) % n
                best_dist_to_mid = float('inf')

                for gap_size, gap_pos in gaps_by_size:
                    if gap_pos in selected:
                        continue
                    dist_to_mid = min(abs(gap_pos - mid), n - abs(gap_pos - mid))
                    if dist_to_mid < best_dist_to_mid:
                        best_dist_to_mid = dist_to_mid
                        best_gap_pos = gap_pos

            if best_gap_pos is not None:
                selected.append(best_gap_pos)
            else:
                # Fallback: just pick the largest unused gap
                for gap_size, gap_pos in gaps_by_size:
                    if gap_pos not in selected:
                        selected.append(gap_pos)
                        break

        return selected

    # ══════════════════════════════════════════════
    # PHASE 2: SWAP — Road-distance-based refinement
    # ══════════════════════════════════════════════

    def _find_violations(self, labels: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """
        Find clients closer (by median road distance) to an adjacent
        petal than to their own.

        Returns list of (client_idx, current_cluster, better_cluster, severity)
        sorted most severe first.
        """
        violations = []

        for client_idx in range(self.n_clients):
            current_cluster = labels[client_idx]

            # Median road distance to own cluster
            own_members = np.where(labels == current_cluster)[0]
            own_members = own_members[own_members != client_idx]
            if len(own_members) == 0:
                own_median = self.office_distances[client_idx]
            else:
                own_median = np.median(
                    self.client_matrix[client_idx, own_members])

            # Check adjacent petals only
            neighbors = self._get_adjacent_petals(current_cluster, labels)

            best_alt_cluster = None
            best_alt_median = own_median

            for alt_cluster in neighbors:
                alt_members = np.where(labels == alt_cluster)[0]
                if len(alt_members) == 0:
                    continue
                alt_median = np.median(
                    self.client_matrix[client_idx, alt_members])

                if alt_median < best_alt_median:
                    best_alt_median = alt_median
                    best_alt_cluster = alt_cluster

            if best_alt_cluster is not None:
                severity = own_median - best_alt_median
                violations.append(
                    (client_idx, current_cluster, best_alt_cluster, severity))

        violations.sort(key=lambda x: -x[3])
        return violations

    def _get_adjacent_petals(self, petal_id: int,
                              labels: np.ndarray) -> List[int]:
        """
        Get petals adjacent to this one in angular order.
        Wraps around. Skips empty petals to find actual neighbors.
        """
        neighbors = []

        # Search forward
        for offset in range(1, self.n_clusters):
            candidate = (petal_id + offset) % self.n_clusters
            if np.sum(labels == candidate) > 0:
                neighbors.append(candidate)
                break

        # Search backward
        for offset in range(1, self.n_clusters):
            candidate = (petal_id - offset) % self.n_clusters
            if np.sum(labels == candidate) > 0:
                if candidate not in neighbors:
                    neighbors.append(candidate)
                break

        return neighbors

    def _swap(self, labels: np.ndarray) -> np.ndarray:
        """
        Road-distance-based swap refinement.
        Only swaps between adjacent petals.
        Runs until violations stop decreasing.
        """
        print(f"\nPhase 2 (Swap): Road-distance refinement")
        print(f"  Size guardrails: [{self.min_size}, {self.max_size}]")
        print(f"  Only swapping between adjacent petals")

        prev_violation_count = float('inf')

        for pass_num in range(self.max_swap_passes):
            violations = self._find_violations(labels)
            current_count = len(violations)

            if current_count == 0:
                print(f"  Pass {pass_num + 1}: 0 violations — clean petals!")
                break

            if current_count >= prev_violation_count:
                print(f"  Pass {pass_num + 1}: {current_count} violations "
                      f"(not improving, stopping)")
                break

            # Execute swaps
            sizes = self._get_cluster_sizes(labels)
            swaps_made = 0
            swaps_skipped = 0

            for client_idx, src_cluster, tgt_cluster, severity in violations:
                src_size_after = sizes[src_cluster] - 1
                tgt_size_after = sizes[tgt_cluster] + 1

                if src_size_after < self.min_size:
                    swaps_skipped += 1
                    continue
                if tgt_size_after > self.max_size:
                    swaps_skipped += 1
                    continue

                labels[client_idx] = tgt_cluster
                sizes[src_cluster] -= 1
                sizes[tgt_cluster] += 1
                swaps_made += 1

            self.history.append(labels.copy())

            sizes = self._get_cluster_sizes(labels)
            print(f"  Pass {pass_num + 1}: {current_count} violations, "
                  f"{swaps_made} swapped, {swaps_skipped} skipped, "
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
            member_bearings = self.bearings[members]

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
                'bearing_range': (member_bearings.min(), member_bearings.max()),
            })

        sizes = [c['size'] for c in stats['clusters']]
        stats['size_std'] = np.std(sizes)
        stats['size_range'] = (min(sizes), max(sizes))
        return stats

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def fit(self, visualize: bool = True) -> np.ndarray:
        """Run the full Slice → Swap pipeline."""
        print(f"\n{'=' * 60}")
        print(f"PETAL CLUSTERING v2 (Gap-Based Cutting)")
        print(f"{'=' * 60}")

        # Phase 1
        self.labels = self._slice()
        self.labels_after_slice = self.labels.copy()
        self.history.append(self.labels.copy())

        # Phase 2
        self.labels = self._swap(self.labels)

        # Final report
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Cluster sizes: {sorted(sizes)}")
        print(f"  Range: [{sizes.min()}, {sizes.max()}]")
        print(f"  Std: {sizes.std():.2f}")

        remaining = self._find_violations(self.labels)
        print(f"  Remaining violations: {len(remaining)}")

        stats = self.get_cluster_stats()
        intra_dists = [c['avg_intra_dist'] for c in stats['clusters']]
        print(f"  Avg intra-cluster distance: {np.mean(intra_dists):.1f}")
        print(f"{'=' * 60}")

        if visualize:
            self.visualize_petals()
            self.visualize_comparison()
            self.visualize_polar()

        return self.labels

    # ══════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════

    def _draw_petals(self, ax, labels, colors, show_violations=True):
        """Draw petal visualization on an axis. Returns violation count."""
        lons = self.client_coords[:, 1]
        lats = self.client_coords[:, 0]
        office_lon = self.office_coords[1]
        office_lat = self.office_coords[0]

        # Draw cut lines from office outward
        if self.cut_bearings is not None:
            max_dist = max(
                np.max(np.abs(lons - office_lon)),
                np.max(np.abs(lats - office_lat))
            ) * 1.2

            for bearing in self.cut_bearings:
                rad = np.radians(bearing)
                end_lon = office_lon + max_dist * np.sin(rad)
                end_lat = office_lat + max_dist * np.cos(rad)
                ax.plot(
                    [office_lon, end_lon], [office_lat, end_lat],
                    color='gray', alpha=0.3, linewidth=1,
                    linestyle='--', zorder=1
                )

        # Find violations
        violations = self._find_violations(labels) if show_violations else []

        # Plot client dots
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() > 0:
                ax.scatter(lons[mask], lats[mask],
                           c=[colors[cluster_id]], s=40, alpha=0.8,
                           edgecolors='white', linewidths=0.5, zorder=3,
                           label=f"P{cluster_id + 1} ({mask.sum()})")

        # Highlight violations
        if violations:
            v_idx = [v[0] for v in violations]
            ax.scatter(lons[v_idx], lats[v_idx],
                       c='none', s=150, edgecolors='red',
                       linewidths=2.5, zorder=4,
                       label=f"Violations ({len(violations)})")

        # Office
        ax.scatter(office_lon, office_lat,
                   c='red', marker='*', s=300, zorder=5,
                   edgecolors='black', linewidths=1.5,
                   label='Farm')

        return len(violations)

    def visualize_petals(self, filename: str = "Output/petal_v2_final.png"):
        """Visualize final petal clustering."""
        if self.labels is None:
            print("Run fit() first.")
            return

        fig, ax = plt.subplots(figsize=(14, 10))

        if self.n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:self.n_clusters]
        else:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.n_clusters))

        n_violations = self._draw_petals(ax, self.labels, colors)

        sizes = self._get_cluster_sizes(self.labels)
        ax.set_title(
            f"Petal Clustering v2 | K={self.n_clusters}\n"
            f"Sizes: [{sizes.min()}-{sizes.max()}] | "
            f"Std: {sizes.std():.1f} | "
            f"Violations: {n_violations}",
            fontsize=12
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    def visualize_comparison(self, filename: str = "Output/petal_v2_comparison.png"):
        """Side-by-side: after Slice vs after Swap."""
        if self.labels_after_slice is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        if self.n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:self.n_clusters]
        else:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.n_clusters))

        for ax, lbls, title in zip(
            axes,
            [self.labels_after_slice, self.labels],
            ["After Slice (before swapping)", "After Swap (final)"]
        ):
            n_v = self._draw_petals(ax, lbls, colors, show_violations=True)
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

    def visualize_polar(self, filename: str = "Output/petal_v2_polar.png"):
        """
        Polar plot: angle = bearing from office, radius = road distance.
        Shows the petal slices as angular wedges.
        """
        if self.labels is None:
            print("Run fit() first.")
            return

        fig, ax = plt.subplots(figsize=(10, 10),
                                subplot_kw={'projection': 'polar'})

        if self.n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:self.n_clusters]
        else:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, self.n_clusters))

        bearing_rads = np.radians(self.bearings)

        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            if mask.sum() > 0:
                ax.scatter(
                    bearing_rads[mask],
                    self.office_distances[mask],
                    c=[colors[cluster_id]],
                    s=30, alpha=0.7,
                    label=f"P{cluster_id + 1} ({mask.sum()})"
                )

        # Draw cut lines on polar plot
        if self.cut_bearings is not None:
            max_r = np.max(self.office_distances) * 1.1
            for bearing in self.cut_bearings:
                rad = np.radians(bearing)
                ax.plot([rad, rad], [0, max_r],
                        color='gray', alpha=0.4, linewidth=1,
                        linestyle='--')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title("Petal View (Polar)\nRadius = Road Distance from Office",
                     pad=20, fontsize=12)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    # ══════════════════════════════════════════════
    # New client assignment
    # ══════════════════════════════════════════════

    def assign_new_client(self, new_location: Location,
                          new_client_distances: np.ndarray) -> int:
        """
        Assign a new client to the best petal.
        Primary: bearing from office (which petal direction).
        Secondary: road distance to petal members.
        Tertiary: size constraint.
        """
        if self.labels is None:
            raise ValueError("Run fit() first.")

        # Compute bearing for new client
        office_lat = np.radians(self.office_coords[0])
        office_lon = np.radians(self.office_coords[1])
        client_lat = np.radians(new_location.lat)
        client_lon = np.radians(new_location.lon)

        dlon = client_lon - office_lon
        x = np.sin(dlon) * np.cos(client_lat)
        y = (np.cos(office_lat) * np.sin(client_lat) -
             np.sin(office_lat) * np.cos(client_lat) * np.cos(dlon))
        bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360

        sizes = self._get_cluster_sizes(self.labels)
        best_cluster = None
        best_score = np.inf

        for cluster_id in range(self.n_clusters):
            members = np.where(self.labels == cluster_id)[0]
            if len(members) == 0:
                continue

            # Angular distance to petal's mean bearing
            petal_bearings = self.bearings[members]
            mean_bearing = np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(petal_bearings))),
                np.mean(np.cos(np.radians(petal_bearings)))
            ))
            mean_bearing = (mean_bearing + 360) % 360
            angular_dist = min(abs(bearing - mean_bearing),
                              360 - abs(bearing - mean_bearing))

            # Road distance to petal members
            road_dist = np.median(new_client_distances[members])

            # Size penalty
            size_factor = 1.0
            if sizes[cluster_id] >= self.max_size:
                size_factor = 3.0
            elif sizes[cluster_id] > self.target_size:
                size_factor = 1.3

            score = (0.4 * angular_dist + 0.6 * road_dist) * size_factor

            if score < best_score:
                best_score = score
                best_cluster = cluster_id

        return best_cluster

    def get_labels(self) -> np.ndarray:
        return self.labels

    def get_history(self) -> list:
        return self.history
