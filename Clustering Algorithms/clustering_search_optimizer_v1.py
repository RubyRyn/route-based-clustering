"""
Search-Based Clustering Optimization (v2)

Uses HDBSCAN natural clusters as initialization, then optimizes using
a multi-phase search algorithm.

Improvements:
- HDBSCAN baseline: Calculate loss before modifying outliers, keep as baseline
- Median + IQR normalization: Robust to outliers, data-driven

Loss Function (100 points):
- Convex Hull Overlap: 25 points
- Max Internal Road Distance: 20 points  
- Avg Internal Road Distance: 20 points
- Sum of Distances to Medoid: 15 points
- Workload Balance: 20 points (50% clients, 50% travel distance)

Search Strategy:
- Phase 1 (0-30%): Exploration - random moves + swaps
- Phase 2 (30-70%): Structuring - neighbor moves + swaps
- Phase 3 (70-100%): Refinement - swaps only, neighbor-aware
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from location import Location
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.validation import make_valid


class ClusteringSearchOptimizer:
    """
    Optimize cluster assignments using search algorithm.
    Initialized with HDBSCAN clusters, then refined through multi-phase search.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 max_clusters: int = 14,
                 min_clients_per_cluster: int = 1,
                 max_clients_per_cluster: int = 50,
                 no_improvement_limit: int = 500,
                 calibration_samples: int = 200,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            max_clusters: Maximum number of clusters allowed
            min_clients_per_cluster: Minimum clients per cluster
            max_clients_per_cluster: Maximum clients per cluster
            no_improvement_limit: Stop after N iterations with no improvement
            calibration_samples: Number of random solutions for normalization calibration
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_clusters = max_clusters
        self.min_clients_per_cluster = min_clients_per_cluster
        self.max_clients_per_cluster = max_clients_per_cluster
        self.no_improvement_limit = no_improvement_limit
        self.calibration_samples = calibration_samples
        
        np.random.seed(random_seed)
        
        self.num_locations = len(locations)
        self.num_clients = self.num_locations - 1
        
        # Client-only road distance matrix (exclude office)
        self.client_road_matrix = road_matrix[1:, 1:]
        
        # Office to clients distances
        self.office_to_clients = road_matrix[0, 1:]
        
        # Client coordinates for convex hull
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        
        # Results
        self.labels = None
        self.best_labels = None
        self.best_loss = np.inf
        self.hdbscan_baseline_labels = None
        self.hdbscan_baseline_loss = np.inf
        self.loss_history = []
        self.iteration_count = 0
        
        # Loss weights
        self.weight_overlap = 25
        self.weight_max_internal = 20
        self.weight_avg_internal = 20
        self.weight_medoid = 15
        self.weight_balance = 20
        
        # Normalization parameters (Median + IQR)
        self.norm_params = {
            'overlap': {'median': 0, 'iqr': 1},
            'max_internal': {'median': 0, 'iqr': 1},
            'avg_internal': {'median': 0, 'iqr': 1},
            'medoid': {'median': 0, 'iqr': 1},
            'balance': {'median': 0, 'iqr': 1}
        }
        self.is_calibrated = False
    
    def initialize_from_hdbscan(self, hdbscan_labels: np.ndarray):
        """
        Initialize with HDBSCAN labels.
        
        1. Calculate HDBSCAN baseline loss (outliers as individual clusters)
        2. Smartly assign outliers to nearest clusters with room
        
        Args:
            hdbscan_labels: Labels from HDBSCAN clustering (-1 = outlier)
        """
        print(f"\n{'-'*70}")
        print(f"INITIALIZING LABELS FROM HDBSCAN")
        print(f"{'-'*70}")
        outlier_indices = np.where(hdbscan_labels == -1)[0]
        cluster_labels = hdbscan_labels[hdbscan_labels >= 0]
        n_original_clusters = len(np.unique(cluster_labels)) if len(cluster_labels) > 0 else 0
        n_outliers = len(outlier_indices)
        print(f"HDBSCAN result: {n_original_clusters} clusters + {n_outliers} outliers")
        self.hdbscan_baseline_labels = hdbscan_labels.copy()
        next_cluster_id = n_original_clusters
        
        for idx in outlier_indices:
            self.hdbscan_baseline_labels[idx] = next_cluster_id
            next_cluster_id += 1
        self.hdbscan_baseline_labels = self._renumber_labels_static(self.hdbscan_baseline_labels)
        
        print(f"Baseline: {len(np.unique(self.hdbscan_baseline_labels))} total clusters (each outlier = 1 cluster)")
        
        # Step 2: Calculate baseline loss (will be calculated after calibration)
        # For now, store the labels
        
        # Step 3: Smart outlier assignment for search starting point
        # self.labels = hdbscan_labels.copy()
        # self._assign_outliers_smartly()
        # # Renumber labels
        # self.labels = self._renumber_labels_static(self.labels)
        # print(f"Search start: {len(np.unique(self.labels))} clusters (outliers assigned to nearest)")
        # # Print cluster sizes
        # unique = np.unique(self.labels)
        # sizes = [np.sum(self.labels == c) for c in unique]
        # print(f"Cluster sizes: {sizes}")

        # Use baseline labels as starting point (outliers as individual clusters)
        self.labels = self.hdbscan_baseline_labels.copy()
        print(f"Search start: {len(np.unique(self.labels))} clusters (outliers as individual clusters)")


    # def _assign_outliers_smartly(self):
    #     """
    #     Assign outliers to nearest cluster that has room.
    #     """
    #     outlier_indices = np.where(self.labels == -1)[0]
        
    #     if len(outlier_indices) == 0:
    #         return
        
    #     # Get existing clusters
    #     existing_clusters = np.unique(self.labels[self.labels >= 0])
        
    #     if len(existing_clusters) == 0:
    #         # No clusters exist, create one
    #         self.labels[outlier_indices[0]] = 0
    #         existing_clusters = [0]
    #         outlier_indices = outlier_indices[1:]
        
    #     for outlier_idx in outlier_indices:
    #         # Find nearest cluster with room
    #         best_cluster = None
    #         best_distance = np.inf
            
    #         for cluster_id in existing_clusters:
    #             cluster_indices = np.where(self.labels == cluster_id)[0]
    #             cluster_size = len(cluster_indices)
                
    #             # Check if cluster has room
    #             if cluster_size >= self.max_clients_per_cluster:
    #                 continue
                
    #             # Find minimum distance to any client in this cluster
    #             min_dist = min(self.client_road_matrix[outlier_idx, idx] for idx in cluster_indices)
                
    #             if min_dist < best_distance:
    #                 best_distance = min_dist
    #                 best_cluster = cluster_id
            
    #         if best_cluster is not None:
    #             self.labels[outlier_idx] = best_cluster
    #         else:
    #             # No cluster has room, assign to smallest cluster (violate constraint, search will fix)
    #             cluster_sizes = [(c, np.sum(self.labels == c)) for c in existing_clusters]
    #             smallest_cluster = min(cluster_sizes, key=lambda x: x[1])[0]
    #             self.labels[outlier_idx] = smallest_cluster
    
    def _renumber_labels_static(self, labels: np.ndarray) -> np.ndarray:
        """Renumber labels to be consecutive starting from 0."""
        unique_labels = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        return np.array([mapping[l] for l in labels])
    
    # ==================== CALIBRATION ====================
    
    def calibrate_normalization(self):
        """
        Generate random solutions and calculate Median + IQR for each metric.
        """
        print(f"\nCalibrating normalization with {self.calibration_samples} samples...")
        
        # Storage for raw metrics
        metrics = {
            'overlap': [],
            'max_internal': [],
            'avg_internal': [],
            'medoid': [],
            'balance': []
        }
        
        # Generate random valid solutions
        valid_samples = 0
        attempts = 0
        max_attempts = self.calibration_samples * 10
        
        while valid_samples < self.calibration_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate random labels
            n_clusters = np.random.randint(2, min(self.max_clusters + 1, self.num_clients))
            random_labels = np.random.randint(0, n_clusters, size=self.num_clients)
            
            # Check constraints
            if not self._check_constraints(random_labels):
                continue
            
            # Calculate raw metrics
            raw = self._calculate_raw_metrics(random_labels)
            
            if raw is None:
                continue
            
            for key in metrics:
                metrics[key].append(raw[key])
            
            valid_samples += 1
        
        print(f"  Generated {valid_samples} valid samples in {attempts} attempts")
        
        # Calculate Median and IQR for each metric
        for key in metrics:
            values = np.array(metrics[key])
            
            if len(values) == 0:
                continue
            
            median = np.median(values)
            q75 = np.percentile(values, 75)
            q25 = np.percentile(values, 25)
            iqr = q75 - q25
            
            # Avoid division by zero
            if iqr == 0:
                iqr = 1.0
            
            self.norm_params[key] = {'median': median, 'iqr': iqr}
            
            print(f"  {key}: median={median:.4f}, IQR={iqr:.4f}")
        
        self.is_calibrated = True
        
        # Now calculate HDBSCAN baseline loss
        self.hdbscan_baseline_loss, baseline_breakdown = self.calculate_total_loss(self.hdbscan_baseline_labels)
        print(f"\nHDBSCAN Baseline Loss: {self.hdbscan_baseline_loss:.4f}")
        self._print_breakdown(baseline_breakdown)
    
    # ==================== LOSS FUNCTIONS ====================
    
    def _calculate_raw_metrics(self, labels: np.ndarray) -> Optional[Dict]:
        """
        Calculate raw (unnormalized) metrics.
        
        Returns:
            Dictionary of raw metric values, or None if calculation fails
        """
        try:
            overlap = self._calculate_overlap_loss(labels)
            max_internal = self._calculate_max_internal_loss(labels)
            avg_internal = self._calculate_avg_internal_loss(labels)
            medoid = self._calculate_medoid_loss(labels)
            balance = self._calculate_balance_loss(labels)
            
            return {
                'overlap': overlap,
                'max_internal': max_internal,
                'avg_internal': avg_internal,
                'medoid': medoid,
                'balance': balance
            }
        except:
            return None
    
    def calculate_total_loss(self, labels: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate total loss for a labeling solution.
        Uses Median + IQR normalization.
        
        Returns:
            Tuple of (total_loss, breakdown_dict)
        """
        # Check constraints first
        if not self._check_constraints(labels):
            return np.inf, {}
        
        # Calculate raw metrics
        raw = self._calculate_raw_metrics(labels)
        
        if raw is None:
            return np.inf, {}
        
        # Normalize using Median + IQR
        # normalized = (value - median) / IQR
        # Lower (more negative) = better
        
        overlap_norm = (raw['overlap'] - self.norm_params['overlap']['median']) / self.norm_params['overlap']['iqr']
        max_internal_norm = (raw['max_internal'] - self.norm_params['max_internal']['median']) / self.norm_params['max_internal']['iqr']
        avg_internal_norm = (raw['avg_internal'] - self.norm_params['avg_internal']['median']) / self.norm_params['avg_internal']['iqr']
        medoid_norm = (raw['medoid'] - self.norm_params['medoid']['median']) / self.norm_params['medoid']['iqr']
        balance_norm = (raw['balance'] - self.norm_params['balance']['median']) / self.norm_params['balance']['iqr']
        
        # Apply weights
        total_loss = (
            self.weight_overlap * overlap_norm +
            self.weight_max_internal * max_internal_norm +
            self.weight_avg_internal * avg_internal_norm +
            self.weight_medoid * medoid_norm +
            self.weight_balance * balance_norm
        )
        
        breakdown = {
            'overlap': raw['overlap'],
            'max_internal': raw['max_internal'],
            'avg_internal': raw['avg_internal'],
            'medoid': raw['medoid'],
            'balance': raw['balance'],
            'overlap_norm': overlap_norm,
            'max_internal_norm': max_internal_norm,
            'avg_internal_norm': avg_internal_norm,
            'medoid_norm': medoid_norm,
            'balance_norm': balance_norm,
            'overlap_weighted': self.weight_overlap * overlap_norm,
            'max_internal_weighted': self.weight_max_internal * max_internal_norm,
            'avg_internal_weighted': self.weight_avg_internal * avg_internal_norm,
            'medoid_weighted': self.weight_medoid * medoid_norm,
            'balance_weighted': self.weight_balance * balance_norm,
            'total': total_loss
        }
        
        return total_loss, breakdown
    
    def _check_constraints(self, labels: np.ndarray) -> bool:
        """Check if labels satisfy all constraints."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Max clusters
        if n_clusters > self.max_clusters:
            return False
        
        # Min/max clients per cluster
        for cluster_id in unique_labels:
            cluster_size = np.sum(labels == cluster_id)
            if cluster_size < self.min_clients_per_cluster:
                return False
            if cluster_size > self.max_clients_per_cluster:
                return False
        
        return True
    
    def _calculate_overlap_loss(self, labels: np.ndarray) -> float:
        """
        Calculate convex hull overlap between clusters.
        Uses lat/lon coordinates.
        
        Returns:
            Total overlap area (in lat/lon units squared)
        """
        unique_labels = np.unique(labels)
        hulls = {}
        
        # Build convex hulls for each cluster
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 3:
                continue
            
            coords = self.client_coords[cluster_indices]
            
            try:
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                polygon = Polygon(hull_points)
                polygon = make_valid(polygon)
                hulls[cluster_id] = polygon
            except:
                continue
        
        # Calculate pairwise overlaps
        total_overlap = 0.0
        cluster_ids = list(hulls.keys())
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                poly1 = hulls[cluster_ids[i]]
                poly2 = hulls[cluster_ids[j]]
                
                try:
                    if poly1.intersects(poly2):
                        intersection = poly1.intersection(poly2)
                        total_overlap += intersection.area
                except:
                    continue
        
        return total_overlap
    
    def _calculate_max_internal_loss(self, labels: np.ndarray) -> float:
        """
        Calculate maximum internal road distance across all clusters.
        """
        max_distance = 0.0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        dist = self.client_road_matrix[i, j]
                        max_distance = max(max_distance, dist)
        
        return max_distance
    
    def _calculate_avg_internal_loss(self, labels: np.ndarray) -> float:
        """
        Calculate average internal road distance across all clusters.
        """
        total_distance = 0.0
        total_pairs = 0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        total_distance += self.client_road_matrix[i, j]
                        total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return total_distance / total_pairs
    
    def _calculate_medoid_loss(self, labels: np.ndarray) -> float:
        """
        Calculate sum of distances to medoid for all clusters.
        """
        total_loss = 0.0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if len(cluster_indices) == 1:
                continue
            
            # Find medoid
            min_total_dist = np.inf
            medoid_idx = cluster_indices[0]
            
            for candidate in cluster_indices:
                total_dist = sum(self.client_road_matrix[candidate, other] 
                               for other in cluster_indices)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    medoid_idx = candidate
            
            # Sum distances to medoid
            for idx in cluster_indices:
                total_loss += self.client_road_matrix[idx, medoid_idx]
        
        return total_loss
    
    def _calculate_balance_loss(self, labels: np.ndarray) -> float:
        """
        Calculate workload balance loss.
        50% client count, 50% travel distance.
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Client count balance
        client_counts = [np.sum(labels == cid) for cid in unique_labels]
        max_clients = max(client_counts)
        min_clients = min(client_counts)
        client_imbalance = (max_clients - min_clients) / max(max_clients, 1)
        
        # Travel distance balance
        travel_distances = []
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            travel_dist = self._estimate_travel_distance(cluster_indices)
            travel_distances.append(travel_dist)
        
        max_travel = max(travel_distances)
        min_travel = min(travel_distances)
        travel_imbalance = (max_travel - min_travel) / max(max_travel, 1)
        
        return 0.5 * client_imbalance + 0.5 * travel_imbalance
    
    def _estimate_travel_distance(self, cluster_indices: np.ndarray) -> float:
        """
        Estimate travel distance using nearest neighbor heuristic.
        """
        if len(cluster_indices) == 0:
            return 0.0
        
        if len(cluster_indices) == 1:
            return 2 * self.office_to_clients[cluster_indices[0]]
        
        unvisited = set(cluster_indices)
        total_distance = 0.0
        
        # Start from office
        current = min(unvisited, key=lambda x: self.office_to_clients[x])
        total_distance += self.office_to_clients[current]
        unvisited.remove(current)
        
        # Greedy nearest neighbor
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.client_road_matrix[current, x])
            total_distance += self.client_road_matrix[current, nearest]
            current = nearest
            unvisited.remove(current)
        
        # Return to office
        total_distance += self.office_to_clients[current]
        
        return total_distance
    
    # ==================== SEARCH MOVES ====================
    
    def _move_random_reassign(self, labels: np.ndarray) -> np.ndarray:
        """Move type A: Randomly reassign one client to a random cluster."""
        new_labels = labels.copy()
        
        client_idx = np.random.randint(0, self.num_clients)
        unique_labels = np.unique(labels)
        new_cluster = np.random.choice(unique_labels)
        
        new_labels[client_idx] = new_cluster
        
        return new_labels
    
    def _move_swap(self, labels: np.ndarray) -> np.ndarray:
        """Move type B: Swap two clients between different clusters."""
        new_labels = labels.copy()
        
        idx1 = np.random.randint(0, self.num_clients)
        cluster1 = labels[idx1]
        
        other_cluster_indices = np.where(labels != cluster1)[0]
        
        if len(other_cluster_indices) == 0:
            return new_labels
        
        idx2 = np.random.choice(other_cluster_indices)
        
        new_labels[idx1] = labels[idx2]
        new_labels[idx2] = labels[idx1]
        
        return new_labels
    
    def _move_neighbor(self, labels: np.ndarray) -> np.ndarray:
        """Move type C: Move client to a neighboring cluster."""
        new_labels = labels.copy()
        
        client_idx = np.random.randint(0, self.num_clients)
        current_cluster = labels[client_idx]
        
        other_cluster_indices = np.where(labels != current_cluster)[0]
        
        if len(other_cluster_indices) == 0:
            return new_labels
        
        # Find nearest neighbor in other clusters
        nearest_idx = min(other_cluster_indices, 
                         key=lambda x: self.client_road_matrix[client_idx, x])
        neighbor_cluster = labels[nearest_idx]
        
        new_labels[client_idx] = neighbor_cluster
        
        return new_labels
    
    def _get_phase(self, iteration: int, total_iterations: int) -> int:
        """Determine current phase based on iteration progress."""
        progress = iteration / total_iterations
        
        if progress < 0.3:
            return 1
        elif progress < 0.7:
            return 2
        else:
            return 3
    
    def _select_move(self, phase: int, current_loss: float) -> str:
        """Select move type based on phase and current loss."""
        # Adaptive: worse solution -> more random moves
        loss_factor = min(abs(current_loss) / 50.0, 1.0) if current_loss != np.inf else 1.0
        
        if phase == 1:
            r = np.random.random()
            if r < 0.6 + 0.2 * loss_factor:
                return 'random'
            else:
                return 'swap'
        
        elif phase == 2:
            r = np.random.random()
            if r < 0.1 * loss_factor:
                return 'random'
            elif r < 0.5:
                return 'neighbor'
            else:
                return 'swap'
        
        else:
            r = np.random.random()
            if r < 0.6:
                return 'swap'
            else:
                return 'neighbor'
    
    # ==================== MAIN SEARCH ====================
    
    def search(self, max_iterations: int = 10000, verbose: bool = True) -> np.ndarray:
        """
        Run the search algorithm.
        
        Returns:
            Optimized labels (best of HDBSCAN baseline and search result)
        """
        if self.labels is None:
            raise RuntimeError("Call initialize_from_hdbscan() first!")
        
        print(f"\n")
        print(f"SEARCH-BASED CLUSTERING OPTIMIZATION")
        print(f"\n")
        print(f"Clients: {self.num_clients}")
        print(f"Max clusters: {self.max_clusters}")
        print(f"Min clients per cluster: {self.min_clients_per_cluster}")
        print(f"Max clients per cluster: {self.max_clients_per_cluster}")
        print(f"No improvement limit: {self.no_improvement_limit}")
        print(f"Max iterations: {max_iterations}")
        
        # if not self.is_calibrated:
        #     self.calibrate_normalization()
        
        self.best_labels = self.labels.copy()
        self.best_loss, breakdown = self.calculate_total_loss(self.labels)
        self.loss_history = [self.best_loss]
        
        print(f"\nInitial loss (after outlier assignment): {self.best_loss:.4f}")
        self._print_breakdown(breakdown)
        
        print(f"\nHDBSCAN baseline loss: {self.hdbscan_baseline_loss:.4f}")
        
        if self.hdbscan_baseline_loss < self.best_loss:
            print(f"→ HDBSCAN baseline is better, using as starting point")
            self.best_loss = self.hdbscan_baseline_loss
            self.best_labels = self.hdbscan_baseline_labels.copy()
            self.labels = self.hdbscan_baseline_labels.copy()
        
        no_improvement_count = 0
        self.iteration_count = 0
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration
            
            phase = self._get_phase(iteration, max_iterations)
            move_type = self._select_move(phase, self.best_loss)
            
            if move_type == 'random':
                new_labels = self._move_random_reassign(self.labels)
            elif move_type == 'swap':
                new_labels = self._move_swap(self.labels)
            else:
                new_labels = self._move_neighbor(self.labels)
            
            new_loss, new_breakdown = self.calculate_total_loss(new_labels)
            
            if new_loss < self.best_loss:
                self.labels = new_labels
                self.best_labels = new_labels.copy()
                self.best_loss = new_loss
                no_improvement_count = 0
                
                if verbose and iteration % 100 == 0:
                    print(f"\nIteration {iteration} (Phase {phase}): New best = {self.best_loss:.4f}")
            else:
                no_improvement_count += 1
            
            self.loss_history.append(self.best_loss)
            
            if no_improvement_count >= self.no_improvement_limit:
                print(f"\nStopping: No improvement for {self.no_improvement_limit} iterations")
                break
            
            if verbose and iteration % 500 == 0 and iteration > 0:
                print(f"Iteration {iteration} (Phase {phase}): Best loss = {self.best_loss:.4f}")
        
        # Final comparison with HDBSCAN baseline
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {self.iteration_count + 1}")
        print(f"Search best loss: {self.best_loss:.4f}")
        print(f"HDBSCAN baseline loss: {self.hdbscan_baseline_loss:.4f}")
        
        if self.hdbscan_baseline_loss < self.best_loss:
            print(f"\n→ HDBSCAN BASELINE IS BETTER. Returning baseline.")
            self.best_labels = self.hdbscan_baseline_labels.copy()
            self.best_loss = self.hdbscan_baseline_loss
        else:
            print(f"\n→ SEARCH RESULT IS BETTER. Returning optimized solution.")
        
        _, final_breakdown = self.calculate_total_loss(self.best_labels)
        print(f"\nFinal loss: {self.best_loss:.4f}")
        self._print_breakdown(final_breakdown)
        self._print_cluster_summary()
        
        return self.best_labels
    
    def _print_breakdown(self, breakdown: Dict):
        """Print loss breakdown."""
        if not breakdown:
            print("  (No breakdown available)")
            return
        
        print(f"\nLoss Breakdown:")
        print(f"  Overlap:      {breakdown.get('overlap_weighted', 0):>8.2f} / {self.weight_overlap} (raw: {breakdown.get('overlap', 0):.6f}, norm: {breakdown.get('overlap_norm', 0):.2f})")
        print(f"  Max Internal: {breakdown.get('max_internal_weighted', 0):>8.2f} / {self.weight_max_internal} (raw: {breakdown.get('max_internal', 0):.2f} km, norm: {breakdown.get('max_internal_norm', 0):.2f})")
        print(f"  Avg Internal: {breakdown.get('avg_internal_weighted', 0):>8.2f} / {self.weight_avg_internal} (raw: {breakdown.get('avg_internal', 0):.2f} km, norm: {breakdown.get('avg_internal_norm', 0):.2f})")
        print(f"  Medoid:       {breakdown.get('medoid_weighted', 0):>8.2f} / {self.weight_medoid} (raw: {breakdown.get('medoid', 0):.2f} km, norm: {breakdown.get('medoid_norm', 0):.2f})")
        print(f"  Balance:      {breakdown.get('balance_weighted', 0):>8.2f} / {self.weight_balance} (raw: {breakdown.get('balance', 0):.4f}, norm: {breakdown.get('balance_norm', 0):.2f})")
    
    def _print_cluster_summary(self):
        """Print summary of final clusters."""
        print(f"\nCluster Summary:")
        
        unique_labels = np.unique(self.best_labels)
        clients = self.locations[1:]
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(self.best_labels == cluster_id)[0]
            travel_dist = self._estimate_travel_distance(cluster_indices)
            
            print(f"\n  Cluster {cluster_id + 1}: {len(cluster_indices)} clients, ~{travel_dist:.1f} km travel")
            
            client_names = [clients[idx].name for idx in cluster_indices]
            if len(client_names) <= 5:
                print(f"    Members: {', '.join(client_names)}")
            else:
                print(f"    Members: {', '.join(client_names[:5])} + {len(client_names) - 5} more")
    
    def get_labels(self) -> np.ndarray:
        """Get optimized cluster labels."""
        return self.best_labels
    
    def get_hdbscan_baseline_labels(self) -> np.ndarray:
        """Get HDBSCAN baseline labels (outliers as individual clusters)."""
        return self.hdbscan_baseline_labels
    
    def get_loss_history(self) -> List[float]:
        """Get loss history for plotting."""
        return self.loss_history