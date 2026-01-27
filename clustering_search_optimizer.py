"""
Search-Based Clustering Optimization

Uses HDBSCAN natural clusters as initialization, then optimizes using
a multi-phase search algorithm.

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
                 min_clients_per_cluster: int = 3,
                 max_clients_per_cluster: int = 10,
                 no_improvement_limit: int = 500,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            max_clusters: Maximum number of clusters allowed
            min_clients_per_cluster: Minimum clients per cluster
            max_clients_per_cluster: Maximum clients per cluster
            no_improvement_limit: Stop after N iterations with no improvement
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_clusters = max_clusters
        self.min_clients_per_cluster = min_clients_per_cluster
        self.max_clients_per_cluster = max_clients_per_cluster
        self.no_improvement_limit = no_improvement_limit
        
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
        self.loss_history = []
        self.iteration_count = 0
        
        # Loss weights
        self.weight_overlap = 25
        self.weight_max_internal = 20
        self.weight_avg_internal = 20
        self.weight_medoid = 15
        self.weight_balance = 20
    
    def initialize_from_hdbscan(self, hdbscan_labels: np.ndarray):
        """
        Initialize with HDBSCAN labels.
        Outliers (-1) will be assigned by the search algorithm.
        
        Args:
            hdbscan_labels: Labels from HDBSCAN clustering
        """
        self.labels = hdbscan_labels.copy()
        
        # Assign outliers to random clusters initially
        outlier_indices = np.where(self.labels == -1)[0]
        n_clusters = len(np.unique(self.labels[self.labels >= 0]))
        
        if n_clusters == 0:
            n_clusters = 1
        
        for idx in outlier_indices:
            self.labels[idx] = np.random.randint(0, max(n_clusters, 1))
        
        # Renumber labels to be consecutive (0, 1, 2, ...)
        self._renumber_labels()
        
        print(f"Initialized with HDBSCAN labels")
        print(f"  Clusters: {len(np.unique(self.labels))}")
        print(f"  Outliers assigned: {len(outlier_indices)}")
    
    def _renumber_labels(self):
        """Renumber labels to be consecutive starting from 0."""
        unique_labels = np.unique(self.labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        self.labels = np.array([mapping[l] for l in self.labels])
    
    # ==================== LOSS FUNCTIONS ====================
    
    def calculate_total_loss(self, labels: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate total loss for a labeling solution.
        
        Returns:
            Tuple of (total_loss, breakdown_dict)
        """
        # Check constraints first
        if not self._check_constraints(labels):
            return np.inf, {}
        
        # Calculate each component
        overlap_loss = self._calculate_overlap_loss(labels)
        max_internal_loss = self._calculate_max_internal_loss(labels)
        avg_internal_loss = self._calculate_avg_internal_loss(labels)
        medoid_loss = self._calculate_medoid_loss(labels)
        balance_loss = self._calculate_balance_loss(labels)
        
        # Normalize losses to 0-1 scale, then apply weights
        # (Normalization factors based on typical values - can be tuned)
        overlap_normalized = min(overlap_loss / 0.001, 1.0)  # Area in lat/lon units
        max_internal_normalized = min(max_internal_loss / 50.0, 1.0)  # 50km max
        avg_internal_normalized = min(avg_internal_loss / 20.0, 1.0)  # 20km avg
        medoid_normalized = min(medoid_loss / 100.0, 1.0)  # 100km sum
        balance_normalized = min(balance_loss / 1.0, 1.0)  # Already 0-1 scale
        
        total_loss = (
            self.weight_overlap * overlap_normalized +
            self.weight_max_internal * max_internal_normalized +
            self.weight_avg_internal * avg_internal_normalized +
            self.weight_medoid * medoid_normalized +
            self.weight_balance * balance_normalized
        )
        
        breakdown = {
            'overlap': overlap_loss,
            'max_internal': max_internal_loss,
            'avg_internal': avg_internal_loss,
            'medoid': medoid_loss,
            'balance': balance_loss,
            'overlap_weighted': self.weight_overlap * overlap_normalized,
            'max_internal_weighted': self.weight_max_internal * max_internal_normalized,
            'avg_internal_weighted': self.weight_avg_internal * avg_internal_normalized,
            'medoid_weighted': self.weight_medoid * medoid_normalized,
            'balance_weighted': self.weight_balance * balance_normalized,
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
                # Need at least 3 points for a hull
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
        
        Returns:
            The longest road distance between any two clients in the same cluster
        """
        max_distance = 0.0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Find max pairwise distance in this cluster
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        dist = self.client_road_matrix[i, j]
                        max_distance = max(max_distance, dist)
        
        return max_distance
    
    def _calculate_avg_internal_loss(self, labels: np.ndarray) -> float:
        """
        Calculate average internal road distance across all clusters.
        
        Returns:
            Average of all pairwise road distances within clusters
        """
        total_distance = 0.0
        total_pairs = 0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            # Sum all pairwise distances
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
        
        Returns:
            Total sum of distances from each client to its cluster's medoid
        """
        total_loss = 0.0
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Find medoid (client with minimum total distance to others)
            min_total_dist = np.inf
            medoid_idx = cluster_indices[0]
            
            for candidate in cluster_indices:
                total_dist = sum(self.client_road_matrix[candidate, other] 
                               for other in cluster_indices)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    medoid_idx = candidate
            
            # Sum distances from all clients to medoid
            for idx in cluster_indices:
                total_loss += self.client_road_matrix[idx, medoid_idx]
        
        return total_loss
    
    def _calculate_balance_loss(self, labels: np.ndarray) -> float:
        """
        Calculate workload balance loss.
        50% based on client count, 50% based on travel distance.
        
        Returns:
            Balance loss (0-1 scale, lower is better)
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Client count balance
        client_counts = [np.sum(labels == cid) for cid in unique_labels]
        max_clients = max(client_counts)
        min_clients = min(client_counts)
        client_imbalance = (max_clients - min_clients) / max(max_clients, 1)
        
        # Travel distance balance (using nearest neighbor heuristic)
        travel_distances = []
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            travel_dist = self._estimate_travel_distance(cluster_indices)
            travel_distances.append(travel_dist)
        
        max_travel = max(travel_distances)
        min_travel = min(travel_distances)
        travel_imbalance = (max_travel - min_travel) / max(max_travel, 1)
        
        # 50-50 weighted
        balance_loss = 0.5 * client_imbalance + 0.5 * travel_imbalance
        
        return balance_loss
    
    def _estimate_travel_distance(self, cluster_indices: np.ndarray) -> float:
        """
        Estimate travel distance using nearest neighbor heuristic.
        Office -> nearest client -> nearest unvisited -> ... -> Office
        
        Args:
            cluster_indices: Indices of clients in the cluster (client space)
            
        Returns:
            Estimated total travel distance in km
        """
        if len(cluster_indices) == 0:
            return 0.0
        
        if len(cluster_indices) == 1:
            # Office -> client -> Office
            return 2 * self.office_to_clients[cluster_indices[0]]
        
        # Start from office, find nearest client
        unvisited = set(cluster_indices)
        total_distance = 0.0
        
        # Find nearest client to office
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
        """
        Move type A: Randomly reassign one client to a random cluster.
        """
        new_labels = labels.copy()
        
        client_idx = np.random.randint(0, self.num_clients)
        unique_labels = np.unique(labels)
        new_cluster = np.random.choice(unique_labels)
        
        new_labels[client_idx] = new_cluster
        
        return new_labels
    
    def _move_swap(self, labels: np.ndarray) -> np.ndarray:
        """
        Move type B: Swap two clients between different clusters.
        """
        new_labels = labels.copy()
        
        # Pick two clients from different clusters
        idx1 = np.random.randint(0, self.num_clients)
        cluster1 = labels[idx1]
        
        # Find clients in other clusters
        other_cluster_indices = np.where(labels != cluster1)[0]
        
        if len(other_cluster_indices) == 0:
            return new_labels
        
        idx2 = np.random.choice(other_cluster_indices)
        
        # Swap
        new_labels[idx1] = labels[idx2]
        new_labels[idx2] = labels[idx1]
        
        return new_labels
    
    def _move_neighbor(self, labels: np.ndarray) -> np.ndarray:
        """
        Move type C: Move client to a neighboring cluster 
        (cluster that has a client nearby).
        """
        new_labels = labels.copy()
        
        # Pick a random client
        client_idx = np.random.randint(0, self.num_clients)
        current_cluster = labels[client_idx]
        
        # Find nearest client in a different cluster
        other_cluster_indices = np.where(labels != current_cluster)[0]
        
        if len(other_cluster_indices) == 0:
            return new_labels
        
        # Find nearest neighbor in other clusters
        nearest_idx = min(other_cluster_indices, 
                         key=lambda x: self.client_road_matrix[client_idx, x])
        neighbor_cluster = labels[nearest_idx]
        
        # Move to neighbor's cluster
        new_labels[client_idx] = neighbor_cluster
        
        return new_labels
    
    def _get_phase(self, iteration: int, total_iterations: int) -> int:
        """
        Determine current phase based on iteration progress.
        
        Returns:
            1, 2, or 3
        """
        progress = iteration / total_iterations
        
        if progress < 0.3:
            return 1  # Exploration
        elif progress < 0.7:
            return 2  # Structuring
        else:
            return 3  # Refinement
    
    def _select_move(self, phase: int, current_loss: float) -> str:
        """
        Select move type based on phase and current loss.
        Worse solutions allow bolder moves.
        """
        # Adaptive: worse solution -> more random moves
        loss_factor = min(current_loss / 50.0, 1.0)  # Normalize
        
        if phase == 1:
            # Exploration: mostly random + some swaps
            r = np.random.random()
            if r < 0.6 + 0.2 * loss_factor:
                return 'random'
            else:
                return 'swap'
        
        elif phase == 2:
            # Structuring: neighbor moves + regular swaps + rare random
            r = np.random.random()
            if r < 0.1 * loss_factor:
                return 'random'
            elif r < 0.5:
                return 'neighbor'
            else:
                return 'swap'
        
        else:
            # Refinement: swaps + neighbor-aware only
            r = np.random.random()
            if r < 0.6:
                return 'swap'
            else:
                return 'neighbor'
    
    # ==================== MAIN SEARCH ====================
    
    def search(self, max_iterations: int = 1000, verbose: bool = True) -> np.ndarray:
        """
        Run the search algorithm.
        
        Args:
            max_iterations: Maximum total iterations
            verbose: Print progress
            
        Returns:
            Optimized labels
        """
        if self.labels is None:
            raise RuntimeError("Call initialize_from_hdbscan() first!")
        
        print(f"\n{'='*70}")
        print(f"SEARCH-BASED CLUSTERING OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Clients: {self.num_clients}")
        print(f"Max clusters: {self.max_clusters}")
        print(f"Min clients/cluster: {self.min_clients_per_cluster}")
        print(f"Max clients/cluster: {self.max_clients_per_cluster}")
        print(f"No improvement limit: {self.no_improvement_limit}")
        print(f"Max iterations: {max_iterations}")
        
        # Initial loss
        self.best_labels = self.labels.copy()
        self.best_loss, breakdown = self.calculate_total_loss(self.labels)
        self.loss_history = [self.best_loss]
        
        print(f"\nInitial loss: {self.best_loss:.4f}")
        self._print_breakdown(breakdown)
        
        no_improvement_count = 0
        self.iteration_count = 0
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration
            
            # Determine phase
            phase = self._get_phase(iteration, max_iterations)
            
            # Select and apply move
            move_type = self._select_move(phase, self.best_loss)
            
            if move_type == 'random':
                new_labels = self._move_random_reassign(self.labels)
            elif move_type == 'swap':
                new_labels = self._move_swap(self.labels)
            else:
                new_labels = self._move_neighbor(self.labels)
            
            # Evaluate new solution
            new_loss, new_breakdown = self.calculate_total_loss(new_labels)
            
            # Accept if better
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
            
            # Check stopping condition
            if no_improvement_count >= self.no_improvement_limit:
                print(f"\nStopping: No improvement for {self.no_improvement_limit} iterations")
                break
            
            # Progress update
            if verbose and iteration % 500 == 0 and iteration > 0:
                print(f"Iteration {iteration} (Phase {phase}): Best loss = {self.best_loss:.4f}")
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {self.iteration_count + 1}")
        print(f"Final loss: {self.best_loss:.4f}")
        
        _, final_breakdown = self.calculate_total_loss(self.best_labels)
        self._print_breakdown(final_breakdown)
        self._print_cluster_summary()
        
        return self.best_labels
    
    def _print_breakdown(self, breakdown: Dict):
        """Print loss breakdown."""
        print(f"\nLoss Breakdown:")
        print(f"  Overlap:      {breakdown.get('overlap_weighted', 0):.2f} / {self.weight_overlap} (raw: {breakdown.get('overlap', 0):.6f})")
        print(f"  Max Internal: {breakdown.get('max_internal_weighted', 0):.2f} / {self.weight_max_internal} (raw: {breakdown.get('max_internal', 0):.2f} km)")
        print(f"  Avg Internal: {breakdown.get('avg_internal_weighted', 0):.2f} / {self.weight_avg_internal} (raw: {breakdown.get('avg_internal', 0):.2f} km)")
        print(f"  Medoid:       {breakdown.get('medoid_weighted', 0):.2f} / {self.weight_medoid} (raw: {breakdown.get('medoid', 0):.2f} km)")
        print(f"  Balance:      {breakdown.get('balance_weighted', 0):.2f} / {self.weight_balance} (raw: {breakdown.get('balance', 0):.4f})")
    
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
    
    def get_loss_history(self) -> List[float]:
        """Get loss history for plotting."""
        return self.loss_history
