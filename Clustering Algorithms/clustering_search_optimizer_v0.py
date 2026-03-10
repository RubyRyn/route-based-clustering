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

        self.top_10_best = []  # Store top 10 best (loss, labels) pairs
    
    
    def initialize_from_hdbscan(self, hdbscan_labels: np.ndarray):
        """
        Initialize with HDBSCAN labels.
        
        1. Convert outliers to individual clusters
        2. Calculate baseline loss (skip constraints)
        3. Re-assign outliers randomly to natural clusters for search
        
        Args:
            hdbscan_labels: Labels from HDBSCAN clustering (-1 = outlier)
        """
        # Count original clusters and outliers
        outlier_indices = np.where(hdbscan_labels == -1)[0]
        n_original_clusters = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
        n_outliers = len(outlier_indices)
        
        print(f"HDBSCAN result: {n_original_clusters} clusters + {n_outliers} outliers")
        
        # Step 1: Create baseline labels (outliers as individual clusters)
        baseline_labels = hdbscan_labels.copy()
        next_cluster_id = n_original_clusters
        
        for idx in outlier_indices:
            baseline_labels[idx] = next_cluster_id
            next_cluster_id += 1
    
        baseline_labels = self._renumber_labels_static(baseline_labels)
        
        print(f"Baseline: {len(np.unique(baseline_labels))} total clusters (each outlier = 1 cluster)")
        
        # Step 2: Calculate baseline loss (skip constraints)
        self.baseline_labels = baseline_labels.copy()
        self.baseline_loss, breakdown = self._calculate_loss_no_constraints(baseline_labels)
        
        print(f"\nBaseline loss: {self.baseline_loss:.4f}")
        self._print_breakdown(breakdown)
        
        # Step 3: Re-assign outliers randomly to natural clusters for search
        self.labels = hdbscan_labels.copy()
        
        for idx in outlier_indices:
            self.labels[idx] = np.random.randint(0, n_original_clusters)
        
        self.labels = self._renumber_labels_static(self.labels)
        
        print(f"\nSearch start: {len(np.unique(self.labels))} clusters (outliers assigned randomly)")
        
        # Set initial best as baseline
        self.best_labels = self.baseline_labels.copy()
        self.best_loss = self.baseline_loss

    def _renumber_labels_static(self, labels: np.ndarray) -> np.ndarray:
        """Renumber labels to be consecutive starting from 0."""
        unique_labels = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        return np.array([mapping[l] for l in labels])
    
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

        if not self._check_constraints(labels):
            # print("Constraints violated!")
            return np.inf, {}
        
        # print("No constraints violated.")
        overlap_loss = self._calculate_overlap_loss(labels)
        # print(f"Overlap loss: {overlap_loss:.4f} km")
        avg_internal_loss = self._calculate_avg_internal_loss(labels)
        # print(f"Avg internal loss: {avg_internal_loss:.4f} km")
        

        overlap_normalized = overlap_loss 
        avg_internal_normalized = avg_internal_loss 

        total_loss = (
            overlap_normalized + avg_internal_normalized
        )
        # print(f"Total loss: {total_loss:.4f}")

        breakdown = {
            'overlap': overlap_loss,
            'avg_internal': avg_internal_loss,
            'total': total_loss
        }
        
        return total_loss, breakdown
    
    def _check_constraints(self, labels: np.ndarray) -> bool:
        """Check if labels satisfy all constraints."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Max clusters
        if n_clusters > self.max_clusters:
            print("Max clusters exceeded")
            return False
        
        # Min/max clients per cluster
        for cluster_id in unique_labels:
            cluster_size = np.sum(labels == cluster_id)
            if cluster_size < self.min_clients_per_cluster:
                print("Min clients per cluster violated")
                return False
            if cluster_size > self.max_clients_per_cluster:
                print("Max clients per cluster violated")
                return False
        
        return True
    
    def _calculate_overlap_loss(self, labels: np.ndarray) -> float:
        """
        Calculate convex hull overlap between clusters.
        Uses lat/lon coordinates, returns area in km².
        
        Returns:
            Total overlap area in km²
        """
        from pyproj import Geod
        
        unique_labels = np.unique(labels)
        hulls = {}
        
        # Build convex hulls for each cluster
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) < 3:
                # print("Not enough points for convex hull")
                continue
            
            coords = self.client_coords[cluster_indices]
            
            try:
                # print("Calculating convex hull")
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                # Shapely expects (lon, lat) order for geographic calculations
                polygon = Polygon([(coords[v, 1], coords[v, 0]) for v in hull.vertices])
                polygon = make_valid(polygon)
                hulls[cluster_id] = polygon
                # print(f"Cluster {cluster_id}: Convex hull with {len(hull.vertices)} vertices")
            except:
                continue
        
        # Calculate pairwise overlaps
        total_overlap_km2 = 0.0
        cluster_ids = list(hulls.keys())
        geod = Geod(ellps="WGS84")
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                poly1 = hulls[cluster_ids[i]]
                poly2 = hulls[cluster_ids[j]]
                
                try:
                    if poly1.intersects(poly2):
                        intersection = poly1.intersection(poly2)
                        # Calculate geodesic area in m², convert to km²
                        area_m2, _ = geod.geometry_area_perimeter(intersection)
                        area_km2 = abs(area_m2) / 1_000_000
                        # print(f"Clusters {cluster_ids[i]} & {cluster_ids[j]} overlap: {area_km2:.4f} km²")
                        total_overlap_km2 += area_km2
                        # print(f"Total overlap so far: {total_overlap_km2:.4f} km²")
                except:
                    continue
        
        # return squre root of total overlap to reduce impact
        return np.sqrt(total_overlap_km2)
    
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
        
        return max_distance # in km
    
    def _calculate_avg_internal_loss(self, labels: np.ndarray) -> float:
        total_distance = []
        unique_labels = np.unique(labels)
    
        # print(f"DEBUG: unique_labels = {unique_labels}")
        # print(f"DEBUG: num unique labels = {len(unique_labels)}")
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            # print(f"DEBUG: cluster {cluster_id} has {len(cluster_indices)} clients")
            
            if len(cluster_indices) < 2:
                continue
            
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        total_distance.append(self.client_road_matrix[i, j])
        
        # print(f"DEBUG: total_distance length = {len(total_distance)}")
        
        if len(total_distance) == 0:
            return 0.0
        
        return np.mean(total_distance)
    

    

    def _calculate_loss_no_constraints(self, labels: np.ndarray) -> Tuple[float, Dict]:
        """Calculate loss without checking constraints. Used for HDBSCAN baseline."""
        overlap_loss = self._calculate_overlap_loss(labels)
        avg_internal_loss = self._calculate_avg_internal_loss(labels)
        
        # Apply same normalization/weighting as calculate_total_loss
        overlap_normalized = overlap_loss
        avg_internal_normalized = avg_internal_loss

        
        total_loss = (
            overlap_normalized +
            avg_internal_normalized 
        )
        
        breakdown = {
            'overlap': overlap_loss,
            'avg_internal': avg_internal_loss,
            'total': total_loss
        }
        
        return total_loss, breakdown


    # ==================== SEARCH MOVES ====================
    
    def _move_random_reassign(self, labels: np.ndarray) -> np.ndarray:
        """
        Randomly reassign one client to a random cluster.
        """
        new_labels = labels.copy()
        
        client_idx = np.random.randint(0, self.num_clients)
        unique_labels = np.unique(labels)
        new_cluster = np.random.choice(unique_labels)
        
        new_labels[client_idx] = new_cluster
        
        return new_labels
    
    
    
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
        # Initial best already set in initialize_from_hdbscan()
        print(f"\nBaseline loss: {self.baseline_loss:.4f}")
        print(f"Starting search from {len(np.unique(self.labels))} clusters...")
        self.loss_history = [self.best_loss]
        
        print(f"\nInitial loss: {self.best_loss:.4f}")
        # self._print_breakdown(breakdown)
        
        no_improvement_count = 0
        self.iteration_count = 0
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration
            
            new_labels = self._move_random_reassign(self.labels)
            
            # Evaluate new solution
            new_loss, new_breakdown = self.calculate_total_loss(new_labels)

            if new_loss != np.inf:
                self._update_top_10(new_loss, new_labels)
            
            # Accept if better
            if new_loss < self.best_loss:
                self.labels = new_labels
                self.best_labels = new_labels.copy()
                self.best_loss = new_loss
                no_improvement_count = 0
                # Only add when we find a NEW best
                
                
                
                if verbose and iteration % 100 == 0:
                    print(f"\nIteration {iteration} : New best = {self.best_loss:.4f}")
            else:
                no_improvement_count += 1
            
            self.loss_history.append(self.best_loss)
            
            # Check stopping condition
            if no_improvement_count >= self.no_improvement_limit:
                print(f"\nStopping: No improvement for {self.no_improvement_limit} iterations")
                break
            
            # Progress update
            if verbose and iteration % 500 == 0 and iteration > 0:
                self._print_breakdown(new_breakdown)
                print(f"Iteration {iteration} : New loss = {new_loss:.4f}: Best loss = {self.best_loss:.4f}")
        
            # self._print_breakdown(new_breakdown)


        print(f"OPTIMIZATION COMPLETE")
        print(f"Total iterations: {self.iteration_count + 1}")
        print(f"Final loss: {self.best_loss:.4f}")
        
        _, final_breakdown = self.calculate_total_loss(self.best_labels)
        self._print_breakdown(final_breakdown)
        # self._print_cluster_summary()
        
        return self.best_labels
    
    def _print_breakdown(self, breakdown: Dict):
        """Print loss breakdown."""
        print(f"Loss Breakdown:")
        print(f"  Overlap:      {breakdown.get('overlap', 0):.6f}")
        print(f"  Avg Internal: {breakdown.get('avg_internal', 0):.6f} km")
        print(f"  Total Loss:   {breakdown.get('total', 0):.6f}")
        print("\n")
    
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

    def _update_top_10(self, loss: float, labels: np.ndarray):
        """Keep track of top 10 best UNIQUE solutions."""
        
        # Check if this exact loss already exists
        for solution in self.top_10_best:
            if abs(solution['loss'] - loss) < 0.0001:  # Same loss (within tolerance)
                # Check if labels are also the same
                if np.array_equal(solution['labels'], labels):
                    return  # Skip duplicate
        
        self.top_10_best.append({
            'loss': loss,
            'labels': labels.copy()
        })
        
        # Sort by loss (ascending) and keep only top 10
        self.top_10_best = sorted(self.top_10_best, key=lambda x: x['loss'])[:10]


    def visualize_top_10(self, locations, road_matrix, road_calculator):
        """Generate visualization for top 10 best solutions."""
        from visualizations import plot_road_distance_clusters
        
        print(f"\nGenerating top {len(self.top_10_best)} best visualizations...")
        # print(f"Length of top_10_best: {len(self.top_10_best)}")
        # print(f"Top losses: {[sol['loss'] for sol in self.top_10_best]}")
        
        for i, solution in enumerate(self.top_10_best):
            filename = f"top_{i+1}_loss_{solution['loss']:.4f}.html"
            print(f"filename: {filename}")
            
            plot_road_distance_clusters(
                locations=locations,
                labels=solution['labels'],
                road_matrix=road_matrix,
                road_calculator=road_calculator,
                filename=filename
            )
            
            # print(f"  Saved: {filename} (loss: {solution['loss']:.4f})")
        print("Top 10 visualizations generated.")