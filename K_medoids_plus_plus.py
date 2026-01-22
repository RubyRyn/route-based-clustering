"""
K-Medoids++ clustering using road distance matrix
Optimizes for:
1. Dynamic number of clusters (up to max_clusters)
2. Minimize total travel distance per cluster
3. Minimize maximum travel distance across all clusters (minimax)
"""

import numpy as np
from typing import List, Tuple, Optional
from location import Location


class KMedoidsPlusPlus:
    """
    K-Medoids++ clustering algorithm for delivery route optimization
    Uses road distance matrix to cluster clients with dynamic cluster count
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location], 
                 max_clusters: int, max_iterations: int = 100, random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients)
            locations: List of Location objects (office first, then clients)
            max_clusters: Maximum number of clusters (number of employees)
            max_iterations: Maximum iterations for convergence
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_clusters = max_clusters
        self.max_iterations = max_iterations
        
        np.random.seed(random_seed)
        
        # Extract client-only matrix (exclude office row/col)
        self.office_idx = 0
        self.client_matrix = road_matrix[1:, 1:]  # Clients to clients
        self.office_to_clients = road_matrix[0, 1:]  # Office to each client
        self.n_clients = len(self.client_matrix)
        
        # Actual number of clusters used (may be less than max_clusters)
        self.n_clusters = min(max_clusters, self.n_clients)
        
        self.medoid_indices = None  # Indices of medoids (in client space)
        self.labels = None  # Cluster assignment for each client
        self.cluster_distances = None  # Total distance for each cluster
        
    def initialize_medoids_plusplus(self, k: int) -> np.ndarray:
        """
        K-Medoids++ initialization: spread medoids out
        
        Args:
            k: Number of medoids to initialize
            
        Returns:
            Array of k medoid indices
        """
        medoids = np.zeros(k, dtype=int)
        
        # Choose first medoid randomly
        medoids[0] = np.random.randint(0, self.n_clients)
        print(f"  Medoid 1: Client {medoids[0]} ({self.locations[medoids[0] + 1].name})")
        
        # Choose remaining medoids
        for m in range(1, k):
            # Calculate distance from each client to nearest existing medoid
            min_distances = np.full(self.n_clients, np.inf)
            
            for client_idx in range(self.n_clients):
                for medoid_idx in medoids[:m]:
                    dist = self.client_matrix[client_idx, medoid_idx]
                    min_distances[client_idx] = min(min_distances[client_idx], dist)
            
            # Don't choose already selected medoids
            min_distances[medoids[:m]] = 0
            
            # Choose next medoid with probability proportional to distance²
            probabilities = min_distances ** 2
            probabilities_sum = probabilities.sum()
            
            if probabilities_sum > 0:
                probabilities = probabilities / probabilities_sum
                medoids[m] = np.random.choice(self.n_clients, p=probabilities)
            else:
                # Fallback: choose randomly from remaining
                remaining = [i for i in range(self.n_clients) if i not in medoids[:m]]
                medoids[m] = np.random.choice(remaining)
            
            print(f"  Medoid {m+1}: Client {medoids[m]} ({self.locations[medoids[m] + 1].name})")
        
        return medoids
    
    def assign_clients_to_medoids(self, medoids: np.ndarray) -> np.ndarray:
        """
        Assign each client to nearest medoid
        
        Args:
            medoids: Array of medoid indices
            
        Returns:
            Array of cluster labels for each client
        """
        k = len(medoids)
        labels = np.zeros(self.n_clients, dtype=int)
        
        for client_idx in range(self.n_clients):
            min_dist = np.inf
            best_cluster = 0
            
            for cluster_id, medoid_idx in enumerate(medoids):
                dist = self.client_matrix[client_idx, medoid_idx]
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_id
            
            labels[client_idx] = best_cluster
        
        return labels
    
    def update_medoids(self, labels: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Update medoids by finding client that minimizes total distance within each cluster
        
        Args:
            labels: Current cluster assignments
            medoids: Current medoid indices
            
        Returns:
            Tuple of (new_medoids, changed)
        """
        k = len(medoids)
        new_medoids = np.zeros(k, dtype=int)
        changed = False
        
        for cluster_id in range(k):
            # Get all clients in this cluster
            cluster_members = np.where(labels == cluster_id)[0]
            
            if len(cluster_members) == 0:
                # Empty cluster - keep current medoid
                new_medoids[cluster_id] = medoids[cluster_id]
                continue
            
            # Find client that minimizes sum of distances to all others in cluster
            best_medoid = cluster_members[0]
            min_total_dist = np.inf
            
            for candidate in cluster_members:
                total_dist = sum(self.client_matrix[candidate, member] 
                               for member in cluster_members)
                
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_medoid = candidate
            
            new_medoids[cluster_id] = best_medoid
            
            if new_medoids[cluster_id] != medoids[cluster_id]:
                changed = True
        
        return new_medoids, changed
    
    def calculate_cluster_distance(self, cluster_members: np.ndarray, medoid: int) -> float:
        """
        Calculate total travel distance for a cluster.
        Approximation: Office -> Medoid -> visit all members (star topology) -> Office
        
        Args:
            cluster_members: Array of client indices in the cluster
            medoid: Index of the cluster medoid
            
        Returns:
            Total estimated travel distance for the cluster
        """
        if len(cluster_members) == 0:
            return 0.0
        
        # Distance: Office -> Medoid
        dist_from_office = self.office_to_clients[medoid]
        # Distance: Medoid -> Office (return)
        dist_to_office = self.office_to_clients[medoid]
        
        # Sum of round-trip distances from medoid to each cluster member
        # (medoid to member and back)
        internal_dist = 2 * sum(self.client_matrix[medoid, member] 
                                for member in cluster_members if member != medoid)
        
        return dist_from_office + internal_dist + dist_to_office
    
    def calculate_cluster_metrics(self, labels: np.ndarray, medoids: np.ndarray) -> dict:
        """
        Calculate metrics for clusters
        
        Args:
            labels: Cluster assignments
            medoids: Medoid indices
            
        Returns:
            Dictionary with cluster metrics
        """
        k = len(medoids)
        cluster_distances = []
        cluster_sizes = []
        
        for cluster_id in range(k):
            cluster_members = np.where(labels == cluster_id)[0]
            cluster_sizes.append(len(cluster_members))
            
            if len(cluster_members) == 0:
                cluster_distances.append(0)
                continue
            
            medoid = medoids[cluster_id]
            total_cluster_dist = self.calculate_cluster_distance(cluster_members, medoid)
            cluster_distances.append(total_cluster_dist)
        
        non_empty_distances = [d for d in cluster_distances if d > 0]
        
        return {
            'cluster_distances': cluster_distances,
            'cluster_sizes': cluster_sizes,
            'total_distance': sum(cluster_distances),
            'max_cluster_distance': max(non_empty_distances) if non_empty_distances else 0,
            'min_cluster_distance': min(non_empty_distances) if non_empty_distances else 0,
            'avg_cluster_distance': np.mean(non_empty_distances) if non_empty_distances else 0,
            'n_active_clusters': len(non_empty_distances)
        }
    
    def run_kmedoids(self, k: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run K-Medoids++ for a specific k value
        
        Args:
            k: Number of clusters
            verbose: Whether to print progress
            
        Returns:
            Tuple of (labels, medoids, metrics)
        """
        if verbose:
            print(f"\n--- Running K-Medoids++ with k={k} ---")
        
        # Initialize medoids
        medoids = self.initialize_medoids_plusplus(k)
        
        for iteration in range(self.max_iterations):
            # Assign clients to nearest medoid
            labels = self.assign_clients_to_medoids(medoids)
            
            # Update medoids
            new_medoids, changed = self.update_medoids(labels, medoids)
            
            # Calculate metrics
            metrics = self.calculate_cluster_metrics(labels, new_medoids)
            
            if verbose and (iteration % 10 == 0 or not changed):
                print(f"  Iteration {iteration}: "
                      f"Total dist: {metrics['total_distance']:.1f} km, "
                      f"Max cluster: {metrics['max_cluster_distance']:.1f} km")
            
            if not changed:
                if verbose:
                    print(f"  Converged after {iteration} iterations")
                break
            
            medoids = new_medoids
        
        # Remove empty clusters
        labels, medoids, metrics = self._remove_empty_clusters(labels, medoids)
        
        return labels, medoids, metrics
    
    def _remove_empty_clusters(self, labels: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Remove empty clusters and reindex
        
        Returns:
            Tuple of (new_labels, new_medoids, metrics)
        """
        k = len(medoids)
        
        # Find non-empty clusters
        non_empty = []
        for cluster_id in range(k):
            if np.sum(labels == cluster_id) > 0:
                non_empty.append(cluster_id)
        
        if len(non_empty) == k:
            # No empty clusters
            metrics = self.calculate_cluster_metrics(labels, medoids)
            return labels, medoids, metrics
        
        # Reindex
        new_medoids = medoids[non_empty]
        new_labels = labels.copy()
        
        mapping = {old: new for new, old in enumerate(non_empty)}
        for i in range(len(new_labels)):
            new_labels[i] = mapping[labels[i]]
        
        metrics = self.calculate_cluster_metrics(new_labels, new_medoids)
        return new_labels, new_medoids, metrics
    
    def fit(self) -> np.ndarray:
        """
        Run K-Medoids++ clustering with dynamic cluster selection.
        
        Strategy:
        1. Try different values of k from 1 to max_clusters
        2. Select k that best balances:
           - Minimizing total distance (constraint 2)
           - Minimizing maximum cluster distance (constraint 3)
        
        Returns:
            Array of cluster labels for each client
        """
        print(f"\n{'='*70}")
        print(f"K-MEDOIDS++ WITH DYNAMIC CLUSTER SELECTION")
        print(f"{'='*70}")
        print(f"Max clusters (employees): {self.max_clusters}")
        print(f"Number of clients: {self.n_clients}")
        
        # Try different values of k
        results = {}
        min_k = 1
        max_k = min(self.max_clusters, self.n_clients)
        
        print(f"\nTesting k from {min_k} to {max_k}...")
        
        for k in range(min_k, max_k + 1):
            # Reset random seed for fair comparison
            np.random.seed(42 + k)
            
            labels, medoids, metrics = self.run_kmedoids(k, verbose=False)
            results[k] = {
                'labels': labels,
                'medoids': medoids,
                'metrics': metrics
            }
            
            print(f"  k={k}: Total={metrics['total_distance']:.1f}km, "
                  f"Max={metrics['max_cluster_distance']:.1f}km, "
                  f"Active clusters={metrics['n_active_clusters']}")
        
        # Select best k using multi-objective optimization
        best_k = self._select_best_k(results)
        
        print(f"\n>>> Selected k={best_k} as optimal <<<")
        
        # Store results
        best_result = results[best_k]
        self.labels = best_result['labels']
        self.medoid_indices = best_result['medoids']
        self.n_clusters = len(self.medoid_indices)
        
        metrics = best_result['metrics']
        self.cluster_distances = metrics['cluster_distances']
        
        return self.labels
    
    def _select_best_k(self, results: dict) -> int:
        """
        Select the best k using multi-objective optimization.
        
        Objectives:
        1. Minimize total travel distance
        2. Minimize maximum cluster distance (minimax)
        
        Strategy: Find the "elbow" point where adding more clusters gives 
        diminishing returns on reducing max cluster distance.
        
        Args:
            results: Dictionary mapping k to (labels, medoids, metrics)
            
        Returns:
            Best k value
        """
        k_values = sorted(results.keys())
        
        if len(k_values) == 1:
            return k_values[0]
        
        # Extract metrics
        max_dists = [results[k]['metrics']['max_cluster_distance'] for k in k_values]
        total_dists = [results[k]['metrics']['total_distance'] for k in k_values]
        
        # Normalize both objectives to [0, 1]
        max_dist_range = max(max_dists) - min(max_dists) if max(max_dists) != min(max_dists) else 1
        total_dist_range = max(total_dists) - min(total_dists) if max(total_dists) != min(total_dists) else 1
        
        # Calculate combined score (lower is better)
        # Weight max_distance more heavily since constraint 3 focuses on minimax
        scores = []
        for i, k in enumerate(k_values):
            # Normalized max distance (want to minimize)
            norm_max = (max_dists[i] - min(max_dists)) / max_dist_range
            # Normalized total distance (want to minimize)
            norm_total = (total_dists[i] - min(total_dists)) / total_dist_range
            
            # Combined score: prioritize minimax (constraint 3) but consider total
            # Also add small penalty for using more clusters (parsimony)
            cluster_penalty = 0.05 * (k / max(k_values))
            
            score = 0.6 * norm_max + 0.3 * norm_total + 0.1 * cluster_penalty
            scores.append(score)
        
        # Find k with best score
        best_idx = np.argmin(scores)
        best_k = k_values[best_idx]
        
        print(f"\nOptimization scores (lower is better):")
        for i, k in enumerate(k_values):
            marker = " <<< BEST" if k == best_k else ""
            print(f"  k={k}: score={scores[i]:.4f}{marker}")
        
        return best_k
    
    def fit_minimax(self) -> np.ndarray:
        """
        Alternative fit method that strictly minimizes maximum cluster distance.
        Uses iterative approach: starts with max clusters and merges if beneficial.
        
        Returns:
            Array of cluster labels for each client
        """
        print(f"\n{'='*70}")
        print(f"K-MEDOIDS++ WITH MINIMAX OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Max clusters (employees): {self.max_clusters}")
        print(f"Number of clients: {self.n_clients}")
        
        # Start with max clusters
        k = min(self.max_clusters, self.n_clients)
        
        print(f"\nPhase 1: Initial clustering with k={k}")
        labels, medoids, metrics = self.run_kmedoids(k, verbose=True)
        
        best_labels = labels.copy()
        best_medoids = medoids.copy()
        best_max_dist = metrics['max_cluster_distance']
        best_k = len(medoids)
        
        print(f"\nPhase 2: Trying to reduce clusters while maintaining minimax...")
        
        # Try reducing k
        for try_k in range(best_k - 1, 0, -1):
            np.random.seed(42 + try_k)
            labels, medoids, metrics = self.run_kmedoids(try_k, verbose=False)
            
            print(f"  k={try_k}: Max cluster distance = {metrics['max_cluster_distance']:.1f} km")
            
            # Accept if max distance doesn't increase significantly (within 10%)
            if metrics['max_cluster_distance'] <= best_max_dist * 1.1:
                best_labels = labels.copy()
                best_medoids = medoids.copy()
                best_max_dist = metrics['max_cluster_distance']
                best_k = len(medoids)
                print(f"    -> Accepted (within tolerance)")
            else:
                print(f"    -> Rejected (max distance increased beyond tolerance)")
                break  # Stop reducing k once quality degrades
        
        print(f"\n>>> Final selection: k={best_k} with max distance={best_max_dist:.1f} km <<<")
        
        # Store results
        self.labels = best_labels
        self.medoid_indices = best_medoids
        self.n_clusters = len(self.medoid_indices)
        
        final_metrics = self.calculate_cluster_metrics(self.labels, self.medoid_indices)
        self.cluster_distances = final_metrics['cluster_distances']
        
        return self.labels
    
    def print_results(self):
        """Print detailed clustering results"""
        if self.labels is None:
            print("No clustering results. Run fit() first.")
            return
        
        print(f"\n{'='*70}")
        print(f"CLUSTERING RESULTS")
        print(f"{'='*70}")
        
        metrics = self.calculate_cluster_metrics(self.labels, self.medoid_indices)
        
        print(f"\nOverall Metrics:")
        print(f"  Number of clusters used: {self.n_clusters} (max was {self.max_clusters})")
        print(f"  Total distance (all clusters): {metrics['total_distance']:.2f} km")
        print(f"  Average cluster distance: {metrics['avg_cluster_distance']:.2f} km")
        print(f"  Max cluster distance: {metrics['max_cluster_distance']:.2f} km")
        print(f"  Min cluster distance: {metrics['min_cluster_distance']:.2f} km")
        
        if metrics['min_cluster_distance'] > 0:
            print(f"  Imbalance ratio: {metrics['max_cluster_distance'] / metrics['min_cluster_distance']:.2f}x")
        
        print(f"\nCluster Details:")
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(self.labels == cluster_id)[0]
            medoid_idx = self.medoid_indices[cluster_id]
            
            print(f"\n  Cluster {cluster_id + 1}:")
            print(f"    Medoid: {self.locations[medoid_idx + 1].name}")
            print(f"    Size: {len(cluster_members)} clients")
            print(f"    Total distance: {self.cluster_distances[cluster_id]:.2f} km")
            print(f"    Members: ", end="")
            
            member_names = [self.locations[idx + 1].name for idx in cluster_members]
            if len(member_names) <= 5:
                print(", ".join(member_names))
            else:
                print(", ".join(member_names[:5]) + f" + {len(member_names) - 5} more")
    
    def get_cluster_assignments(self) -> dict:
        """
        Get cluster assignments in readable format
        
        Returns:
            Dictionary mapping cluster_id to list of client names
        """
        assignments = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(self.labels == cluster_id)[0]
            client_names = [self.locations[idx + 1].name for idx in cluster_members]
            assignments[f"Cluster_{cluster_id + 1}"] = client_names
        
        return assignments


def optimize_workload_balance(clusterer: KMedoidsPlusPlus, 
                              max_swap_iterations: int = 100) -> np.ndarray:
    """
    Post-processing: Balance workload by swapping clients between clusters
    Addresses Constraint 3: Minimize maximum cluster distance
    
    Args:
        clusterer: Fitted KMedoidsPlusPlus instance
        max_swap_iterations: Maximum number of swap attempts
        
    Returns:
        Improved cluster labels
    """
    print(f"\n{'-'*70}")
    print(f"WORKLOAD BALANCING (Post-processing)")
    print(f"{'-'*70}")
    
    labels = clusterer.labels.copy()
    medoids = clusterer.medoid_indices.copy()
    n_clusters = clusterer.n_clusters
    road_matrix = clusterer.client_matrix
    office_to_clients = clusterer.office_to_clients
    
    def calculate_cluster_distance(cluster_members, medoid):
        """Calculate total distance for cluster"""
        if len(cluster_members) == 0:
            return 0
        dist_from_office = office_to_clients[medoid]
        internal = 2 * sum(road_matrix[medoid, m] for m in cluster_members if m != medoid)
        return dist_from_office * 2 + internal
    
    # Initial state
    initial_dists = []
    for cluster_id in range(n_clusters):
        members = np.where(labels == cluster_id)[0]
        initial_dists.append(calculate_cluster_distance(members, medoids[cluster_id]))
    
    initial_max = max(initial_dists)
    print(f"Initial max cluster distance: {initial_max:.1f} km")
    
    best_labels = labels.copy()
    best_max_dist = initial_max
    
    for iteration in range(max_swap_iterations):
        # Calculate current cluster distances
        cluster_dists = []
        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]
            cluster_dists.append(calculate_cluster_distance(members, medoids[cluster_id]))
        
        current_max = max(cluster_dists)
        
        if current_max < best_max_dist:
            best_max_dist = current_max
            best_labels = labels.copy()
        
        # Find overloaded and underloaded clusters
        max_cluster = np.argmax(cluster_dists)
        min_cluster = np.argmin([d if d > 0 else np.inf for d in cluster_dists])
        
        if max_cluster == min_cluster:
            break
        
        max_members = np.where(labels == max_cluster)[0]
        
        if len(max_members) <= 1:
            break
        
        # Try swapping each client from max_cluster to min_cluster
        best_swap_client = None
        best_improvement = 0
        
        for client in max_members:
            if client == medoids[max_cluster]:
                continue  # Don't move medoid
            
            # Simulate the swap
            new_max_members = [m for m in max_members if m != client]
            min_members = np.where(labels == min_cluster)[0]
            new_min_members = list(min_members) + [client]
            
            new_max_dist = calculate_cluster_distance(new_max_members, medoids[max_cluster])
            new_min_dist = calculate_cluster_distance(new_min_members, medoids[min_cluster])
            
            new_overall_max = max(new_max_dist, new_min_dist, 
                                  *[d for i, d in enumerate(cluster_dists) 
                                    if i not in [max_cluster, min_cluster]])
            
            improvement = current_max - new_overall_max
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_swap_client = client
        
        if best_swap_client is not None and best_improvement > 0:
            labels[best_swap_client] = min_cluster
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: Max distance = {current_max:.1f} km")
        else:
            break
    
    improvement = initial_max - best_max_dist
    print(f"Final max cluster distance: {best_max_dist:.1f} km")
    print(f"Improvement: {improvement:.1f} km ({100*improvement/initial_max:.1f}%)")
    
    return best_labels