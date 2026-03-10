"""
K-Medoids++ Clustering with Smart Initialization

Simple K-Medoids clustering using road distance matrix.
- K-Medoids++ initialization (spread medoids out)
- Assigns clients to nearest medoid by road distance
- Iteratively improves medoid positions
"""

import numpy as np
from typing import List, Tuple
from location import Location


class KMedoidsPlusPlus:
    """
    K-Medoids++ clustering using road distance matrix.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location], 
                 n_clusters: int, max_iterations: int = 100, random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            n_clusters: Number of clusters
            max_iterations: Maximum iterations for convergence
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
        np.random.seed(random_seed)
        
        # Extract client-only matrix (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.office_to_clients = road_matrix[0, 1:]
        self.n_clients = len(self.client_matrix)
        
        
        # Results
        self.medoid_indices = None
        self.labels = None
    
    def _initialize_medoids(self) -> np.ndarray:
        """
        K-Medoids++ initialization: spread medoids out.
        First medoid random, subsequent chosen with probability proportional to distance².
        
        Returns:
            Array of medoid indices (in client space)
        """
        medoids = np.zeros(self.n_clusters, dtype=int)
        
        # First medoid: random
        medoids[0] = np.random.randint(0, self.n_clients)
        
        # Remaining medoids: probability proportional to distance²
        for m in range(1, self.n_clusters):
            # Distance from each client to nearest existing medoid
            min_distances = np.full(self.n_clients, np.inf)
            
            for client_idx in range(self.n_clients):
                for medoid_idx in medoids[:m]:
                    dist = self.client_matrix[client_idx, medoid_idx]
                    min_distances[client_idx] = min(min_distances[client_idx], dist)
            
            # Don't choose already selected medoids
            min_distances[medoids[:m]] = 0
            
            # Choose with probability proportional to distance²
            probabilities = min_distances ** 2
            prob_sum = probabilities.sum()
            
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
                medoids[m] = np.random.choice(self.n_clients, p=probabilities)
            else:
                # Fallback: random from remaining
                remaining = [i for i in range(self.n_clients) if i not in medoids[:m]]
                medoids[m] = np.random.choice(remaining)
        
        return medoids
    
    def _assign_to_medoids(self, medoids: np.ndarray) -> np.ndarray:
        """
        Assign each client to nearest medoid.
        
        Returns:
            Array of cluster labels for each client
        """
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
    
    def _update_medoids(self, labels: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Update medoids: find client that minimizes total distance within each cluster.
        
        Returns:
            Tuple of (new_medoids, changed)
        """
        new_medoids = np.zeros(self.n_clusters, dtype=int)
        changed = False
        
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(labels == cluster_id)[0]
            
            if len(cluster_members) == 0:
                new_medoids[cluster_id] = medoids[cluster_id]
                continue
            
            # Find client that minimizes sum of distances to others in cluster
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
    
    def fit(self) -> np.ndarray:
        """
        Run K-Medoids++ clustering.
        
        Returns:
            Array of cluster labels for each client
        """
        print(f"\n{'='*60}")
        print(f"K-MEDOIDS++ CLUSTERING")
        print(f"{'='*60}")
        print(f"Clients: {self.n_clients}")
        print(f"Clusters: {self.n_clusters}")
        
        # Initialize medoids
        print(f"\nInitializing medoids (K-Medoids++)...")
        self.medoid_indices = self._initialize_medoids()
        
        # Iterative refinement
        print(f"Running clustering (max {self.max_iterations} iterations)...")
        
        for iteration in range(self.max_iterations):
            # Assign clients to nearest medoid
            self.labels = self._assign_to_medoids(self.medoid_indices)
            
            # Update medoids
            new_medoids, changed = self._update_medoids(self.labels, self.medoid_indices)
            self.medoid_indices = new_medoids
            
            if not changed:
                print(f"Converged at iteration {iteration + 1}")
                break
        else:
            print(f"Reached max iterations ({self.max_iterations})")
        
        # Final assignment
        self.labels = self._assign_to_medoids(self.medoid_indices)
        
        self._print_results()
        
        return self.labels
    
    def _print_results(self):
        """Print clustering results."""
        print(f"\n{'-'*60}")
        print(f"RESULTS")
        print(f"{'-'*60}")
        
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(self.labels == cluster_id)[0]
            medoid_idx = self.medoid_indices[cluster_id]
            medoid_name = self.locations[medoid_idx + 1].name
            
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Medoid: {medoid_name}")
            print(f"  Size: {len(cluster_members)} clients")
            
            member_names = [self.locations[idx + 1].name for idx in cluster_members]
            if len(member_names) <= 5:
                print(f"  Members: {', '.join(member_names)}")
            else:
                print(f"  Members: {', '.join(member_names[:5])} + {len(member_names) - 5} more")
    
    def get_labels(self) -> np.ndarray:
        """Get cluster labels for each client."""
        return self.labels
    
    def get_medoid_indices(self) -> np.ndarray:
        """Get medoid indices (in client space, 0-indexed)."""
        return self.medoid_indices
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get list of client names in a cluster."""
        cluster_members = np.where(self.labels == cluster_id)[0]
        return [self.locations[idx + 1].name for idx in cluster_members]