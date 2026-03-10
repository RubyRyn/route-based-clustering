"""
K-Medoids++ Clustering with Min/Max Constraints

K-Medoids clustering using road distance matrix with cluster size constraints.
- K-Medoids++ initialization (spread medoids out)
- Assigns clients to nearest medoid by road distance
- Iteratively improves medoid positions
- Balances cluster sizes to respect min/max constraints
"""

import numpy as np
from typing import List, Tuple
from location import Location


class KMedoidsPlusPlus:
    """
    K-Medoids++ clustering using road distance matrix with size constraints.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location], 
                 n_clusters: int, 
                 min_clients_per_cluster: int = None,
                 max_clients_per_cluster: int = None,
                 max_iterations: int = 1000, 
                 max_balance_iterations: int = 1000,
                 n_neighbors: int = 3,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            n_clusters: Number of clusters
            min_clients_per_cluster: Minimum clients per cluster (None = no limit)
            max_clients_per_cluster: Maximum clients per cluster (None = no limit)
            max_iterations: Maximum iterations for K-Medoids convergence
            max_balance_iterations: Maximum iterations for balancing
            n_neighbors: Number of nearest clusters to consider as neighbors
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.max_balance_iterations = max_balance_iterations
        self.n_neighbors = n_neighbors
        
        np.random.seed(random_seed)
        
        # Extract client-only matrix (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.office_to_clients = road_matrix[0, 1:]
        self.n_clients = len(self.client_matrix)
        
        # Set default min/max if not provided
        avg_size = self.n_clients / n_clusters
        self.min_clients = min_clients_per_cluster if min_clients_per_cluster is not None else 1
        self.max_clients = max_clients_per_cluster if max_clients_per_cluster is not None else self.n_clients
        
        # Validate constraints
        self._validate_constraints()
        
        # Results
        self.medoid_indices = None
        self.labels = None
    
    def _validate_constraints(self):
        """Validate that constraints are feasible."""
        if self.min_clients * self.n_clusters > self.n_clients:
            raise ValueError(
                f"Infeasible: {self.n_clusters} clusters × {self.min_clients} min = "
                f"{self.n_clusters * self.min_clients} > {self.n_clients} clients"
            )
        
        if self.max_clients * self.n_clusters < self.n_clients:
            raise ValueError(
                f"Infeasible: {self.n_clusters} clusters × {self.max_clients} max = "
                f"{self.n_clusters * self.max_clients} < {self.n_clients} clients"
            )
        
        avg = self.n_clients / self.n_clusters
        print(f"Constraints: min={self.min_clients}, max={self.max_clients}, avg={avg:.1f}")
    
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
    
    def _get_cluster_sizes(self, labels: np.ndarray) -> np.ndarray:
        """Get size of each cluster."""
        sizes = np.zeros(self.n_clusters, dtype=int)
        for cluster_id in range(self.n_clusters):
            sizes[cluster_id] = np.sum(labels == cluster_id)
        return sizes
    
    def _get_neighboring_clusters(self, client_idx: int, exclude_cluster: int = None) -> List[int]:
        """
        Get neighboring clusters for a client (nearest medoids).
        
        Args:
            client_idx: Client index
            exclude_cluster: Cluster to exclude (current cluster)
            
        Returns:
            List of cluster IDs sorted by distance to client
        """
        distances = []
        for cluster_id in range(self.n_clusters):
            if cluster_id == exclude_cluster:
                continue
            medoid_idx = self.medoid_indices[cluster_id]
            dist = self.client_matrix[client_idx, medoid_idx]
            distances.append((cluster_id, dist))
        
        # Sort by distance and return top n_neighbors
        distances.sort(key=lambda x: x[1])
        return [cluster_id for cluster_id, _ in distances[:self.n_neighbors]]
    
    def _fix_oversized_clusters(self, labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Fix clusters that exceed max_clients.
        Move furthest clients to nearest neighboring clusters with room.
        
        Returns:
            Tuple of (updated_labels, num_moves)
        """
        labels = labels.copy()
        total_moves = 0
        sizes = self._get_cluster_sizes(labels)
        
        for cluster_id in range(self.n_clusters):
            while sizes[cluster_id] > self.max_clients:
                # Get members of this cluster
                cluster_members = np.where(labels == cluster_id)[0]
                medoid_idx = self.medoid_indices[cluster_id]
                
                # Find client furthest from medoid
                distances = [(idx, self.client_matrix[idx, medoid_idx]) for idx in cluster_members]
                distances.sort(key=lambda x: x[1], reverse=True)
                print(f"  Cluster {cluster_id + 1} oversized: size={sizes[cluster_id]}, max={self.max_clients}")
                
                moved = False
                for client_idx, _ in distances:
                    # Skip medoid
                    if client_idx == medoid_idx:
                        continue
                    
                    # Find neighboring clusters with room
                    neighbors = self._get_neighboring_clusters(client_idx, exclude_cluster=cluster_id)
                    
                    for neighbor_id in neighbors:
                        if sizes[neighbor_id] < self.max_clients:
                            # Move client to neighbor
                            labels[client_idx] = neighbor_id
                            sizes[cluster_id] -= 1
                            sizes[neighbor_id] += 1
                            total_moves += 1
                            moved = True
                            break
                    
                    if moved:
                        break
                
                if not moved:
                    # No valid move found, try any cluster with room
                    for client_idx, _ in distances:
                        if client_idx == medoid_idx:
                            continue
                        for other_cluster in range(self.n_clusters):
                            if other_cluster != cluster_id and sizes[other_cluster] < self.max_clients:
                                labels[client_idx] = other_cluster
                                sizes[cluster_id] -= 1
                                sizes[other_cluster] += 1
                                total_moves += 1
                                moved = True
                                break
                        if moved:
                            break
                
                if not moved:
                    print(f"Warning: Could not fix oversized cluster {cluster_id}")
                    break
        
        return labels, total_moves
    
    def _fix_undersized_clusters(self, labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Fix clusters that are below min_clients.
        Steal nearest clients from neighboring clusters that can spare.
        
        Returns:
            Tuple of (updated_labels, num_moves)
        """
        labels = labels.copy()
        total_moves = 0
        sizes = self._get_cluster_sizes(labels)
        
        for cluster_id in range(self.n_clusters):
            while sizes[cluster_id] < self.min_clients:
                medoid_idx = self.medoid_indices[cluster_id]
                
                # Find neighboring clusters that can spare clients
                neighbors = self._get_neighboring_clusters(medoid_idx, exclude_cluster=cluster_id)
                
                # Expand to all clusters if neighbors can't spare
                candidate_clusters = [n for n in neighbors if sizes[n] > self.min_clients]
                if not candidate_clusters:
                    candidate_clusters = [c for c in range(self.n_clusters) 
                                         if c != cluster_id and sizes[c] > self.min_clients]
                
                if not candidate_clusters:
                    print(f"Warning: Could not fix undersized cluster {cluster_id}")
                    break
                
                # Find nearest client from candidate clusters
                best_client = None
                best_distance = np.inf
                best_source_cluster = None
                
                for source_cluster in candidate_clusters:
                    source_members = np.where(labels == source_cluster)[0]
                    source_medoid = self.medoid_indices[source_cluster]
                    
                    for client_idx in source_members:
                        # Don't steal medoids
                        if client_idx == source_medoid:
                            continue
                        
                        dist = self.client_matrix[client_idx, medoid_idx]
                        if dist < best_distance:
                            best_distance = dist
                            best_client = client_idx
                            best_source_cluster = source_cluster
                
                if best_client is not None:
                    # Steal the client
                    labels[best_client] = cluster_id
                    sizes[cluster_id] += 1
                    sizes[best_source_cluster] -= 1
                    total_moves += 1
                else:
                    print(f"Warning: Could not find client to steal for cluster {cluster_id}")
                    break
        
        return labels, total_moves
    
    def _balance_clusters(self, labels: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Balance clusters to respect min/max constraints.
        
        Returns:
            Tuple of (balanced_labels, changes_made)
        """
        labels = labels.copy()
        
        # Fix oversized first
        labels, oversized_moves = self._fix_oversized_clusters(labels)
        
        # Then fix undersized
        labels, undersized_moves = self._fix_undersized_clusters(labels)
        
        total_moves = oversized_moves + undersized_moves
        
        if total_moves > 0:
            print(f"  Balancing: {oversized_moves} oversized moves, {undersized_moves} undersized moves")
        
        return labels, total_moves > 0
    
    def _check_constraints(self, labels: np.ndarray) -> bool:
        """Check if all constraints are satisfied."""
        sizes = self._get_cluster_sizes(labels)
        
        for cluster_id in range(self.n_clusters):
            if sizes[cluster_id] < self.min_clients:
                return False
            if sizes[cluster_id] > self.max_clients:
                return False
        
        return True
    
    def fit(self) -> np.ndarray:
        """
        Run K-Medoids++ clustering with constraints.
        
        Returns:
            Array of cluster labels for each client
        """
        print(f"\n{'='*60}")
        print(f"K-MEDOIDS++ CLUSTERING (with constraints)")
        print(f"{'='*60}")
        print(f"Clients: {self.n_clients}")
        print(f"Clusters: {self.n_clusters}")
        print(f"Min clients/cluster: {self.min_clients}")
        print(f"Max clients/cluster: {self.max_clients}")
        
        # Step 1: Initialize medoids
        print(f"\nPhase 1: Initializing medoids (K-Medoids++)...")
        self.medoid_indices = self._initialize_medoids()
        
        # Step 2: Standard K-Medoids iterations
        print(f"\nPhase 2: Running K-Medoids (max {self.max_iterations} iterations)...")
        
        for iteration in range(self.max_iterations):
            self.labels = self._assign_to_medoids(self.medoid_indices)
            new_medoids, changed = self._update_medoids(self.labels, self.medoid_indices)
            self.medoid_indices = new_medoids
            
            if not changed:
                print(f"  Converged at iteration {iteration + 1}")
                break
        else:
            print(f"  Reached max iterations ({self.max_iterations})")
        
        # Final assignment before balancing
        self.labels = self._assign_to_medoids(self.medoid_indices)
        
        # Print pre-balance sizes
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\nPre-balance cluster sizes: min={sizes.min()}, max={sizes.max()}")
        # print(f"Sizes: {sorted(sizes, reverse=True)}")
        
        # Step 3: Balance clusters
        print(f"\nPhase 3: Balancing clusters (max {self.max_balance_iterations} iterations)...")
        
        for balance_iter in range(self.max_balance_iterations):
            self.labels, changes_made = self._balance_clusters(self.labels)
            
            # Update medoids after balancing
            self.medoid_indices, _ = self._update_medoids(self.labels, self.medoid_indices)
            
            if not changes_made:
                print(f"  Balancing complete at iteration {balance_iter + 1}")
                break
            
            if self._check_constraints(self.labels):
                print(f"  All constraints satisfied at iteration {balance_iter + 1}")
                break
        else:
            print(f"  Reached max balance iterations ({self.max_balance_iterations})")
        
        # Final check
        if self._check_constraints(self.labels):
            print(f"\n✓ All constraints satisfied!")
        else:
            print(f"\n✗ Warning: Some constraints not satisfied")
            sizes = self._get_cluster_sizes(self.labels)
            violations = []
            for i, size in enumerate(sizes):
                if size < self.min_clients:
                    violations.append(f"Cluster {i+1}: {size} < {self.min_clients}")
                if size > self.max_clients:
                    violations.append(f"Cluster {i+1}: {size} > {self.max_clients}")
            for v in violations:
                print(f"  {v}")
        
        self._print_results()
        
        return self.labels
    
    def _print_results(self):
        """Print clustering results."""
        print(f"\n{'-'*60}")
        print(f"RESULTS")
        print(f"{'-'*60}")
        
        sizes = self._get_cluster_sizes(self.labels)
        print(f"\nCluster sizes: min={sizes.min()}, max={sizes.max()}, avg={sizes.mean():.1f}")
        
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(self.labels == cluster_id)[0]
            medoid_idx = self.medoid_indices[cluster_id]
            medoid_name = self.locations[medoid_idx + 1].name
            
            # Calculate internal distances
            if len(cluster_members) > 1:
                internal_dists = [self.client_matrix[medoid_idx, m] for m in cluster_members if m != medoid_idx]
                avg_dist = np.mean(internal_dists) if internal_dists else 0
                max_dist = np.max(internal_dists) if internal_dists else 0
            else:
                avg_dist = 0
                max_dist = 0
            
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Medoid: {medoid_name}")
            print(f"  Size: {len(cluster_members)} clients")
            print(f"  Avg dist to medoid: {avg_dist:.2f} km")
            print(f"  Max dist to medoid: {max_dist:.2f} km")
            
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
    
    def get_cluster_sizes(self) -> np.ndarray:
        """Get size of each cluster."""
        return self._get_cluster_sizes(self.labels)

