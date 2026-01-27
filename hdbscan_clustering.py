"""
HDBSCAN Clustering based on Road Distance Matrix

Density-based clustering that:
- Automatically finds the number of clusters
- Uses road distance matrix (handles rivers, mountains, etc.)
- Identifies outliers (isolated clients)
- Enforces max cluster constraint (merges outliers if needed)
- Can split large clusters (optional, for future use)
"""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.cluster import HDBSCAN
from location import Location


class HDBSCANClustering:
    """
    HDBSCAN clustering using road distance matrix.
    
    Automatically determines the number of clusters based on data density.
    Outliers (isolated clients) are labeled as -1.
    Can enforce max cluster limit and split large clusters.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 max_clusters: int = None,
                 min_cluster_size: int = 2, min_samples: int = 1,
                 max_clients_per_cluster: int = None):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            max_clusters: Maximum total clusters + outliers allowed (e.g., 14 employees)
            min_cluster_size: Minimum number of clients to form a cluster (default: 2)
            min_samples: How conservative clustering is. Higher = more outliers (default: 1)
            max_clients_per_cluster: If a cluster exceeds this, split it (optional, for future use)
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.max_clients_per_cluster = max_clients_per_cluster
        
        self.num_locations = len(locations)
        self.num_clients = self.num_locations - 1
        
        # Client-only road distance matrix (exclude office)
        self.client_road_matrix = road_matrix[1:, 1:]
        
        # Results
        self.labels = None
        self.n_clusters = None
        self.n_outliers = None
        self.probabilities = None
        self.clusterer = None
    
    def cluster(self) -> np.ndarray:
        """
        Cluster clients using HDBSCAN on road distances.
        If max_clusters is set, merges outliers until constraint is satisfied.
        
        Returns:
            Array of cluster labels for each client (-1 = outlier)
        """
        print(f"\n{'='*70}")
        print(f"HDBSCAN CLUSTERING (Road Distance Matrix)")
        print(f"{'='*70}")
        print(f"Clients: {self.num_clients}")
        print(f"Min cluster size: {self.min_cluster_size}")
        print(f"Min samples: {self.min_samples}")
        if self.max_clusters:
            print(f"Max clusters (including outliers): {self.max_clusters}")
        
        # Make matrix symmetric
        symmetric_matrix = (self.client_road_matrix + self.client_road_matrix.T) / 2
        np.fill_diagonal(symmetric_matrix, 0)
        
        # Run HDBSCAN
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        
        self.labels = self.clusterer.fit_predict(symmetric_matrix)
        self.probabilities = self.clusterer.probabilities_
        
        # Count initial results
        self._update_counts()
        
        print(f"\nInitial HDBSCAN result:")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Outliers: {self.n_outliers}")
        print(f"  Total: {self.n_clusters + self.n_outliers}")
        
        # Enforce max_clusters constraint
        if self.max_clusters:
            self._enforce_max_clusters()
        
        # Split large clusters if needed (future use)
        if self.max_clients_per_cluster:
            self._split_large_clusters()
        
        self._update_counts()
        self._print_cluster_stats()
        
        return self.labels
    
    def _update_counts(self):
        """Update cluster and outlier counts."""
        unique_labels = np.unique(self.labels)
        self.n_clusters = len(unique_labels[unique_labels >= 0])
        self.n_outliers = np.sum(self.labels == -1)
    
    def _enforce_max_clusters(self):
        """
        Merge outliers into nearest clusters until total <= max_clusters.
        Each outlier counts as 1 toward the total.
        """
        total = self.n_clusters + self.n_outliers
        
        if total <= self.max_clusters:
            print(f"\nTotal ({total}) <= max ({self.max_clusters}). No merging needed.")
            return
        
        print(f"\n{'-'*70}")
        print(f"ENFORCING MAX CLUSTERS CONSTRAINT")
        print(f"{'-'*70}")
        print(f"Total ({total}) > max ({self.max_clusters}). Merging outliers...")
        
        # How many outliers need to be merged?
        to_merge = total - self.max_clusters
        
        if to_merge > self.n_outliers:
            print(f"Warning: Need to merge {to_merge} but only {self.n_outliers} outliers.")
            print(f"Will merge all outliers, but constraint cannot be fully satisfied.")
            to_merge = self.n_outliers
        
        # Get outlier indices sorted by distance to nearest cluster
        outlier_indices = np.where(self.labels == -1)[0]
        
        # Calculate distance from each outlier to nearest cluster
        outlier_distances = []
        for outlier_idx in outlier_indices:
            nearest_cluster, min_dist = self._find_nearest_cluster(outlier_idx)
            outlier_distances.append((outlier_idx, nearest_cluster, min_dist))
        
        # Sort by distance (merge closest outliers first)
        outlier_distances.sort(key=lambda x: x[2])
        
        # Merge outliers
        for i in range(to_merge):
            outlier_idx, nearest_cluster, dist = outlier_distances[i]
            client_name = self.locations[outlier_idx + 1].name
            
            self.labels[outlier_idx] = nearest_cluster
            print(f"  Merged: {client_name} → Cluster {nearest_cluster + 1} (dist: {dist:.2f} km)")
        
        self._update_counts()
        print(f"\nAfter merging:")
        print(f"  Clusters: {self.n_clusters}")
        print(f"  Outliers: {self.n_outliers}")
        print(f"  Total: {self.n_clusters + self.n_outliers}")
    
    def _find_nearest_cluster(self, client_idx: int) -> Tuple[int, float]:
        """
        Find the nearest cluster to a client based on average road distance.
        
        Args:
            client_idx: Client index (in client space, 0-indexed)
            
        Returns:
            Tuple of (cluster_id, average_distance)
        """
        best_cluster = 0
        best_dist = np.inf
        
        unique_labels = np.unique(self.labels)
        cluster_labels = unique_labels[unique_labels >= 0]
        
        for cluster_id in cluster_labels:
            cluster_indices = np.where(self.labels == cluster_id)[0]
            
            # Average distance from client to cluster members
            dists = [self.client_road_matrix[client_idx, idx] for idx in cluster_indices]
            avg_dist = np.mean(dists)
            
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_cluster = cluster_id
        
        return best_cluster, best_dist
    
    def _split_large_clusters(self):
        """
        Split clusters that exceed max_clients_per_cluster.
        Uses hierarchical clustering within the large cluster.
        (For future use)
        """
        if not self.max_clients_per_cluster:
            return
        
        print(f"\n{'-'*70}")
        print(f"CHECKING FOR LARGE CLUSTERS (max {self.max_clients_per_cluster} clients)")
        print(f"{'-'*70}")
        
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        unique_labels = np.unique(self.labels)
        cluster_labels = unique_labels[unique_labels >= 0]
        
        splits_made = False
        new_cluster_id = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
        
        for cluster_id in cluster_labels:
            cluster_indices = np.where(self.labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size > self.max_clients_per_cluster:
                splits_made = True
                print(f"\nCluster {cluster_id + 1} has {cluster_size} clients. Splitting...")
                
                # Determine how many sub-clusters needed
                n_subclusters = int(np.ceil(cluster_size / self.max_clients_per_cluster))
                
                # Get sub-matrix for this cluster
                sub_matrix = self.client_road_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Make symmetric
                sub_matrix = (sub_matrix + sub_matrix.T) / 2
                np.fill_diagonal(sub_matrix, 0)
                
                # Hierarchical clustering within this cluster
                condensed = squareform(sub_matrix)
                linkage_matrix = linkage(condensed, method='average')
                sub_labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust') - 1
                
                # Assign new labels
                # Keep first sub-cluster with original ID, give others new IDs
                for sub_cluster in range(n_subclusters):
                    sub_indices = cluster_indices[sub_labels == sub_cluster]
                    
                    if sub_cluster == 0:
                        # Keep original cluster ID
                        for idx in sub_indices:
                            self.labels[idx] = cluster_id
                        print(f"  Sub-cluster 1: {len(sub_indices)} clients → Cluster {cluster_id + 1}")
                    else:
                        # Assign new cluster ID
                        for idx in sub_indices:
                            self.labels[idx] = new_cluster_id
                        print(f"  Sub-cluster {sub_cluster + 1}: {len(sub_indices)} clients → Cluster {new_cluster_id + 1}")
                        new_cluster_id += 1
        
        if not splits_made:
            print("No clusters exceed the limit. No splitting needed.")
    
    def _print_cluster_stats(self):
        """Print statistics for each cluster."""
        print(f"\n{'-'*70}")
        print(f"FINAL CLUSTER STATISTICS")
        print(f"{'-'*70}")
        
        clients = self.locations[1:]
        
        # Print outliers first
        if self.n_outliers > 0:
            outlier_indices = np.where(self.labels == -1)[0]
            
            print(f"\nOUTLIERS ({self.n_outliers} clients):")
            print(f"  Each outlier = 1 employee assignment")
            for idx in outlier_indices:
                client = clients[idx]
                office_dist = self.road_matrix[0, idx + 1]
                print(f"  - {client.name} ({office_dist:.2f} km from office)")
        
        # Print each cluster
        unique_labels = np.unique(self.labels)
        cluster_labels = sorted(unique_labels[unique_labels >= 0])
        
        for cluster_id in cluster_labels:
            cluster_mask = self.labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)
            
            # Calculate internal distances
            if cluster_size > 1:
                internal_dists = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i < j:
                            internal_dists.append(self.client_road_matrix[i, j])
                avg_internal = np.mean(internal_dists)
                max_internal = np.max(internal_dists)
            else:
                avg_internal = 0
                max_internal = 0
            
            # Distance from office
            office_dists = [self.road_matrix[0, idx + 1] for idx in cluster_indices]
            avg_office_dist = np.mean(office_dists)
            
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Clients: {cluster_size}")
            print(f"  Avg distance from office: {avg_office_dist:.2f} km")
            print(f"  Avg internal distance: {avg_internal:.2f} km")
            print(f"  Max internal distance: {max_internal:.2f} km")
            
            # List client names
            client_names = [clients[idx].name for idx in cluster_indices]
            if len(client_names) <= 5:
                print(f"  Members: {', '.join(client_names)}")
            else:
                print(f"  Members: {', '.join(client_names[:5])} + {len(client_names) - 5} more")
    
    def get_labels(self) -> np.ndarray:
        """Get cluster labels for each client (-1 = outlier)."""
        return self.labels
    
    def get_total_assignments(self) -> int:
        """Get total assignments (clusters + outliers)."""
        return self.n_clusters + self.n_outliers
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get list of client names in a cluster."""
        clients = self.locations[1:]
        cluster_indices = np.where(self.labels == cluster_id)[0]
        return [clients[idx].name for idx in cluster_indices]
    
    def get_outliers(self) -> List[str]:
        """Get list of outlier client names."""
        clients = self.locations[1:]
        outlier_indices = np.where(self.labels == -1)[0]
        return [clients[idx].name for idx in outlier_indices]
    
    def get_cluster_assignments(self) -> dict:
        """Get all cluster assignments as dictionary."""
        assignments = {}
        
        # Add clusters
        unique_labels = np.unique(self.labels)
        cluster_labels = sorted(unique_labels[unique_labels >= 0])
        
        for cluster_id in cluster_labels:
            assignments[f"Cluster_{cluster_id + 1}"] = self.get_cluster_members(cluster_id)
        
        # Add outliers (each as separate assignment)
        outliers = self.get_outliers()
        for i, outlier_name in enumerate(outliers):
            assignments[f"Outlier_{i + 1}"] = [outlier_name]
        
        return assignments