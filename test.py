"""
HDBSCAN Clustering based on Road Distance Matrix (Simple Version)

Density-based clustering that:
- Automatically finds the number of clusters
- Uses road distance matrix (handles rivers, mountains, etc.)
- Identifies outliers (isolated clients)
"""

import numpy as np
from typing import List
from sklearn.cluster import HDBSCAN
from location import Location


class HDBSCANClusteringSimple:
    """
    Simple HDBSCAN clustering using road distance matrix.
    
    Automatically determines the number of clusters based on data density.
    Outliers (isolated clients) are labeled as -1.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 min_cluster_size: int = 2, min_samples: int = 1):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            min_cluster_size: Minimum number of clients to form a cluster (default: 2)
            min_samples: How conservative clustering is. Higher = more outliers (default: 1)
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
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
        
        Returns:
            Array of cluster labels for each client (-1 = outlier)
        """
        print(f"\n{'='*70}")
        print(f"HDBSCAN CLUSTERING (Road Distance Matrix)")
        print(f"{'='*70}")
        print(f"Clients: {self.num_clients}")
        print(f"Min cluster size: {self.min_cluster_size}")
        print(f"Min samples: {self.min_samples}")
        
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
        
        # Count clusters and outliers
        unique_labels = np.unique(self.labels)
        self.n_clusters = len(unique_labels[unique_labels >= 0])
        self.n_outliers = np.sum(self.labels == -1)
        
        print(f"\nClustering complete!")
        print(f"Clusters found: {self.n_clusters}")
        print(f"Outliers: {self.n_outliers}")
        
        self._print_cluster_stats()
        
        return self.labels
    
    def _print_cluster_stats(self):
        """Print statistics for each cluster."""
        print(f"\n{'-'*70}")
        print(f"CLUSTER STATISTICS")
        print(f"{'-'*70}")
        
        clients = self.locations[1:]
        
        # Print outliers first
        if self.n_outliers > 0:
            outlier_indices = np.where(self.labels == -1)[0]
            
            print(f"\nOUTLIERS ({self.n_outliers} clients):")
            for idx in outlier_indices:
                client = clients[idx]
                office_dist = self.road_matrix[0, idx + 1]
                print(f"  - {client.name} ({office_dist:.2f} km from office)")
        
        # Print each cluster
        for cluster_id in range(self.n_clusters):
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
        
        # Add outliers
        outliers = self.get_outliers()
        if outliers:
            assignments["Outliers"] = outliers
        
        # Add clusters
        for cluster_id in range(self.n_clusters):
            assignments[f"Cluster_{cluster_id + 1}"] = self.get_cluster_members(cluster_id)
        
        return assignments