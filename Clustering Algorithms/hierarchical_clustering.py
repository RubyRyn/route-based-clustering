"""
Hierarchical Clustering based on Road Distance Matrix

Groups clients that are close BY ROAD (not Euclidean distance).
This handles real-world obstacles like rivers, mountains, no direct routes.
"""

import numpy as np
from typing import List, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from location import Location


class RoadDistance_HierarchicalClustering:
    """
    Cluster clients using hierarchical clustering on road distances.
    
    Unlike K-Medoids++ which assigns to nearest medoid,
    this groups clients that are close to EACH OTHER by road.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 max_clusters: int, linkage_method: str = 'average'):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            max_clusters: Maximum number of clusters (employees)
            linkage_method: Hierarchical clustering linkage method
                           'average' - balanced clusters (recommended)
                           'complete' - compact clusters (minimizes max distance within cluster)
                           'ward' - minimizes variance (tends to create equal-sized clusters)
                           'single' - can create chain-like clusters
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_clusters = max_clusters
        self.linkage_method = linkage_method
        
        self.num_locations = len(locations)
        self.num_clients = self.num_locations - 1
        
        # Client-only road distance matrix (exclude office)
        self.client_road_matrix = road_matrix[1:, 1:]
        
        # Results
        self.labels = None
        self.n_clusters = None
        self.linkage_matrix = None
    
    def cluster(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Cluster clients using hierarchical clustering on road distances.
        
        Args:
            n_clusters: Number of clusters (default: max_clusters)
            
        Returns:
            Array of cluster labels for each client
        """
        if n_clusters is None:
            n_clusters = self.max_clusters
        
        # Don't create more clusters than clients
        n_clusters = min(n_clusters, self.num_clients)
        
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL CLUSTERING (Road Distance Matrix)")
        print(f"{'='*70}")
        print(f"Clients: {self.num_clients}")
        print(f"Target clusters: {n_clusters}")
        print(f"Linkage method: {self.linkage_method}")
        
        # Make matrix symmetric (average of both directions)
        symmetric_matrix = (self.client_road_matrix + self.client_road_matrix.T) / 2
        np.fill_diagonal(symmetric_matrix, 0)
        
        # Convert to condensed form for scipy
        condensed = squareform(symmetric_matrix)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(condensed, method=self.linkage_method)
        
        # Cut tree to get desired number of clusters
        self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
        self.n_clusters = len(np.unique(self.labels))
        
        print(f"\nClustering complete!")
        self._print_cluster_stats()
        
        return self.labels
    
    def _print_cluster_stats(self):
        """Print statistics for each cluster."""
        print(f"\n{'-'*70}")
        print(f"CLUSTER STATISTICS")
        print(f"{'-'*70}")
        
        clients = self.locations[1:]
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)
            
            # Calculate internal distances (within cluster)
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
        """Get cluster labels for each client."""
        return self.labels
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Get list of client names in a cluster."""
        clients = self.locations[1:]
        cluster_indices = np.where(self.labels == cluster_id)[0]
        return [clients[idx].name for idx in cluster_indices]
    
    def get_cluster_assignments(self) -> dict:
        """Get all cluster assignments as dictionary."""
        assignments = {}
        for cluster_id in range(self.n_clusters):
            assignments[f"Cluster_{cluster_id + 1}"] = self.get_cluster_members(cluster_id)
        return assignments