"""
HDBSCAN Clustering based on Road Distance Matrix

Density-based clustering that:
- Automatically finds the number of clusters
- Uses road distance matrix
- Identifies outliers (isolated clients)
"""

import numpy as np
from typing import List

from sklearn.metrics import silhouette_score
# from sklearn.cluster import HDBSCAN
import hdbscan
from hdbscan import HDBSCAN
from hdbscan import validity_index
from location import Location
import pandas as pd
from itertools import product

class HDBSCANClusteringSimple:
    """
    Simple HDBSCAN clustering using road distance matrix.
    
    Automatically determines the number of clusters based on data density.
    Outliers (isolated clients) are labeled as -1.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 min_cluster_size: int = 3, min_samples: int = 1):
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
        print(f"\n")
        print(f"HDBSCAN CLUSTERING (Road Distance Matrix)")
        print(f"\n")
        print(f"Total Clients: {self.num_clients}")
        print(f"Min cluster size: {self.min_cluster_size}")
        print(f"Min samples: {self.min_samples}")
        
        symmetric_matrix = (self.client_road_matrix + self.client_road_matrix.T) / 2
        np.fill_diagonal(symmetric_matrix, 0)
        
        # Ensure non-negative
        symmetric_matrix = np.clip(symmetric_matrix, 0, None)
        
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        
        self.labels = self.clusterer.fit_predict(symmetric_matrix)
        self.probabilities = self.clusterer.probabilities_
        
        unique_labels = np.unique(self.labels)
        self.n_clusters = len(unique_labels[unique_labels >= 0])
        self.n_outliers = np.sum(self.labels == -1)
        
        # Calculate DBCV
        dbcv_score = None
        if self.n_clusters > 0:
            try:
                dbcv_score = validity_index(
                    symmetric_matrix, 
                    self.labels, 
                    metric='precomputed', 
                    d=2
                )
            except Exception as e:
                print(f"Warning: Could not calculate DBCV: {e}")

        # Calculate silhouette score (excluding outliers)
        sil = None
        mask = self.labels != -1
        if mask.sum() > 1 and self.n_clusters > 1:
            try:
                filtered_matrix = symmetric_matrix[mask][:, mask]
                filtered_matrix = np.clip(filtered_matrix, 0, None)
                np.fill_diagonal(filtered_matrix, 0)
                sil = silhouette_score(filtered_matrix, self.labels[mask], metric='precomputed')
            except Exception as e:
                print(f"Warning: Could not calculate silhouette score: {e}")
                
        print(f"\nHDBSCAN Clustering complete!")
        print(f"Clusters found: {self.n_clusters}")
        print(f"Outliers: {self.n_outliers}")
        print(f"DBCV Score: {dbcv_score}")
        print(f"Silhouette Score: {sil if sil is not None else 'N/A'}")
        
        return self.labels


    def tune_hdbscan(self, min_cluster_sizes=range(2, 10),min_samples_range=range(1, 10)):
        """
        Grid search for optimal HDBSCAN parameters.
        """
        
        # Use self.client_road_matrix directly
        symmetric_matrix = (self.client_road_matrix + self.client_road_matrix.T) / 2
        np.fill_diagonal(symmetric_matrix, 0)
        
        results = []
        
        for min_cluster_size, min_samples in product(min_cluster_sizes, min_samples_range):
            
            # Skip invalid combinations
            if min_samples > min_cluster_size:
                continue
            
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='precomputed'
            )
            
            labels = clusterer.fit_predict(symmetric_matrix)
            
            # Calculate metrics
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_outliers = np.sum(labels == -1)
            outlier_pct = n_outliers / len(labels) * 100
            
            # DBCV
            if n_clusters > 0:
                try:
                    dbcv = validity_index(symmetric_matrix, labels, metric='precomputed', d=2)
                except:
                    dbcv = np.nan
            else:
                dbcv = np.nan
            
            # Silhouette (excluding noise)
            mask = labels != -1
            if n_clusters > 1 and mask.sum() > 1:
                try:
                    sil = silhouette_score(symmetric_matrix[mask][:, mask], labels[mask], metric='precomputed')
                except:
                    sil = np.nan
            else:
                sil = np.nan
            
            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'outlier_pct': round(outlier_pct, 1),
                'dbcv': round(dbcv, 4) if not np.isnan(dbcv) else np.nan,
                'silhouette': round(sil, 4) if not np.isnan(sil) else np.nan
            })
        
        df = pd.DataFrame(results)
        return df


    def _print_cluster_stats(self):
        """Print statistics for each cluster."""
        print(f"\n")
        print(f"CLUSTER STATISTICS")
        print(f"\n")
        
        clients = self.locations[1:]
        
        # Print outliers
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
        """Get all cluster assignments as dictionary. Each outlier is its own cluster."""
        assignments = {}
        
        # Add clusters
        for cluster_id in range(self.n_clusters):
            assignments[f"Cluster_{cluster_id + 1}"] = self.get_cluster_members(cluster_id)
        
        # Add each outlier as its own cluster
        outliers = self.get_outliers()
        for i, outlier_name in enumerate(outliers):
            assignments[f"Outlier_{i + 1}"] = [outlier_name]
        
        return assignments
    
    def get_labels_with_outliers_as_clusters(self) -> np.ndarray:
        """
        Get labels where each outlier is assigned its own cluster ID.
        Useful for visualization - no missing points.
        
        Returns:
            Array of cluster labels (no -1 values)
        """
        new_labels = self.labels.copy()
        
        # Find outliers
        outlier_indices = np.where(self.labels == -1)[0]
        
        # Assign each outlier a unique cluster ID starting after existing clusters
        next_cluster_id = self.n_clusters
        for idx in outlier_indices:
            new_labels[idx] = next_cluster_id
            next_cluster_id += 1
        
        return new_labels
    
    def merge_nearby_outliers(self, max_distance_km: float = 7.2) -> np.ndarray:
        """
        Merge outliers into nearest cluster if shortest road distance to any 
        client in that cluster is <= max_distance_km.
        
        Args:
            max_distance_km: Maximum road distance to merge (default 7.2km = 4.5 miles)
            
        Returns:
            Updated labels array
        """
        if self.labels is None:
            raise RuntimeError("Run cluster() first!")
        
        if self.n_outliers == 0:
            print("No outliers to merge.")
            return self.labels
        
        print(f"\n{'-'*70}")
        print(f"MERGING NEARBY OUTLIERS (max distance: {max_distance_km} km)")
        print(f"{'-'*70}")
        
        clients = self.locations[1:]
        outlier_indices = np.where(self.labels == -1)[0]
        merged_count = 0
        
        for outlier_idx in outlier_indices:
            outlier_name = clients[outlier_idx].name
            
            # Find shortest distance to any client in any cluster
            best_cluster = None
            best_distance = np.inf
            best_neighbor = None
            
            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(self.labels == cluster_id)[0]
                
                for cluster_member_idx in cluster_indices:
                    dist = self.client_road_matrix[outlier_idx, cluster_member_idx]
                    
                    if dist < best_distance:
                        best_distance = dist
                        best_cluster = cluster_id
                        best_neighbor = clients[cluster_member_idx].name
            
            # Merge if within threshold
            if best_distance <= max_distance_km:
                self.labels[outlier_idx] = best_cluster
                merged_count += 1
                print(f"  ✓ {outlier_name} → Cluster {best_cluster + 1}")
                print(f"    (nearest: {best_neighbor}, dist: {best_distance:.2f} km)")
            else:
                print(f"  ✗ {outlier_name} remains outlier")
                print(f"    (nearest: {best_neighbor}, dist: {best_distance:.2f} km > {max_distance_km} km)")
        
        # Update counts
        self.n_outliers = np.sum(self.labels == -1)
        
        print(f"\nMerged: {merged_count} outliers")
        print(f"Remaining outliers: {self.n_outliers}")
        
        return self.labels