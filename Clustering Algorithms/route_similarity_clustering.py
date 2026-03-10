"""
Cluster clients based on route similarity (routes that share common paths)
"""

import numpy as np
from typing import List, Tuple, Optional
from location import Location
from road_distance_calculator import RoadDistanceCalculator


class RouteSimilarityClustering:
    """Cluster clients based on how similar their routes are from office"""
    
    def __init__(self, distance_calculator: RoadDistanceCalculator, locations: List[Location]):
        self.distance_calculator = distance_calculator
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:]
        self.road_matrix: Optional[np.ndarray] = None 
    
    def calculate_route_similarity(self, client1: Location, client2: Location) -> float:
        """Calculate similarity between routes to two clients (0-1 scale)"""
        route1 = self.distance_calculator.get_route_geometry(self.office, client1)
        route2 = self.distance_calculator.get_route_geometry(self.office, client2)
        
        if not route1 or not route2:
            return 0.0
        
        shared_ratio = self._calculate_shared_path_ratio(route1, route2)
        return shared_ratio
    
    def _calculate_shared_path_ratio(self, route1: List[Tuple[float, float]], 
                                    route2: List[Tuple[float, float]]) -> float:
        """Calculate what ratio of the routes overlap"""
        shared_points = 0
        threshold = 0.001 # ~100m in lat/lon degrees
        min_length = min(len(route1), len(route2))
        
        for i in range(min_length):
            lat1, lon1 = route1[i]
            lat2, lon2 = route2[i]
            dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
            
            if dist < threshold:
                shared_points += 1
            else:
                break
        
        avg_length = (len(route1) + len(route2)) / 2
        return shared_points / avg_length if avg_length > 0 else 0
    
    def build_similarity_matrix(self) -> np.ndarray:
        """Build similarity matrix between all clients"""
        n_clients = len(self.clients)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        print("\nCalculating route similarities...")
        
        for i in range(n_clients):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, n_clients):
                sim = self.calculate_route_similarity(self.clients[i], self.clients[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def cluster(self, n_clusters: int = 5, method: str = 'hierarchical') -> np.ndarray:
        """Cluster clients based on route similarity"""
        similarity_matrix = self.build_similarity_matrix()
        distance_matrix = 1 - similarity_matrix
        
        if method == 'hierarchical':
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            condensed = squareform(distance_matrix)
            linkage_matrix = linkage(condensed, method='average')
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
        elif method == 'spectral':
            from sklearn.cluster import SpectralClustering
            
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            labels = clustering.fit_predict(similarity_matrix)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return labels