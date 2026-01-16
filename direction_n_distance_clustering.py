"""
Cluster clients based on direction and distance from office
"""

import numpy as np
from typing import List, Optional
from sklearn.cluster import KMeans
from location import Location
from road_distance_calculator import RoadDistanceCalculator


class DirectionDistanceClustering:
    """Cluster clients by direction and distance from office"""
    
    def __init__(self, distance_calculator: RoadDistanceCalculator, locations: List[Location]):
        self.distance_calculator = distance_calculator
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:]
        self.road_matrix: Optional[np.ndarray] = None 
    
    def cluster(self, n_clusters: int = 5) -> np.ndarray:
        """Cluster based on polar coordinates from office"""
        features = []
        
        for i, client in enumerate(self.clients):
            dx = client.lon - self.office.lon
            dy = client.lat - self.office.lat
            angle = np.arctan2(dy, dx)
            
            if self.road_matrix is not None:
                client_idx = i + 1
                distance = self.road_matrix[0, client_idx]
            else:
                distance = np.sqrt(dx**2 + dy**2)
            
            features.append([angle * 10, distance])
        
        features = np.array(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        return labels