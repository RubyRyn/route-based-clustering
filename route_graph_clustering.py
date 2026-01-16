"""
Cluster using graph community detection based on route overlap
"""

import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import List, Optional
from location import Location
from road_distance_calculator import RoadDistanceCalculator


class RouteGraphClustering:
    """Use graph-based clustering on route similarities"""
    
    def __init__(self, distance_calculator: RoadDistanceCalculator, locations: List[Location]):
        self.distance_calculator = distance_calculator
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:]
        self.road_matrix: Optional[np.ndarray] = None
    
    def calculate_route_similarity(self, client1: Location, client2: Location) -> float:
        """Calculate route similarity"""
        route1 = self.distance_calculator.get_route_geometry(self.office, client1)
        route2 = self.distance_calculator.get_route_geometry(self.office, client2)
        
        if not route1 or not route2:
            return 0.0
        
        shared_points = 0
        threshold = 0.001
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
    
    def cluster(self, similarity_threshold: float = 0.5) -> np.ndarray:
        """Cluster using graph community detection"""
        n_clients = len(self.clients)
        G = nx.Graph()
        
        for i in range(n_clients):
            G.add_node(i, name=self.clients[i].name)
        
        print("\nBuilding similarity graph...")
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                sim = self.calculate_route_similarity(self.clients[i], self.clients[j])
                if sim > similarity_threshold:
                    G.add_edge(i, j, weight=sim)
        
        communities = community.louvain_communities(G, weight='weight', seed=42)
        
        labels = np.zeros(n_clients, dtype=int)
        for cluster_id, comm in enumerate(communities):
            for client_idx in comm:
                labels[client_idx] = cluster_id
        
        print(f"Found {len(communities)} natural clusters")
        return labels