"""
Build routes sequentially by assigning clients to minimize detours
"""

import numpy as np
from typing import List,Optional
from location import Location
from road_distance_calculator import RoadDistanceCalculator


class SequentialRoutingClustering:
    """Build routes by sequential assignment"""
    
    def __init__(self, distance_calculator: RoadDistanceCalculator, locations: List[Location]):
        self.distance_calculator = distance_calculator
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:]
        self.road_matrix: Optional[np.ndarray] = None 
    
    def cluster(self, n_routes: int = 5) -> np.ndarray:
        """Build routes sequentially"""
        unassigned = list(self.clients)
        routes = [[] for _ in range(n_routes)]
        
        seeds = self._select_diverse_seeds(n_routes)
        for i, seed in enumerate(seeds):
            routes[i].append(seed)
            unassigned.remove(seed)
        
        while unassigned:
            best_assignment = None 
            min_cost = float('inf')
            
            for client in unassigned:
                for route_idx, route in enumerate(routes):
                    cost = self._calculate_addition_cost(route, client)
                    if cost < min_cost:
                        min_cost = cost
                        best_assignment = (client, route_idx)
            # Type assertion for linters
            assert best_assignment is not None, "No valid assignment found"
            client, route_idx = best_assignment
            routes[route_idx].append(client)
            unassigned.remove(client)
        
        labels = np.zeros(len(self.clients), dtype=int)
        for route_idx, route in enumerate(routes):
            for client in route:
                client_idx = self.clients.index(client)
                labels[client_idx] = route_idx
        
        return labels
    
    def _select_diverse_seeds(self, n_seeds: int) -> List[Location]:
        """Select geographically diverse seed clients"""
        angles = []
        for client in self.clients:
            dx = client.lon - self.office.lon
            dy = client.lat - self.office.lat
            angle = np.arctan2(dy, dx)
            angles.append((angle, client))
        
        angles.sort()
        seeds = []
        step = len(angles) // n_seeds
        for i in range(n_seeds):
            idx = i * step
            seeds.append(angles[idx][1])
        
        return seeds
    
    def _calculate_addition_cost(self, route: List[Location], new_client: Location) -> float:
        """Calculate cost of adding client to route"""
        if not route:
            client_idx = self.locations.index(new_client)
            return self.road_matrix[0, client_idx] if self.road_matrix is not None else 0
        
        last_client = route[-1]
        last_idx = self.locations.index(last_client)
        new_idx = self.locations.index(new_client)
        
        if self.road_matrix is not None:
            return self.road_matrix[last_idx, new_idx]
        else:
            dx = new_client.lon - last_client.lon
            dy = new_client.lat - last_client.lat
            return np.sqrt(dx**2 + dy**2)