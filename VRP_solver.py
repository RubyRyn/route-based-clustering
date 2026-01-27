"""
Vehicle Routing Problem (VRP) Solver using Google OR-Tools

Objectives:
1. Minimize total travel distance
2. Minimize difference between longest and shortest route (workload balance)

Features:
- Dynamic number of vehicles (uses fewer if optimal)
- Soft workload balancing via GlobalSpanCostCoefficient
- Configurable time limit
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from location import Location


class VRPSolver:
    """
    Vehicle Routing Problem solver for delivery route optimization.
    Uses road distance matrix and balances workload across vehicles.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 max_vehicles: int, time_limit_seconds: int = 180,
                 balance_coefficient: int = 100):
        """
        Args:
            road_matrix: Full distance matrix (office + clients), in km
            locations: List of Location objects (office first, then clients)
            max_vehicles: Maximum number of vehicles (employees)
            time_limit_seconds: Solver time limit (default 180 seconds)
            balance_coefficient: Higher = more balanced routes (default 100)
                                 0 = no balancing, just minimize total distance
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.max_vehicles = max_vehicles
        self.time_limit = time_limit_seconds
        self.balance_coefficient = balance_coefficient
        
        self.depot = 0  # Office is at index 0
        self.num_locations = len(locations)
        self.num_clients = self.num_locations - 1
        
        # Convert km to meters (OR-Tools works better with integers)
        # Multiply by 1000 and round to avoid floating point issues
        self.distance_matrix_int = (road_matrix * 1000).astype(int)
        
        # Results
        self.solution = None
        self.routes = None
        self.route_distances = None
        self.labels = None  # Cluster labels for visualization compatibility
    
    def _create_data_model(self) -> dict:
        """Create data model for OR-Tools."""
        data = {
            'distance_matrix': self.distance_matrix_int.tolist(),
            'num_vehicles': self.max_vehicles,
            'depot': self.depot
        }
        return data
    
    def _distance_callback(self, from_index: int, to_index: int, manager) -> int:
        """Returns distance between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return self.distance_matrix_int[from_node][to_node]
    
    def solve(self, verbose: bool = True) -> bool:
        """
        Solve the VRP.
        
        Args:
            verbose: Print progress and results
            
        Returns:
            True if solution found, False otherwise
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"VEHICLE ROUTING PROBLEM (VRP) SOLVER")
            print(f"{'='*70}")
            print(f"Clients: {self.num_clients}")
            print(f"Max vehicles: {self.max_vehicles}")
            print(f"Time limit: {self.time_limit} seconds")
            print(f"Balance coefficient: {self.balance_coefficient}")
            print(f"\nSolving...")
        
        # Create data model
        data = self._create_data_model()
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Create distance callback
        def distance_callback(from_index, to_index):
            return self._distance_callback(from_index, to_index, manager)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc (distance)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add distance dimension for tracking route distances
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # No slack
            999999999,  # Maximum distance per vehicle (arbitrary large number),
            True,  # Start cumul at zero
            dimension_name
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        
        # Set global span cost coefficient for workload balancing
        # This minimizes: Max(route_distance) - Min(route_distance)
        distance_dimension.SetGlobalSpanCostCoefficient(self.balance_coefficient)
        
        # Allow vehicles to be unused (for dynamic fleet size)
        for vehicle_id in range(self.max_vehicles):
            routing.SetFixedCostOfVehicle(0, vehicle_id)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # First solution strategy
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Metaheuristic for improvement
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # Time limit
        search_parameters.time_limit.seconds = self.time_limit
        
        # Log search progress
        search_parameters.log_search = verbose
        
        # Solve
        self.solution = routing.SolveWithParameters(search_parameters)
        
        if self.solution:
            self._extract_routes(manager, routing)
            if verbose:
                self.print_results()
            return True
        else:
            if verbose:
                print("No solution found!")
            return False
    
    def _extract_routes(self, manager, routing):
        """Extract routes from solution."""
        self.routes = []
        self.route_distances = []
        
        # Create labels array for visualization compatibility
        # labels[i] = vehicle_id for client i
        self.labels = np.zeros(self.num_clients, dtype=int)
        
        active_vehicle_id = 0  # Renumber to only count active vehicles
        
        for vehicle_id in range(self.max_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != self.depot:
                    route.append(node)
                    # Assign label (client index is node - 1)
                    self.labels[node - 1] = active_vehicle_id
                
                previous_index = index
                index = self.solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            # Convert distance back to km
            route_distance_km = route_distance / 1000.0
            
            if len(route) > 0:  # Only add non-empty routes
                self.routes.append(route)
                self.route_distances.append(route_distance_km)
                active_vehicle_id += 1
        
        # Renumber labels to be consecutive (0, 1, 2, ...)
        if active_vehicle_id < self.max_vehicles:
            # Some vehicles unused, labels are already correct
            pass
    
    def print_results(self):
        """Print detailed results."""
        if not self.routes:
            print("No solution. Run solve() first.")
            return
        
        print(f"\n{'='*70}")
        print(f"VRP SOLUTION")
        print(f"{'='*70}")
        
        total_distance = sum(self.route_distances)
        vehicles_used = len(self.routes)
        
        print(f"\nSummary:")
        print(f"  Total distance: {total_distance:.2f} km")
        print(f"  Vehicles used: {vehicles_used} of {self.max_vehicles}")
        
        if self.route_distances:
            max_dist = max(self.route_distances)
            min_dist = min(self.route_distances)
            avg_dist = np.mean(self.route_distances)
            
            print(f"\nWorkload Balance:")
            print(f"  Longest route: {max_dist:.2f} km")
            print(f"  Shortest route: {min_dist:.2f} km")
            print(f"  Difference: {max_dist - min_dist:.2f} km")
            print(f"  Average route: {avg_dist:.2f} km")
            if min_dist > 0:
                print(f"  Imbalance ratio: {max_dist / min_dist:.2f}x")
        
        print(f"\n{'-'*70}")
        print(f"ROUTES:")
        print(f"{'-'*70}")
        
        for i, (route, distance) in enumerate(zip(self.routes, self.route_distances)):
            print(f"\nVehicle {i + 1}: {distance:.2f} km ({len(route)} clients)")
            
            # Build route string
            route_str = "  Office"
            for node in route:
                client_name = self.locations[node].name
                route_str += f" → {client_name}"
            route_str += " → Office"
            
            print(route_str)
    
    def get_route_assignments(self) -> Dict[str, List[str]]:
        """
        Get route assignments in readable format.
        
        Returns:
            Dictionary mapping vehicle_id to ordered list of client names
        """
        assignments = {}
        
        for i, route in enumerate(self.routes):
            client_names = [self.locations[node].name for node in route]
            assignments[f"Vehicle_{i + 1}"] = client_names
        
        return assignments
    
    def get_labels(self) -> np.ndarray:
        """
        Get cluster labels for visualization compatibility.
        labels[i] = vehicle_id for client i (0-indexed)
        
        Returns:
            Array of labels for each client
        """
        return self.labels
    
    def get_routes_with_coordinates(self) -> List[List[Tuple[float, float]]]:
        """
        Get routes as list of coordinates for visualization.
        
        Returns:
            List of routes, each route is list of (lat, lon) tuples
        """
        routes_coords = []
        office = self.locations[self.depot]
        
        for route in self.routes:
            coords = [(office.lat, office.lon)]  # Start at office
            
            for node in route:
                loc = self.locations[node]
                coords.append((loc.lat, loc.lon))
            
            coords.append((office.lat, office.lon))  # Return to office
            routes_coords.append(coords)
        
        return routes_coords
    
    def get_medoid_indices(self) -> np.ndarray:
        """
        Get medoid-like indices for visualization compatibility.
        For VRP, we use the first client in each route as the "medoid".
        
        Returns:
            Array of first client index for each route
        """
        medoids = []
        for route in self.routes:
            if route:
                # First client in route (convert from location index to client index)
                medoids.append(route[0] - 1)
            else:
                medoids.append(0)
        return np.array(medoids)