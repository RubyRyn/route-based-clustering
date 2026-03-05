import requests
import numpy as np
from typing import List, Tuple, Optional
from location import Location


class RoadDistanceCalculator:
    """Calculate actual road distances using routing APIs"""
    
    def __init__(self, api_type: str = 'osrm_demo', server_ip: str = 'localhost', port: int = 5000):
        """
        Args:
            api_type: 'osrm_demo' or 'osrm_local'
            server_ip: IP address for local OSRM server
            port: Port for local OSRM server (default 5000)
        """
        self.api_type = api_type
        self.server_ip = server_ip
        self.port = port
        self.cache = {}
        self.route_geometries = {}
    
    def get_distance_matrix_table(self, locations: List[Location], 
                                   batch_size: int = 100) -> np.ndarray:
        """
        Get full distance matrix using OSRM Table API (much faster).
        
        Args:
            locations: List of Location objects
            batch_size: Max locations per batch (OSRM limit ~100 for demo, higher for self-hosted)
            
        Returns:
            Distance matrix in km (n x n)
        """
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        
        if self.api_type == 'osrm_demo':
            base_url = "http://router.project-osrm.org"
            batch_size = min(batch_size, 100)  # Demo server limit
        else:
            base_url = f"http://{self.server_ip}:{self.port}"
        
        # For small location sets, do it in one call
        if n <= batch_size:
            print(f"Fetching {n}x{n} matrix in single call...")
            distance_matrix = self._table_api_call(base_url, locations, locations)
            return distance_matrix
        
        # For larger sets, batch the requests
        total_batches = (n // batch_size + 1) ** 2
        current_batch = 0
        
        print(f"Fetching {n}x{n} matrix in batches of {batch_size}...")
        
        for i in range(0, n, batch_size):
            for j in range(0, n, batch_size):
                current_batch += 1
                
                # Get source and destination slices
                sources = locations[i:i + batch_size]
                destinations = locations[j:j + batch_size]
                
                print(f"  Batch {current_batch}: rows {i}-{min(i+batch_size, n)}, cols {j}-{min(j+batch_size, n)}")
                
                # Get submatrix
                submatrix = self._table_api_call(base_url, sources, destinations)
                
                if submatrix is not None:
                    # Place submatrix in correct position
                    i_end = min(i + batch_size, n)
                    j_end = min(j + batch_size, n)
                    distance_matrix[i:i_end, j:j_end] = submatrix
                else:
                    print(f"WARNING: Batch failed, using fallback")
                    # Fallback: estimate using Euclidean * 1.3
                    for si, src in enumerate(sources):
                        for di, dst in enumerate(destinations):
                            if i + si < n and j + di < n:
                                euc = self._haversine(src.lat, src.lon, dst.lat, dst.lon)
                                distance_matrix[i + si, j + di] = euc * 1.3
        
        print(f"Distance matrix complete: {n}x{n}")
        return distance_matrix
    
    def _table_api_call(self, base_url: str, sources: List[Location], 
                        destinations: List[Location], max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Make a single Table API call.
        
        Returns:
            Submatrix of distances in km, or None if failed
        """
        import time
        
        # Build coordinates string
        all_locations = sources + destinations
        coords = ";".join([f"{loc.lon},{loc.lat}" for loc in all_locations])
        
        # Source and destination indices
        n_sources = len(sources)
        n_destinations = len(destinations)
        source_indices = ";".join([str(i) for i in range(n_sources)])
        dest_indices = ";".join([str(i) for i in range(n_sources, n_sources + n_destinations)])
        
        url = f"{base_url}/table/v1/driving/{coords}"
        params = {
            'sources': source_indices,
            'destinations': dest_indices,
            'annotations': 'distance'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if data['code'] == 'Ok':
                    # Convert to numpy array (meters → km)
                    distances = np.array(data['distances']) / 1000.0
                    return distances
                else:
                    print(f"Table API error: {data.get('code', 'Unknown')}")
                    return None
            
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"    Connection issue, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"    Failed after {max_retries} attempts")
                    return None
            
            except requests.exceptions.RequestException as e:
                print(f"    Request failed: {e}")
                return None
        
        return None
    
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Euclidean distance using Haversine formula"""
        R = 6371
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_distance(self, loc1: Location, loc2: Location) -> Optional[float]:
        """Get single road distance (for compatibility)"""
        cache_key = f"{loc1.id}_{loc2.id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.api_type == 'osrm_demo':
            distance, geometry = self._osrm_demo(loc1, loc2)
        elif self.api_type == 'osrm_local':
            distance, geometry = self._osrm_local(loc1, loc2)
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")
        
        if distance is not None:
            self.cache[cache_key] = distance
            self.cache[f"{loc2.id}_{loc1.id}"] = distance
        
        if geometry is not None:
            self.route_geometries[cache_key] = geometry
            self.route_geometries[f"{loc2.id}_{loc1.id}"] = list(reversed(geometry))
        
        return distance
    
    def get_route_geometry(self, loc1: Location, loc2: Location) -> Optional[List[Tuple[float, float]]]:
        """Get route geometry (fetches on-demand if not cached)"""
        cache_key = f"{loc1.id}_{loc2.id}"
        
        # Return cached if available
        if cache_key in self.route_geometries:
            return self.route_geometries[cache_key]
        
        # Fetch geometry on-demand
        if self.api_type == 'osrm_demo':
            _, geometry = self._osrm_demo(loc1, loc2)
        else:
            _, geometry = self._osrm_local(loc1, loc2)
        
        if geometry is not None:
            self.route_geometries[cache_key] = geometry
            self.route_geometries[f"{loc2.id}_{loc1.id}"] = list(reversed(geometry))
        
        return geometry
    
    def _osrm_demo(self, loc1: Location, loc2: Location) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """Use OSRM public demo server"""
        url = f"http://router.project-osrm.org/route/v1/driving/{loc1.lon},{loc1.lat};{loc2.lon},{loc2.lat}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'alternatives': 'false',
            'steps': 'false',
            'annotations': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and len(data['routes']) > 0:
                distance_km = data['routes'][0]['distance'] / 1000.0
                geometry_coords = data['routes'][0]['geometry']['coordinates']
                route_geometry = [(lat, lon) for lon, lat in geometry_coords]
                return distance_km, route_geometry
            else:
                return None, None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {loc1.id}->{loc2.id}: {e}")
            return None, None
    
    def _osrm_local(self, loc1: Location, loc2: Location) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """Use local/self-hosted OSRM server with retry"""
        import time
        
        url = f"http://{self.server_ip}:{self.port}/route/v1/driving/{loc1.lon},{loc1.lat};{loc2.lon},{loc2.lat}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'alternatives': 'false',
            'steps': 'false',
            'annotations': 'false'
        }
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if data['code'] == 'Ok' and len(data['routes']) > 0:
                    distance_km = data['routes'][0]['distance'] / 1000.0
                    geometry_coords = data['routes'][0]['geometry']['coordinates']
                    route_geometry = [(lat, lon) for lon, lat in geometry_coords]
                    return distance_km, route_geometry
                else:
                    return None, None
            
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, None
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return None, None
        
        return None, None