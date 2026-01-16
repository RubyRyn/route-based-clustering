import requests
from typing import List, Tuple, Optional
from location import Location


class RoadDistanceCalculator:
    """Calculate actual road distances using routing APIs"""
    
    def __init__(self, api_type: str = 'osrm_demo'):
        """
        Args:
            api_type: 'osrm_demo', 'osrm_local', 'mapbox', or 'graphhopper'
        """
        self.api_type = api_type
        self.cache = {}  # Cache API results to avoid redundant calls
        self.route_geometries = {}  # Cache route geometries for visualization
        
    def get_distance(self, loc1: Location, loc2: Location) -> Optional[float]:
        """
        Get actual road distance between two locations
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in kilometers, or None if API call fails
        """
        # Check cache first
        cache_key = f"{loc1.id}_{loc2.id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Call appropriate API
        if self.api_type == 'osrm_demo':
            distance, geometry = self._osrm_demo(loc1, loc2)
        elif self.api_type == 'osrm_local':
            distance, geometry = self._osrm_local(loc1, loc2)
        elif self.api_type == 'mapbox':
            distance, geometry = self._mapbox(loc1, loc2)
        elif self.api_type == 'graphhopper':
            distance, geometry = self._graphhopper(loc1, loc2)
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")
        
        # Cache results
        if distance is not None:
            self.cache[cache_key] = distance
            self.cache[f"{loc2.id}_{loc1.id}"] = distance  # Symmetric
        
        if geometry is not None:
            self.route_geometries[cache_key] = geometry
            # Reverse geometry for opposite direction
            self.route_geometries[f"{loc2.id}_{loc1.id}"] = list(reversed(geometry))
        
        return distance
    
    def get_route_geometry(self, loc1: Location, loc2: Location) -> Optional[List[Tuple[float, float]]]:
        """
        Get the actual route geometry (list of lat/lon points) between two locations
        
        Returns:
            List of (lat, lon) tuples representing the route path, or None
        """
        cache_key = f"{loc1.id}_{loc2.id}"
        return self.route_geometries.get(cache_key)
    
    def _osrm_demo(self, loc1: Location, loc2: Location) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """
        Use OSRM public demo server
        WARNING: Rate-limited and may be slow. For testing only!
        For production, use local OSRM server instead.
        
        Returns:
            Tuple of (distance_km, route_geometry)
        """
        url = f"http://router.project-osrm.org/route/v1/driving/{loc1.lon},{loc1.lat};{loc2.lon},{loc2.lat}"
        params = {
            'overview': 'full',  # Changed from 'false' to get full route geometry
            'geometries': 'geojson',  # Get geometry in GeoJSON format
            'alternatives': 'false',
            'steps': 'false',
            'annotations': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and len(data['routes']) > 0:
                # Distance is in meters, convert to km
                distance_km = data['routes'][0]['distance'] / 1000.0
                
                # Extract route geometry (list of [lon, lat] pairs)
                geometry_coords = data['routes'][0]['geometry']['coordinates']
                # Convert to list of (lat, lon) tuples for folium
                route_geometry = [(lat, lon) for lon, lat in geometry_coords]
                
                return distance_km, route_geometry
            else:
                print(f"OSRM error for {loc1.id}->{loc2.id}: {data.get('code', 'Unknown')}")
                return None, None
                
        except requests.exceptions.Timeout:
            print(f"Timeout for {loc1.id}->{loc2.id} (public server may be slow)")
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {loc1.id}->{loc2.id}: {e}")
            return None, None
    
    def _osrm_local(self, loc1: Location, loc2: Location, 
                    host: str = 'localhost', port: int = 5000) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """
        Use local OSRM server
        Requires OSRM running on localhost:5000 (see setup instructions at top of file)
        
        Returns:
            Tuple of (distance_km, route_geometry)
        """
        url = f"http://{host}:{port}/route/v1/driving/{loc1.lon},{loc1.lat};{loc2.lon},{loc2.lat}"
        params = {
            'overview': 'full',  # Get full route geometry
            'geometries': 'geojson',  # Get geometry in GeoJSON format
            'alternatives': 'false',
            'steps': 'false',
            'annotations': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and len(data['routes']) > 0:
                distance_km = data['routes'][0]['distance'] / 1000.0
                
                # Extract route geometry
                geometry_coords = data['routes'][0]['geometry']['coordinates']
                route_geometry = [(lat, lon) for lon, lat in geometry_coords]
                
                return distance_km, route_geometry
            else:
                print(f"OSRM local error for {loc1.id}->{loc2.id}: {data.get('code', 'Unknown')}")
                return None, None
                
        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Cannot connect to local OSRM server at {host}:{port}")
            print(f"Make sure OSRM is running. See setup instructions at top of file.")
            print(f"Or change api_choice to 'osrm_demo' to use public server.\n")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Local OSRM request failed: {e}")
            return None, None
    
    def _mapbox(self, loc1: Location, loc2: Location, 
                api_key: Optional[str] = None) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """
        Use Mapbox Directions API
        Requires API key set in environment or passed as parameter
        
        Returns:
            Tuple of (distance_km, route_geometry)
        """
        if api_key is None:
            import os
            api_key = os.getenv('MAPBOX_API_KEY')
            if not api_key:
                print("Mapbox API key not found. Set MAPBOX_API_KEY environment variable.")
                return None, None
        
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{loc1.lon},{loc1.lat};{loc2.lon},{loc2.lat}"
        params = {
            'access_token': api_key,
            'overview': 'full',
            'geometries': 'geojson'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'routes' in data and len(data['routes']) > 0:
                distance_km = data['routes'][0]['distance'] / 1000.0
                
                # Extract route geometry
                geometry_coords = data['routes'][0]['geometry']['coordinates']
                route_geometry = [(lat, lon) for lon, lat in geometry_coords]
                
                return distance_km, route_geometry
            else:
                return None, None
                
        except requests.exceptions.RequestException as e:
            print(f"Mapbox request failed: {e}")
            return None, None
    
    def _graphhopper(self, loc1: Location, loc2: Location,
                     api_key: Optional[str] = None) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
        """
        Use GraphHopper Routing API
        Requires API key
        
        Returns:
            Tuple of (distance_km, route_geometry)
        """
        if api_key is None:
            import os
            api_key = os.getenv('GRAPHHOPPER_API_KEY')
            if not api_key:
                print("GraphHopper API key not found.")
                return None, None
        
        url = "https://graphhopper.com/api/1/route"
        params = {
            'point': [f"{loc1.lat},{loc1.lon}", f"{loc2.lat},{loc2.lon}"],
            'vehicle': 'car',
            'key': api_key,
            'points_encoded': 'false'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'paths' in data and len(data['paths']) > 0:
                distance_km = data['paths'][0]['distance'] / 1000.0
                
                # Extract route geometry
                points = data['paths'][0]['points']['coordinates']
                route_geometry = [(lat, lon) for lon, lat in points]
                
                return distance_km, route_geometry
            else:
                return None, None
                
        except requests.exceptions.RequestException as e:
            print(f"GraphHopper request failed: {e}")
            return None, None
