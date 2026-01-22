import numpy as np
import pandas as pd
import time
import json
from typing import List, Tuple
from location import Location
from road_distance_calculator import RoadDistanceCalculator


class DistanceMatrixGenerator:
    """Generate Euclidean and road distance matrices"""
    
    def __init__(self, random_seed: int = 42, api_type: str = 'osrm_demo', 
                 use_fallback: bool = True):
        """
        Args:
            random_seed: Random seed for reproducibility
            api_type: Which routing API to use
            use_fallback: If True, fall back to approximation if API fails
        """
        self.use_fallback = use_fallback
        np.random.seed(random_seed)
        
        self.locations: List[Location] = []
        self.euclidean_matrix: np.ndarray = None
        self.road_matrix: np.ndarray = None
        self.road_calculator = RoadDistanceCalculator(api_type=api_type)
    
    def load_locations_from_csv(self, csv_file: str, 
                                lat_col: str = 'Latitude', 
                                lon_col: str = 'Longitude',
                                name_col: str = 'Name',
                                type_col: str = 'Type') -> List[Location]:
        """Load locations from CSV file"""
        df = pd.read_csv(csv_file)
        
        locations = []
        for idx, row in df.iterrows():
            loc_id = row[name_col].lower().replace(' ', '_')
            loc = Location(
                id=loc_id,
                name=row[name_col],
                lat=float(row[lat_col]),
                lon=float(row[lon_col]),
                loc_type=row[type_col]
            )
            locations.append(loc)
        
        locations.sort(key=lambda x: (x.type != 'Office', x.name))
        self.locations = locations
        print(f"Loaded {len(locations)} locations from {csv_file}")
        
        return locations
    
    def calculate_euclidean_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance using Haversine formula"""
        R = 6371
        lat1_rad = np.radians(loc1.lat)
        lat2_rad = np.radians(loc2.lat)
        dlat = np.radians(loc2.lat - loc1.lat)
        dlon = np.radians(loc2.lon - loc1.lon)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def generate_distance_matrices(self, delay_between_calls: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate both Euclidean and road distance matrices"""
        if not self.locations:
            raise RuntimeError("No locations loaded. Call load_locations_from_csv first.")
        
        n = len(self.locations)
        euclidean_matrix = np.zeros((n, n))
        road_matrix = np.zeros((n, n))
        
        total_calls = n * (n - 1) // 2
        completed_calls = 0
        
        print(f"\nCalculating distances for {n} locations ({total_calls} API calls)...")
        
        for i in range(n):
            for j in range(i + 1, n):
                euc_dist = self.calculate_euclidean_distance(self.locations[i], self.locations[j])
                euclidean_matrix[i, j] = euc_dist
                euclidean_matrix[j, i] = euc_dist
                
                road_dist = self.road_calculator.get_distance(self.locations[i], self.locations[j])
                
                if road_dist is None:
                    if self.use_fallback:
                        road_dist = euc_dist * 1.3
                        print(f"Using fallback for {self.locations[i].name} -> {self.locations[j].name}")
                    else:
                        raise RuntimeError(f"Failed to get road distance")
                
                road_matrix[i, j] = road_dist
                road_matrix[j, i] = road_dist
                
                completed_calls += 1
                if completed_calls % 10 == 0:
                    print(f"Progress: {completed_calls}/{total_calls}")
                
                if delay_between_calls > 0:
                    time.sleep(delay_between_calls)
        
        print(f"All distance calculations complete!")
        
        self.euclidean_matrix = euclidean_matrix
        self.road_matrix = road_matrix
        
        return euclidean_matrix, road_matrix
    
    def get_location_names(self) -> List[str]:
        """Get list of location names"""
        return [loc.name for loc in self.locations]
    

    def export_to_json(self, filename: str = 'delivery_data.json'):
        """Export all data to JSON"""
        data = {
            'metadata': {
                'api_type': self.road_calculator.api_type,
                'total_locations': len(self.locations)
            },
            'locations': [loc.to_dict() for loc in self.locations],
            'euclidean_distance_matrix': self.euclidean_matrix.tolist(),
            'road_distance_matrix': self.road_matrix.tolist()
        }

        with open(f"Output\\{filename}", 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {filename}")
    
    def export_to_csv(self, filename: str = 'distance_matrix.csv'):
        """Export matrices to CSV"""
        names = self.get_location_names()
        df_euc = pd.DataFrame(self.euclidean_matrix, index=names, columns=names)
        df_road = pd.DataFrame(self.road_matrix, index=names, columns=names)

        df_euc.to_csv(f'Output\\euclidean_{filename}')
        df_road.to_csv(f'Output\\road_{filename}')
        print(f"Exported matrices to euclidean_{filename} and road_{filename}")