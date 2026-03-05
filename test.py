"""
Test OSRM Table API + Route Geometry Visualization
"""

import requests
import numpy as np
import folium
from folium import plugins

# Your server
SERVER_IP = "91.99.169.156"
PORT = 5000
BASE_URL = f"http://{SERVER_IP}:{PORT}"

# Test locations (name, lat, lon)
locations = [
    ("Client A", 16.76033,95.227028),
    ("Client B", 16.529137,95.266893),
    ("Client C", 16.77478,95.293839),
]

def test_table_api():
    """Test Table API for distance matrix"""
    print("=" * 60)
    print("TEST 1: Table API (Distance Matrix)")
    print("=" * 60)
    
    # Build coordinates string: lon,lat;lon,lat;...
    coords = ";".join([f"{lon},{lat}" for name, lat, lon in locations])
    
    url = f"{BASE_URL}/table/v1/driving/{coords}?annotations=distance"
    print(f"URL: {url}\n")
    
    response = requests.get(url, timeout=30)
    data = response.json()
    
    if data['code'] == 'Ok':
        # Convert to km
        distances = np.array(data['distances']) / 1000.0
        
        print("Distance Matrix (km):")
        print("-" * 60)
        
        # Header
        print(f"{'':>12}", end="")
        for name, _, _ in locations:
            print(f"{name:>12}", end="")
        print()
        
        # Rows
        for i, (name, _, _) in enumerate(locations):
            print(f"{name:>12}", end="")
            for j in range(len(locations)):
                print(f"{distances[i][j]:>12.2f}", end="")
            print()
        
        return distances
    else:
        print(f"Error: {data}")
        return None


def get_route_geometry(origin, destination):
    """
    Get route geometry between two points.
    
    Args:
        origin: (name, lat, lon)
        destination: (name, lat, lon)
    
    Returns:
        List of [lat, lon] points for the route
    """
    origin_name, origin_lat, origin_lon = origin
    dest_name, dest_lat, dest_lon = destination
    
    url = f"{BASE_URL}/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    params = {
        'overview': 'full',
        'geometries': 'geojson'
    }
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    
    if data['code'] == 'Ok' and len(data['routes']) > 0:
        # Extract geometry (OSRM returns [lon, lat], we need [lat, lon] for folium)
        geometry_coords = data['routes'][0]['geometry']['coordinates']
        route_points = [[lat, lon] for lon, lat in geometry_coords]
        
        distance_km = data['routes'][0]['distance'] / 1000.0
        duration_min = data['routes'][0]['duration'] / 60.0
        
        return route_points, distance_km, duration_min
    else:
        print(f"Error getting route: {data.get('code', 'Unknown')}")
        return None, None, None


def test_route_geometry():
    """Test Route API for geometry"""
    print("\n" + "=" * 60)
    print("TEST 2: Route API (Geometry)")
    print("=" * 60)
    
    origin = locations[0]  # Office
    destination = locations[1]  # Client A
    
    route_points, distance, duration = get_route_geometry(origin, destination)
    
    if route_points:
        print(f"Route: {origin[0]} → {destination[0]}")
        print(f"Distance: {distance:.2f} km")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Route points: {len(route_points)} coordinates")
        return True
    return False


def visualize_routes():
    """Visualize all routes on a map"""
    print("\n" + "=" * 60)
    print("TEST 3: Visualize Routes on Map")
    print("=" * 60)
    
    # Create map centered on locations
    center_lat = np.mean([lat for _, lat, _ in locations])
    center_lon = np.mean([lon for _, _, lon in locations])
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Colors for routes
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    # Add markers for all locations
    for i, (name, lat, lon) in enumerate(locations):
        if i == 0:
            # Office marker
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{name}</b><br>Office",
                tooltip=name,
                icon=folium.Icon(color='red', icon='building', prefix='fa')
            ).add_to(m)
        else:
            # Client marker
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{name}</b>",
                tooltip=name,
                icon=folium.Icon(color='blue', icon='home', prefix='fa')
            ).add_to(m)
    
    # Draw routes from Office to each client
    office = locations[0]
    
    for i, client in enumerate(locations[1:]):
        route_points, distance, duration = get_route_geometry(office, client)
        
        if route_points:
            color = colors[i % len(colors)]
            
            folium.PolyLine(
                locations=route_points,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"<b>{office[0]} → {client[0]}</b><br>"
                      f"Distance: {distance:.2f} km<br>"
                      f"Duration: {duration:.1f} min",
                tooltip=f"{office[0]} → {client[0]}: {distance:.2f} km"
            ).add_to(m)
            
            print(f"  Route: {office[0]} → {client[0]}: {distance:.2f} km, {duration:.1f} min")
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Save map
    output_file = "Output\\test_routes.html"
    m.save(output_file)
    print(f"\nMap saved to: {output_file}")
    
    return m


if __name__ == "__main__":
    # Test 1: Table API
    distances = test_table_api()
    
    # Test 2: Route Geometry
    test_route_geometry()
    
    # Test 3: Visualize on Map
    visualize_routes()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)
