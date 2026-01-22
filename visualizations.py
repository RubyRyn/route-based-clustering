
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import folium
from folium import plugins
FOLIUM_AVAILABLE = True
from location import Location
from road_distance_calculator import RoadDistanceCalculator


def plot_static_map(locations: List[Location], road_matrix: np.ndarray, 
                    euclidean_matrix: np.ndarray, figsize=(12, 5)):
    """Create static matplotlib visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    office = locations[0]
    clients = locations[1:]
    
    ax1.scatter([office.lon], [office.lat], c='red', s=200, marker='s', 
               label='Office', zorder=3, edgecolors='black', linewidths=2)
    
    client_lons = [c.lon for c in clients]
    client_lats = [c.lat for c in clients]
    ax1.scatter(client_lons, client_lats, c='blue', s=100, 
               label='Clients', zorder=2, edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Delivery Locations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if road_matrix is not None:
        ratio_matrix = np.divide(road_matrix, euclidean_matrix, 
                                where=euclidean_matrix != 0)
        np.fill_diagonal(ratio_matrix, 0)
        
        im = ax2.imshow(ratio_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=3.0)
        ax2.set_title('Road/Euclidean Distance Ratio')
        plt.colorbar(im, ax=ax2, label='Ratio')
    
    plt.tight_layout()
    plt.savefig('Output\\delivery_map_static.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_openstreetmap(locations: List[Location], road_calculator: RoadDistanceCalculator,
                       road_matrix: np.ndarray, euclidean_matrix: np.ndarray,
                       filename: str = 'routes_map.html'):
    """Create interactive OpenStreetMap"""
    if not FOLIUM_AVAILABLE:
        print("Folium not available. Install with: pip install folium")
        return
    
    lats = [loc.lat for loc in locations]
    lons = [loc.lon for loc in locations]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    office = locations[0]
    folium.Marker(
        location=[office.lat, office.lon],
        popup=f"<b>{office.name}</b><br>Main Office",
        tooltip=office.name,
        icon=folium.Icon(color='red', icon='building', prefix='fa')
    ).add_to(m)

    for i, client in enumerate(locations[1:], 1):
        if road_matrix is not None:
            road_dist = road_matrix[0, i]
            euc_dist = euclidean_matrix[0, i]
            ratio = road_dist / euc_dist if euc_dist > 0 else 0
            popup_html = f"<b>{client.name}</b><br>Road: {road_dist:.2f}km<br>Direct: {euc_dist:.2f}km<br>Ratio: {ratio:.2f}x"
        else:
            popup_html = f"<b>{client.name}</b>"
        
        folium.Marker(
            location=[client.lat, client.lon],
            popup=popup_html,
            tooltip=client.name,
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)

    if road_matrix is not None:
        for i, client in enumerate(locations[1:], 1):
            road_dist = road_matrix[0, i]
            euc_dist = euclidean_matrix[0, i]
            ratio = road_dist / euc_dist if euc_dist > 0 else 1
            
            color = 'red' if ratio > 2.0 else ('orange' if ratio > 1.5 else 'green')
            weight = 4 if ratio > 2.0 else 3
            
            route_geometry = road_calculator.get_route_geometry(office, client)
            
            if route_geometry and len(route_geometry) > 0:
                folium.PolyLine(
                    locations=route_geometry,
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    popup=f"{office.name} → {client.name}<br>{road_dist:.2f}km"
                ).add_to(m)

    plugins.Fullscreen().add_to(m)
    m.save(f"Output\\{filename}")



def plot_clustered_routes(
    locations: List[Location], 
    labels: np.ndarray,
    road_calculator: RoadDistanceCalculator,
    road_matrix: np.ndarray,
    euclidean_matrix: np.ndarray,
    filename: str = 'clustered_routes.html'
):
    """
    Visualize clusters with routes from office to each client
    (Similar to plot_openstreetmap but colored by cluster)
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available. Install with: pip install folium")
        return
    
    # Create map
    lats = [loc.lat for loc in locations]
    lons = [loc.lon for loc in locations]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12, 
        tiles='OpenStreetMap'
    )

    # Office marker
    office = locations[0]
    folium.Marker(
        location=[office.lat, office.lon],
        popup=f"<b>{office.name}</b><br>Main Office",
        tooltip=office.name,
        icon=folium.Icon(color='red', icon='building', prefix='fa')
    ).add_to(m)

    # Color scheme for clusters
    colors = ['#0C2C55', '#F63049','#FA8112','#5B23FF','#84934A',
              '#C40C0C','#F075AE','#5DD3B6','#9929EA','#E5BA41',
              '#3F9AAE','#8F0177','#EF7722','#78C841','#6B3F69',
               '#D92C54','#932F67','#5A9CB5','#6AECE1','#005461',
                '#CF0F0F','#434E78','#C47BE4','#DD88CF','#006A67',
                 '#229799','#7886C7','#F72C5B','#EF9C66','#FFA1F5',
                  '#FFC55A','#00DFA2','#14B1AB' ]
    # colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
    #            'darkblue', 'darkgreen', 'cadetblue']

    # Plot client markers
    clients = locations[1:]
    for i, client in enumerate(clients):
        cluster_id = labels[i]
        color = colors[cluster_id % len(colors)]
        
        # Get distance info
        road_dist = road_matrix[0, i + 1]
        euc_dist = euclidean_matrix[0, i + 1]
        ratio = road_dist / euc_dist if euc_dist > 0 else 0
        
        popup_html = f"""
            <b>{client.name}</b><br>
            <b>Cluster:</b> {cluster_id + 1}<br>
            <hr style="margin: 5px 0;">
            <b>Road Distance:</b> {road_dist:.2f} km<br>
            <b>Direct Distance:</b> {euc_dist:.2f} km<br>
            <b>Ratio:</b> {ratio:.2f}x
        """
        
        folium.Marker(
            location=[client.lat, client.lon],
            popup=popup_html,
            tooltip=f"{client.name} (Cluster {cluster_id + 1})",
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(m)

    # Plot routes from office to each client (colored by cluster)
    print(f"\nDrawing routes from office to {len(clients)} clients...")
    
    for i, client in enumerate(clients):
        cluster_id = labels[i]
        color = colors[cluster_id % len(colors)]
        
        road_dist = road_matrix[0, i + 1]
        euc_dist = euclidean_matrix[0, i + 1]
        ratio = road_dist / euc_dist if euc_dist > 0 else 1
        
        # Get actual road geometry
        route_geometry = road_calculator.get_route_geometry(office, client)
        
        if route_geometry and len(route_geometry) > 0:
            # Draw route with actual road path
            folium.PolyLine(
                locations=route_geometry,
                color=color,
                weight=4,
                opacity=0.7,
                popup=f"""
                    <b>Cluster {cluster_id + 1}</b><br>
                    {office.name} → {client.name}<br>
                    <b>Distance:</b> {road_dist:.2f} km<br>
                    <b>Ratio:</b> {ratio:.2f}x
                """,
                tooltip=f"Cluster {cluster_id + 1}: {office.name} → {client.name}"
            ).add_to(m)
            
        else:
            # Fallback to straight line if geometry not available
            folium.PolyLine(
                locations=[[office.lat, office.lon], [client.lat, client.lon]],
                color=color,
                weight=3,
                opacity=0.5,
                dash_array='5, 5',
                popup=f"{office.name} → {client.name}<br>{road_dist:.2f} km (approximate)"
            ).add_to(m)

    # Add legend
    num_clusters = len(set(labels))
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 280px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; font-size: 16px;">Route Clusters</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><i class="fa fa-building" style="color:red"></i> <b>Office</b> - Start Point</p>
    <p style="margin: 5px 0;"><i class="fa fa-home" style="color:blue"></i> <b>Clients</b> - Delivery Locations</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0; font-weight: bold;">Clusters (by route similarity):</p>
    """
    
    # Count clients per cluster
    for cluster_id in range(num_clusters):
        color = colors[cluster_id % len(colors)]
        num_clients = sum(1 for label in labels if label == cluster_id)
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="color:{color}; font-weight:bold; font-size:18px;">━━━</span> 
            Cluster {cluster_id + 1} ({num_clients} clients)
        </p>
        '''
    
    legend_html += """
    <hr style="margin: 5px 0;">
    <p style="margin: 3px 0; font-size: 11px; font-style: italic;">
        Routes with similar paths are in the same cluster
    </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add fullscreen plugin
    plugins.Fullscreen().add_to(m)
    m.save(f"Output\\{filename}")