import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import folium
from folium import plugins
FOLIUM_AVAILABLE = True
from location import Location
from road_distance_calculator import RoadDistanceCalculator


def plot_road_distance_clusters(locations: List[Location], 
                                 labels: np.ndarray,
                                 road_matrix: np.ndarray,
                                 road_calculator: RoadDistanceCalculator,
                                 filename: str = 'road_distance_clusters.html'):
    """
    Visualize clusters based on road distance with boundaries on OpenStreetMap.
    Handles outliers (-1) by showing them as gray markers.
    
    Args:
        locations: List of Location objects (office first, then clients)
        labels: Cluster labels for each client (-1 = outlier)
        road_matrix: Road distance matrix
        road_calculator: RoadDistanceCalculator for getting route geometry
        filename: Output HTML filename
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available.")
        return
    
    from scipy.spatial import ConvexHull
    
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
        popup=f"<b>{office.name}</b><br>Office",
        tooltip="Office",
        icon=folium.Icon(color='red', icon='building', prefix='fa')
    ).add_to(m)
    
    # Colors for clusters
    colors = ['blue', 'green', 'orange', 'purple', 'darkred',
              'darkblue', 'darkgreen', 'cadetblue', 'pink', 'black',
              'lightgreen', 'lightblue', 'beige',
              'salmon', 'cyan', 'magenta', 'lime', 'teal']
    
    clients = locations[1:]
    
    # Get unique labels (excluding -1 for outliers)
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    has_outliers = -1 in unique_labels
    n_outliers = np.sum(labels == -1)
    
    # Draw cluster boundaries (convex hull) and client markers
    for i, cluster_id in enumerate(cluster_labels):
        cluster_mask = labels == cluster_id
        cluster_client_indices = np.where(cluster_mask)[0]
        color = colors[i % len(colors)]
        
        # Get coordinates of clients in this cluster
        cluster_coords = []
        for idx in cluster_client_indices:
            client = clients[idx]
            cluster_coords.append([client.lat, client.lon])
        
        # Draw cluster boundary if we have enough points
        if len(cluster_coords) >= 3:
            cluster_coords_arr = np.array(cluster_coords)
            
            try:
                hull = ConvexHull(cluster_coords_arr)
                hull_points = cluster_coords_arr[hull.vertices].tolist()
                hull_points.append(hull_points[0])  # Close polygon
                
                folium.Polygon(
                    locations=hull_points,
                    color=color,
                    weight=3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.15,
                    popup=f"Cluster {cluster_id + 1} ({len(cluster_client_indices)} clients)"
                ).add_to(m)
            except:
                pass
        
        # Plot client markers
        for idx in cluster_client_indices:
            client = clients[idx]
            
            # Calculate distance from office
            office_dist = road_matrix[0, idx + 1]
            
            popup_html = f"""
                <b>{client.name}</b><br>
                <b>Cluster:</b> {cluster_id + 1}<br>
                <b>Distance from office:</b> {office_dist:.2f} km
            """
            
            folium.Marker(
                location=[client.lat, client.lon],
                popup=popup_html,
                tooltip=f"Cluster {cluster_id + 1}: {client.name}",
                icon=folium.Icon(color=color, icon='home', prefix='fa')
            ).add_to(m)
    
    # Plot outliers (label == -1) as gray markers
    if has_outliers:
        outlier_indices = np.where(labels == -1)[0]
        for idx in outlier_indices:
            client = clients[idx]
            office_dist = road_matrix[0, idx + 1]
            
            popup_html = f"""
                <b>{client.name}</b><br>
                <b>Status:</b> OUTLIER<br>
                <b>Distance from office:</b> {office_dist:.2f} km
            """
            
            folium.Marker(
                location=[client.lat, client.lon],
                popup=popup_html,
                tooltip=f"OUTLIER: {client.name}",
                icon=folium.Icon(color='gray', icon='question', prefix='fa')
            ).add_to(m)
    
    # Add legend
    n_clusters = len(cluster_labels)
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 250px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; font-size: 16px;">Road Distance Clusters</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><b>Total clusters:</b> {n_clusters}</p>
    <p style="margin: 5px 0;"><b>Outliers:</b> {n_outliers}</p>
    <p style="margin: 5px 0;"><b>Total clients:</b> {len(clients)}</p>
    <hr style="margin: 5px 0;">
    """
    
    for i, cluster_id in enumerate(cluster_labels):
        color = colors[i % len(colors)]
        n_clients = np.sum(labels == cluster_id)
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="background-color:{color}; padding: 2px 8px; color: white; border-radius: 3px;">
                Cluster {cluster_id + 1}
            </span> {n_clients} clients
        </p>
        '''
    
    if has_outliers:
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="background-color:gray; padding: 2px 8px; color: white; border-radius: 3px;">
                Outliers
            </span> {n_outliers} clients
        </p>
        '''
    
    legend_html += """
    <hr style="margin: 5px 0;">
    <p style="margin: 3px 0; font-size: 11px; font-style: italic;">
        Clusters based on road distance (not straight-line)
    </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    plugins.Fullscreen().add_to(m)
    m.save(f"Output\\{filename}")
    print(f"Saved cluster map to Output\\{filename}")


def plot_road_distance_clusters_static(locations: List[Location],
                                        labels: np.ndarray,
                                        road_matrix: np.ndarray,
                                        filename: str = 'road_distance_clusters_static.png'):
    """
    Static matplotlib visualization of road distance clusters with boundaries.
    
    Args:
        locations: List of Location objects
        labels: Cluster labels for each client
        road_matrix: Road distance matrix
        filename: Output filename
    """
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    office = locations[0]
    clients = locations[1:]
    
    # Handle outliers (-1 labels)
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(cluster_labels)
    has_outliers = -1 in unique_labels
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 2)))
    
    # Plot office
    ax.scatter(office.lon, office.lat, c='red', s=300, marker='s',
               label='Office', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate('Office', (office.lon, office.lat), fontsize=10,
                xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Plot outliers first (if any)
    if has_outliers:
        outlier_indices = np.where(labels == -1)[0]
        outlier_lons = [clients[idx].lon for idx in outlier_indices]
        outlier_lats = [clients[idx].lat for idx in outlier_indices]
        
        ax.scatter(outlier_lons, outlier_lats, c='gray', s=100, marker='x',
                  label=f'Outliers ({len(outlier_indices)})', zorder=3, linewidths=2)
    
    # Plot each cluster with boundary
    for i, cluster_id in enumerate(cluster_labels):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        color = colors[i]
        
        # Get coordinates
        cluster_lons = [clients[idx].lon for idx in cluster_indices]
        cluster_lats = [clients[idx].lat for idx in cluster_indices]
        
        # Plot clients
        ax.scatter(cluster_lons, cluster_lats, c=[color], s=100, marker='o',
                  label=f'Cluster {cluster_id + 1} ({len(cluster_indices)} clients)',
                  zorder=3, edgecolors='black', linewidths=0.5)
        
        # Draw convex hull boundary
        if len(cluster_indices) >= 3:
            points = np.column_stack([cluster_lons, cluster_lats])
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                polygon = Polygon(hull_points, alpha=0.2, facecolor=color, 
                                 edgecolor=color, linewidth=2)
                ax.add_patch(polygon)
            except:
                pass
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Clustering based on Road Distance\n'
                 '(Boundaries show cluster regions)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Output\\{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved static plot to Output\\{filename}")


def plot_hdbscan_clusters(locations: List[Location],
                          labels: np.ndarray,
                          probabilities: np.ndarray,
                          road_matrix: np.ndarray,
                          road_calculator: RoadDistanceCalculator,
                          filename: str = 'hdbscan_clusters.html'):
    """
    Visualize HDBSCAN clusters on OpenStreetMap.
    Shows clusters with boundaries and outliers marked differently.
    
    Args:
        locations: List of Location objects (office first, then clients)
        labels: Cluster labels (-1 = outlier)
        probabilities: Cluster membership probabilities
        road_matrix: Road distance matrix
        road_calculator: RoadDistanceCalculator
        filename: Output HTML filename
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available.")
        return
    
    from scipy.spatial import ConvexHull
    
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
        popup=f"<b>{office.name}</b><br>Office",
        tooltip="Office",
        icon=folium.Icon(color='red', icon='building', prefix='fa')
    ).add_to(m)
    
    # Colors for clusters
    colors = ['blue', 'green', 'orange', 'purple', 'darkred',
              'darkblue', 'darkgreen', 'cadetblue', 'pink', 'black',
              'lightgreen', 'lightblue', 'beige', 'salmon', 'cyan']
    
    clients = locations[1:]
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(cluster_labels)
    n_outliers = np.sum(labels == -1)
    
    # Draw cluster boundaries and markers
    for i, cluster_id in enumerate(cluster_labels):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        color = colors[i % len(colors)]
        
        # Get coordinates
        cluster_coords = []
        for idx in cluster_indices:
            client = clients[idx]
            cluster_coords.append([client.lat, client.lon])
        
        # Draw boundary
        if len(cluster_coords) >= 3:
            cluster_coords_arr = np.array(cluster_coords)
            try:
                hull = ConvexHull(cluster_coords_arr)
                hull_points = cluster_coords_arr[hull.vertices].tolist()
                hull_points.append(hull_points[0])
                
                folium.Polygon(
                    locations=hull_points,
                    color=color,
                    weight=3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.15,
                    popup=f"Cluster {cluster_id + 1} ({len(cluster_indices)} clients)"
                ).add_to(m)
            except:
                pass
        
        # Plot client markers
        for idx in cluster_indices:
            client = clients[idx]
            office_dist = road_matrix[0, idx + 1]
            prob = probabilities[idx]
            
            popup_html = f"""
                <b>{client.name}</b><br>
                <b>Cluster:</b> {cluster_id + 1}<br>
                <b>Membership probability:</b> {prob:.2f}<br>
                <b>Distance from office:</b> {office_dist:.2f} km
            """
            
            folium.Marker(
                location=[client.lat, client.lon],
                popup=popup_html,
                tooltip=f"Cluster {cluster_id + 1}: {client.name}",
                icon=folium.Icon(color=color, icon='home', prefix='fa')
            ).add_to(m)
    
    # Plot outliers
    outlier_indices = np.where(labels == -1)[0]
    for idx in outlier_indices:
        client = clients[idx]
        office_dist = road_matrix[0, idx + 1]
        
        popup_html = f"""
            <b>{client.name}</b><br>
            <b>Status:</b> OUTLIER<br>
            <b>Distance from office:</b> {office_dist:.2f} km<br>
            <i>This client doesn't fit well into any cluster</i>
        """
        
        folium.Marker(
            location=[client.lat, client.lon],
            popup=popup_html,
            tooltip=f"OUTLIER: {client.name}",
            icon=folium.Icon(color='gray', icon='question', prefix='fa')
        ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 280px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; font-size: 16px;">HDBSCAN Clustering</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><b>Clusters found:</b> {n_clusters}</p>
    <p style="margin: 5px 0;"><b>Outliers:</b> {n_outliers}</p>
    <p style="margin: 5px 0;"><b>Total clients:</b> {len(clients)}</p>
    <hr style="margin: 5px 0;">
    """
    
    for i, cluster_id in enumerate(cluster_labels):
        color = colors[i % len(colors)]
        n_clients = np.sum(labels == cluster_id)
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="background-color:{color}; padding: 2px 8px; color: white; border-radius: 3px;">
                Cluster {cluster_id + 1}
            </span> {n_clients} clients
        </p>
        '''
    
    if n_outliers > 0:
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="background-color:gray; padding: 2px 8px; color: white; border-radius: 3px;">
                Outliers
            </span> {n_outliers} clients
        </p>
        '''
    
    legend_html += """
    <hr style="margin: 5px 0;">
    <p style="margin: 3px 0; font-size: 11px; font-style: italic;">
        HDBSCAN finds natural clusters automatically
    </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    plugins.Fullscreen().add_to(m)
    m.save(f"Output\\{filename}")
    print(f"Saved HDBSCAN cluster map to Output\\{filename}")


def plot_kmedoids_clusters(locations: List[Location], labels: np.ndarray,
                           medoid_indices: np.ndarray, road_matrix: np.ndarray,
                           filename: str = 'kmedoids_clusters.png'):
    """
    Visualize K-Medoids++ clustering using MDS (Multidimensional Scaling).
    Converts road distance matrix into 2D coordinates where visual distances
    approximate actual road distances.
    
    Args:
        locations: List of Location objects (office first, then clients)
        labels: Cluster labels for each client
        medoid_indices: Indices of medoids (in client space, 0-indexed)
        road_matrix: Road distance matrix
        filename: Output filename
    """
    from sklearn.manifold import MDS
    
    # Use MDS to convert road distance matrix to 2D coordinates
    # This makes visual distance ≈ road distance
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords_2d = mds.fit_transform(road_matrix)
    
    # Separate office and clients
    office_coord = coords_2d[0]
    client_coords = coords_2d[1:]
    
    clients = locations[1:]
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 2)))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot office
    ax.scatter(office_coord[0], office_coord[1], c='red', s=300, marker='s', 
               label='Office', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate('Office', (office_coord[0], office_coord[1]), fontsize=9, 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_client_indices = np.where(cluster_mask)[0]
        medoid_client_idx = medoid_indices[cluster_id]
        
        color = colors[cluster_id]
        
        # Get medoid coordinate
        medoid_coord = client_coords[medoid_client_idx]
        
        # Plot non-medoid clients
        for idx in cluster_client_indices:
            client_coord = client_coords[idx]
            
            if idx != medoid_client_idx:
                ax.scatter(client_coord[0], client_coord[1], c=[color], s=100, 
                          marker='o', zorder=3, edgecolors='black', linewidths=0.5)
                
                # Draw line from client to medoid
                ax.plot([client_coord[0], medoid_coord[0]], 
                       [client_coord[1], medoid_coord[1]],
                       color=color, linewidth=1, alpha=0.4, zorder=1)
                
                # Show road distance on line
                mid_x = (client_coord[0] + medoid_coord[0]) / 2
                mid_y = (client_coord[1] + medoid_coord[1]) / 2
                road_dist = road_matrix[idx + 1, medoid_client_idx + 1]
        
        # Plot medoid (larger, star marker)
        ax.scatter(medoid_coord[0], medoid_coord[1], c=[color], s=400, marker='*',
                  label=f'Cluster {cluster_id + 1} ({sum(cluster_mask)} clients)', 
                  zorder=4, edgecolors='black', linewidths=1.5)
        
        # Annotate medoid
        medoid_name = clients[medoid_client_idx].name
        ax.annotate(f'C{cluster_id + 1}: {medoid_name}', 
                   (medoid_coord[0], medoid_coord[1]), fontsize=8,
                   xytext=(8, 8), textcoords='offset points',
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))
    
    ax.set_xlabel('MDS Dimension 1 (based on road distance)', fontsize=11)
    ax.set_ylabel('MDS Dimension 2 (based on road distance)', fontsize=11)
    ax.set_title('K-Medoids++ Clustering (MDS projection of Road Distance Matrix)\n'
                 'Visual distance ≈ Road distance | Stars = Medoids', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'Output\\{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved K-Medoids MDS plot to Output\\{filename}")


def print_kmedoids_distance_table(locations: List[Location], labels: np.ndarray,
                                   medoid_indices: np.ndarray, road_matrix: np.ndarray,
                                   filename: str = 'kmedoids_distance_table.csv'):
    """
    Print and export a distance table showing:
    - Each client
    - Its assigned cluster/medoid
    - Distance to assigned medoid
    - Distances to all other medoids (to show why it wasn't assigned there)
    
    Args:
        locations: List of Location objects (office first, then clients)
        labels: Cluster labels for each client
        medoid_indices: Indices of medoids (in client space, 0-indexed)
        road_matrix: Road distance matrix
        filename: Output CSV filename
    """
    import pandas as pd
    
    clients = locations[1:]
    n_clusters = len(medoid_indices)
    
    # Build table data
    rows = []
    for client_idx, client in enumerate(clients):
        assigned_cluster = labels[client_idx]
        assigned_medoid_idx = medoid_indices[assigned_cluster]
        assigned_medoid = clients[assigned_medoid_idx]
        
        # Distance to assigned medoid
        dist_to_assigned = road_matrix[client_idx + 1, assigned_medoid_idx + 1]
        
        # Distances to all medoids
        distances_to_medoids = {}
        for cluster_id, medoid_idx in enumerate(medoid_indices):
            medoid_name = clients[medoid_idx].name
            dist = road_matrix[client_idx + 1, medoid_idx + 1]
            distances_to_medoids[f'Dist to C{cluster_id + 1} ({medoid_name})'] = round(dist, 2)
        
        # Check if correctly assigned (is assigned medoid the closest?)
        all_dists = [road_matrix[client_idx + 1, m_idx + 1] for m_idx in medoid_indices]
        closest_medoid_cluster = np.argmin(all_dists)
        is_correct = '✓' if closest_medoid_cluster == assigned_cluster else '✗'
        
        row = {
            'Client': client.name,
            'Assigned Cluster': f'C{assigned_cluster + 1}',
            'Medoid': assigned_medoid.name,
            'Dist to Medoid (km)': round(dist_to_assigned, 2),
            'Correct?': is_correct,
            **distances_to_medoids
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by cluster, then by distance to medoid
    df = df.sort_values(['Assigned Cluster', 'Dist to Medoid (km)'])
    
    # Print summary
    print("\n" + "="*80)
    print("K-MEDOIDS++ DISTANCE TABLE")
    print("="*80)
    
    # Print medoids summary
    print("\nMEDOIDS:")
    for cluster_id, medoid_idx in enumerate(medoid_indices):
        medoid = clients[medoid_idx]
        cluster_size = sum(labels == cluster_id)
        print(f"  Cluster {cluster_id + 1}: {medoid.name} ({cluster_size} clients)")
    
    # Print table by cluster
    print("\n" + "-"*80)
    print("CLIENT ASSIGNMENTS (sorted by cluster and distance to medoid):")
    print("-"*80)
    
    for cluster_id in range(n_clusters):
        cluster_df = df[df['Assigned Cluster'] == f'C{cluster_id + 1}']
        medoid_name = clients[medoid_indices[cluster_id]].name
        
        print(f"\n>>> CLUSTER {cluster_id + 1} (Medoid: {medoid_name}) <<<")
        print(f"{'Client':<30} {'Dist to Medoid':<15} {'Next Closest':<25}")
        print("-" * 70)
        
        for _, row in cluster_df.iterrows():
            client_name = row['Client']
            dist_to_medoid = row['Dist to Medoid (km)']
            
            # Find next closest medoid
            medoid_cols = [c for c in df.columns if c.startswith('Dist to C')]
            dists = [(c, row[c]) for c in medoid_cols]
            dists_sorted = sorted(dists, key=lambda x: x[1])
            
            # Skip assigned medoid, get next closest
            next_closest = None
            for col, dist in dists_sorted:
                if dist != dist_to_medoid or col != f'Dist to C{cluster_id + 1}':
                    next_closest = f"{col.split('(')[0].strip()}: {dist} km"
                    break
            
            is_medoid = " (MEDOID)" if client_name == medoid_name else ""
            print(f"{client_name:<30} {dist_to_medoid:<15} {next_closest or 'N/A':<25}{is_medoid}")
    
    # Export to CSV
    df.to_csv(f'Output\\{filename}', index=False)
    print(f"\n\nExported full table to Output\\{filename}")
    
    return df


def plot_vrp_routes(locations: List[Location], 
                    routes: List[List[int]], 
                    route_distances: List[float],
                    road_calculator,
                    filename: str = 'vrp_routes.html'):
    """
    Visualize VRP solution with ordered routes on OpenStreetMap.
    Shows actual road paths with arrows indicating direction.
    
    Args:
        locations: List of Location objects (office first, then clients)
        routes: List of routes, each route is list of location indices
        route_distances: Distance for each route in km
        road_calculator: RoadDistanceCalculator for getting route geometry
        filename: Output HTML filename
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
        popup=f"<b>{office.name}</b><br>Depot (Start/End)",
        tooltip="Office (Depot)",
        icon=folium.Icon(color='red', icon='building', prefix='fa')
    ).add_to(m)
    
    # Colors for routes
    colors = ['blue', 'green', 'orange', 'purple', 'darkred', 
              'darkblue', 'darkgreen', 'cadetblue', 'pink', 'lightblue',
              'lightgreen', 'gray', 'black', 'lightgray', 'beige']
    
    # Plot each route
    for route_idx, (route, distance) in enumerate(zip(routes, route_distances)):
        color = colors[route_idx % len(colors)]
        
        # Build full route: Office -> clients -> Office
        full_route = [0] + route + [0]
        
        # Plot route segments with actual road geometry
        for i in range(len(full_route) - 1):
            from_loc = locations[full_route[i]]
            to_loc = locations[full_route[i + 1]]
            
            # Get actual road geometry
            route_geometry = road_calculator.get_route_geometry(from_loc, to_loc)
            
            if route_geometry and len(route_geometry) > 0:
                # Draw route with actual road path
                line = folium.PolyLine(
                    locations=route_geometry,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=f"Route {route_idx + 1}: {from_loc.name} → {to_loc.name}"
                )
                line.add_to(m)
                
                # Add arrow at midpoint to show direction
                mid_idx = len(route_geometry) // 2
                if mid_idx > 0:
                    plugins.AntPath(
                        locations=route_geometry,
                        color=color,
                        weight=4,
                        opacity=0.6,
                        delay=1000
                    ).add_to(m)
            else:
                # Fallback to straight line
                folium.PolyLine(
                    locations=[[from_loc.lat, from_loc.lon], [to_loc.lat, to_loc.lon]],
                    color=color,
                    weight=3,
                    opacity=0.6,
                    dash_array='5, 5'
                ).add_to(m)
        
        # Plot client markers for this route
        for stop_order, node in enumerate(route):
            client = locations[node]
            
            popup_html = f"""
                <b>{client.name}</b><br>
                <b>Route:</b> {route_idx + 1}<br>
                <b>Stop:</b> {stop_order + 1} of {len(route)}<br>
            """
            
            # Use numbered markers to show order
            folium.Marker(
                location=[client.lat, client.lon],
                popup=popup_html,
                tooltip=f"Route {route_idx + 1}, Stop {stop_order + 1}: {client.name}",
                icon=folium.DivIcon(
                    html=f'''
                        <div style="
                            background-color: {color};
                            color: white;
                            border-radius: 50%;
                            width: 24px;
                            height: 24px;
                            text-align: center;
                            line-height: 24px;
                            font-weight: bold;
                            font-size: 12px;
                            border: 2px solid white;
                            box-shadow: 1px 1px 3px rgba(0,0,0,0.4);
                        ">{stop_order + 1}</div>
                    ''',
                    icon_size=(24, 24),
                    icon_anchor=(12, 12)
                )
            ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 300px;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; font-size: 16px;">VRP Solution</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0;"><b>Total routes:</b> {len(routes)}</p>
    <p style="margin: 5px 0;"><b>Total distance:</b> {sum(route_distances):.1f} km</p>
    <hr style="margin: 5px 0;">
    """
    
    for route_idx, distance in enumerate(route_distances):
        color = colors[route_idx % len(colors)]
        n_clients = len(routes[route_idx])
        legend_html += f'''
        <p style="margin: 3px 0;">
            <span style="color:{color}; font-weight:bold;">━━━</span>
            Route {route_idx + 1}: {distance:.1f} km ({n_clients} clients)
        </p>
        '''
    
    legend_html += """
    <hr style="margin: 5px 0;">
    <p style="margin: 3px 0; font-size: 11px;">
        Numbers show visit order within each route
    </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen plugin
    plugins.Fullscreen().add_to(m)
    
    m.save(f"Output\\{filename}")
    print(f"Saved VRP routes map to Output\\{filename}")


def plot_vrp_clusters_static(locations: List[Location],
                             routes: List[List[int]],
                             route_distances: List[float],
                             filename: str = 'vrp_clusters_static.png'):
    """
    Static matplotlib visualization of VRP routes (lat/lon based).
    
    Args:
        locations: List of Location objects
        routes: List of routes
        route_distances: Distance for each route
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    office = locations[0]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 2)))
    
    # Plot office
    ax.scatter(office.lon, office.lat, c='red', s=300, marker='s',
               label='Office (Depot)', zorder=5, edgecolors='black', linewidths=2)
    ax.annotate('Office', (office.lon, office.lat), fontsize=10,
                xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    # Plot each route
    for route_idx, (route, distance) in enumerate(zip(routes, route_distances)):
        color = colors[route_idx]
        
        # Build full route coordinates
        route_lons = [office.lon]
        route_lats = [office.lat]
        
        for node in route:
            loc = locations[node]
            route_lons.append(loc.lon)
            route_lats.append(loc.lat)
        
        route_lons.append(office.lon)
        route_lats.append(office.lat)
        
        # Plot route line
        ax.plot(route_lons, route_lats, color=color, linewidth=2, alpha=0.7,
                label=f'Route {route_idx + 1}: {distance:.1f} km')
        
        # Plot arrows to show direction
        for i in range(len(route_lons) - 1):
            dx = route_lons[i + 1] - route_lons[i]
            dy = route_lats[i + 1] - route_lats[i]
            ax.annotate('', xy=(route_lons[i + 1], route_lats[i + 1]),
                       xytext=(route_lons[i], route_lats[i]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                       zorder=2)
        
        # Plot client markers with stop numbers
        for stop_order, node in enumerate(route):
            loc = locations[node]
            ax.scatter(loc.lon, loc.lat, c=[color], s=150, marker='o',
                      zorder=3, edgecolors='black', linewidths=1)
            ax.annotate(str(stop_order + 1), (loc.lon, loc.lat),
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       color='white')
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('VRP Solution - Delivery Routes\n(Numbers show visit order)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Output\\{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved VRP static plot to Output\\{filename}")


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
    # colors = ['#0C2C55', '#F63049','#FA8112','#5B23FF','#84934A',
    #           '#C40C0C','#F075AE','#5DD3B6','#9929EA','#E5BA41',
    #           '#3F9AAE','#8F0177','#EF7722','#78C841','#6B3F69',
    #            '#D92C54','#932F67','#5A9CB5','#6AECE1','#005461',
    #             '#CF0F0F','#434E78','#C47BE4','#DD88CF','#006A67',
    #              '#229799','#7886C7','#F72C5B','#EF9C66','#FFA1F5',
    #               '#FFC55A','#00DFA2','#14B1AB' ]
    colors = ['black', 'orange', 'lightred', 'red', 'beige', 'gray', 'lightgreen',
            'darkpurple', 'green', 'lightblue', 'lightgray', 'pink', 'darkred', 
            'darkblue', 'white', 'purple', 'cadetblue', 'blue', 'darkgreen']


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