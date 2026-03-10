import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import folium
from folium import plugins
FOLIUM_AVAILABLE = True
from location import Location
from road_distance_calculator import RoadDistanceCalculator


def plot_kmedoids_clusters_mds(locations: List[Location], labels: np.ndarray,
                           medoid_indices: np.ndarray, road_matrix: np.ndarray,
                           filename: str = 'kmedoids_clusters_mds.png'):
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
