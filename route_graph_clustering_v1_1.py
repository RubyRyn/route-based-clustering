"""
Cluster using graph community detection based on route overlap
WITH VISUALIZATION
"""

import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from location import Location
from road_distance_calculator import RoadDistanceCalculator


class RouteGraphClustering:
    """Use graph-based clustering on route similarities with visualization"""
    
    def __init__(self, distance_calculator: RoadDistanceCalculator, locations: List[Location]):
        self.distance_calculator = distance_calculator
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:]
        self.road_matrix: Optional[np.ndarray] = None
        self.graph = None  # Store graph for visualization
        self.similarity_matrix = None  # Store similarities
    
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
    
    def build_similarity_matrix(self) -> np.ndarray:
        """Build and store full similarity matrix"""
        n_clients = len(self.clients)
        similarity_matrix = np.zeros((n_clients, n_clients))
        
        print("\nCalculating route similarities...")
        total = n_clients * (n_clients - 1) // 2
        completed = 0
        
        for i in range(n_clients):
            similarity_matrix[i, i] = 1.0
            for j in range(i + 1, n_clients):
                sim = self.calculate_route_similarity(self.clients[i], self.clients[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total}")
        
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def cluster(self, similarity_threshold: float = 0.5, visualize: bool = True) -> np.ndarray:
        """Cluster using graph community detection
        similarity_threshold: float = 0.5 : Consider two routes similar 
        if at least 50% of their points overlap consecutively from the start.”
        """
        n_clients = len(self.clients)
        G = nx.Graph()
        
        # Add nodes with positions (use actual lat/lon)
        for i in range(n_clients):
            G.add_node(i, 
                      name=self.clients[i].name,
                      pos=(self.clients[i].lon, self.clients[i].lat))
        
        # Build similarity matrix if not already done
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        print(f"\nBuilding similarity graph (threshold: {similarity_threshold})...")
        edge_count = 0
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                sim = self.similarity_matrix[i, j]
                if sim > similarity_threshold:
                    G.add_edge(i, j, weight=sim)
                    edge_count += 1
        
        print(f"  Created graph with {n_clients} nodes and {edge_count} edges")
        self.graph = G
        
        # Detect communities
        print("\nDetecting communities...")
        communities = community.louvain_communities(G, weight='weight', seed=42)
        
        # Convert to labels array
        labels = np.zeros(n_clients, dtype=int)
        for cluster_id, comm in enumerate(communities):
            for client_idx in comm:
                labels[client_idx] = cluster_id
        
        print(f"✓ Found {len(communities)} natural clusters")
        
        # Print cluster summary
        for cluster_id in range(len(communities)):
            members = [self.clients[i].name for i in range(n_clients) if labels[i] == cluster_id]
            print(f"  Cluster {cluster_id + 1}: {len(members)} clients - {', '.join(members[:3])}" + 
                  (f" + {len(members)-3} more" if len(members) > 3 else ""))
        
        # Visualize if requested
        if visualize:
            self.visualize_graph(labels, similarity_threshold)
            self.visualize_similarity_matrix(labels)
            self.visualize_clusters_geographic(labels)
        
        return labels
    
    def visualize_graph(self, labels: np.ndarray, threshold: float, 
                       figsize: Tuple[int, int] = (14, 10)):
        """Visualize the graph with nodes colored by cluster"""
        if self.graph is None:
            print("No graph to visualize. Run cluster() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get positions (use geographic coordinates)
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Color map for clusters
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        node_colors = [colors[labels[node]] for node in self.graph.nodes()]
        
        # SUBPLOT 1: Geographic layout (actual positions)
        ax1.set_title(f'Graph Clustering (Geographic Layout)\nThreshold: {threshold}', 
                     fontsize=14, fontweight='bold')
        
        # Draw edges with transparency based on weight
        edges = self.graph.edges()
        weights = [self.graph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(self.graph, pos, ax=ax1,
                              width=[w*3 for w in weights],
                              alpha=0.3,
                              edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, ax=ax1,
                              node_color=node_colors,
                              node_size=500,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        labels_dict = {i: self.clients[i].name.split()[-1] for i in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels_dict, ax=ax1,
                               font_size=8, font_weight='bold')
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, alpha=0.3)
        
        # SUBPLOT 2: Spring layout (shows connectivity better)
        ax2.set_title('Graph Clustering (Force-Directed Layout)\nShows Connectivity', 
                     fontsize=14, fontweight='bold')
        
        # Use spring layout to show community structure
        pos_spring = nx.spring_layout(self.graph, weight='weight', seed=42, k=0.5, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos_spring, ax=ax2,
                              width=[w*3 for w in weights],
                              alpha=0.3,
                              edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos_spring, ax=ax2,
                              node_color=node_colors,
                              node_size=500,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos_spring, labels_dict, ax=ax2,
                               font_size=8, font_weight='bold')
        
        ax2.axis('off')
        
        # Add legend
        legend_elements = [mpatches.Patch(facecolor=colors[i], 
                                         edgecolor='black',
                                         label=f'Cluster {i+1}')
                          for i in range(n_clusters)]
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=min(n_clusters, 5), bbox_to_anchor=(0.5, 0.98))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('Output\\graph_clustering_visualization.png', dpi=150, bbox_inches='tight')
        print("Graph visualization saved to: graph_clustering_visualization.png")
        plt.show()
    
    def visualize_similarity_matrix(self, labels: np.ndarray, figsize: Tuple[int, int] = (12, 10)):
        """Visualize the similarity matrix sorted by clusters"""
        if self.similarity_matrix is None:
            print("No similarity matrix. Run cluster() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        n_clients = len(self.clients)
        
        # SUBPLOT 1: Original similarity matrix
        im1 = ax1.imshow(self.similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title('Similarity Matrix (Original Order)', fontweight='bold')
        ax1.set_xlabel('Client Index')
        ax1.set_ylabel('Client Index')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Route Similarity', rotation=270, labelpad=20)
        
        # SUBPLOT 2: Sorted by cluster
        # Sort indices by cluster
        sorted_indices = np.argsort(labels)
        sorted_matrix = self.similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        im2 = ax2.imshow(sorted_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title('Similarity Matrix (Sorted by Cluster)', fontweight='bold')
        ax2.set_xlabel('Client Index (sorted)')
        ax2.set_ylabel('Client Index (sorted)')
        
        # Draw cluster boundaries
        cluster_boundaries = []
        for i in range(1, len(sorted_labels)):
            if sorted_labels[i] != sorted_labels[i-1]:
                cluster_boundaries.append(i)
        
        for boundary in cluster_boundaries:
            ax2.axhline(y=boundary - 0.5, color='blue', linewidth=2)
            ax2.axvline(x=boundary - 0.5, color='blue', linewidth=2)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Route Similarity', rotation=270, labelpad=20)
        
        # Add cluster labels on the side
        cluster_positions = []
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(sorted_labels == cluster_id)[0]
            cluster_positions.append((cluster_id, cluster_indices[0], cluster_indices[-1]))
        
        for cluster_id, start, end in cluster_positions:
            mid = (start + end) / 2
            ax2.text(n_clients + 0.5, mid, f'C{cluster_id+1}', 
                    ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('Output\\similarity_matrix_visualization.png', dpi=150, bbox_inches='tight')
        print("Similarity matrix visualization saved to: similarity_matrix_visualization.png")
        plt.show()
    
    def visualize_clusters_geographic(self, labels: np.ndarray, figsize: Tuple[int, int] = (12, 8)):
        """Visualize clusters on geographic map"""
        fig, ax = plt.subplots(figsize=figsize)
        
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # Plot office
        ax.scatter(self.office.lon, self.office.lat, 
                  c='red', s=300, marker='s', 
                  edgecolors='black', linewidths=2,
                  label='Office', zorder=3)
        
        # Plot clients by cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_clients = [self.clients[i] for i in cluster_indices]
            
            lons = [c.lon for c in cluster_clients]
            lats = [c.lat for c in cluster_clients]
            
            ax.scatter(lons, lats, 
                      c=[colors[cluster_id]], s=200,
                      edgecolors='black', linewidths=1.5,
                      label=f'Cluster {cluster_id + 1} ({len(cluster_clients)} clients)',
                      zorder=2)
            
            # Add labels
            for client in cluster_clients:
                ax.annotate(client.name.split()[-1], 
                           (client.lon, client.lat),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Geographic Distribution of Clusters', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Output\\geographic_clusters_visualization.png', dpi=150, bbox_inches='tight')
        print("Geographic clusters visualization saved to: geographic_clusters_visualization.png")
        plt.show()
    
    def print_clustering_stats(self, labels: np.ndarray):
        """Print detailed clustering statistics"""
        n_clusters = len(np.unique(labels))
        
        print("\n" + "="*70)
        print("GRAPH CLUSTERING STATISTICS")
        print("="*70)
        
        # Graph statistics
        if self.graph is not None:
            print(f"\nGraph Structure:")
            print(f"  Nodes (clients): {self.graph.number_of_nodes()}")
            print(f"  Edges (similarities above threshold): {self.graph.number_of_edges()}")
            
            avg_degree = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            print(f"  Average connections per client: {avg_degree:.1f}")
            
            # Connected components
            components = list(nx.connected_components(self.graph))
            print(f"  Connected components: {len(components)}")
            if len(components) > 1:
                print(f"    (Some clusters are isolated - no route similarity with others)")
        
        # Cluster statistics
        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Internal similarity (how similar are clients within cluster)
            if cluster_size > 1 and self.similarity_matrix is not None:
                internal_sims = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx_i = cluster_indices[i]
                        idx_j = cluster_indices[j]
                        internal_sims.append(self.similarity_matrix[idx_i, idx_j])
                
                avg_internal_sim = np.mean(internal_sims) if internal_sims else 0
                
                print(f"\n  Cluster {cluster_id + 1}:")
                print(f"    Size: {cluster_size} clients")
                print(f"    Avg internal similarity: {avg_internal_sim:.3f}")
                print(f"    Members: {', '.join([self.clients[i].name for i in cluster_indices])}")