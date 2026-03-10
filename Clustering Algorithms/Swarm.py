"""
Swarm-Based Clustering

Each client decides which cluster to join based on where their neighbors are.
No central control - emergent behavior from local decisions.

Rules:
- Cohesion: Join cluster where most neighbors are
- Separation: Prefer clusters that aren't too full (soft limit)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from location import Location


class SwarmClustering:
    """
    Swarm-based clustering using local neighbor decisions.
    """
    
    def __init__(self, road_matrix: np.ndarray, locations: List[Location],
                 n_clusters: int = 14,
                 neighbor_percent: float = 0.1,
                 max_clients_per_cluster: int = None,
                 max_iterations: int = 100,
                 random_seed: int = 42):
        """
        Args:
            road_matrix: Full distance matrix (office + clients) in km
            locations: List of Location objects (office first, then clients)
            n_clusters: Number of clusters
            neighbor_percent: Percentage of clients to consider as neighbors (0.05 = 5%)
            max_clients_per_cluster: Soft max limit (None = no limit)
            max_iterations: Maximum iterations
            random_seed: Random seed for reproducibility
        """
        self.road_matrix = road_matrix
        self.locations = locations
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
        np.random.seed(random_seed)
        
        # Extract client-only data (exclude office)
        self.client_matrix = road_matrix[1:, 1:]
        self.n_clients = len(self.client_matrix)
        
        # Calculate K neighbors (minimum 3)
        self.k_neighbors = max(3, int(self.n_clients * neighbor_percent))
        print(f"Using K={self.k_neighbors} neighbors for decisions.")
        
        # Soft max limit
        avg_size = self.n_clients / n_clusters
        self.max_clients =avg_size  #max_clients_per_cluster if max_clients_per_cluster else int(avg_size * 1.5)
        print(f"Average clients per cluster: {avg_size:.2f}")
        print(f"Soft max clients per cluster: {self.max_clients}")
        
        # Client coordinates for plotting
        self.client_coords = np.array([
            [loc.lat, loc.lon] for loc in locations[1:]
        ])
        
        # Results
        self.labels = None
        self.history = []  # Track labels at each iteration
    
    def _initialize_random(self) -> np.ndarray:
        """Randomly assign each client to a cluster."""
        return np.random.randint(0, self.n_clusters, size=self.n_clients)
    
    def _get_k_nearest_neighbors(self, client_idx: int) -> np.ndarray:
        """Get indices of K nearest neighbors for a client."""
        distances = self.client_matrix[client_idx].copy()
        distances[client_idx] = np.inf  # Exclude self
        
        # Get indices of K smallest distances
        neighbor_indices = np.argsort(distances)[:self.k_neighbors]
        return neighbor_indices
    
    def _get_cluster_sizes(self, labels: np.ndarray) -> np.ndarray:
        """Get size of each cluster."""
        sizes = np.zeros(self.n_clusters, dtype=int)
        for cluster_id in range(self.n_clusters):
            sizes[cluster_id] = np.sum(labels == cluster_id)
        return sizes
    
    def _decide_cluster(self, client_idx: int, labels: np.ndarray) -> int:
        """
        Decide which cluster a client should join based on neighbors.
        
        Returns:
            Best cluster ID
        """
        # Get neighbors
        neighbors = self._get_k_nearest_neighbors(client_idx)
        
        # Count neighbors in each cluster
        neighbor_counts = np.zeros(self.n_clusters)
        for neighbor_idx in neighbors:
            cluster_id = labels[neighbor_idx]
            neighbor_counts[cluster_id] += 1
        
        # Get cluster sizes
        cluster_sizes = self._get_cluster_sizes(labels)
        
        # Calculate attraction to each cluster
        # More neighbors = more attraction
        # Fuller cluster = less attraction (soft penalty)
        attraction = np.zeros(self.n_clusters)
        
        for cluster_id in range(self.n_clusters):
            # Base attraction = number of neighbors
            attraction[cluster_id] = neighbor_counts[cluster_id]
            
            # Soft penalty for full clusters
            if cluster_sizes[cluster_id] >= self.max_clients:
                attraction[cluster_id] *= 0.5  # Reduce attraction by half
        
        # Choose cluster with highest attraction
        best_cluster = np.argmax(attraction)
        
        # If no attraction anywhere (no neighbors), stay in current cluster
        if attraction[best_cluster] == 0:
            return labels[client_idx]
        
        return best_cluster
    
    def _run_iteration(self, labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run one iteration of swarm decisions.
        
        Returns:
            Tuple of (new_labels, num_switches)
        """
        new_labels = labels.copy()
        switches = 0
        
        # Each client makes a decision
        for client_idx in range(self.n_clients):
            current_cluster = labels[client_idx]
            best_cluster = self._decide_cluster(client_idx, labels)
            
            if best_cluster != current_cluster:
                new_labels[client_idx] = best_cluster
                switches += 1
        
        return new_labels, switches
    
    def fit(self, visualize: bool = True, save_every: int = 1) -> np.ndarray:
        """
        Run swarm clustering.
        """
        print(f"\n{'='*60}")
        print(f"SWARM CLUSTERING")
        # ... print statements ...
        
        # Initialize
        self.labels = self._initialize_random()
        self.history = [self.labels.copy()]
        
        # === ADD LIVE PLOT SETUP HERE ===
        if visualize:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))
        
        # Run iterations
        for iteration in range(self.max_iterations):
            new_labels, switches = self._run_iteration(self.labels)
            self.labels = new_labels
            self.history.append(self.labels.copy())
            
            # === UPDATE PLOT EACH ITERATION ===
            if visualize:
                ax.clear()
                for cluster_id in range(self.n_clusters):
                    mask = self.labels == cluster_id
                    if mask.sum() > 0:
                        ax.scatter(
                            self.client_coords[mask, 1],
                            self.client_coords[mask, 0],
                            c=[colors[cluster_id]],
                            s=30,
                            alpha=0.7
                        )
                sizes = self._get_cluster_sizes(self.labels)
                ax.set_title(f"Iteration {iteration + 1} | Switches: {switches} | Min: {sizes.min()}, Max: {sizes.max()}")
                plt.pause(0.3)  # Pause to see the update
            
            if switches == 0:
                print(f"\nConverged at iteration {iteration + 1}!")
                break
    
        # # === TURN OFF INTERACTIVE MODE ===
        # if visualize:
        #     plt.ioff()
        #     plt.show()
        
        return self.labels
    
    def visualize_iterations(self, save_every: int = 1):
        """Visualize cluster evolution over iterations."""
        
        # Select iterations to show
        iterations_to_show = list(range(0, len(self.history), save_every))
        if len(self.history) - 1 not in iterations_to_show:
            iterations_to_show.append(len(self.history) - 1)
        
        # Limit to max 12 plots
        if len(iterations_to_show) > 12:
            step = len(iterations_to_show) // 12
            iterations_to_show = iterations_to_show[::step]
            if len(self.history) - 1 not in iterations_to_show:
                iterations_to_show.append(len(self.history) - 1)
        
        n_plots = len(iterations_to_show)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Colors for clusters
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))
        
        for plot_idx, iter_idx in enumerate(iterations_to_show):
            ax = axes[plot_idx]
            labels = self.history[iter_idx]
            
            # Plot each cluster
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                if mask.sum() > 0:
                    ax.scatter(
                        self.client_coords[mask, 1],  # lon
                        self.client_coords[mask, 0],  # lat
                        c=[colors[cluster_id]],
                        s=30,
                        alpha=0.7
                    )
            
            sizes = self._get_cluster_sizes(labels)
            ax.set_title(f"Iter {iter_idx}\nmin={sizes.min()}, max={sizes.max()}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig("Output\\swarm_iterations.png", dpi=150)
        plt.show()
        print(f"Saved: Output\\swarm_iterations.png")
    
    def visualize_final(self):
        """Visualize final clustering result."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            if mask.sum() > 0:
                ax.scatter(
                    self.client_coords[mask, 1],  # lon
                    self.client_coords[mask, 0],  # lat
                    c=[colors[cluster_id]],
                    s=50,
                    alpha=0.7,
                    label=f"Cluster {cluster_id + 1} ({mask.sum()})"
                )
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Swarm Clustering - Final Result")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("Output\\swarm_final.png", dpi=150)
        plt.show()
        print(f"Saved: Output\\swarm_final.png")
    
    def get_labels(self) -> np.ndarray:
        """Get cluster labels for each client."""
        return self.labels
    
    def get_history(self) -> List[np.ndarray]:
        """Get labels at each iteration."""
        return self.history