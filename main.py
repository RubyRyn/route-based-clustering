
import sys
from K_medoids_plus_plus import KMedoidsPlusPlus, optimize_workload_balance
from distance_matrix_generator import DistanceMatrixGenerator
from hdbscan_clustering import HDBSCANClustering
from route_similarity_clustering import RouteSimilarityClustering
from direction_n_distance_clustering import DirectionDistanceClustering # wrost 
from sequential_routing_clustering import SequentialRoutingClustering
from route_graph_clustering import RouteGraphClustering
from visualizations import (plot_static_map, plot_openstreetmap, 
                            plot_clustered_routes, plot_kmedoids_clusters)
from plot_kmedoid_mds import plot_kmedoids_clusters_mds
from VRP_solver import VRPSolver
from visualizations import plot_vrp_routes, plot_vrp_clusters_static
from hierarchical_clustering import RoadDistance_HierarchicalClustering
from visualizations import plot_road_distance_clusters, plot_road_distance_clusters_static
from hdbscan_clustering import HDBSCANClustering
from visualizations import plot_hdbscan_clusters, plot_road_distance_clusters_static
from hdbscan_clustering_simple import HDBSCANClusteringSimple
from clustering_search_optimizer import ClusteringSearchOptimizer
from visualizations import plot_road_distance_clusters

def main():
    
    # Configuration
    method_choice = 10
    api_choice = 'osrm_demo'
    due_date = '2025-09-02'
    n_clusters = 14
    # sample_data = 'Sample data for route-awareness.csv'
    # csv_filename = f'Input\\{sample_data}'
    csv_filename = f'Input\\Einme_clients_{due_date}.csv'
    API_or_CSV = 'CSV'

    if API_or_CSV == 'CSV':
        #___________ Load locations and generate distance matrices ___________#
        generator = DistanceMatrixGenerator(random_seed=42)
        
        try:
            locations = generator.load_locations_from_csv(
                csv_file=csv_filename,
                lat_col='Latitude',
                lon_col='Longitude',
                name_col='Name',
                type_col='Type'
            )
        except FileNotFoundError:
            print(f"\nError: CSV file '{csv_filename}' not found!")
            return 1

        # Load matrices from previously exported files
        euclidean_matrix, road_matrix = generator.import_from_csv(
            euclidean_file='Output\\euclidean_distance_matrix.csv',
            road_file='Output\\road_distance_matrix.csv'
        )
    elif API_or_CSV == 'API':
        #____________ Generate from API calls __________________________________
        generator = DistanceMatrixGenerator(random_seed=42,api_type=api_choice,use_fallback=True)
        try:
            locations = generator.load_locations_from_csv(
                csv_file=csv_filename,
                lat_col='Latitude',
                lon_col='Longitude',
                name_col='Name',
                type_col='Type'
            )
        except FileNotFoundError:
            print(f"\nError: CSV file '{csv_filename}' not found!")
            return 1
        delay = 0.5 if api_choice == 'osrm_demo' else 0.0
        euclidean_matrix, road_matrix = generator.generate_distance_matrices(
            delay_between_calls=delay
        )
    

    #______________________________________________________________________
    
    generator.export_to_json('delivery_data.json')
    generator.export_to_csv('distance_matrix.csv')
    
    
    plot_static_map(locations, road_matrix, euclidean_matrix)
    plot_openstreetmap(locations, generator.road_calculator, 
                      road_matrix, euclidean_matrix, 'routes_map.html')
    
    
    if method_choice == 1:
        clusterer = RouteSimilarityClustering(generator.road_calculator, locations)
        clusterer.road_matrix = road_matrix
        labels = clusterer.cluster(n_clusters=n_clusters, method='hierarchical') #spectral hierarchical
        
    elif method_choice == 2:
        clusterer = DirectionDistanceClustering(generator.road_calculator, locations)
        clusterer.road_matrix = road_matrix
        labels = clusterer.cluster(n_clusters=n_clusters)
        
    elif method_choice == 3:
        clusterer = SequentialRoutingClustering(generator.road_calculator, locations)
        clusterer.road_matrix = road_matrix
        labels = clusterer.cluster(n_routes=n_clusters)

    elif method_choice == 4:
        clusterer = RouteGraphClustering(generator.road_calculator, locations)
        clusterer.road_matrix = road_matrix
        labels = clusterer.cluster(similarity_threshold=0.5, visualize=True)
        clusterer.print_clustering_stats(labels)

    elif method_choice == 5:
        clusterer = KMedoidsPlusPlus(road_matrix, locations, max_clusters=n_clusters)
        labels = clusterer.fit() # OR labels = clusterer.fit_minimax() #Strict minimax optimization
        # balanced_labels = optimize_workload_balance(clusterer, max_swap_iterations=50)
        # clusterer.labels = balanced_labels
        # clusterer.print_results()
        assignments = clusterer.get_cluster_assignments()
        plot_kmedoids_clusters_mds(
            locations=locations,
            labels=labels,
            medoid_indices=clusterer.medoid_indices,
            road_matrix=road_matrix
            )
        plot_kmedoids_clusters(
            locations=locations,
            labels=labels,
            medoid_indices=clusterer.medoid_indices,
            road_matrix=road_matrix
            )
        
    elif method_choice == 6:
        solver = VRPSolver(
            road_matrix=road_matrix,
            locations=locations,
            max_vehicles=14,
            time_limit_seconds=180,
            balance_coefficient=100  # Higher = more balanced
            )
        # Solve
        solver.solve()
        # Visualize
        plot_vrp_routes(
            locations=locations,
            routes=solver.routes,
            route_distances=solver.route_distances,
            road_calculator=generator.road_calculator
            )
        plot_vrp_clusters_static(
            locations=locations,
            routes=solver.routes,
            route_distances=solver.route_distances
            )
        # Get labels for compatibility with existing visualizations
        labels = solver.get_labels()

    elif method_choice == 7:
        #Hierarchical Clustering

        # Create clusterer
        clusterer = RoadDistance_HierarchicalClustering(
            road_matrix=road_matrix,
            locations=locations,
            max_clusters=14,
            linkage_method='average'  # or 'complete', 'ward', 'single'
        )

        # Run clustering
        labels = clusterer.cluster(n_clusters=14)

        # Visualize - Interactive map with boundaries
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='hierarchical_clusters.html'
        )

        # Visualize - Static plot
        plot_road_distance_clusters_static(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix
        )

    elif method_choice == 8:

        clusterer = HDBSCANClustering(
            road_matrix=road_matrix,
            locations=locations,
            max_clusters=n_clusters,              # Total (clusters + outliers) ≤ 14
            min_cluster_size=3,
            min_samples=1,
            max_clients_per_cluster=None  # Set a number to enable splitting (future use)
        )

        labels = clusterer.cluster()

        # Get total assignments
        total = clusterer.get_total_assignments()

        # Visualize - Interactive map
        plot_hdbscan_clusters(
            locations=locations,
            labels=labels,
            probabilities=clusterer.probabilities,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='hdbscan_clusters.html'
        )

        # Visualize - Static plot
        plot_road_distance_clusters_static(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix
        )

    elif method_choice == 9:

        clusterer = HDBSCANClusteringSimple(
            road_matrix=road_matrix,
            locations=locations,
            min_cluster_size=2,
            min_samples=1
        )

        labels = clusterer.cluster()

        # Get results
        # outliers = clusterer.get_outliers()
        # assignments = clusterer.get_cluster_assignments()  
        
 
        # Visualize - Interactive map with boundaries
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='hdbscan_simple_clusters_before_merge.html'
        )     
        labels = clusterer.merge_nearby_outliers(max_distance_km=7.2)
        # Visualize - Interactive map with boundaries
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='hdbscan_simple_clusters_after_merge.html'
        ) 

    elif method_choice == 10:

        # Step 1: Run HDBSCAN
        hdbscan_clusterer = HDBSCANClusteringSimple(
            road_matrix=road_matrix,
            locations=locations,
            min_cluster_size=3,
            min_samples=1
        )
        hdbscan_labels = hdbscan_clusterer.cluster()

        # Visualize BEFORE
        plot_road_distance_clusters(
            locations=locations,
            labels=hdbscan_labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='before_optimization.html'
        )
        print(f"\n{'-'*70}")
        print(f"Optimizing {max(hdbscan_labels)+1} clusters from HDBSCAN...")
        # Step 2: Run Search Optimization
        optimizer = ClusteringSearchOptimizer(
            road_matrix=road_matrix,
            locations=locations,
            max_clusters=14,
            min_clients_per_cluster=3,
            max_clients_per_cluster=10,
            no_improvement_limit=500
        )

        optimizer.initialize_from_hdbscan(hdbscan_labels)
        optimized_labels = optimizer.search(max_iterations=1000)

        # Visualize AFTER
        plot_road_distance_clusters(
            locations=locations,
            labels=optimized_labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename='after_optimization.html'
        )

        # Optional: Plot loss history
        import matplotlib.pyplot as plt
        plt.plot(optimizer.get_loss_history())
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Optimization Progress')
        plt.savefig('Output\\loss_history.png')
        plt.show()



            
    # print("\nVisualizing clusters...")

    # plot_clustered_routes(
    #     locations=locations,
    #     labels=labels,
    #     road_calculator=generator.road_calculator,
    #     road_matrix=road_matrix,
    #     euclidean_matrix=euclidean_matrix,
    #     filename='clustered_routes.html'
    # )
    
    
    # print("\nGenerated files:")
    # print("  • delivery_data.json")
    # print("  • euclidean_distance_matrix.csv")
    # print("  • road_distance_matrix.csv")
    # print("  • delivery_map_static.png")
    # print("  • routes_map.html")
    # print("  • clustered_routes.html")
    # print("COMPLETE!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
