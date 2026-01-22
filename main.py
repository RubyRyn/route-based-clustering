
import sys
from K_medoids_plus_plus import KMedoidsPlusPlus, optimize_workload_balance
from distance_matrix_generator import DistanceMatrixGenerator
from route_similarity_clustering import RouteSimilarityClustering
from direction_n_distance_clustering import DirectionDistanceClustering # wrost 
from sequential_routing_clustering import SequentialRoutingClustering
from route_graph_clustering import RouteGraphClustering
from visualizations import plot_static_map, plot_openstreetmap, plot_clustered_routes
# from route_graph_clustering_v1_1 import RouteGraphClustering

def main():
    
    # Configuration
    api_choice = 'osrm_demo'
    due_date = '2025-09-02'
    n_clusters = 14
    # sample_data = 'Sample data for route-awareness.csv'
    # csv_filename = f'Input\\{sample_data}'
    csv_filename = f'Input\\Einme_clients_{due_date}.csv'
    
    
    
    print("\nLoading locations and generating distance matrix...")
    generator = DistanceMatrixGenerator(
        random_seed=42,
        api_type=api_choice,
        use_fallback=True
    )
    
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
    except KeyError as e:
        print(f"\nError: Column {e} not found in CSV!")
        return 1
    
    delay = 0.5 if api_choice == 'osrm_demo' else 0.0
    euclidean_matrix, road_matrix = generator.generate_distance_matrices(
        delay_between_calls=delay
    )
    
    # generator.print_summary()
    
    generator.export_to_json('delivery_data.json')
    generator.export_to_csv('distance_matrix.csv')
    
    
    plot_static_map(locations, road_matrix, euclidean_matrix)
    plot_openstreetmap(locations, generator.road_calculator, 
                      road_matrix, euclidean_matrix, 'routes_map.html')
    
    method_choice = 5
    
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
        balanced_labels = optimize_workload_balance(clusterer, max_swap_iterations=50)
        clusterer.labels = balanced_labels
        # clusterer.print_results()
        assignments = clusterer.get_cluster_assignments()


    print("\nVisualizing clusters...")
    plot_clustered_routes(
        locations=locations,
        labels=labels,
        road_calculator=generator.road_calculator,
        road_matrix=road_matrix,
        euclidean_matrix=euclidean_matrix,
        filename='clustered_routes.html'
    )
    
    
    print("\nGenerated files:")
    print("  • delivery_data.json")
    print("  • euclidean_distance_matrix.csv")
    print("  • road_distance_matrix.csv")
    print("  • delivery_map_static.png")
    print("  • routes_map.html")
    print("  • clustered_routes.html")
    print("COMPLETE!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
