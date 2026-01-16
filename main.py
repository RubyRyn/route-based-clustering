
import sys
from distance_matrix_generator import DistanceMatrixGenerator
from route_similarity_clustering import RouteSimilarityClustering
from direction_n_distance_clustering import DirectionDistanceClustering
from sequential_routing_clustering import SequentialRoutingClustering
from route_graph_clustering import RouteGraphClustering
from visualizations import plot_static_map, plot_openstreetmap, plot_clustered_routes


def main():
    
    api_choice = 'osrm_demo'
    csv_filename = 'Input\\Sample data for route-awareness.csv'
    n_clusters = 5
    
    
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
    
    # Step 2: Export data
    generator.export_to_json('delivery_data.json')
    generator.export_to_csv('distance_matrix.csv')
    
    # Step 3: Visualize base map
    plot_static_map(locations, road_matrix, euclidean_matrix)
    plot_openstreetmap(locations, generator.road_calculator, 
                      road_matrix, euclidean_matrix, 'routes_map.html')
    
    
    # Choose clustering method
    # print("\nSelect clustering method:")
    # print("  1. Route Similarity (best for shared routes)")
    # print("  2. Direction & Distance (fast, simple)")
    # print("  3. Sequential Routing (practical)")
    # print("  4. Graph Community Detection (automatic)")
    
    method_choice = 4
    
    if method_choice == 1:
        clusterer = RouteSimilarityClustering(generator.road_calculator, locations)
        clusterer.road_matrix = road_matrix
        labels = clusterer.cluster(n_clusters=n_clusters, method='spectral') #spectral hierarchical
        
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
        labels = clusterer.cluster(similarity_threshold=0.5)
    
    
    clients = locations[1:]
    for cluster_id in range(max(labels) + 1):
        cluster_clients = [clients[i] for i, label in enumerate(labels) if label == cluster_id]
        print(f"\nRoute {cluster_id + 1}: {len(cluster_clients)} clients")
        for client in cluster_clients:
            client_idx = locations.index(client)
            dist = road_matrix[0, client_idx]
            print(f"  - {client.name:20} ({dist:.2f} km from office)")

    print("\nVisualizing clusters...")

    # Visualize (no optimization needed!)
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
