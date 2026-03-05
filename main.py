
import sys
from K_medoids_plus_plus import KMedoidsPlusPlus
from SGB_v1 import SeedGrowBalance
from WardVisualization import WardVisualizer
from distance_matrix_generator import DistanceMatrixGenerator
from hdbscan_clustering import HDBSCANClustering
from route_similarity_clustering import RouteSimilarityClustering
from direction_n_distance_clustering import DirectionDistanceClustering # wrost 
from sequential_routing_clustering import SequentialRoutingClustering
from route_graph_clustering import RouteGraphClustering
from visualizations import (plot_static_map, plot_openstreetmap,plot_clustered_routes, plot_kmedoids_clusters)
from plot_kmedoid_mds import plot_kmedoids_clusters_mds
from VRP_solver import VRPSolver
from visualizations import plot_vrp_routes, plot_vrp_clusters_static
from hierarchical_clustering import RoadDistance_HierarchicalClustering
from visualizations import plot_road_distance_clusters, plot_road_distance_clusters_static
from hdbscan_clustering import HDBSCANClustering
from visualizations import plot_hdbscan_clusters, plot_road_distance_clusters_static
from hdbscan_clustering_simple import HDBSCANClusteringSimple
from clustering_search_optimizer_v0 import ClusteringSearchOptimizer
from visualizations import plot_road_distance_clusters
from kmedoids_constrained import KMedoidsPlusPlus
from TerritoryViz import visualize_territories, visualize_territories_comparison

def calculate_cluster_size_bounds(n_locations: int, n_clusters: int, 
                                   tolerance: float = 0.3) -> tuple:
    """
    Calculate min and max cluster size around average.
    
    Args:
        n_locations: Total number of clients
        n_clusters: Number of clusters
        tolerance: Percentage tolerance (0.3 = ±30%)
    
    Returns:
        (min_size, max_size, average)
    """
    average = n_locations / n_clusters
    
    min_size = int(average * (1 - tolerance))
    max_size = int(average * (1 + tolerance)) + 1  # +1 to round up
    
    # Ensure minimum is at least 1
    min_size = max(1, min_size)
    
    # Verify feasibility
    if n_clusters * min_size > n_locations:
        min_size = n_locations // n_clusters
    
    if n_clusters * max_size < n_locations:
        max_size = (n_locations // n_clusters) + 1
    
    return min_size, max_size, round(average, 1)

def main():
    
    # Configuration
    method_choice = 19
    api_choice = 'osrm_local'
    branch_name = "Kangyidaunt"
    n_clusters = 12 # number of employees
    n_locations = 474
    # max_clients_per_cluster = 30
    # min_clients_per_cluster = 3
    # tolerance=0.25 means ±25% from the average.
    min_clients_per_cluster, max_clients_per_cluster, avg = calculate_cluster_size_bounds(n_locations, n_clusters, tolerance=0.2)
    print(f"Average: {avg}")
    print(f"Min: {min_clients_per_cluster}")
    print(f"Max: {max_clients_per_cluster}")

    # sample_data = 'Sample data for route-awareness.csv'
    csv_filename = f'Input\\Sample data - {branch_name}.csv'
    # csv_filename = f'Input\\Einme_clients_{branch_name}.csv'
    API_or_CSV = 'API'

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
            euclidean_file=f'Output\\euclidean_distance_matrix (Einme).csv',
            road_file=f'Output\\road_distance_matrix (Einme).csv'
        )
    elif API_or_CSV == 'API':
        # For demo server
        if api_choice == 'osrm_demo':
            generator = DistanceMatrixGenerator(
                random_seed=42,
                api_type='osrm_demo',
                use_fallback=True
            )
            delay = 1
        
        # For your self-hosted server
        elif api_choice == 'osrm_local':
            print(f"Using local OSRM server at 91.99.169.156:5000")
            generator = DistanceMatrixGenerator(
                random_seed=42,
                api_type='osrm_local',
                server_ip='91.99.169.156',  
                port=5000,
                use_fallback=True
            )
            delay = 0.0
    
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
        
        # Generate distance matrices
        euclidean_matrix, road_matrix = generator.generate_distance_matrices_tableAPI()
        
    

    #______________________________________________________________________
    
    generator.export_to_json(f'data_{branch_name}.json')
    generator.export_to_csv(f'distance_matrix_{branch_name}.csv')
    print("\nDistance matrices generated and exported successfully!")

    
    # plot_static_map(locations, road_matrix, euclidean_matrix)
    # plot_openstreetmap(locations, generator.road_calculator, 
    #                   road_matrix, euclidean_matrix, 'routes_map.html')
    
    
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
        clusterer = KMedoidsPlusPlus(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            max_iterations=1000,
            random_seed=42
        )
        labels = clusterer.fit()

        plot_kmedoids_clusters(
            locations=locations,
            labels=labels,
            medoid_indices=clusterer.get_medoid_indices(),
            road_matrix=road_matrix, 
            filename=f'kmedoids_clusters_{branch_name}.html'
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
            min_cluster_size=max_clients_per_cluster,
            min_samples=3
        )

        labels = clusterer.cluster()

        # Run tuning (no need to pass client_road_matrix anymore)
        results_df = clusterer.tune_hdbscan()

        # Display all results
        print(results_df.to_string(index=False))
        print("\n--- Top 5 by DBCV ---")
        print(results_df.nlargest(5, 'dbcv').to_string(index=False))

        # Top 5 by Silhouette
        print("\n--- Top 5 by Silhouette ---")
        print(results_df.nlargest(5, 'silhouette').to_string(index=False))

        # Balanced: good DBCV + reasonable outlier count
        print("\n--- Balanced (DBCV > 0.4, outliers < 30%) ---")
        balanced = results_df[(results_df['dbcv'] > 0.4) & (results_df['outlier_pct'] < 30)]
        print(balanced.sort_values('dbcv', ascending=False).to_string(index=False))
 
        # Visualize - Interactive map with boundaries
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'hdbscan_simple_clusters_before_merge_{branch_name}.html'
        )     
        # labels = clusterer.merge_nearby_outliers(max_distance_km=7.2)
        # # Visualize - Interactive map with boundaries
        # plot_road_distance_clusters(
        #     locations=locations,
        #     labels=labels,
        #     road_matrix=road_matrix,
        #     road_calculator=generator.road_calculator,
        #     filename=f'hdbscan_simple_clusters_after_merge_{branch_name}.html'
        # ) 

    elif method_choice == 10:


        # from clustering_search_optimizer_v1 import ClusteringSearchOptimizer

        # Step 1: HDBSCAN
        hdbscan = HDBSCANClusteringSimple(road_matrix, locations, min_cluster_size=min_clients_per_cluster, min_samples=1)
        hdbscan_labels = hdbscan.cluster()

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
            max_clusters=n_clusters,
            min_clients_per_cluster=min_clients_per_cluster,
            max_clients_per_cluster=max_clients_per_cluster,  # Increased!
            no_improvement_limit=25000,
            #calibration_samples=200
        )

        optimizer.initialize_from_hdbscan(hdbscan_labels)
        optimized_labels = optimizer.search(max_iterations=50000)
        optimizer.visualize_top_10(locations, road_matrix, generator.road_calculator)

        # # Visualize AFTER
        # plot_road_distance_clusters(
        #     locations=locations,
        #     labels=optimized_labels,
        #     road_matrix=road_matrix,
        #     road_calculator=generator.road_calculator,
        #     filename='after_optimization.html'
        # )

        # Optional: Plot loss history
        import matplotlib.pyplot as plt
        plt.plot(optimizer.get_loss_history())
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Optimization Progress')
        plt.savefig('Output\\loss_history.png')
        plt.show()

    elif method_choice == 11:

        # Run clustering
        clusterer = KMedoidsPlusPlus(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            min_clients_per_cluster=min_clients_per_cluster,
            max_clients_per_cluster=max_clients_per_cluster,
            n_neighbors=3  # Consider top 3 nearest clusters as neighbors
        )

        labels = clusterer.fit()
        plot_kmedoids_clusters(
            locations=locations,
            labels=labels,
            medoid_indices=clusterer.get_medoid_indices(),
            road_matrix=road_matrix, 
            filename=f'kmedoids_constraint_clusters_{branch_name}.html'
        )

    elif method_choice == 12:
        from Swarm import SwarmClustering

        clusterer = SwarmClustering(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            neighbor_percent=0.05,  # 5% of clients as neighbors
            max_iterations=100
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'swarm_{branch_name}.html'
        )  

    elif method_choice == 13: 
        from SGB_v1 import SeedGrowBalance
        clusterer = SeedGrowBalance(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            balance_weight=0.1,       
            max_refine_iterations=100
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'sgb_{branch_name}_v1.html'
        )
        violations = visualize_territories(
        labels=labels,
        client_coords=clusterer.client_coords,
        office_coords=clusterer.office_coords,
        n_clusters=n_clusters,
        title="V1 Grow Result",
        filename="Output/territories.png"
        )

    elif method_choice == 14:
        from SGB_v2 import SeedGrowBalance

        clusterer = SeedGrowBalance(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            max_imbalance=0.25,
            max_refine_iterations=100
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'sgb_{branch_name}_v2.html'
        )

        violations = visualize_territories(
        labels=labels,
        client_coords=clusterer.client_coords,
        office_coords=clusterer.office_coords,
        n_clusters=n_clusters,
        title="V1 Grow Result",
        filename="Output/territories.png"
        )
    
    elif method_choice == 15:
        from SGB_v3 import SeedGrowBalance

        clusterer = SeedGrowBalance(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            balance_weight=0.4,
            size_tolerance=0.20,
            max_swap_passes=5000
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'sgb_{branch_name}_v3.html'
        )
        violations = visualize_territories(
        labels=labels,
        client_coords=clusterer.client_coords,
        office_coords=clusterer.office_coords,
        n_clusters=n_clusters,
        title="V1 Grow Result",
        filename="Output/territories.png"
        )

    elif method_choice == 16:
        from petal import PetalClustering

        clusterer = PetalClustering(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            size_tolerance=0.20,
            max_swap_passes=5000
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'petal_{branch_name}.html'
        )
            
    elif method_choice == 17:
        from petal_v2 import PetalClustering

        clusterer = PetalClustering(
            road_matrix=road_matrix,
            locations=locations,
            n_clusters=n_clusters,
            size_tolerance=0.20,
            max_swap_passes=50
        )

        labels = clusterer.fit(visualize=True)
        plot_road_distance_clusters(
            locations=locations,
            labels=labels,
            road_matrix=road_matrix,
            road_calculator=generator.road_calculator,
            filename=f'petal_{branch_name}.html'
        )
           
    elif method_choice == 18:
        # MIMU

        from ClientWardsMapping import WardMapper

        # Step 1: Load wards and map clients
        mapper = WardMapper(geojson_path="Input\\AyeyarwadyVillageTract.json")
        client_ward_map = mapper.map_clients(locations)
        # Fix any clients that fell outside ward boundaries
        mapper.fix_unmapped_clients(locations)

        # See the results
        mapper.summary()

        # These will be useful for Step 2:
        ward_clients = mapper.get_ward_client_map()    # ward_pcode -> [client indices]
        client_wards = mapper.get_client_ward_labels()  # client_idx -> ward_pcode

        from WardVisualization import WardVisualizer

        viz = WardVisualizer(mapper, locations)

        # # Three coloring modes:
        # viz.color_by_client_count(filename="Output\\ward_by_count.html")
        # viz.color_by_township(filename="Output\\ward_by_township.html")
        # viz.color_by_distance(filename="Output\\ward_by_distance.html")

        from WardAdjacency import WardAdjacency

        # Step 2: Build adjacency and territory tree
        adjacency = WardAdjacency(mapper, locations, road_matrix)
        adjacency.build_adjacency()
        adjacency.build_territory_tree()

                # ward_clients = mapper.get_ward_client_map()
                # adj = adjacency.get_adjacency()

                # # Check neighbor counts for all client wards
                # zero_neighbors = []
                # low_neighbors = []
                # for pcode in ward_clients.keys():
                #     neighbors = adj.get(pcode, {})
                #     if len(neighbors) == 0:
                #         zero_neighbors.append(pcode)
                #     elif len(neighbors) <= 2:
                #         low_neighbors.append((pcode, len(neighbors)))

                # print(f"Client wards with 0 neighbors: {len(zero_neighbors)}")
                # print(f"Client wards with 1-2 neighbors: {len(low_neighbors)}")
        print(f"Total wards in GeoJSON: {len(mapper.wards)}")
        print(f"Relevant wards (from territory tree): {len(adjacency.get_relevant_wards())}")
        # Visualize
        # adjacency.visualize_adjacency(filename="Output\\ward_adjacency.html")
        # adjacency.visualize_tree(filename="Output\\ward_tree.html")

                # from WardZones import WardZoneAssignment
                # zoner = WardZoneAssignment(
                #     mapper=mapper,
                #     adjacency_builder=adjacency,
                #     locations=locations,
                #     road_matrix=road_matrix,
                #     n_zones=n_clusters,
                #     size_tolerance=0.20
                # )

        # from WardZones_v2 import WardZoneAssignment
        # from WardZones_v3 import WardZoneAssignment
        # from WardZones_v4 import WardZoneAssignment
        # from WardZones_v5 import WardZoneAssignment
        from WardZones_v6 import WardZoneAssignment
        zoner = WardZoneAssignment(
            mapper=mapper,
            adjacency_builder=adjacency,
            locations=locations,
            road_matrix=road_matrix,
            n_zones=n_clusters,
            size_tolerance=0.20,
            empty_connecting_ward_allowance=0 # try 0, 1, 2, 3
        )

        labels = zoner.assign_zones()
        zoner.visualize_zones(filename="Output/ward_zones.html")

    elif method_choice == 19:
        # MIMU + WardZones v7 (improved balancing)

        from ClientWardsMapping import WardMapper

        # Step 1: Load wards and map clients
        mapper = WardMapper(geojson_path="Input\\Myanmar_village_tract.json")
        client_ward_map = mapper.map_clients(locations)
        mapper.fix_unmapped_clients(locations)
        mapper.summary()

        ward_clients = mapper.get_ward_client_map()
        client_wards = mapper.get_client_ward_labels()

        from WardAdjacency import WardAdjacency

        # Step 2: Build adjacency and territory tree
        adjacency = WardAdjacency(mapper, locations, road_matrix)
        adjacency_file = "Output/ward_adjacency_graph.json"
        if not adjacency.load_adjacency(adjacency_file):
            adjacency.build_adjacency()
            adjacency.save_adjacency(adjacency_file)
        adjacency.build_territory_tree()

        print(f"Total wards in GeoJSON: {len(mapper.wards)}")
        print(f"Relevant wards (from territory tree): {len(adjacency.get_relevant_wards())}")

        from WardZones_v7 import WardZoneAssignment
        zoner = WardZoneAssignment(
            mapper=mapper,
            adjacency_builder=adjacency,
            locations=locations,
            road_matrix=road_matrix,
            n_zones=n_clusters,
            size_tolerance=0.20,
            empty_connecting_ward_allowance=0
        )

        labels = zoner.assign_zones()
        zoner.visualize_zones(filename=f"Output/ward_zones_v7_allowance_0 - {branch_name}.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
