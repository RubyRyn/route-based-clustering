"""
Usage example in main.py
"""

from distance_matrix_generator import DistanceMatrixGenerator
from K_medoids_plus_plus import KMedoidsPlusPlus, optimize_workload_balance
import sys

from visualizations import plot_clustered_routes


def main():
    """Main execution function"""
    
    # Configuration
    api_choice = 'osrm_demo'
    due_date = '2025-09-02'
    n_clusters = 7
    sample_data = 'Sample data for route-awareness.csv'
    csv_filename = f'Input\\{sample_data}'
    # csv_filename = 'Input\\Einme_clients_{due_date}.csv'
    # csv_filename = csv_filename.format(due_date=due_date)
    
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
    
    delay = 0.5 if api_choice == 'osrm_demo' else 0.0
    euclidean_matrix, road_matrix = generator.generate_distance_matrices(
        delay_between_calls=delay
    )
    
    print("\n[STEP 2] Exporting data...")
    generator.export_to_json('delivery_data.json')
    generator.export_to_csv('distance_matrix.csv')
    
    
    clusterer = KMedoidsPlusPlus(
        road_matrix=road_matrix,
        locations=locations,
        n_clusters=n_clusters,
        max_iterations=100,
        random_seed=42
    )
    
    
    labels = clusterer.fit()
    clusterer.print_results()
    balanced_labels = optimize_workload_balance(clusterer, max_swap_iterations=50)
    
    clusterer.labels = balanced_labels
    clusterer.print_results()
    assignments = clusterer.get_cluster_assignments()

    plot_clustered_routes(
        locations=locations,
        labels=labels,
        road_calculator=generator.road_calculator,
        road_matrix=road_matrix,
        euclidean_matrix=euclidean_matrix,
        filename='clustered_routes.html'
    )
    
    
    return 0


if __name__ == "__main__":
    sys.exit(main())