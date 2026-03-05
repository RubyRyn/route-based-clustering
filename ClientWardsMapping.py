"""
Ward Mapping (Step 1)

Maps each client to their ward/village tract using point-in-polygon
against GeoJSON administrative boundaries.

Usage:
    from ward_mapping import WardMapper

    mapper = WardMapper(geojson_path="path/to/wards.geojson")
    mapping = mapper.map_clients(locations)
    mapper.summary()
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from shapely.geometry import shape, Point
from location import Location


class WardMapper:
    """
    Maps client locations to ward/village tract polygons.
    """

    def __init__(self, geojson_path: str,
                 pcode_field: str = "VT_PCODE",
                 name_field: str = "VT",
                 township_field: str = "TS",
                 township_pcode_field: str = "TS_PCODE"):
        """
        Args:
            geojson_path: Path to the GeoJSON file with ward boundaries.
            pcode_field: Property name for the ward/village tract pcode.
            name_field: Property name for the ward/village tract name.
            township_field: Property name for the township name.
            township_pcode_field: Property name for the township pcode.
        """
        self.pcode_field = pcode_field
        self.name_field = name_field
        self.township_field = township_field
        self.township_pcode_field = township_pcode_field

        # Load and parse GeoJSON
        print(f"Loading ward boundaries from: {geojson_path}")
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson = json.load(f)

        # Build ward lookup: list of (shapely_polygon, properties)
        self.wards = []
        for feature in geojson['features']:
            try:
                geom = shape(feature['geometry'])
                props = feature.get('properties', {})
                self.wards.append({
                    'geometry': geom,
                    'pcode': props.get(pcode_field, 'UNKNOWN'),
                    'name': props.get(name_field, 'UNKNOWN'),
                    'township': props.get(township_field, 'UNKNOWN'),
                    'township_pcode': props.get(township_pcode_field, 'UNKNOWN'),
                    'properties': props,
                })
            except Exception as e:
                print(f"  Warning: Skipped a feature due to: {e}")

        print(f"  Loaded {len(self.wards)} ward/village tract boundaries")

        # Results (populated after map_clients)
        self.client_ward_map = {}      # client_idx -> ward info dict
        self.ward_client_counts = {}   # pcode -> count of clients
        self.unmapped_clients = []     # client indices that fell outside all wards

    def map_clients(self, locations: List[Location],
                    skip_office: bool = True) -> Dict[int, dict]:
        """
        Map each client to their ward.

        Args:
            locations: List of Location objects (office first, then clients).
            skip_office: If True, skip locations[0] (the office).

        Returns:
            Dict mapping client_idx (0-based, excluding office) to ward info:
            {
                'pcode': 'MMR017008054',
                'name': 'Ah Dar Sin Gaung',
                'township': 'Hinthada',
                'township_pcode': 'MMR017008',
            }
            Clients outside all wards map to None values.
        """
        start_idx = 1 if skip_office else 0
        clients = locations[start_idx:]
        n_clients = len(clients)

        print(f"\nMapping {n_clients} clients to wards...")

        self.client_ward_map = {}
        self.ward_client_counts = {}
        self.unmapped_clients = []

        for client_idx, loc in enumerate(clients):
            point = Point(loc.lon, loc.lat)  # Shapely uses (x, y) = (lon, lat)
            matched_ward = None

            for ward in self.wards:
                if ward['geometry'].contains(point):
                    matched_ward = ward
                    break

            if matched_ward is not None:
                ward_info = {
                    'pcode': matched_ward['pcode'],
                    'name': matched_ward['name'],
                    'township': matched_ward['township'],
                    'township_pcode': matched_ward['township_pcode'],
                }
                self.client_ward_map[client_idx] = ward_info

                # Count clients per ward
                pcode = matched_ward['pcode']
                self.ward_client_counts[pcode] = \
                    self.ward_client_counts.get(pcode, 0) + 1
            else:
                self.client_ward_map[client_idx] = None
                self.unmapped_clients.append(client_idx)

            # Progress for large datasets
            if (client_idx + 1) % 100 == 0:
                print(f"  Processed {client_idx + 1}/{n_clients} clients...")

        print(f"  Done. Mapped {n_clients - len(self.unmapped_clients)}/{n_clients} clients.")

        return self.client_ward_map

    def fix_unmapped_clients(self, locations: List[Location],
                              skip_office: bool = True) -> int:
        """
        Try to map unmapped clients to the nearest ward boundary.
        Uses distance to ward polygon boundary for clients that fell
        outside all wards (likely GPS drift or boundary edge cases).

        Returns:
            Number of clients that were fixed.
        """
        if not self.unmapped_clients:
            print("No unmapped clients to fix.")
            return 0

        start_idx = 1 if skip_office else 0
        clients = locations[start_idx:]
        fixed = 0

        print(f"\nFixing {len(self.unmapped_clients)} unmapped clients "
              f"(assigning to nearest ward)...")

        for client_idx in self.unmapped_clients[:]:  # Copy list since we modify it
            loc = clients[client_idx]
            point = Point(loc.lon, loc.lat)

            nearest_ward = None
            nearest_dist = float('inf')

            for ward in self.wards:
                dist = ward['geometry'].distance(point)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_ward = ward

            if nearest_ward is not None:
                ward_info = {
                    'pcode': nearest_ward['pcode'],
                    'name': nearest_ward['name'],
                    'township': nearest_ward['township'],
                    'township_pcode': nearest_ward['township_pcode'],
                    'was_unmapped': True,
                    'distance_to_ward': nearest_dist,
                }
                self.client_ward_map[client_idx] = ward_info

                pcode = nearest_ward['pcode']
                self.ward_client_counts[pcode] = \
                    self.ward_client_counts.get(pcode, 0) + 1

                self.unmapped_clients.remove(client_idx)
                fixed += 1

        print(f"  Fixed {fixed} clients. "
              f"Remaining unmapped: {len(self.unmapped_clients)}")

        return fixed

    def summary(self):
        """Print a summary of the mapping results."""
        total = len(self.client_ward_map)
        mapped = total - len(self.unmapped_clients)
        n_wards_with_clients = len(self.ward_client_counts)

        print(f"\n{'=' * 60}")
        print(f"WARD MAPPING SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total clients: {total}")
        print(f"  Mapped to wards: {mapped}")
        print(f"  Unmapped (outside boundaries): {len(self.unmapped_clients)}")
        print(f"  Unique wards with clients: {n_wards_with_clients}")

        if self.ward_client_counts:
            counts = list(self.ward_client_counts.values())
            print(f"\n  Clients per ward:")
            print(f"    Min: {min(counts)}")
            print(f"    Max: {max(counts)}")
            print(f"    Avg: {np.mean(counts):.1f}")
            print(f"    Median: {np.median(counts):.1f}")

            # Show wards with most clients
            sorted_wards = sorted(self.ward_client_counts.items(),
                                   key=lambda x: -x[1])
            print(f"\n  Top 10 wards by client count:")
            for pcode, count in sorted_wards[:10]:
                # Find ward name
                name = pcode
                for ward in self.wards:
                    if ward['pcode'] == pcode:
                        name = f"{ward['name']} ({ward['township']})"
                        break
                print(f"    {name}: {count} clients")

        # Township-level summary
        township_counts = {}
        for client_idx, ward_info in self.client_ward_map.items():
            if ward_info is not None:
                ts = ward_info['township']
                township_counts[ts] = township_counts.get(ts, 0) + 1

        if township_counts:
            print(f"\n  Clients per township:")
            for ts, count in sorted(township_counts.items(), key=lambda x: -x[1]):
                print(f"    {ts}: {count}")

        print(f"{'=' * 60}")

    def get_ward_client_map(self) -> Dict[str, List[int]]:
        """
        Get reverse mapping: ward pcode -> list of client indices.
        Useful for Step 2 (grouping wards).
        """
        ward_clients = {}
        for client_idx, ward_info in self.client_ward_map.items():
            if ward_info is not None:
                pcode = ward_info['pcode']
                if pcode not in ward_clients:
                    ward_clients[pcode] = []
                ward_clients[pcode].append(client_idx)
        return ward_clients

    def get_client_ward_labels(self) -> Dict[int, str]:
        """
        Get simple mapping: client_idx -> ward pcode.
        Returns None for unmapped clients.
        """
        return {
            idx: (info['pcode'] if info is not None else None)
            for idx, info in self.client_ward_map.items()
        }