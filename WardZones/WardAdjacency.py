"""
Ward Adjacency & Territory Tree (Step 2 Foundation)

Builds a ward adjacency graph from GeoJSON polygon borders:
- Detects which wards share borders
- Measures shared border length
- Combines with road distance for connection scoring
- Grows a territory tree outward from the office ward
- Visualizes the adjacency and tree on a Folium map

Usage:
    from ward_mapping import WardMapper
    from ward_adjacency import WardAdjacency

    mapper = WardMapper(geojson_path="path/to/wards.geojson")
    mapper.map_clients(locations)
    mapper.fix_unmapped_clients(locations)

    adjacency = WardAdjacency(mapper, locations, road_matrix)
    adjacency.build_adjacency()
    adjacency.build_territory_tree()
    adjacency.visualize_adjacency(filename="Output/ward_adjacency.html")
    adjacency.visualize_tree(filename="Output/ward_tree.html")
"""

import json
import os
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import folium
import colorsys
from shapely.geometry import shape, MultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import shared_paths, nearest_points
from location import Location
from ClientWardsMapping import WardMapper


class WardAdjacency:
    """
    Builds ward adjacency graph and territory tree from the office.
    """

    def __init__(self, mapper: WardMapper, locations: List[Location],
                 road_matrix: np.ndarray, skip_office: bool = True):
        """
        Args:
            mapper: WardMapper that has already run map_clients().
            locations: List of Location objects (office first, then clients).
            road_matrix: Full road distance matrix (office at index 0).
            skip_office: If True, locations[0] is the office.
        """
        self.mapper = mapper
        self.locations = locations
        self.road_matrix = road_matrix
        self.office = locations[0]
        self.clients = locations[1:] if skip_office else locations

        # Ward-client mappings
        self.ward_clients = mapper.get_ward_client_map()  # pcode -> [client indices]
        self.wards_with_clients = set(self.ward_clients.keys())

        # Find which ward the office is in
        self.office_ward = self._find_office_ward()
        print(f"Office is in ward: {self.office_ward}")

        # Build pcode -> ward lookup
        self.ward_lookup = {}
        for ward in mapper.wards:
            self.ward_lookup[ward['pcode']] = ward

        # Results
        self.adjacency = {}       # pcode -> {neighbor_pcode: {border_length, road_distance, score}}
        self.territory_tree = {}  # pcode -> parent_pcode (None for root)
        self.tree_order = []      # Order in which wards were visited
        self.relevant_wards = set()  # Wards included in the territory

    def _find_office_ward(self) -> Optional[str]:
        """Find which ward polygon contains the office."""
        from shapely.geometry import Point
        office_point = Point(self.office.lon, self.office.lat)

        for ward in self.mapper.wards:
            if ward['geometry'].contains(office_point):
                return ward['pcode']

        # If office falls outside all wards, find nearest
        print("  Warning: Office not inside any ward, finding nearest...")
        best_pcode = None
        best_dist = float('inf')
        for ward in self.mapper.wards:
            dist = ward['geometry'].distance(office_point)
            if dist < best_dist:
                best_dist = dist
                best_pcode = ward['pcode']

        return best_pcode

    # ══════════════════════════════════════════════
    # STEP 1: BUILD ADJACENCY GRAPH
    # ══════════════════════════════════════════════

    def build_adjacency(self):
        """
        Detect which wards share borders and measure border lengths.
        Two wards are adjacent if their polygon boundaries overlap/touch
        with a non-zero length (not just a single point).
        """
        print(f"\nBuilding ward adjacency graph...")

        wards = self.mapper.wards
        n = len(wards)
        self.adjacency = {ward['pcode']: {} for ward in wards}

        # Compare each pair of wards
        # This is O(n^2) but we only need to do it once
        processed = 0
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            ward_a = wards[i]
            geom_a = ward_a['geometry']
            pcode_a = ward_a['pcode']

            for j in range(i + 1, n):
                ward_b = wards[j]
                geom_b = ward_b['geometry']
                pcode_b = ward_b['pcode']

                # Quick bounding box check first (fast rejection)
                if not geom_a.bounds_intersect(geom_b) if hasattr(geom_a, 'bounds_intersect') else not self._bounds_overlap(geom_a, geom_b):
                    continue

                # Check if they share a border
                try:
                    intersection = geom_a.boundary.intersection(geom_b.boundary)

                    if intersection.is_empty:
                        continue

                    # Measure shared border length
                    border_length = intersection.length

                    # Skip point-only touches (length = 0)
                    if border_length < 1e-8:
                        continue

                    # Convert to approximate km
                    # At Myanmar's latitude (~17°N), 1 degree ≈ 111 km lat, ~106 km lon
                    border_length_km = border_length * 110  # rough approximation

                    # Compute road distance between wards
                    road_dist = self._ward_road_distance(pcode_a, pcode_b)

                    # Compute straight-line distance between ward centroids
                    centroid_a = geom_a.centroid
                    centroid_b = geom_b.centroid
                    straight_dist = self._haversine(
                        centroid_a.y, centroid_a.x,
                        centroid_b.y, centroid_b.x
                    )

                    # Detour ratio
                    detour_ratio = road_dist / straight_dist if straight_dist > 0 else float('inf')

                    # Connection score: higher = better connection
                    # Long border + short road distance = strong connection
                    # Short border + long road distance = weak connection
                    if road_dist > 0:
                        score = border_length_km / road_dist
                    else:
                        score = border_length_km

                    edge_data = {
                        'border_length_km': border_length_km,
                        'road_distance': road_dist,
                        'straight_distance': straight_dist,
                        'detour_ratio': detour_ratio,
                        'score': score,
                    }

                    self.adjacency[pcode_a][pcode_b] = edge_data
                    self.adjacency[pcode_b][pcode_a] = edge_data

                except Exception as e:
                    # Some geometries may cause topology errors
                    continue

            processed += n - i - 1
            if (i + 1) % 100 == 0:
                print(f"  Processed ward {i + 1}/{n}...")

        # Summary
        total_edges = sum(len(v) for v in self.adjacency.values()) // 2
        wards_with_neighbors = sum(1 for v in self.adjacency.values() if len(v) > 0)
        print(f"  Done. Found {total_edges} adjacency connections "
              f"across {wards_with_neighbors} wards.")

        # Stats on connection scores
        all_scores = [e['score'] for neighbors in self.adjacency.values()
                      for e in neighbors.values()]
        if all_scores:
            print(f"  Connection scores: min={min(all_scores):.4f}, "
                  f"max={max(all_scores):.4f}, "
                  f"median={np.median(all_scores):.4f}")

    def _bounds_overlap(self, geom_a, geom_b) -> bool:
        """Quick check if two geometries' bounding boxes overlap."""
        a = geom_a.bounds  # (minx, miny, maxx, maxy)
        b = geom_b.bounds
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def _ward_road_distance(self, pcode_a: str, pcode_b: str) -> float:
        """
        Compute road distance between two wards.
        Uses minimum road distance between any client in ward A
        and any client in ward B.
        If either ward has no clients, uses centroid-based estimation.
        """
        clients_a = self.ward_clients.get(pcode_a, [])
        clients_b = self.ward_clients.get(pcode_b, [])

        if clients_a and clients_b:
            # Min road distance between any pair of clients
            # road_matrix indices: 0=office, 1..N=clients
            # client indices in ward_clients are 0-based (client 0 = road_matrix index 1)
            min_dist = float('inf')
            for ca in clients_a:
                for cb in clients_b:
                    dist = self.road_matrix[ca + 1, cb + 1]
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        # If one or both wards have no clients, use centroid haversine as estimate
        ward_a = self.ward_lookup.get(pcode_a)
        ward_b = self.ward_lookup.get(pcode_b)
        if ward_a and ward_b:
            ca = ward_a['geometry'].centroid
            cb = ward_b['geometry'].centroid
            return self._haversine(ca.y, ca.x, cb.y, cb.x) * 1.4  # Rough road factor

        return float('inf')

    # ══════════════════════════════════════════════
    # STEP 2: BUILD TERRITORY TREE FROM OFFICE
    # ══════════════════════════════════════════════

    def build_territory_tree(self):
        """
        Grow territory outward from the office ward.
        Uses connection score (border length / road distance) to prioritize
        strongest connections first.

        Includes empty wards only if they're on the path to reach
        wards with clients.
        """
        print(f"\nBuilding territory tree from office ward: {self.office_ward}")

        if self.office_ward is None:
            print("  Error: Office ward not found.")
            return

        # Priority queue: (negative_score, pcode, parent_pcode)
        # Using negative because we want highest score first
        import heapq

        visited = set()
        self.territory_tree = {}
        self.tree_order = []
        self.relevant_wards = set()

        # Track which wards with clients have been reached
        reached_client_wards = set()
        total_client_wards = self.wards_with_clients.copy()

        # Start from office ward
        heap = [(-float('inf'), self.office_ward, None)]

        while heap and len(reached_client_wards) < len(total_client_wards):
            neg_score, current_pcode, parent_pcode = heapq.heappop(heap)

            if current_pcode in visited:
                continue

            visited.add(current_pcode)
            self.territory_tree[current_pcode] = parent_pcode
            self.tree_order.append(current_pcode)

            # Track if this ward has clients
            if current_pcode in self.wards_with_clients:
                reached_client_wards.add(current_pcode)

            # Add neighbors to heap
            for neighbor_pcode, edge_data in self.adjacency.get(current_pcode, {}).items():
                if neighbor_pcode not in visited:
                    heapq.heappush(heap,
                                   (-edge_data['score'], neighbor_pcode, current_pcode))

        # Now determine relevant wards: wards with clients + wards on paths to them
        self.relevant_wards = set()

        # Start with all wards that have clients
        for pcode in self.wards_with_clients:
            if pcode in self.territory_tree:
                # Trace back to root, adding all intermediate wards
                current = pcode
                while current is not None:
                    self.relevant_wards.add(current)
                    current = self.territory_tree.get(current)

        # Stats
        client_wards_reached = len(reached_client_wards)
        client_wards_missed = total_client_wards - reached_client_wards
        print(f"  Wards in territory: {len(self.relevant_wards)}")
        print(f"  Wards with clients reached: {client_wards_reached}/{len(total_client_wards)}")
        if client_wards_missed:
            print(f"  ⚠️ Unreached client wards: {client_wards_missed}")
        print(f"  Empty connecting wards: {len(self.relevant_wards) - client_wards_reached}")

        # Tree depth stats
        depths = {}
        for pcode in self.tree_order:
            parent = self.territory_tree[pcode]
            if parent is None:
                depths[pcode] = 0
            else:
                depths[pcode] = depths.get(parent, 0) + 1
        if depths:
            max_depth = max(depths.values())
            print(f"  Tree depth: {max_depth}")

        self.tree_depths = depths

    # ══════════════════════════════════════════════
    # VISUALIZATION: ADJACENCY MAP
    # ══════════════════════════════════════════════

    def visualize_adjacency(self, filename: str = "Output/ward_adjacency.html"):
        """
        Folium map showing ward adjacency connections.
        Lines between adjacent ward centroids, colored by connection strength.
        """
        print(f"\nGenerating adjacency map...")

        m = folium.Map(
            location=[self.office.lat, self.office.lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # ── Ward polygons ──
        ward_layer = folium.FeatureGroup(name="Wards")
        for ward in self.mapper.wards:
            pcode = ward['pcode']
            has_clients = pcode in self.wards_with_clients
            client_count = len(self.ward_clients.get(pcode, []))

            geojson = json.loads(json.dumps(
                ward['geometry'].__geo_interface__
            ))

            fill_color = '#a1dab4' if has_clients else '#f0f0f0'
            fill_opacity = 0.3 if has_clients else 0.05

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{ward['name']}</b><br>
                Township: {ward['township']}<br>
                Pcode: {pcode}<br>
                Clients: {client_count}<br>
                Neighbors: {len(self.adjacency.get(pcode, {}))}
            </div>
            """

            folium.GeoJson(
                geojson,
                style_function=lambda x, fc=fill_color, fo=fill_opacity: {
                    'fillColor': fc,
                    'fillOpacity': fo,
                    'color': '#666',
                    'weight': 0.8,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{ward['name']} ({client_count} clients)",
            ).add_to(ward_layer)
        ward_layer.add_to(m)

        # ── Adjacency lines ──
        adj_layer = folium.FeatureGroup(name="Adjacency Connections")
        drawn = set()

        # Get score range for color scaling
        all_scores = [e['score'] for neighbors in self.adjacency.values()
                      for e in neighbors.values()]
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
        else:
            min_score, max_score = 0, 1

        for pcode_a, neighbors in self.adjacency.items():
            ward_a = self.ward_lookup.get(pcode_a)
            if not ward_a:
                continue
            centroid_a = ward_a['geometry'].centroid

            for pcode_b, edge_data in neighbors.items():
                edge_key = tuple(sorted([pcode_a, pcode_b]))
                if edge_key in drawn:
                    continue
                drawn.add(edge_key)

                ward_b = self.ward_lookup.get(pcode_b)
                if not ward_b:
                    continue
                centroid_b = ward_b['geometry'].centroid

                # Color by score: green=strong, red=weak
                score = edge_data['score']
                norm = (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
                r = int(255 * (1 - norm))
                g = int(255 * norm)
                color = f'#{r:02x}{g:02x}44'

                # Line weight by border length
                weight = max(1, min(5, edge_data['border_length_km'] * 2))

                tooltip = (f"{ward_a['name']} ↔ {ward_b['name']}<br>"
                          f"Border: {edge_data['border_length_km']:.2f} km<br>"
                          f"Road dist: {edge_data['road_distance']:.1f} km<br>"
                          f"Detour ratio: {edge_data['detour_ratio']:.1f}<br>"
                          f"Score: {score:.4f}")

                folium.PolyLine(
                    locations=[
                        [centroid_a.y, centroid_a.x],
                        [centroid_b.y, centroid_b.x]
                    ],
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    tooltip=tooltip,
                ).add_to(adj_layer)

        adj_layer.add_to(m)

        # ── Office ──
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="Office / Farm",
            tooltip="Office / Farm",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(filename)
        print(f"  Saved: {filename}")
        print(f"  Total adjacency edges: {len(drawn)}")
        return m

    # ══════════════════════════════════════════════
    # VISUALIZATION: TERRITORY TREE
    # ══════════════════════════════════════════════

    def visualize_tree(self, filename: str = "Output/ward_tree.html"):
        """
        Folium map showing the territory tree growing from the office.
        Wards colored by tree depth (distance from office in the tree).
        Lines showing parent-child connections.
        """
        if not self.territory_tree:
            print("Run build_territory_tree() first.")
            return

        print(f"\nGenerating territory tree map...")

        m = folium.Map(
            location=[self.office.lat, self.office.lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # Color by tree depth
        max_depth = max(self.tree_depths.values()) if self.tree_depths else 1
        depth_colors = self._generate_depth_colors(max_depth + 1)

        # ── Ward polygons colored by depth ──
        relevant_layer = folium.FeatureGroup(name="Territory Wards")
        other_layer = folium.FeatureGroup(name="Other Wards", show=True)

        for ward in self.mapper.wards:
            pcode = ward['pcode']
            geojson = json.loads(json.dumps(
                ward['geometry'].__geo_interface__
            ))

            if pcode in self.relevant_wards:
                depth = self.tree_depths.get(pcode, 0)
                fill_color = depth_colors[depth]
                fill_opacity = 0.45
                weight = 1.5
                border_color = fill_color
                layer = relevant_layer
                has_clients = pcode in self.wards_with_clients
                client_count = len(self.ward_clients.get(pcode, []))
                parent = self.territory_tree.get(pcode)
                parent_name = self.ward_lookup[parent]['name'] if parent and parent in self.ward_lookup else 'ROOT'

                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>{ward['name']}</b><br>
                    Township: {ward['township']}<br>
                    Pcode: {pcode}<br>
                    Tree depth: {depth}<br>
                    Parent: {parent_name}<br>
                    Clients: {client_count}<br>
                    {'📦 Has clients' if has_clients else '🔗 Connecting ward'}
                </div>
                """
                tooltip = f"{ward['name']} (depth {depth}, {client_count} clients)"
            else:
                fill_color = '#f0f0f0'
                fill_opacity = 0.05
                weight = 0.3
                border_color = '#cccccc'
                layer = other_layer
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>{ward['name']}</b><br>
                    Township: {ward['township']}<br>
                    (Not in delivery territory)
                </div>
                """
                tooltip = f"{ward['name']} (outside territory)"

            folium.GeoJson(
                geojson,
                style_function=lambda x, fc=fill_color, fo=fill_opacity,
                    bc=border_color, w=weight: {
                    'fillColor': fc,
                    'fillOpacity': fo,
                    'color': bc,
                    'weight': w,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=tooltip,
            ).add_to(layer)

        relevant_layer.add_to(m)
        other_layer.add_to(m)

        # ── Tree edges (parent → child lines) ──
        tree_layer = folium.FeatureGroup(name="Tree Connections")

        for pcode, parent_pcode in self.territory_tree.items():
            if parent_pcode is None:
                continue  # Root node
            if pcode not in self.relevant_wards:
                continue

            ward_child = self.ward_lookup.get(pcode)
            ward_parent = self.ward_lookup.get(parent_pcode)
            if not ward_child or not ward_parent:
                continue

            centroid_child = ward_child['geometry'].centroid
            centroid_parent = ward_parent['geometry'].centroid

            depth = self.tree_depths.get(pcode, 0)
            color = depth_colors[min(depth, len(depth_colors) - 1)]

            # Edge data
            edge_data = self.adjacency.get(pcode, {}).get(parent_pcode, {})
            border_len = edge_data.get('border_length_km', 0)
            road_dist = edge_data.get('road_distance', 0)
            score = edge_data.get('score', 0)

            tooltip = (f"{ward_parent['name']} → {ward_child['name']}<br>"
                      f"Border: {border_len:.2f} km<br>"
                      f"Road dist: {road_dist:.1f} km<br>"
                      f"Score: {score:.4f}")

            folium.PolyLine(
                locations=[
                    [centroid_parent.y, centroid_parent.x],
                    [centroid_child.y, centroid_child.x]
                ],
                color=color,
                weight=2.5,
                opacity=0.8,
                tooltip=tooltip,
            ).add_to(tree_layer)

        tree_layer.add_to(m)

        # ── Client dots ──
        client_layer = folium.FeatureGroup(name="Clients")
        client_ward_labels = self.mapper.get_client_ward_labels()

        for client_idx, loc in enumerate(self.clients):
            pcode = client_ward_labels.get(client_idx)
            depth = self.tree_depths.get(pcode, 0) if pcode else 0

            folium.CircleMarker(
                location=[loc.lat, loc.lon],
                radius=4,
                color='white',
                fill=True,
                fill_color='#333333',
                fill_opacity=0.8,
                weight=1,
                tooltip=f"Client {client_idx}",
            ).add_to(client_layer)

        client_layer.add_to(m)

        # ── Office ──
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="Office / Farm (Tree Root)",
            tooltip="Office / Farm (Tree Root)",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
        ).add_to(m)

        # ── Depth legend ──
        legend_items = {}
        for d in range(min(max_depth + 1, 15)):
            legend_items[f"Depth {d}"] = depth_colors[d]
        legend_html = self._build_legend("Tree Depth from Office", legend_items)
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(filename)
        print(f"  Saved: {filename}")
        print(f"  Relevant wards: {len(self.relevant_wards)}")
        return m

    # ══════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Haversine distance in km."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _generate_depth_colors(self, n: int) -> list:
        """Generate colors from warm (near office) to cool (far from office)."""
        colors = []
        for i in range(n):
            # Hue: 0 (red) -> 0.3 (green) -> 0.65 (blue)
            hue = 0.0 + (i / max(n - 1, 1)) * 0.65
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def _build_legend(self, title: str, items: Dict[str, str]) -> str:
        """Build an HTML legend."""
        rows = ""
        for label, color in items.items():
            rows += f"""
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <div style="width: 14px; height: 14px; background: {color};
                            border: 1px solid #666; margin-right: 6px;
                            border-radius: 2px; flex-shrink: 0;"></div>
                <span style="font-size: 11px;">{label}</span>
            </div>
            """

        return f"""
        <div style="position: fixed; bottom: 30px; right: 10px; z-index: 1000;
                    background: white; padding: 10px 14px; border-radius: 6px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); max-height: 400px;
                    overflow-y: auto; font-family: Arial;">
            <div style="font-size: 12px; font-weight: bold; margin-bottom: 6px;
                        border-bottom: 1px solid #ddd; padding-bottom: 4px;">
                {title}
            </div>
            {rows}
        </div>
        """

    # ══════════════════════════════════════════════
    # SAVE / LOAD ADJACENCY TO FILE
    # ══════════════════════════════════════════════

    def save_adjacency(self, filepath: str = "Output/ward_adjacency_graph.json"):
        """
        Save the adjacency graph to a JSON file so it can be reused
        without re-computing the expensive O(n²) polygon intersections.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.adjacency, f, ensure_ascii=False)
        total_edges = sum(len(v) for v in self.adjacency.values()) // 2
        print(f"  Saved adjacency graph ({total_edges} edges) → {filepath}")

    def load_adjacency(self, filepath: str = "Output/ward_adjacency_graph.json") -> bool:
        """
        Load a previously saved adjacency graph from a JSON file.
        Returns True if loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            print(f"  No saved adjacency file found at {filepath}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.adjacency = json.load(f)
            total_edges = sum(len(v) for v in self.adjacency.values()) // 2
            print(f"  Loaded adjacency graph ({total_edges} edges) from {filepath}")
            return True
        except Exception as e:
            print(f"  Error loading adjacency file: {e}")
            return False

    # ══════════════════════════════════════════════
    # Getters for Step 2 (zone assignment)
    # ══════════════════════════════════════════════

    def get_adjacency(self) -> dict:
        """Get the full adjacency graph."""
        return self.adjacency

    def get_territory_tree(self) -> dict:
        """Get the territory tree (pcode -> parent_pcode)."""
        return self.territory_tree

    def get_tree_order(self) -> list:
        """Get the order in which wards were visited during tree building."""
        return self.tree_order

    def get_relevant_wards(self) -> set:
        """Get the set of wards included in the delivery territory."""
        return self.relevant_wards

    def get_tree_children(self) -> Dict[str, List[str]]:
        """Get children for each ward in the tree."""
        children = {pcode: [] for pcode in self.territory_tree}
        for pcode, parent in self.territory_tree.items():
            if parent is not None and parent in children:
                children[parent].append(pcode)
        return children

    def get_subtree_client_count(self, pcode: str) -> int:
        """Get total client count in the subtree rooted at a ward."""
        children = self.get_tree_children()
        count = len(self.ward_clients.get(pcode, []))

        def _count_subtree(node):
            total = len(self.ward_clients.get(node, []))
            for child in children.get(node, []):
                total += _count_subtree(child)
            return total

        return _count_subtree(pcode)