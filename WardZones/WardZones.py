"""
Ward Zone Assignment (Step 3)

Grows K zones from K seed wards simultaneously.
At each step, the zone with fewest clients picks the best adjacent unclaimed ward.
Zones are always contiguous (only adjacent wards can be added).

Phase 1 (Seed)  — Pick K spread-out seed wards from wards with clients
Phase 2 (Grow)  — Smallest zone picks next, claiming best adjacent ward
Phase 3 (Assign) — Map zone labels back to individual clients

Usage:
    from ward_zones import WardZoneAssignment

    zoner = WardZoneAssignment(mapper, adjacency_builder, locations, road_matrix, n_zones=14)
    zoner.assign_zones()
    zoner.visualize_zones(filename="Output/ward_zones.html")
"""

import json
import numpy as np
import heapq
import colorsys
from typing import List, Dict, Set, Tuple, Optional
import folium
from location import Location
from ClientWardsMapping import WardMapper
from WardAdjacency import WardAdjacency


class WardZoneAssignment:
    """
    Assign wards to K employee zones by growing from seed wards.
    """

    def __init__(self, mapper: WardMapper, adjacency_builder: WardAdjacency,
                 locations: List[Location], road_matrix: np.ndarray,
                 n_zones: int = 14, size_tolerance: float = 0.20,
                 skip_office: bool = True):
        """
        Args:
            mapper: WardMapper that has already run map_clients().
            adjacency_builder: WardAdjacency that has run build_adjacency().
            locations: List of Location objects (office first, then clients).
            road_matrix: Full road distance matrix (office at index 0).
            n_zones: Number of employee zones (K).
            size_tolerance: Allowed deviation from average client count (±20%).
            skip_office: If True, locations[0] is the office.
        """
        self.mapper = mapper
        self.adj = adjacency_builder
        self.locations = locations
        self.road_matrix = road_matrix
        self.n_zones = n_zones
        self.size_tolerance = size_tolerance
        self.office = locations[0]
        self.clients = locations[1:] if skip_office else locations

        # Ward data
        self.ward_clients = mapper.get_ward_client_map()  # pcode -> [client indices]
        self.wards_with_clients = set(self.ward_clients.keys())
        self.adjacency = adjacency_builder.get_adjacency()
        self.relevant_wards = adjacency_builder.get_relevant_wards()
        self.ward_lookup = adjacency_builder.ward_lookup

        # Office distances per ward (average road distance of clients from office)
        self.ward_office_dist = {}
        for pcode, client_indices in self.ward_clients.items():
            dists = [road_matrix[0, idx + 1] for idx in client_indices]
            self.ward_office_dist[pcode] = np.mean(dists) if dists else 0

        # Target
        total_clients = sum(len(v) for v in self.ward_clients.values())
        self.target_clients = total_clients / n_zones
        self.min_clients = max(1, int(np.floor(self.target_clients * (1 - size_tolerance))))
        self.max_clients = int(np.ceil(self.target_clients * (1 + size_tolerance)))

        # Results
        self.ward_zones = {}       # pcode -> zone_id
        self.zone_wards = {}       # zone_id -> [pcodes]
        self.zone_clients = {}     # zone_id -> [client_indices]
        self.seeds = []            # seed ward pcodes
        self.client_labels = None  # client_idx -> zone_id

        print(f"WardZoneAssignment initialized:")
        print(f"  Wards with clients: {len(self.wards_with_clients)}")
        print(f"  Relevant wards (incl. connectors): {len(self.relevant_wards)}")
        print(f"  Total clients: {total_clients}")
        print(f"  Zones: {n_zones}")
        print(f"  Target clients per zone: {self.target_clients:.1f}")
        print(f"  Allowed range: [{self.min_clients}, {self.max_clients}]")

    # ══════════════════════════════════════════════
    # PHASE 1: SEED — Pick K spread-out wards
    # ══════════════════════════════════════════════

    def _select_seeds(self) -> List[str]:
        """
        Pick K seed wards using furthest-first on road distance.
        Only considers wards with clients.
        First seed = ward closest to office.
        """
        candidate_wards = list(self.wards_with_clients)

        if len(candidate_wards) <= self.n_zones:
            print(f"  Warning: Only {len(candidate_wards)} wards with clients "
                  f"for {self.n_zones} zones.")
            return candidate_wards

        # First seed: ward closest to office
        first_seed = min(candidate_wards,
                         key=lambda p: self.ward_office_dist.get(p, float('inf')))
        seeds = [first_seed]

        # Compute min road distance from each ward to any seed
        # Use average road distance between wards' clients
        def ward_distance(pcode_a, pcode_b):
            clients_a = self.ward_clients.get(pcode_a, [])
            clients_b = self.ward_clients.get(pcode_b, [])
            if not clients_a or not clients_b:
                return float('inf')
            dists = []
            for ca in clients_a:
                for cb in clients_b:
                    dists.append(self.road_matrix[ca + 1, cb + 1])
            return np.mean(dists)

        # Track min distance to nearest seed for each candidate
        min_dist = {}
        for pcode in candidate_wards:
            min_dist[pcode] = ward_distance(pcode, first_seed)

        for k in range(1, self.n_zones):
            # Pick ward furthest from all existing seeds
            best_ward = None
            best_dist = -1
            for pcode in candidate_wards:
                if pcode in seeds:
                    continue
                if min_dist[pcode] > best_dist:
                    best_dist = min_dist[pcode]
                    best_ward = pcode

            if best_ward is None:
                break

            seeds.append(best_ward)

            # Update min distances
            for pcode in candidate_wards:
                if pcode in seeds:
                    continue
                d = ward_distance(pcode, best_ward)
                if d < min_dist[pcode]:
                    min_dist[pcode] = d

            if (k + 1) % 5 == 0:
                print(f"    Selected {k + 1}/{self.n_zones} seeds...")

        print(f"\nPhase 1 (Seed): Selected {len(seeds)} seed wards")
        return seeds

    # ══════════════════════════════════════════════
    # PHASE 2: GROW — Smallest zone picks next
    # ══════════════════════════════════════════════

    def _grow(self, seeds: List[str]):
        """
        Grow zones simultaneously from seeds.
        At each step, the zone with fewest clients picks the best
        adjacent unclaimed ward.

        "Best" = adjacent ward with strongest connection score
        to any ward already in the zone.
        """
        # Initialize zones
        self.zone_wards = {i: [seed] for i, seed in enumerate(seeds)}
        self.ward_zones = {seed: i for i, seed in enumerate(seeds)}

        # Client count per zone
        zone_client_count = {}
        for zone_id, wards in self.zone_wards.items():
            count = sum(len(self.ward_clients.get(p, [])) for p in wards)
            zone_client_count[zone_id] = count

        # Track unclaimed wards (only relevant ones)
        unclaimed = self.relevant_wards - set(seeds)

        # For each zone, maintain a frontier: adjacent unclaimed wards
        # frontier[zone_id] = {pcode: best_connection_score}
        frontiers = {}
        for zone_id, wards in self.zone_wards.items():
            frontiers[zone_id] = {}
            for ward_pcode in wards:
                for neighbor, edge_data in self.adjacency.get(ward_pcode, {}).items():
                    if neighbor in unclaimed:
                        # Keep the best score if neighbor appears multiple times
                        current_score = frontiers[zone_id].get(neighbor, -1)
                        if edge_data['score'] > current_score:
                            frontiers[zone_id][neighbor] = edge_data['score']

        print(f"\nPhase 2 (Grow): Growing zones...")

        iteration = 0
        while unclaimed:
            iteration += 1

            # Find the zone with fewest clients that has frontier wards
            best_zone = None
            min_count = float('inf')
            for zone_id in range(self.n_zones):
                if frontiers[zone_id] and zone_client_count[zone_id] < min_count:
                    min_count = zone_client_count[zone_id]
                    best_zone = zone_id

            if best_zone is None:
                # No zone has any frontier left
                break

            # This zone picks the best frontier ward (highest connection score)
            frontier = frontiers[best_zone]
            best_ward = max(frontier, key=frontier.get)
            best_score = frontier[best_ward]

            # Claim the ward
            self.zone_wards[best_zone].append(best_ward)
            self.ward_zones[best_ward] = best_zone
            unclaimed.discard(best_ward)

            # Update client count
            new_clients = len(self.ward_clients.get(best_ward, []))
            zone_client_count[best_zone] += new_clients

            # Remove this ward from all frontiers
            for zone_id in range(self.n_zones):
                frontiers[zone_id].pop(best_ward, None)

            # Add new frontier wards from the claimed ward's neighbors
            for neighbor, edge_data in self.adjacency.get(best_ward, {}).items():
                if neighbor in unclaimed and neighbor not in self.ward_zones:
                    current_score = frontiers[best_zone].get(neighbor, -1)
                    if edge_data['score'] > current_score:
                        frontiers[best_zone][neighbor] = edge_data['score']

            if iteration % 50 == 0:
                counts = sorted(zone_client_count.values())
                print(f"    Iteration {iteration}: {len(unclaimed)} unclaimed, "
                      f"zone sizes: [{min(counts)}-{max(counts)}]")

        # Handle any remaining unclaimed wards (disconnected from all zones)
        if unclaimed:
            print(f"  Warning: {len(unclaimed)} wards unreachable from any zone.")
            # Assign to nearest zone by road distance
            for pcode in unclaimed:
                best_zone = self._find_nearest_zone(pcode)
                if best_zone is not None:
                    self.zone_wards[best_zone].append(pcode)
                    self.ward_zones[pcode] = best_zone
                    zone_client_count[best_zone] += len(
                        self.ward_clients.get(pcode, []))

        # Summary
        counts = [zone_client_count[z] for z in range(self.n_zones)]
        print(f"\n  Phase 2 complete:")
        print(f"  Zone client counts: {sorted(counts)}")
        print(f"  Range: [{min(counts)}, {max(counts)}]")
        print(f"  Std: {np.std(counts):.1f}")

    def _find_nearest_zone(self, pcode: str) -> Optional[int]:
        """Find nearest zone for a disconnected ward."""
        ward = self.ward_lookup.get(pcode)
        if not ward:
            return 0

        centroid = ward['geometry'].centroid
        best_zone = None
        best_dist = float('inf')

        for zone_id, wards in self.zone_wards.items():
            for zw_pcode in wards:
                zw = self.ward_lookup.get(zw_pcode)
                if zw:
                    zw_centroid = zw['geometry'].centroid
                    dist = self._haversine(
                        centroid.y, centroid.x,
                        zw_centroid.y, zw_centroid.x)
                    if dist < best_dist:
                        best_dist = dist
                        best_zone = zone_id

        return best_zone

    # ══════════════════════════════════════════════
    # PHASE 3: ASSIGN — Map zones back to clients
    # ══════════════════════════════════════════════

    def _assign_clients(self):
        """Map zone labels to individual clients."""
        n_clients = len(self.clients)
        self.client_labels = np.full(n_clients, -1, dtype=int)
        self.zone_clients = {z: [] for z in range(self.n_zones)}

        client_ward_labels = self.mapper.get_client_ward_labels()

        for client_idx in range(n_clients):
            pcode = client_ward_labels.get(client_idx)
            if pcode and pcode in self.ward_zones:
                zone_id = self.ward_zones[pcode]
                self.client_labels[client_idx] = zone_id
                self.zone_clients[zone_id].append(client_idx)
            else:
                # Client not in any zone — assign to nearest zone
                loc = self.clients[client_idx]
                best_zone = 0
                best_dist = float('inf')
                for zone_id, client_list in self.zone_clients.items():
                    if client_list:
                        for other_idx in client_list[:5]:  # Check a few
                            dist = self.road_matrix[client_idx + 1, other_idx + 1]
                            if dist < best_dist:
                                best_dist = dist
                                best_zone = zone_id
                self.client_labels[client_idx] = best_zone
                self.zone_clients[best_zone].append(client_idx)

        assigned = np.sum(self.client_labels >= 0)
        print(f"\nPhase 3 (Assign): Mapped zones to {assigned}/{n_clients} clients")

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def assign_zones(self) -> np.ndarray:
        """Run the full Seed → Grow → Assign pipeline."""
        print(f"\n{'=' * 60}")
        print(f"WARD ZONE ASSIGNMENT")
        print(f"{'=' * 60}")

        # Phase 1
        self.seeds = self._select_seeds()

        # Phase 2
        self._grow(self.seeds)

        # Phase 3
        self._assign_clients()

        # Final report
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        for zone_id in range(self.n_zones):
            n_wards = len(self.zone_wards.get(zone_id, []))
            n_clients = len(self.zone_clients.get(zone_id, []))
            client_indices = self.zone_clients.get(zone_id, [])
            if client_indices:
                avg_dist = np.mean([self.road_matrix[0, idx + 1]
                                    for idx in client_indices])
            else:
                avg_dist = 0
            print(f"  Zone {zone_id + 1}: {n_wards} wards, "
                  f"{n_clients} clients, avg dist {avg_dist:.1f} km")

        counts = [len(self.zone_clients[z]) for z in range(self.n_zones)]
        print(f"\n  Client counts: {sorted(counts)}")
        print(f"  Range: [{min(counts)}, {max(counts)}]")
        print(f"  Std: {np.std(counts):.1f}")
        print(f"{'=' * 60}")

        return self.client_labels

    # ══════════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════════

    def visualize_zones(self, filename: str = "Output/ward_zones.html"):
        """Folium map showing ward zones with territories."""
        if not self.ward_zones:
            print("Run assign_zones() first.")
            return

        print(f"\nGenerating zone map...")

        m = folium.Map(
            location=[self.office.lat, self.office.lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # Zone colors
        zone_colors = self._generate_zone_colors(self.n_zones)

        # ── Ward polygons colored by zone ──
        zone_layers = {}
        for z in range(self.n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            zone_layers[z] = folium.FeatureGroup(
                name=f"Zone {z + 1} ({n_clients} clients)")

        for ward in self.mapper.wards:
            pcode = ward['pcode']

            # Skip wards not in any zone or wards with no clients
            if pcode not in self.ward_zones or pcode not in self.wards_with_clients:
                continue

            geojson = json.loads(json.dumps(
                ward['geometry'].__geo_interface__
            ))

            zone_id = self.ward_zones[pcode]
            fill_color = zone_colors[zone_id]
            fill_opacity = 0.4
            weight = 1.5
            border_color = fill_color
            layer = zone_layers[zone_id]

            client_count = len(self.ward_clients.get(pcode, []))
            has_clients = pcode in self.wards_with_clients
            is_seed = pcode in self.seeds

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{ward['name']}</b><br>
                Township: {ward['township']}<br>
                Pcode: {pcode}<br>
                <b>Zone: {zone_id + 1}</b><br>
                Clients: {client_count}<br>
                {'⭐ Seed ward' if is_seed else ''}
                {'📦 Has clients' if has_clients else '🔗 Connector'}
            </div>
            """
            tooltip = (f"{ward['name']} — Zone {zone_id + 1} "
                      f"({client_count} clients)")

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

        for z in range(self.n_zones):
            zone_layers[z].add_to(m)

        # ── Client dots ──
        client_layer = folium.FeatureGroup(name="Clients")
        for client_idx, loc in enumerate(self.clients):
            zone_id = self.client_labels[client_idx]
            color = zone_colors[zone_id] if zone_id >= 0 else '#ff0000'

            ward_info = self.mapper.client_ward_map.get(client_idx)
            ward_name = ward_info['name'] if ward_info else 'N/A'

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Client {client_idx}</b><br>
                Zone: {zone_id + 1}<br>
                Ward: {ward_name}<br>
                Office dist: {self.road_matrix[0, client_idx + 1]:.1f} km
            </div>
            """

            folium.CircleMarker(
                location=[loc.lat, loc.lon],
                radius=5,
                color='white',
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=1,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"Client {client_idx} — Zone {zone_id + 1}",
            ).add_to(client_layer)

        client_layer.add_to(m)

        # ── Seed ward markers ──
        seed_layer = folium.FeatureGroup(name="Seed Wards")
        for zone_id, seed_pcode in enumerate(self.seeds):
            ward = self.ward_lookup.get(seed_pcode)
            if ward:
                centroid = ward['geometry'].centroid
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    popup=f"Seed — Zone {zone_id + 1}<br>{ward['name']}",
                    tooltip=f"Seed Zone {zone_id + 1}",
                    icon=folium.Icon(color='black', icon='star', prefix='fa'),
                ).add_to(seed_layer)
        seed_layer.add_to(m)

        # ── Office ──
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="Office / Farm",
            tooltip="Office / Farm",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
        ).add_to(m)

        # ── Legend ──
        legend_items = {}
        for z in range(self.n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            legend_items[f"Zone {z + 1} ({n_clients} clients)"] = zone_colors[z]
        legend_html = self._build_legend("Employee Zones", legend_items)
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(filename)
        print(f"  Saved: {filename}")
        return m

    # ══════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _generate_zone_colors(self, n: int) -> list:
        colors = []
        palette = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
            '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
            '#ffd8b1', '#000075', '#a9a9a9', '#ffe119', '#e6beff'
        ]
        for i in range(n):
            if i < len(palette):
                colors.append(palette[i])
            else:
                hue = i / n
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.85)
                colors.append('#{:02x}{:02x}{:02x}'.format(
                    int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def _build_legend(self, title: str, items: Dict[str, str]) -> str:
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
    # Getters
    # ══════════════════════════════════════════════

    def get_client_labels(self) -> np.ndarray:
        return self.client_labels

    def get_zone_wards(self) -> dict:
        return self.zone_wards

    def get_zone_clients(self) -> dict:
        return self.zone_clients

    def get_ward_zones(self) -> dict:
        return self.ward_zones
