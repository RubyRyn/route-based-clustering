"""
Ward Zone Assignment (v3)

Pre-computes client-ward-to-client-ward adjacency by BFS through ALL wards.
Then grows K zones using only the 150 client wards.
No wards get missed because BFS explores all possible paths.

Phase 1 (Adjacency) — BFS from each client ward through all wards,
                       find other client wards within empty_connecting_ward_allowance
Phase 2 (Seed)      — Pick K spread-out client wards
Phase 3 (Grow)      — Smallest zone picks next adjacent client ward
Phase 4 (Balance)   — Transfer boundary wards from biggest to smallest neighbor
Phase 5 (Assign)    — Map zone labels back to individual clients

Usage:
    from ward_zones import WardZoneAssignment

    zoner = WardZoneAssignment(mapper, adjacency_builder, locations, road_matrix,
                                n_zones=14, empty_connecting_ward_allowance=3)
    zoner.assign_zones()
    zoner.visualize_zones(filename="Output/ward_zones.html")
"""

import json
import numpy as np
import colorsys
from collections import deque
from typing import List, Dict, Set, Tuple, Optional
import folium
from location import Location
from ClientWardsMapping import WardMapper
from WardAdjacency import WardAdjacency


class WardZoneAssignment:
    """
    Assign wards to K employee zones by growing from seed wards.
    Uses pre-computed client-ward adjacency.
    """

    def __init__(self, mapper: WardMapper, adjacency_builder: WardAdjacency,
                 locations: List[Location], road_matrix: np.ndarray,
                 n_zones: int = 14, size_tolerance: float = 0.20,
                 empty_connecting_ward_allowance: int = 3,
                 skip_office: bool = True):
        """
        Args:
            mapper: WardMapper that has already run map_clients().
            adjacency_builder: WardAdjacency that has run build_adjacency().
            locations: List of Location objects (office first, then clients).
            road_matrix: Full road distance matrix (office at index 0).
            n_zones: Number of employee zones (K).
            size_tolerance: Allowed deviation from average client count (±20%).
            empty_connecting_ward_allowance: Max consecutive empty wards between
                two client wards for them to be considered adjacent.
            skip_office: If True, locations[0] is the office.
        """
        self.mapper = mapper
        self.adj = adjacency_builder
        self.locations = locations
        self.road_matrix = road_matrix
        self.n_zones = n_zones
        self.size_tolerance = size_tolerance
        self.empty_allowance = empty_connecting_ward_allowance
        self.office = locations[0]
        self.clients = locations[1:] if skip_office else locations

        # Ward data
        self.ward_clients = mapper.get_ward_client_map()
        self.wards_with_clients = set(self.ward_clients.keys())
        self.full_adjacency = adjacency_builder.get_adjacency()  # All wards
        self.ward_lookup = adjacency_builder.ward_lookup

        # Office distances per ward
        self.ward_office_dist = {}
        for pcode, client_indices in self.ward_clients.items():
            dists = [road_matrix[0, idx + 1] for idx in client_indices]
            self.ward_office_dist[pcode] = np.mean(dists) if dists else 0

        # Target
        total_clients = sum(len(v) for v in self.ward_clients.values())
        self.target_clients = total_clients / n_zones
        self.min_clients = max(1, int(np.floor(self.target_clients * (1 - size_tolerance))))
        self.max_clients = int(np.ceil(self.target_clients * (1 + size_tolerance)))

        # Client ward adjacency (built in Phase 1)
        self.client_ward_adjacency = {}  # pcode -> set of adjacent client ward pcodes

        # Results
        self.ward_zones = {}
        self.zone_wards = {}
        self.zone_clients = {}
        self.seeds = []
        self.client_labels = None
        self.actual_n_zones = n_zones

        print(f"WardZoneAssignment v3 initialized:")
        print(f"  Wards with clients: {len(self.wards_with_clients)}")
        print(f"  Total wards in adjacency: {len(self.full_adjacency)}")
        print(f"  Total clients: {total_clients}")
        print(f"  Zones: {n_zones}")
        print(f"  Target clients per zone: {self.target_clients:.1f}")
        print(f"  Allowed range: [{self.min_clients}, {self.max_clients}]")
        print(f"  Empty ward allowance: {self.empty_allowance}")

    # ══════════════════════════════════════════════
    # PHASE 1: BUILD CLIENT WARD ADJACENCY
    # ══════════════════════════════════════════════

    def _build_client_ward_adjacency(self):
        """
        For each client ward, BFS through ALL wards (including empty ones).
        Find other client wards reachable within empty_connecting_ward_allowance
        empty wards.

        Result: self.client_ward_adjacency[pcode] = set of adjacent client ward pcodes
        """
        print(f"\nPhase 1 (Adjacency): Building client ward connections...")
        print(f"  BFS through all {len(self.full_adjacency)} wards")
        print(f"  Empty ward allowance: {self.empty_allowance}")

        self.client_ward_adjacency = {p: set() for p in self.wards_with_clients}

        for source_pcode in self.wards_with_clients:
            # BFS from this client ward
            # Queue entries: (ward_pcode, empty_count)
            # empty_count = consecutive empty wards crossed since source
            queue = deque()
            visited = set()
            visited.add(source_pcode)

            # Start: add all neighbors of source
            for neighbor in self.full_adjacency.get(source_pcode, {}):
                if neighbor not in visited:
                    if neighbor in self.wards_with_clients:
                        # Direct neighbor with clients — connected with 0 empty wards
                        self.client_ward_adjacency[source_pcode].add(neighbor)
                        self.client_ward_adjacency[neighbor].add(source_pcode)
                        visited.add(neighbor)
                        # Don't continue BFS through this ward — it's a client ward
                        # (we only want direct connections, not chains through client wards)
                    else:
                        # Empty ward — start counting
                        if self.empty_allowance >= 1:
                            queue.append((neighbor, 1))
                            visited.add(neighbor)

            # BFS through empty wards
            while queue:
                current_pcode, empty_count = queue.popleft()

                # Look at neighbors of this empty ward
                for neighbor in self.full_adjacency.get(current_pcode, {}):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    if neighbor in self.wards_with_clients:
                        # Found a client ward — it's connected to source
                        self.client_ward_adjacency[source_pcode].add(neighbor)
                        self.client_ward_adjacency[neighbor].add(source_pcode)
                        # Don't continue through this client ward
                    else:
                        # Another empty ward — continue if within allowance
                        new_empty_count = empty_count + 1
                        if new_empty_count <= self.empty_allowance:
                            queue.append((neighbor, new_empty_count))

        # Summary
        neighbor_counts = [len(v) for v in self.client_ward_adjacency.values()]
        connected = sum(1 for v in self.client_ward_adjacency.values() if len(v) > 0)
        isolated = sum(1 for v in self.client_ward_adjacency.values() if len(v) == 0)

        print(f"  Client ward connections built:")
        print(f"    Connected client wards: {connected}")
        print(f"    Isolated client wards: {isolated}")
        if neighbor_counts:
            print(f"    Neighbors per ward: min={min(neighbor_counts)}, "
                  f"max={max(neighbor_counts)}, "
                  f"avg={np.mean(neighbor_counts):.1f}")

        if isolated > 0:
            print(f"    ⚠️ {isolated} client wards have no connections. "
                  f"Consider increasing empty_connecting_ward_allowance.")

    # ══════════════════════════════════════════════
    # PHASE 2: SEED
    # ══════════════════════════════════════════════

    def _select_seeds(self) -> List[str]:
        """
        Pick K seed wards using furthest-first on road distance.
        Only considers wards with clients.
        """
        candidate_wards = list(self.wards_with_clients)

        if len(candidate_wards) <= self.n_zones:
            return candidate_wards

        first_seed = min(candidate_wards,
                         key=lambda p: self.ward_office_dist.get(p, float('inf')))
        seeds = [first_seed]

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

        min_dist = {}
        for pcode in candidate_wards:
            min_dist[pcode] = ward_distance(pcode, first_seed)

        for k in range(1, self.n_zones):
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

            for pcode in candidate_wards:
                if pcode in seeds:
                    continue
                d = ward_distance(pcode, best_ward)
                if d < min_dist[pcode]:
                    min_dist[pcode] = d

            if (k + 1) % 5 == 0:
                print(f"    Selected {k + 1}/{self.n_zones} seeds...")

        print(f"\nPhase 2 (Seed): Selected {len(seeds)} seed wards")
        return seeds

    # ══════════════════════════════════════════════
    # PHASE 3: GROW — Using client ward adjacency only
    # ══════════════════════════════════════════════

    def _grow(self, seeds: List[str]):
        """
        Grow zones using pre-computed client ward adjacency.
        Only works with the 150 client wards — no empty wards involved.
        Smallest zone picks next. Zones cap at max_clients.
        """
        self.zone_wards = {i: [seed] for i, seed in enumerate(seeds)}
        self.ward_zones = {seed: i for i, seed in enumerate(seeds)}

        zone_client_count = {}
        for zone_id, wards in self.zone_wards.items():
            count = sum(len(self.ward_clients.get(p, [])) for p in wards)
            zone_client_count[zone_id] = count

        zone_capped = {i: False for i in range(len(seeds))}

        unclaimed = self.wards_with_clients - set(seeds)

        # Frontier: zone_id -> set of adjacent unclaimed client wards
        frontiers = {}
        for zone_id, wards in self.zone_wards.items():
            frontiers[zone_id] = set()
            for ward_pcode in wards:
                for neighbor in self.client_ward_adjacency.get(ward_pcode, set()):
                    if neighbor in unclaimed:
                        frontiers[zone_id].add(neighbor)

        print(f"\nPhase 3 (Grow): Growing zones from {len(seeds)} seeds...")
        print(f"  Client wards to claim: {len(unclaimed)}")

        iteration = 0
        while unclaimed:
            iteration += 1

            # Find smallest uncapped zone with frontier
            best_zone = None
            min_count = float('inf')
            for zone_id in range(len(seeds)):
                if zone_capped[zone_id]:
                    continue
                if not frontiers[zone_id]:
                    continue
                if zone_client_count[zone_id] < min_count:
                    min_count = zone_client_count[zone_id]
                    best_zone = zone_id

            if best_zone is None:
                break

            # Pick the best frontier ward
            # Best = closest by road distance to zone's existing clients
            frontier = frontiers[best_zone]
            best_ward = None
            best_road_dist = float('inf')

            zone_members = self.zone_wards[best_zone]
            for candidate in frontier:
                # Average road distance to zone's client wards
                dists = []
                for member in zone_members:
                    member_clients = self.ward_clients.get(member, [])
                    candidate_clients = self.ward_clients.get(candidate, [])
                    for mc in member_clients:
                        for cc in candidate_clients:
                            dists.append(self.road_matrix[mc + 1, cc + 1])
                avg_dist = np.mean(dists) if dists else float('inf')

                if avg_dist < best_road_dist:
                    best_road_dist = avg_dist
                    best_ward = candidate

            if best_ward is None:
                break

            # Claim
            self.zone_wards[best_zone].append(best_ward)
            self.ward_zones[best_ward] = best_zone
            unclaimed.discard(best_ward)

            new_clients = len(self.ward_clients.get(best_ward, []))
            zone_client_count[best_zone] += new_clients

            # Cap check
            if zone_client_count[best_zone] >= self.max_clients:
                zone_capped[best_zone] = True
                frontiers[best_zone] = set()

            # Remove from all frontiers
            for zone_id in range(len(seeds)):
                frontiers[zone_id].discard(best_ward)

            # Add new frontier from claimed ward (if not capped)
            if not zone_capped[best_zone]:
                for neighbor in self.client_ward_adjacency.get(best_ward, set()):
                    if neighbor in unclaimed:
                        frontiers[best_zone].add(neighbor)

            if iteration % 20 == 0:
                counts = sorted(zone_client_count.values())
                n_capped = sum(1 for v in zone_capped.values() if v)
                print(f"    Iteration {iteration}: {len(unclaimed)} unclaimed, "
                      f"sizes: [{min(counts)}-{max(counts)}], "
                      f"{n_capped} capped")

        # Handle unclaimed client wards → new zones
        if unclaimed:
            print(f"\n  {len(unclaimed)} client wards unreached "
                  f"— creating additional zones...")

            new_zone_groups = self._group_unclaimed_wards(unclaimed)

            for group in new_zone_groups:
                new_zone_id = len(self.zone_wards)
                self.zone_wards[new_zone_id] = list(group)
                zone_client_count[new_zone_id] = 0
                zone_capped[new_zone_id] = False

                for pcode in group:
                    self.ward_zones[pcode] = new_zone_id
                    zone_client_count[new_zone_id] += len(
                        self.ward_clients.get(pcode, []))

                print(f"    New zone {new_zone_id + 1}: "
                      f"{len(group)} wards, "
                      f"{zone_client_count[new_zone_id]} clients")

        self.actual_n_zones = len(self.zone_wards)

        counts = [zone_client_count[z] for z in range(self.actual_n_zones)]
        print(f"\n  Phase 3 complete:")
        print(f"  Total zones: {self.actual_n_zones} "
              f"(started with {self.n_zones} seeds)")
        print(f"  Zone client counts: {sorted(counts)}")
        print(f"  Range: [{min(counts)}, {max(counts)}]")
        print(f"  Std: {np.std(counts):.1f}")

    def _group_unclaimed_wards(self, unclaimed_wards: Set[str]) -> List[Set[str]]:
        """Group unclaimed client wards into connected components."""
        remaining = set(unclaimed_wards)
        groups = []

        while remaining:
            start = next(iter(remaining))
            group = set()
            queue = [start]

            while queue:
                current = queue.pop(0)
                if current in group:
                    continue
                if current not in remaining:
                    continue
                group.add(current)
                remaining.discard(current)

                for neighbor in self.client_ward_adjacency.get(current, set()):
                    if neighbor in remaining:
                        queue.append(neighbor)

            groups.append(group)

        return groups

    # ══════════════════════════════════════════════
    # PHASE 4: BALANCE
    # ══════════════════════════════════════════════

    def _get_zone_client_count(self, zone_id: int) -> int:
        return sum(len(self.ward_clients.get(p, []))
                   for p in self.zone_wards.get(zone_id, []))

    def _get_zone_neighbor_zones(self, zone_id: int) -> Set[int]:
        neighbor_zones = set()
        for pcode in self.zone_wards.get(zone_id, []):
            for neighbor_pcode in self.client_ward_adjacency.get(pcode, set()):
                if neighbor_pcode in self.ward_zones:
                    other_zone = self.ward_zones[neighbor_pcode]
                    if other_zone != zone_id:
                        neighbor_zones.add(other_zone)
        return neighbor_zones

    def _get_boundary_wards(self, from_zone: int, to_zone: int) -> List[str]:
        boundary = []
        for pcode in self.zone_wards.get(from_zone, []):
            for neighbor_pcode in self.client_ward_adjacency.get(pcode, set()):
                if neighbor_pcode in self.ward_zones:
                    if self.ward_zones[neighbor_pcode] == to_zone:
                        boundary.append(pcode)
                        break
        boundary.sort(key=lambda p: len(self.ward_clients.get(p, [])))
        return boundary

    def _would_disconnect_zone(self, zone_id: int, ward_to_remove: str) -> bool:
        remaining_wards = [p for p in self.zone_wards[zone_id]
                          if p != ward_to_remove]

        if len(remaining_wards) <= 1:
            return False

        visited = set()
        queue = [remaining_wards[0]]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.client_ward_adjacency.get(current, set()):
                if neighbor in remaining_wards and neighbor not in visited:
                    queue.append(neighbor)

        return len(visited) < len(remaining_wards)

    def _balance(self, max_iterations: int = 200):
        """
        Transfer boundary wards from biggest zone to its smallest neighbor.
        - Transfer smallest boundary ward (fewest clients)
        - No transfer back rule prevents ping-pong
        - Receiver can exceed max (will be handled in next iteration)
        """
        print(f"\nPhase 4 (Balance): Ward transfer balancing")
        print(f"  Target range: [{self.min_clients}, {self.max_clients}]")

        blocked_transfers = set()

        for iteration in range(max_iterations):
            zone_sizes = {}
            for z in range(self.actual_n_zones):
                zone_sizes[z] = self._get_zone_client_count(z)

            sorted_zones = sorted(zone_sizes.keys(), key=lambda z: -zone_sizes[z])
            biggest_zone = sorted_zones[0]
            biggest_size = zone_sizes[biggest_zone]

            if biggest_size <= self.max_clients:
                print(f"  Converged at iteration {iteration + 1}: "
                      f"max size {biggest_size} within limit {self.max_clients}")
                break

            # Try each oversized zone, starting from biggest
            transferred = False
            for zone_candidate in sorted_zones:
                if zone_sizes[zone_candidate] <= self.max_clients:
                    break

                neighbor_zones = self._get_zone_neighbor_zones(zone_candidate)
                allowed_neighbors = {nz for nz in neighbor_zones
                                     if (zone_candidate, nz) not in blocked_transfers}

                if not allowed_neighbors:
                    continue

                smallest_neighbor = min(allowed_neighbors,
                                        key=lambda z: zone_sizes[z])

                boundary_wards = self._get_boundary_wards(
                    zone_candidate, smallest_neighbor)

                for ward_to_transfer in boundary_wards:
                    if self._would_disconnect_zone(zone_candidate, ward_to_transfer):
                        continue

                    # Transfer
                    self.zone_wards[zone_candidate].remove(ward_to_transfer)
                    self.zone_wards[smallest_neighbor].append(ward_to_transfer)
                    self.ward_zones[ward_to_transfer] = smallest_neighbor

                    blocked_transfers.add((smallest_neighbor, zone_candidate))

                    ward_clients = len(self.ward_clients.get(ward_to_transfer, []))
                    new_giver = zone_sizes[zone_candidate] - ward_clients
                    new_receiver = zone_sizes[smallest_neighbor] + ward_clients

                    print(f"  Iter {iteration + 1}: Zone {zone_candidate + 1} "
                          f"({zone_sizes[zone_candidate]}) → "
                          f"ward ({ward_clients} clients) → "
                          f"Zone {smallest_neighbor + 1} "
                          f"({zone_sizes[smallest_neighbor]}). "
                          f"New: {new_giver}, {new_receiver}")

                    transferred = True
                    break

                if transferred:
                    break

            if not transferred:
                print(f"  No more transfers possible at iteration {iteration + 1}")
                break

        final_sizes = [self._get_zone_client_count(z)
                       for z in range(self.actual_n_zones)]
        print(f"\n  Phase 4 complete:")
        print(f"  Zone client counts: {sorted(final_sizes)}")
        print(f"  Range: [{min(final_sizes)}, {max(final_sizes)}]")
        print(f"  Std: {np.std(final_sizes):.1f}")

    # ══════════════════════════════════════════════
    # PHASE 5: ASSIGN — Map zones back to clients
    # ══════════════════════════════════════════════

    def _assign_clients(self):
        n_clients = len(self.clients)
        self.client_labels = np.full(n_clients, -1, dtype=int)
        self.zone_clients = {z: [] for z in range(self.actual_n_zones)}

        client_ward_labels = self.mapper.get_client_ward_labels()

        for client_idx in range(n_clients):
            pcode = client_ward_labels.get(client_idx)
            if pcode and pcode in self.ward_zones:
                zone_id = self.ward_zones[pcode]
                self.client_labels[client_idx] = zone_id
                self.zone_clients[zone_id].append(client_idx)
            else:
                best_zone = 0
                best_dist = float('inf')
                for zone_id, client_list in self.zone_clients.items():
                    for other_idx in client_list[:10]:
                        dist = self.road_matrix[client_idx + 1, other_idx + 1]
                        if dist < best_dist:
                            best_dist = dist
                            best_zone = zone_id
                self.client_labels[client_idx] = best_zone
                self.zone_clients[best_zone].append(client_idx)

        assigned = np.sum(self.client_labels >= 0)
        print(f"\nPhase 5 (Assign): Mapped zones to {assigned}/{n_clients} clients")

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def assign_zones(self) -> np.ndarray:
        print(f"\n{'=' * 60}")
        print(f"WARD ZONE ASSIGNMENT v3 (Pre-computed Client Ward Adjacency)")
        print(f"{'=' * 60}")

        # Phase 1
        self._build_client_ward_adjacency()

        # Phase 2
        self.seeds = self._select_seeds()

        # Phase 3
        self._grow(self.seeds)

        # Phase 4
        self._balance()

        # Phase 5
        self._assign_clients()

        # Final report
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Total zones: {self.actual_n_zones}")
        for zone_id in range(self.actual_n_zones):
            n_wards = len(self.zone_wards.get(zone_id, []))
            n_clients = len(self.zone_clients.get(zone_id, []))
            client_indices = self.zone_clients.get(zone_id, [])
            if client_indices:
                avg_dist = np.mean([self.road_matrix[0, idx + 1]
                                    for idx in client_indices])
            else:
                avg_dist = 0
            is_new = zone_id >= self.n_zones
            marker = " (NEW)" if is_new else ""
            print(f"  Zone {zone_id + 1}{marker}: {n_wards} wards, "
                  f"{n_clients} clients, avg dist {avg_dist:.1f} km")

        counts = [len(self.zone_clients[z]) for z in range(self.actual_n_zones)]
        print(f"\n  Client counts: {sorted(counts)}")
        print(f"  Range: [{min(counts)}, {max(counts)}]")
        print(f"  Std: {np.std(counts):.1f}")
        if self.actual_n_zones > self.n_zones:
            print(f"\n  ⚠️ Geography required {self.actual_n_zones - self.n_zones} "
                  f"additional zones beyond the requested {self.n_zones}")
        print(f"{'=' * 60}")

        return self.client_labels

    # ══════════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════════

    def visualize_zones(self, filename: str = "Output/ward_zones.html"):
        if not self.ward_zones:
            print("Run assign_zones() first.")
            return

        print(f"\nGenerating zone map...")

        m = folium.Map(
            location=[self.office.lat, self.office.lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        zone_colors = self._generate_zone_colors(self.actual_n_zones)

        zone_layers = {}
        for z in range(self.actual_n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            is_new = z >= self.n_zones
            label = f"Zone {z + 1} ({n_clients} clients)"
            if is_new:
                label += " NEW"
            zone_layers[z] = folium.FeatureGroup(name=label)

        for ward in self.mapper.wards:
            pcode = ward['pcode']
            if pcode not in self.ward_zones or pcode not in self.wards_with_clients:
                continue

            geojson = json.loads(json.dumps(
                ward['geometry'].__geo_interface__
            ))

            zone_id = self.ward_zones[pcode]
            fill_color = zone_colors[zone_id]
            is_seed = pcode in self.seeds
            client_count = len(self.ward_clients.get(pcode, []))

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{ward['name']}</b><br>
                Township: {ward['township']}<br>
                Pcode: {pcode}<br>
                <b>Zone: {zone_id + 1}</b><br>
                Clients: {client_count}<br>
                {'⭐ Seed ward' if is_seed else ''}
            </div>
            """
            tooltip = (f"{ward['name']} — Zone {zone_id + 1} "
                      f"({client_count} clients)")

            folium.GeoJson(
                geojson,
                style_function=lambda x, fc=fill_color: {
                    'fillColor': fc,
                    'fillOpacity': 0.4,
                    'color': fc,
                    'weight': 1.5,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=tooltip,
            ).add_to(zone_layers[zone_id])

        for z in range(self.actual_n_zones):
            zone_layers[z].add_to(m)

        # Client dots
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

        # Seeds
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

        # Office
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="Office / Farm",
            tooltip="Office / Farm",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
        ).add_to(m)

        # Legend
        legend_items = {}
        for z in range(self.actual_n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            is_new = z >= self.n_zones
            label = f"Zone {z + 1} ({n_clients} clients)"
            if is_new:
                label += " ⚠️"
            legend_items[label] = zone_colors[z]
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
        palette = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
            '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
            '#ffd8b1', '#000075', '#a9a9a9', '#ffe119', '#e6beff'
        ]
        colors = []
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

    def get_actual_n_zones(self) -> int:
        return self.actual_n_zones

    def get_client_ward_adjacency(self) -> dict:
        return self.client_ward_adjacency
