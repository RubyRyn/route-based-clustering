"""
Ward Zone Assignment (v2)

Grows K zones from K seed wards simultaneously.
- Smallest zone picks next adjacent ward
- Zones stop growing when they hit max client count
- empty_connecting_ward_allowance limits how many consecutive empty wards
  a zone can cross to reach the next ward with clients
- May produce more than K zones if geography requires it

Usage:
    from ward_zones import WardZoneAssignment

    zoner = WardZoneAssignment(mapper, adjacency_builder, locations, road_matrix,
                                n_zones=14, empty_connecting_ward_allowance=3)
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
            empty_connecting_ward_allowance: Max consecutive empty wards a zone
                can cross to reach the next ward with clients.
                0 = only claim wards directly adjacent to a client ward.
                3 = allow up to 3 empty wards in between.
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
        self.adjacency = adjacency_builder.get_adjacency()
        self.relevant_wards = adjacency_builder.get_relevant_wards()
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

        # Results
        self.ward_zones = {}
        self.zone_wards = {}
        self.zone_clients = {}
        self.seeds = []
        self.client_labels = None
        self.actual_n_zones = n_zones  # May increase if geography splits things

        print(f"WardZoneAssignment v2 initialized:")
        print(f"  Wards with clients: {len(self.wards_with_clients)}")
        print(f"  Total clients: {total_clients}")
        print(f"  Zones: {n_zones}")
        print(f"  Target clients per zone: {self.target_clients:.1f}")
        print(f"  Allowed range: [{self.min_clients}, {self.max_clients}]")
        print(f"  Empty ward allowance: {self.empty_allowance}")

    # ══════════════════════════════════════════════
    # PHASE 1: SEED
    # ══════════════════════════════════════════════

    def _select_seeds(self) -> List[str]:
        """
        Pick K seed wards using furthest-first on road distance.
        Only considers wards with clients.
        First seed = ward closest to office.
        """
        candidate_wards = list(self.wards_with_clients)

        if len(candidate_wards) <= self.n_zones:
            return candidate_wards

        # First seed: ward closest to office
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

        print(f"\nPhase 1 (Seed): Selected {len(seeds)} seed wards")
        return seeds

    # ══════════════════════════════════════════════
    # PHASE 2: GROW — With empty ward allowance and cap
    # ══════════════════════════════════════════════

    def _grow(self, seeds: List[str]):
        """
        Grow zones simultaneously from seeds.

        Rules:
        - Smallest zone (by client count) picks next
        - Zones stop growing when they hit max_clients
        - Empty ward counter tracks consecutive empty wards crossed
        - If counter exceeds empty_connecting_ward_allowance, stop
          expanding in that direction
        - Unclaimed client wards after all zones are done become new zones
        """
        # Initialize zones
        self.zone_wards = {i: [seed] for i, seed in enumerate(seeds)}
        self.ward_zones = {seed: i for i, seed in enumerate(seeds)}

        # Client count per zone
        zone_client_count = {}
        for zone_id, wards in self.zone_wards.items():
            count = sum(len(self.ward_clients.get(p, [])) for p in wards)
            zone_client_count[zone_id] = count

        # Zone capped flag
        zone_capped = {i: False for i in range(len(seeds))}

        # Track unclaimed relevant wards
        unclaimed = self.relevant_wards - set(seeds)

        # Frontier: zone_id -> {pcode: (connection_score, empty_count)}
        # empty_count = consecutive empty wards crossed to reach this frontier ward
        frontiers = {}
        for zone_id, wards in self.zone_wards.items():
            frontiers[zone_id] = {}
            for ward_pcode in wards:
                for neighbor, edge_data in self.adjacency.get(ward_pcode, {}).items():
                    if neighbor in unclaimed:
                        # Seed wards have clients, so neighbors start with empty_count
                        # depending on whether the neighbor itself has clients
                        if neighbor in self.wards_with_clients:
                            empty_count = 0
                        else:
                            empty_count = 1

                        # Only add if within allowance
                        if empty_count <= self.empty_allowance:
                            existing = frontiers[zone_id].get(neighbor)
                            if existing is None or edge_data['score'] > existing[0]:
                                frontiers[zone_id][neighbor] = (edge_data['score'], empty_count)

        print(f"\nPhase 2 (Grow): Growing zones...")
        print(f"  Max clients per zone: {self.max_clients}")
        print(f"  Empty ward allowance: {self.empty_allowance}")

        iteration = 0
        while unclaimed:
            iteration += 1

            # Find the zone with fewest clients that has frontier wards and isn't capped
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

            # Pick the best frontier ward (highest connection score)
            frontier = frontiers[best_zone]
            best_ward = max(frontier, key=lambda p: frontier[p][0])
            best_score, empty_count = frontier[best_ward]

            # Claim the ward
            self.zone_wards[best_zone].append(best_ward)
            self.ward_zones[best_ward] = best_zone
            unclaimed.discard(best_ward)

            # Update client count
            ward_has_clients = best_ward in self.wards_with_clients
            new_clients = len(self.ward_clients.get(best_ward, []))
            zone_client_count[best_zone] += new_clients

            # Check if zone hit the cap
            if zone_client_count[best_zone] >= self.max_clients:
                zone_capped[best_zone] = True
                frontiers[best_zone] = {}  # Clear frontier

            # Remove this ward from all frontiers
            for zone_id in range(len(seeds)):
                frontiers[zone_id].pop(best_ward, None)

            # Add new frontier wards (only if zone not capped)
            if not zone_capped[best_zone]:
                for neighbor, edge_data in self.adjacency.get(best_ward, {}).items():
                    if neighbor in unclaimed and neighbor not in self.ward_zones:
                        # Calculate empty count for this neighbor
                        if ward_has_clients:
                            # We just claimed a ward with clients, reset counter
                            if neighbor in self.wards_with_clients:
                                new_empty_count = 0
                            else:
                                new_empty_count = 1
                        else:
                            # We crossed an empty ward, increment counter
                            if neighbor in self.wards_with_clients:
                                new_empty_count = 0  # Found clients, reset
                            else:
                                new_empty_count = empty_count + 1

                        # Only add if within allowance
                        if new_empty_count <= self.empty_allowance:
                            existing = frontiers[best_zone].get(neighbor)
                            if existing is None or edge_data['score'] > existing[0]:
                                frontiers[best_zone][neighbor] = (
                                    edge_data['score'], new_empty_count)

            if iteration % 50 == 0:
                counts = sorted(zone_client_count.values())
                n_capped = sum(1 for v in zone_capped.values() if v)
                print(f"    Iteration {iteration}: {len(unclaimed)} unclaimed, "
                      f"sizes: [{min(counts)}-{max(counts)}], "
                      f"{n_capped} zones capped")

        # ── Handle unclaimed wards with clients → create new zones ──
        unclaimed_client_wards = unclaimed & self.wards_with_clients
        if unclaimed_client_wards:
            print(f"\n  {len(unclaimed_client_wards)} client wards unreached "
                  f"— creating additional zones...")

            # Group unclaimed client wards by adjacency
            new_zone_groups = self._group_unclaimed_wards(unclaimed_client_wards)

            for group in new_zone_groups:
                new_zone_id = len(self.zone_wards)
                self.zone_wards[new_zone_id] = list(group)
                zone_client_count[new_zone_id] = 0
                zone_capped[new_zone_id] = False

                for pcode in group:
                    self.ward_zones[pcode] = new_zone_id
                    unclaimed.discard(pcode)
                    zone_client_count[new_zone_id] += len(
                        self.ward_clients.get(pcode, []))

                print(f"    New zone {new_zone_id + 1}: "
                      f"{len(group)} wards, "
                      f"{zone_client_count[new_zone_id]} clients")

        self.actual_n_zones = len(self.zone_wards)

        # Summary
        counts = [zone_client_count[z] for z in range(self.actual_n_zones)]
        print(f"\n  Phase 2 complete:")
        print(f"  Total zones: {self.actual_n_zones} "
              f"(started with {self.n_zones} seeds)")
        print(f"  Zone client counts: {sorted(counts)}")
        print(f"  Range: [{min(counts)}, {max(counts)}]")
        print(f"  Std: {np.std(counts):.1f}")

    def _group_unclaimed_wards(self, unclaimed_wards: Set[str]) -> List[Set[str]]:
        """
        Group unclaimed client wards into connected components
        based on adjacency.
        """
        remaining = set(unclaimed_wards)
        groups = []

        while remaining:
            # BFS from an arbitrary starting ward
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

                # Add adjacent unclaimed client wards
                for neighbor in self.adjacency.get(current, {}):
                    if neighbor in remaining:
                        queue.append(neighbor)

            groups.append(group)

        return groups

    # ══════════════════════════════════════════════
    # PHASE 4: BALANCE — Transfer boundary wards from biggest to smallest neighbor
    # ══════════════════════════════════════════════

    def _get_zone_client_count(self, zone_id: int) -> int:
        """Get total client count for a zone."""
        return sum(len(self.ward_clients.get(p, []))
                   for p in self.zone_wards.get(zone_id, [])
                   if p in self.wards_with_clients)

    def _get_zone_neighbor_zones(self, zone_id: int) -> Set[int]:
        """Find zones that share a boundary ward with this zone."""
        neighbor_zones = set()
        for pcode in self.zone_wards.get(zone_id, []):
            for neighbor_pcode in self.adjacency.get(pcode, {}):
                if neighbor_pcode in self.ward_zones:
                    other_zone = self.ward_zones[neighbor_pcode]
                    if other_zone != zone_id:
                        neighbor_zones.add(other_zone)
        return neighbor_zones

    def _get_boundary_wards(self, from_zone: int, to_zone: int) -> List[str]:
        """
        Get wards in from_zone that are on the boundary with to_zone.
        A ward is on the boundary if at least one of its adjacent wards
        belongs to to_zone.

        Returns list of pcodes sorted by client count (smallest first).
        """
        boundary = []
        for pcode in self.zone_wards.get(from_zone, []):
            # Only consider wards with clients
            if pcode not in self.wards_with_clients:
                continue

            # Check if any neighbor belongs to to_zone
            for neighbor_pcode in self.adjacency.get(pcode, {}):
                if neighbor_pcode in self.ward_zones:
                    if self.ward_zones[neighbor_pcode] == to_zone:
                        boundary.append(pcode)
                        break

        # Sort by client count, smallest first
        boundary.sort(key=lambda p: len(self.ward_clients.get(p, [])))
        return boundary

    def _would_disconnect_zone(self, zone_id: int, ward_to_remove: str) -> bool:
        """
        Check if removing a ward from a zone would split it into
        disconnected pieces.
        """
        remaining_wards = [p for p in self.zone_wards[zone_id]
                          if p != ward_to_remove and p in self.wards_with_clients]

        if len(remaining_wards) <= 1:
            return False  # Can't disconnect 0 or 1 ward

        # BFS from the first remaining ward
        visited = set()
        queue = [remaining_wards[0]]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.adjacency.get(current, {}):
                if neighbor in remaining_wards and neighbor not in visited:
                    queue.append(neighbor)

        # If all remaining wards are visited, zone stays connected
        return len(visited) < len(remaining_wards)

    def _balance(self, max_iterations: int = 200):
        """
        Balance zone sizes by transferring boundary wards from
        the biggest zone to its smallest neighbor.

        Rules:
        - Sort zones by client count, highest first
        - Take the biggest zone
        - Find its neighbor zones
        - Transfer the smallest boundary ward to the smallest neighbor
        - No transfer back: the receiver cannot transfer back to the giver
          in the next step
        - Repeat until biggest zone is within max or no more transfers possible
        """
        print(f"\nPhase 4 (Balance): Ward transfer balancing")
        print(f"  Target range: [{self.min_clients}, {self.max_clients}]")

        # Track "no transfer back" pairs: (from_zone, to_zone) that are blocked
        blocked_transfers = set()  # (giver_zone, receiver_zone) -> receiver can't give back

        for iteration in range(max_iterations):
            # Get current zone sizes
            zone_sizes = {}
            for z in range(self.actual_n_zones):
                zone_sizes[z] = self._get_zone_client_count(z)

            # Sort by size, biggest first
            sorted_zones = sorted(zone_sizes.keys(), key=lambda z: -zone_sizes[z])
            biggest_zone = sorted_zones[0]
            biggest_size = zone_sizes[biggest_zone]

            # Check if we're done
            if biggest_size <= self.max_clients:
                print(f"  Converged at iteration {iteration + 1}: "
                      f"max zone size {biggest_size} within limit {self.max_clients}")
                break

            # Find neighbor zones of the biggest zone
            neighbor_zones = self._get_zone_neighbor_zones(biggest_zone)

            # Remove blocked neighbors (no transfer back rule)
            allowed_neighbors = set()
            for nz in neighbor_zones:
                if (biggest_zone, nz) not in blocked_transfers:
                    allowed_neighbors.add(nz)

            if not allowed_neighbors:
                # Try next biggest zone
                transferred = False
                for zone_candidate in sorted_zones[1:]:
                    if zone_sizes[zone_candidate] <= self.max_clients:
                        break  # No more oversized zones

                    cand_neighbors = self._get_zone_neighbor_zones(zone_candidate)
                    cand_allowed = {nz for nz in cand_neighbors
                                    if (zone_candidate, nz) not in blocked_transfers}

                    if not cand_allowed:
                        continue

                    # Use this zone instead
                    biggest_zone = zone_candidate
                    biggest_size = zone_sizes[zone_candidate]
                    allowed_neighbors = cand_allowed
                    transferred = True
                    break

                if not transferred:
                    print(f"  No more transfers possible at iteration {iteration + 1}")
                    break

            # Find the smallest allowed neighbor
            smallest_neighbor = min(allowed_neighbors, key=lambda z: zone_sizes[z])

            # Find boundary wards (smallest first)
            boundary_wards = self._get_boundary_wards(biggest_zone, smallest_neighbor)

            if not boundary_wards:
                # Block this pair and continue
                blocked_transfers.add((smallest_neighbor, biggest_zone))
                continue

            # Try to transfer the smallest boundary ward
            transferred = False
            for ward_to_transfer in boundary_wards:
                # Check if removing this ward would disconnect the zone
                if self._would_disconnect_zone(biggest_zone, ward_to_transfer):
                    continue

                # Transfer the ward
                self.zone_wards[biggest_zone].remove(ward_to_transfer)
                self.zone_wards[smallest_neighbor].append(ward_to_transfer)
                self.ward_zones[ward_to_transfer] = smallest_neighbor

                # Block reverse transfer
                blocked_transfers.add((smallest_neighbor, biggest_zone))

                ward_clients = len(self.ward_clients.get(ward_to_transfer, []))
                new_giver_size = biggest_size - ward_clients
                new_receiver_size = zone_sizes[smallest_neighbor] + ward_clients

                print(f"  Iter {iteration + 1}: Zone {biggest_zone + 1} "
                      f"({biggest_size}) → ward ({ward_clients} clients) → "
                      f"Zone {smallest_neighbor + 1} ({zone_sizes[smallest_neighbor]}). "
                      f"New sizes: {new_giver_size}, {new_receiver_size}")

                transferred = True
                break

            if not transferred:
                # Couldn't transfer any ward without disconnecting
                blocked_transfers.add((smallest_neighbor, biggest_zone))
                continue

        # Final summary
        final_sizes = [self._get_zone_client_count(z)
                       for z in range(self.actual_n_zones)]
        print(f"\n  Phase 4 complete:")
        print(f"  Zone client counts: {sorted(final_sizes)}")
        print(f"  Range: [{min(final_sizes)}, {max(final_sizes)}]")
        print(f"  Std: {np.std(final_sizes):.1f}")

    # ══════════════════════════════════════════════
    # PHASE 3: ASSIGN — Map zones back to clients
    # ══════════════════════════════════════════════

    def _assign_clients(self):
        """Map zone labels to individual clients."""
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
                # Client not in any zone — assign to nearest zone by road distance
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
        print(f"\nPhase 3 (Assign): Mapped zones to {assigned}/{n_clients} clients")

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def assign_zones(self) -> np.ndarray:
        """Run the full Seed → Grow → Assign pipeline."""
        print(f"\n{'=' * 60}")
        print(f"WARD ZONE ASSIGNMENT v2")
        print(f"{'=' * 60}")

        # Phase 1
        self.seeds = self._select_seeds()

        # Phase 2
        self._grow(self.seeds)

        # Phase 4 (before client assignment)
        self._balance()

        # Phase 3
        self._assign_clients()

        # Final report
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULTS:")
        print(f"  Total zones: {self.actual_n_zones}")
        for zone_id in range(self.actual_n_zones):
            n_wards = len([w for w in self.zone_wards.get(zone_id, [])
                          if w in self.wards_with_clients])
            n_clients = len(self.zone_clients.get(zone_id, []))
            client_indices = self.zone_clients.get(zone_id, [])
            if client_indices:
                avg_dist = np.mean([self.road_matrix[0, idx + 1]
                                    for idx in client_indices])
            else:
                avg_dist = 0
            is_new = zone_id >= self.n_zones
            marker = " (NEW)" if is_new else ""
            print(f"  Zone {zone_id + 1}{marker}: {n_wards} client wards, "
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
        """Folium map showing ward zones."""
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

        # Zone layers
        zone_layers = {}
        for z in range(self.actual_n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            is_new = z >= self.n_zones
            label = f"Zone {z + 1} ({n_clients} clients)"
            if is_new:
                label += " NEW"
            zone_layers[z] = folium.FeatureGroup(name=label)

        # Ward polygons (only wards with clients)
        for ward in self.mapper.wards:
            pcode = ward['pcode']

            # Only show wards that are in a zone AND have clients
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

        # Seed markers
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
