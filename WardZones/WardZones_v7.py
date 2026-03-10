"""
Ward Zone Assignment (v7)

Improvements over v6:
  - Phase 2: Density-aware seed selection (covers dense + sparse regions)
  - Phase 3: Size-penalized growth (oversized zones become less greedy)
  - Phase 4: Enhanced balancing (chain transfers, pull-toward-undersized, cooldown blocking)
  - Phase 5: Simulated annealing post-processing (ward-level swaps to optimize balance)

Phase 1 (Adjacency) — BFS from each client ward through all wards
Phase 2 (Seed)      — Density-aware spread: pick seeds covering different density pockets
Phase 3 (Grow)      — Smallest zone picks next, with size penalty on oversized zones
Phase 4 (Balance)   — Chain transfers + pull-toward-undersized + cooldown blocking
Phase 5 (Anneal)    — Simulated annealing ward swaps for final polish
Phase 6 (Assign)    — Map zone labels back to individual clients

Usage:
    from WardZones_v7 import WardZoneAssignment

    zoner = WardZoneAssignment(mapper, adjacency_builder, locations, road_matrix,
                                n_zones=14, empty_connecting_ward_allowance=0)
    zoner.assign_zones()
    zoner.visualize_zones(filename="Output/ward_zones_v7.html")
"""

import json
import math
import random
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
    v7: Density-aware seeds, size-penalized growth, enhanced balancing, simulated annealing.
    """

    def __init__(self, mapper: WardMapper, adjacency_builder: WardAdjacency,
                 locations: List[Location], road_matrix: np.ndarray,
                 n_zones: int = 14, size_tolerance: float = 0.20,
                 empty_connecting_ward_allowance: int = 0,
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

        print(f"WardZoneAssignment v7 initialized:")
        print(f"  Wards with clients: {len(self.wards_with_clients)}")
        print(f"  Total wards in adjacency: {len(self.full_adjacency)}")
        print(f"  Total clients: {total_clients}")
        print(f"  Zones: {n_zones}")
        print(f"  Target clients per zone: {self.target_clients:.1f}")
        print(f"  Allowed range: [{self.min_clients}, {self.max_clients}]")
        print(f"  Empty ward allowance: {self.empty_allowance}")

    # ══════════════════════════════════════════════
    # PHASE 1: BUILD CLIENT WARD ADJACENCY (same as v6)
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
            queue = deque()
            visited = set()
            visited.add(source_pcode)

            for neighbor in self.full_adjacency.get(source_pcode, {}):
                if neighbor not in visited:
                    if neighbor in self.wards_with_clients:
                        self.client_ward_adjacency[source_pcode].add(neighbor)
                        self.client_ward_adjacency[neighbor].add(source_pcode)
                        visited.add(neighbor)
                    else:
                        if self.empty_allowance >= 1:
                            queue.append((neighbor, 1))
                            visited.add(neighbor)

            while queue:
                current_pcode, empty_count = queue.popleft()

                for neighbor in self.full_adjacency.get(current_pcode, {}):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    if neighbor in self.wards_with_clients:
                        self.client_ward_adjacency[source_pcode].add(neighbor)
                        self.client_ward_adjacency[neighbor].add(source_pcode)
                    else:
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
    # PHASE 2: DENSITY-AWARE SEED SELECTION (v7 improved)
    # ══════════════════════════════════════════════

    def _ward_road_distance(self, pcode_a: str, pcode_b: str) -> float:
        """Min road distance between any client pair in two wards."""
        clients_a = self.ward_clients.get(pcode_a, [])
        clients_b = self.ward_clients.get(pcode_b, [])
        if not clients_a or not clients_b:
            return float('inf')
        return min(
            self.road_matrix[ca + 1, cb + 1]
            for ca in clients_a for cb in clients_b
        )

    def _get_ward_client_count(self, pcode: str) -> int:
        """Number of clients in a ward."""
        return len(self.ward_clients.get(pcode, []))

    def _compute_density_potential(self, pcode: str, radius_km: float = None) -> int:
        """
        Count clients reachable from this ward within radius_km via road distance.
        If radius_km is None, uses mean client-office distance.
        """
        if radius_km is None:
            all_client_dists = [self.road_matrix[0, idx + 1]
                                for idx in range(len(self.clients))]
            radius_km = np.mean(all_client_dists) / np.sqrt(self.n_zones)

        total = self._get_ward_client_count(pcode)
        for neighbor_pcode in self.wards_with_clients:
            if neighbor_pcode == pcode:
                continue
            dist = self._ward_road_distance(pcode, neighbor_pcode)
            if dist <= radius_km:
                total += self._get_ward_client_count(neighbor_pcode)
        return total

    def _select_seeds(self) -> List[str]:
        """
        Density-aware seed selection.
        1. Compute density potential for each candidate ward.
        2. Pick first seed = ward with highest density potential.
        3. Each next seed maximizes: min_distance_to_existing_seeds * density_potential.
           This spreads seeds across both dense and sparse areas.
        """
        candidate_wards = list(self.wards_with_clients)

        if len(candidate_wards) <= self.n_zones:
            return candidate_wards

        print(f"\n  Computing density potential for {len(candidate_wards)} candidate wards...")

        # Compute density potential for each ward
        all_client_dists = [self.road_matrix[0, idx + 1]
                            for idx in range(len(self.clients))]
        density_radius = np.mean(all_client_dists) / np.sqrt(self.n_zones)
        print(f"  Density radius: {density_radius:.1f} km")

        density = {}
        for pcode in candidate_wards:
            density[pcode] = self._compute_density_potential(pcode, density_radius)

        # Pick first seed: ward with highest density potential
        seeds = []
        seed_set = set()
        remaining = set(candidate_wards)

        first_seed = max(remaining, key=lambda p: density[p])
        seeds.append(first_seed)
        seed_set.add(first_seed)
        remaining.discard(first_seed)
        seed_name = self.ward_lookup[first_seed]['name'] if first_seed in self.ward_lookup else first_seed
        centroid = self.ward_lookup[first_seed]['geometry'].centroid if first_seed in self.ward_lookup else None
        lat, lon = (centroid.y, centroid.x) if centroid else (0, 0)
        print(f"  Seed 1: density={density[first_seed]}, name={seed_name}, pcode={first_seed}, lat,lon ={lat:.6f},{lon:.6f}")

        # Pick subsequent seeds: maximize min_dist_to_seeds × density
        while len(seeds) < self.n_zones and remaining:
            best_pcode = None
            best_score = -1

            for pcode in remaining:
                # Min road distance to any existing seed
                min_dist = min(
                    self._ward_road_distance(pcode, s) for s in seeds
                )
                if min_dist == float('inf'):
                    min_dist = 0  # isolated ward, low priority

                # Score = distance × density (both matter)
                score = min_dist * density[pcode]

                if score > best_score:
                    best_score = score
                    best_pcode = pcode

            if best_pcode is None:
                break

            seeds.append(best_pcode)
            seed_set.add(best_pcode)
            remaining.discard(best_pcode)

            min_d = min(self._ward_road_distance(best_pcode, s)
                        for s in seeds if s != best_pcode)
            best_seed_name = self.ward_lookup[best_pcode]['name'] if best_pcode in self.ward_lookup else best_pcode
            centroid = self.ward_lookup[best_pcode]['geometry'].centroid if best_pcode in self.ward_lookup else None
            lat, lon = (centroid.y, centroid.x) if centroid else (0, 0)
            print(f"  Seed {len(seeds)}: density={density[best_pcode]}, name={best_seed_name}, pcode={best_pcode}, lat, lon ={lat:.6f},{lon:.6f}, min_dist_to_seeds={min_d:.1f} km")

        # Fill remaining if needed (isolated wards)
        if len(seeds) < self.n_zones:
            print(f"    Got {len(seeds)} seeds, filling remaining "
                  f"{self.n_zones - len(seeds)} from leftover wards...")
            for pcode in candidate_wards:
                if len(seeds) >= self.n_zones:
                    break
                if pcode not in seed_set:
                    seeds.append(pcode)
                    seed_set.add(pcode)

        # Print seed spacing details
        print(f"\n  Seed spacing details:")
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                dist = self._ward_road_distance(seeds[i], seeds[j])
                print(f"    Seed {i+1} la ↔ Seed {j+1}: {dist:.1f} km")

        print(f"\nPhase 2 (Seed): Selected {len(seeds)} seed wards (density-aware spread)")
        return seeds

    # ══════════════════════════════════════════════
    # PHASE 3: SIZE-PENALIZED GROWTH 
    # ══════════════════════════════════════════════

    def _grow(self, seeds: List[str]):
        """
        Grow zones using pre-computed client ward adjacency.
        v7 improvement: quadratic size penalty discourages oversized zones
        from claiming the closest ward, letting smaller zones catch up.
        """
        self.zone_wards = {i: [seed] for i, seed in enumerate(seeds)}
        self.ward_zones = {seed: i for i, seed in enumerate(seeds)}

        zone_client_count = {}
        for zone_id, wards in self.zone_wards.items():
            count = sum(len(self.ward_clients.get(p, [])) for p in wards)
            zone_client_count[zone_id] = count

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
        print(f"  Size penalty: quadratic when zone exceeds target ({self.target_clients:.0f})")

        iteration = 0
        while unclaimed:
            iteration += 1

            # Find smallest zone with frontier
            best_zone = None
            min_count = float('inf')
            for zone_id in range(len(seeds)):
                if not frontiers[zone_id]:
                    continue
                if zone_client_count[zone_id] < min_count:
                    min_count = zone_client_count[zone_id]
                    best_zone = zone_id

            if best_zone is None:
                break

            # Pick the best frontier ward with size penalty
            frontier = frontiers[best_zone]
            best_ward = None
            best_score = float('inf')

            zone_members = self.zone_wards[best_zone]
            zone_ratio = zone_client_count[best_zone] / self.target_clients

            # Quadratic penalty: kicks in when zone exceeds target
            size_penalty = max(1.0, zone_ratio ** 2)

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

                # v7: Apply size penalty — oversized zones see inflated distances
                score = avg_dist * size_penalty

                if score < best_score:
                    best_score = score
                    best_ward = candidate

            if best_ward is None:
                break

            # Claim
            self.zone_wards[best_zone].append(best_ward)
            self.ward_zones[best_ward] = best_zone
            unclaimed.discard(best_ward)

            new_clients = len(self.ward_clients.get(best_ward, []))
            zone_client_count[best_zone] += new_clients

            # Remove from all frontiers
            for zone_id in range(len(seeds)):
                frontiers[zone_id].discard(best_ward)

            # Add new frontier from claimed ward
            for neighbor in self.client_ward_adjacency.get(best_ward, set()):
                if neighbor in unclaimed:
                    frontiers[best_zone].add(neighbor)

            if iteration % 20 == 0:
                counts = sorted(zone_client_count.values())
                print(f"    Iteration {iteration}: {len(unclaimed)} unclaimed, "
                      f"sizes: [{min(counts)}-{max(counts)}]")

        # Handle unclaimed client wards → new zones
        if unclaimed:
            print(f"\n  {len(unclaimed)} client wards unreached "
                  f"— creating additional zones...")

            new_zone_groups = self._group_unclaimed_wards(unclaimed)

            for group in new_zone_groups:
                new_zone_id = len(self.zone_wards)
                self.zone_wards[new_zone_id] = list(group)
                zone_client_count[new_zone_id] = 0

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
    # PHASE 4: ENHANCED BALANCING (v7 improved)
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

    def _find_zone_path(self, from_zone: int, to_zone: int) -> Optional[List[int]]:
        """BFS to find shortest path of zone neighbors from from_zone to to_zone."""
        if from_zone == to_zone:
            return [from_zone]

        visited = {from_zone}
        queue = deque([(from_zone, [from_zone])])

        while queue:
            current, path = queue.popleft()
            for neighbor_zone in self._get_zone_neighbor_zones(current):
                if neighbor_zone == to_zone:
                    return path + [to_zone]
                if neighbor_zone not in visited:
                    visited.add(neighbor_zone)
                    queue.append((neighbor_zone, path + [neighbor_zone]))

        return None  # No path found

    def _try_transfer(self, from_zone: int, to_zone: int,
                      blocked_transfers: Dict[Tuple[int, int], int],
                      current_iter: int, cooldown: int = 10) -> bool:
        """
        Try to transfer a boundary ward from from_zone to to_zone.
        Returns True if transfer succeeded.
        Uses cooldown-based blocking instead of permanent blocking.
        """
        # Check cooldown
        key = (from_zone, to_zone)
        if key in blocked_transfers:
            if current_iter - blocked_transfers[key] < cooldown:
                return False

        boundary_wards = self._get_boundary_wards(from_zone, to_zone)
        if not boundary_wards:
            return False

        for ward_to_transfer in boundary_wards:
            if self._would_disconnect_zone(from_zone, ward_to_transfer):
                continue

            # Don't transfer if it would make from_zone undersized
            from_count = self._get_zone_client_count(from_zone)
            ward_clients = len(self.ward_clients.get(ward_to_transfer, []))
            if from_count - ward_clients < self.min_clients:
                continue

            # Transfer
            self.zone_wards[from_zone].remove(ward_to_transfer)
            self.zone_wards[to_zone].append(ward_to_transfer)
            self.ward_zones[ward_to_transfer] = to_zone

            # Block reverse with cooldown
            blocked_transfers[(to_zone, from_zone)] = current_iter

            return True

        return False

    def _balance(self, max_iterations: int = 500):
        """
        Enhanced balancing with three improvements
        1. Chain transfers: cascade along zone paths if direct transfer not possible
        2. Pull toward undersized: smallest zone pulls from largest neighbor
        3. Cooldown blocking: reverse transfers blocked for 10 iterations, not permanently
        """
        print(f"\nPhase 4 (Balance): Enhanced ward transfer balancing")
        print(f"  Target range: [{self.min_clients}, {self.max_clients}]")

        blocked_transfers = {}  # (from, to) -> iteration when blocked
        cooldown = 10

        transfers_made = 0

        for iteration in range(max_iterations):
            zone_sizes = {z: self._get_zone_client_count(z)
                          for z in range(self.actual_n_zones)}

            sorted_zones = sorted(zone_sizes.keys(), key=lambda z: -zone_sizes[z])
            biggest_zone = sorted_zones[0]
            biggest_size = zone_sizes[biggest_zone]
            smallest_zone = sorted_zones[-1]
            smallest_size = zone_sizes[smallest_zone]

            # Check convergence
            if biggest_size <= self.max_clients and smallest_size >= self.min_clients:
                print(f"  Converged at iteration {iteration + 1}: "
                      f"range [{smallest_size}, {biggest_size}] within "
                      f"[{self.min_clients}, {self.max_clients}]")
                break

            transferred = False

            # Strategy A: Push from oversized zones
            for zone_candidate in sorted_zones:
                if zone_sizes[zone_candidate] <= self.max_clients:
                    break

                # Try direct transfer to smallest neighbor
                neighbor_zones = self._get_zone_neighbor_zones(zone_candidate)
                if neighbor_zones:
                    smallest_neighbor = min(neighbor_zones,
                                            key=lambda z: zone_sizes[z])

                    if self._try_transfer(zone_candidate, smallest_neighbor,
                                          blocked_transfers, iteration, cooldown):
                        ward_transferred = self.zone_wards[smallest_neighbor][-1]
                        wc = len(self.ward_clients.get(ward_transferred, []))
                        print(f"  Iter {iteration+1} [PUSH]: Zone {zone_candidate+1} "
                              f"({zone_sizes[zone_candidate]}) → "
                              f"ward ({wc} clients) → "
                              f"Zone {smallest_neighbor+1} ({zone_sizes[smallest_neighbor]})")
                        transferred = True
                        transfers_made += 1
                        break

                # Try chain transfer if direct didn't work
                # Find path from biggest to smallest zone
                path = self._find_zone_path(zone_candidate, smallest_zone)
                if path and len(path) >= 3:
                    # Try cascading: transfer along the path
                    # Transfer from zone_candidate to first neighbor on path
                    next_zone = path[1]
                    if self._try_transfer(zone_candidate, next_zone,
                                          blocked_transfers, iteration, cooldown):
                        ward_transferred = self.zone_wards[next_zone][-1]
                        wc = len(self.ward_clients.get(ward_transferred, []))
                        print(f"  Iter {iteration+1} [CHAIN]: Zone {zone_candidate+1} "
                              f"({zone_sizes[zone_candidate]}) → "
                              f"ward ({wc} clients) → "
                              f"Zone {next_zone+1} ({zone_sizes[next_zone]}) "
                              f"[path to Zone {smallest_zone+1}]")
                        transferred = True
                        transfers_made += 1
                        break

            # Strategy B: Pull toward undersized zones (if no push happened)
            if not transferred:
                for zone_candidate in reversed(sorted_zones):
                    if zone_sizes[zone_candidate] >= self.min_clients:
                        break

                    neighbor_zones = self._get_zone_neighbor_zones(zone_candidate)
                    if not neighbor_zones:
                        continue

                    # Pull from largest neighbor
                    largest_neighbor = max(neighbor_zones,
                                           key=lambda z: zone_sizes[z])

                    # Only pull if the neighbor is bigger than us
                    if zone_sizes[largest_neighbor] <= zone_sizes[zone_candidate]:
                        continue

                    if self._try_transfer(largest_neighbor, zone_candidate,
                                          blocked_transfers, iteration, cooldown):
                        ward_transferred = self.zone_wards[zone_candidate][-1]
                        wc = len(self.ward_clients.get(ward_transferred, []))
                        print(f"  Iter {iteration+1} [PULL]: Zone {largest_neighbor+1} "
                              f"({zone_sizes[largest_neighbor]}) → "
                              f"ward ({wc} clients) → "
                              f"Zone {zone_candidate+1} ({zone_sizes[zone_candidate]})")
                        transferred = True
                        transfers_made += 1
                        break

            if not transferred:
                print(f"  No more transfers possible at iteration {iteration + 1}")
                break

        final_sizes = [self._get_zone_client_count(z)
                       for z in range(self.actual_n_zones)]
        print(f"\n  Phase 4 complete ({transfers_made} transfers):")
        print(f"  Zone client counts: {sorted(final_sizes)}")
        print(f"  Range: [{min(final_sizes)}, {max(final_sizes)}]")
        print(f"  Std: {np.std(final_sizes):.1f}")

    # ══════════════════════════════════════════════
    # PHASE 5: SIMULATED ANNEALING (v7 new)
    # ══════════════════════════════════════════════

    def _compute_balance_score(self) -> float:
        """
        Combined score: lower is better.
        balance_cost = std deviation of zone sizes (measures imbalance)
        compactness_cost = avg within-zone road distance (measures geographic spread)
        """
        zone_sizes = [self._get_zone_client_count(z)
                      for z in range(self.actual_n_zones)]
        balance_cost = np.std(zone_sizes)

        # Compactness: average road distance within each zone
        compactness_costs = []
        for z in range(self.actual_n_zones):
            zone_client_indices = []
            for pcode in self.zone_wards.get(z, []):
                zone_client_indices.extend(self.ward_clients.get(pcode, []))
            if len(zone_client_indices) < 2:
                continue
            # Sample to keep it fast
            if len(zone_client_indices) > 30:
                sample = random.sample(zone_client_indices, 30)
            else:
                sample = zone_client_indices
            dists = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dists.append(self.road_matrix[sample[i] + 1, sample[j] + 1])
            if dists:
                compactness_costs.append(np.mean(dists))

        compactness_cost = np.mean(compactness_costs) if compactness_costs else 0

        # Weight balance more heavily than compactness
        return balance_cost * 10 + compactness_cost

    def _anneal(self, max_iterations: int = 100000, initial_temp: float = 10.0,
                final_temp: float = 0.01):
        """
        Simulated annealing: randomly propose ward swaps between neighboring zones.
        Accept if improves score, or with decreasing probability if worse.
        """
        print(f"\nPhase 5 (Anneal): Simulated annealing post-processing")
        print(f"  Iterations: {max_iterations}, temp: {initial_temp} → {final_temp}")

        initial_score = self._compute_balance_score()
        best_score = initial_score
        current_score = initial_score

        # Save best state
        best_ward_zones = dict(self.ward_zones)
        best_zone_wards = {z: list(ws) for z, ws in self.zone_wards.items()}

        accepted = 0
        improved = 0

        # Collect all boundary wards (wards adjacent to a different zone)
        def get_boundary_candidates():
            candidates = []
            for pcode in self.ward_zones:
                zone_id = self.ward_zones[pcode]
                for neighbor in self.client_ward_adjacency.get(pcode, set()):
                    if neighbor in self.ward_zones:
                        other_zone = self.ward_zones[neighbor]
                        if other_zone != zone_id:
                            candidates.append((pcode, zone_id, other_zone))
                            break
            return candidates

        for iteration in range(max_iterations):
            temp = initial_temp * (1 - iteration / max_iterations) + final_temp

            # Get current boundary candidates
            candidates = get_boundary_candidates()
            if not candidates:
                break

            # Pick a random boundary ward
            pcode, from_zone, to_zone = random.choice(candidates)

            # Check: would this disconnect the source zone?
            if self._would_disconnect_zone(from_zone, pcode):
                continue

            # Check: source zone must keep at least 1 ward
            if len(self.zone_wards[from_zone]) <= 1:
                continue

            # Perform swap
            self.zone_wards[from_zone].remove(pcode)
            self.zone_wards[to_zone].append(pcode)
            self.ward_zones[pcode] = to_zone

            new_score = self._compute_balance_score()
            delta = new_score - current_score

            # Accept or reject
            if delta < 0:
                # Better — always accept
                current_score = new_score
                accepted += 1
                if new_score < best_score:
                    best_score = new_score
                    best_ward_zones = dict(self.ward_zones)
                    best_zone_wards = {z: list(ws) for z, ws in self.zone_wards.items()}
                    improved += 1
            elif temp > 0 and random.random() < math.exp(-delta / temp):
                # Worse but accepts with probability
                current_score = new_score
                accepted += 1
            else:
                # Reject — revert
                self.zone_wards[to_zone].remove(pcode)
                self.zone_wards[from_zone].append(pcode)
                self.ward_zones[pcode] = from_zone

            if (iteration + 1) % 1000 == 0:
                sizes = [self._get_zone_client_count(z)
                         for z in range(self.actual_n_zones)]
                print(f"    Iter {iteration+1}: score={current_score:.1f}, "
                      f"best={best_score:.1f}, "
                      f"range=[{min(sizes)}, {max(sizes)}], "
                      f"temp={temp:.2f}")

        # Restore best state
        self.ward_zones = best_ward_zones
        self.zone_wards = best_zone_wards

        final_sizes = [self._get_zone_client_count(z)
                       for z in range(self.actual_n_zones)]
        print(f"\n  Phase 5 complete:")
        print(f"  Score: {initial_score:.1f} → {best_score:.1f}")
        print(f"  Accepted: {accepted}/{max_iterations}, "
              f"improved: {improved}")
        print(f"  Zone client counts: {sorted(final_sizes)}")
        print(f"  Range: [{min(final_sizes)}, {max(final_sizes)}]")
        print(f"  Std: {np.std(final_sizes):.1f}")

    # ══════════════════════════════════════════════
    # PHASE 6: ASSIGN — Map zones back to clients
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
        print(f"\nPhase 6 (Assign): Mapped zones to {assigned}/{n_clients} clients")

    # ══════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════

    def assign_zones(self) -> np.ndarray:
        print(f"\n{'=' * 60}")
        print(f"WARD ZONE ASSIGNMENT v7 (Density-Aware + SA)")
        print(f"{'=' * 60}")

        # Phase 1
        self._build_client_ward_adjacency()

        # Phase 2 — Density-aware seeds
        self.seeds = self._select_seeds()

        # Phase 3 — Size-penalized growth
        self._grow(self.seeds)

        # Phase 4 — Enhanced balancing
        self._balance()

        # Phase 5 — Simulated annealing
        self._anneal()

        # Phase 6 — Assign clients
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
    # VISUALIZATION (same as v6)
    # ══════════════════════════════════════════════

    def visualize_zones(self, filename: str = "Output/ward_zones_v7.html"):
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
                    tooltip=f"Seed {zone_id + 1}",
                    icon=folium.Icon(color='black', icon='star', prefix='fa'),
                ).add_to(seed_layer)
        seed_layer.add_to(m)

        # Office
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="<b>Office</b>",
            tooltip="Office",
            icon=folium.Icon(color='red', icon='building', prefix='fa'),
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # Legend
        legend_items = {}
        for z in range(self.actual_n_zones):
            n_clients = len(self.zone_clients.get(z, []))
            is_new = z >= self.n_zones
            label = f"Zone {z + 1}: {n_clients} clients"
            if is_new:
                label += " (NEW)"
            legend_items[label] = zone_colors[z]

        legend_html = self._build_legend("Zone Legend", legend_items)
        m.get_root().html.add_child(folium.Element(legend_html))

        m.save(filename)
        print(f"Map saved to: {filename}")

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
