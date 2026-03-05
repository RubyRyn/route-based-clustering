"""
Ward Visualization (Folium)

Interactive map with three coloring modes:
- color_by_client_count: darker = more clients
- color_by_township: all wards in same township share a color
- color_by_distance: closer to office = warmer, further = cooler

Usage:
    from ward_mapping import WardMapper
    from ward_visualization import WardVisualizer

    mapper = WardMapper(geojson_path="path/to/wards.geojson")
    mapper.map_clients(locations)
    mapper.fix_unmapped_clients(locations)

    viz = WardVisualizer(mapper, locations)
    viz.color_by_client_count(filename="Output/ward_by_count.html")
    viz.color_by_township(filename="Output/ward_by_township.html")
    viz.color_by_distance(filename="Output/ward_by_distance.html")
"""

import json
import numpy as np
import colorsys
from typing import List, Dict, Optional
import folium
import branca.colormap as cm
from ClientWardsMapping import WardMapper
from location import Location


class WardVisualizer:
    """
    Interactive Folium map visualizer for ward-client mapping.
    Supports multiple coloring modes.
    """

    def __init__(self, mapper: WardMapper, locations: List[Location],
                 skip_office: bool = True):
        """
        Args:
            mapper: WardMapper that has already run map_clients().
            locations: List of Location objects (office first, then clients).
            skip_office: If True, locations[0] is the office.
        """
        self.mapper = mapper
        self.locations = locations
        self.office = locations[0]
        self.clients = locations[1:] if skip_office else locations

        # Precompute useful lookups
        self.ward_clients = mapper.get_ward_client_map()
        self.client_ward_labels = mapper.get_client_ward_labels()
        self.active_pcodes = set(self.ward_clients.keys())

    # ══════════════════════════════════════════════
    # COLOR BY CLIENT COUNT
    # ══════════════════════════════════════════════

    def color_by_client_count(self, filename: str = "Output/ward_by_count.html"):
        """
        Color wards by number of clients. Darker = more clients.
        """
        print(f"\nGenerating map: Color by Client Count")

        # Compute client counts per ward
        ward_counts = {}
        for pcode in self.active_pcodes:
            ward_counts[pcode] = len(self.ward_clients[pcode])

        max_count = max(ward_counts.values()) if ward_counts else 1

        # Color scale
        colormap = cm.LinearColormap(
            colors=['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
            vmin=0,
            vmax=max_count,
            caption='Clients per Ward'
        )

        def get_ward_color(pcode):
            count = ward_counts.get(pcode, 0)
            return colormap(count)

        def get_ward_popup_extra(pcode):
            count = ward_counts.get(pcode, 0)
            return f"<b>Clients: {count}</b>"

        m = self._build_map(
            get_ward_color=get_ward_color,
            get_client_color=lambda idx: '#333333',  # Dark dots on colored wards
            get_ward_popup_extra=get_ward_popup_extra,
            title="Wards by Client Count"
        )

        colormap.add_to(m)
        self._save(m, filename)
        return m

    # ══════════════════════════════════════════════
    # COLOR BY TOWNSHIP
    # ══════════════════════════════════════════════

    def color_by_township(self, filename: str = "Output/ward_by_township.html"):
        """
        Color wards by their township. All wards in the same township
        share the same color.
        """
        print(f"\nGenerating map: Color by Township")

        # Get unique townships from active wards
        township_set = set()
        ward_township_map = {}
        for ward in self.mapper.wards:
            pcode = ward['pcode']
            ts = ward['township']
            ts_pcode = ward['township_pcode']
            ward_township_map[pcode] = ts_pcode
            if pcode in self.active_pcodes:
                township_set.add((ts_pcode, ts))

        # Also include townships from wards without clients for context
        all_townships = set()
        for ward in self.mapper.wards:
            all_townships.add((ward['township_pcode'], ward['township']))

        # Assign colors to townships
        township_list = sorted(township_set)
        township_colors = self._generate_distinct_colors(len(township_list))
        township_color_map = {}
        for i, (ts_pcode, ts_name) in enumerate(township_list):
            township_color_map[ts_pcode] = township_colors[i]

        # Inactive townships get gray
        inactive_color = '#d3d3d3'

        def get_ward_color(pcode):
            ts_pcode = ward_township_map.get(pcode)
            if ts_pcode and ts_pcode in township_color_map:
                return township_color_map[ts_pcode]
            return inactive_color

        def get_client_color(client_idx):
            pcode = self.client_ward_labels.get(client_idx)
            if pcode:
                ts_pcode = ward_township_map.get(pcode)
                if ts_pcode and ts_pcode in township_color_map:
                    return township_color_map[ts_pcode]
            return '#ff0000'

        def get_ward_popup_extra(pcode):
            count = len(self.ward_clients.get(pcode, []))
            ts_pcode = ward_township_map.get(pcode, '')
            return f"Township Pcode: {ts_pcode}<br><b>Clients: {count}</b>"

        m = self._build_map(
            get_ward_color=get_ward_color,
            get_client_color=get_client_color,
            get_ward_popup_extra=get_ward_popup_extra,
            title="Wards by Township"
        )

        # Add township legend
        legend_html = self._build_legend(
            "Townships",
            {ts_name: township_color_map[ts_pcode]
             for ts_pcode, ts_name in township_list}
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        self._save(m, filename)
        return m

    # ══════════════════════════════════════════════
    # COLOR BY DISTANCE FROM OFFICE
    # ══════════════════════════════════════════════

    def color_by_distance(self, filename: str = "Output/ward_by_distance.html"):
        """
        Color wards by average road distance of their clients from the office.
        Warm (red/orange) = close, Cool (blue/purple) = far.
        """
        print(f"\nGenerating map: Color by Distance from Office")

        # Compute average office distance per ward
        # Use road matrix if available, otherwise fall back to Euclidean
        ward_avg_dist = {}
        for pcode, client_indices in self.ward_clients.items():
            # Euclidean distance from office to ward's clients
            dists = []
            for idx in client_indices:
                loc = self.clients[idx]
                dist = self._haversine(
                    self.office.lat, self.office.lon,
                    loc.lat, loc.lon
                )
                dists.append(dist)
            ward_avg_dist[pcode] = np.mean(dists) if dists else 0

        if ward_avg_dist:
            max_dist = max(ward_avg_dist.values())
            min_dist = min(ward_avg_dist.values())
        else:
            max_dist, min_dist = 1, 0

        # Warm to cool colormap
        colormap = cm.LinearColormap(
            colors=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60',
                    '#1a9850', '#4575b4', '#313695'],
            vmin=min_dist,
            vmax=max_dist,
            caption='Avg Distance from Office (km)'
        )

        def get_ward_color(pcode):
            dist = ward_avg_dist.get(pcode)
            if dist is not None:
                return colormap(dist)
            return '#d3d3d3'

        def get_client_color(client_idx):
            pcode = self.client_ward_labels.get(client_idx)
            if pcode:
                return get_ward_color(pcode)
            return '#ff0000'

        def get_ward_popup_extra(pcode):
            count = len(self.ward_clients.get(pcode, []))
            dist = ward_avg_dist.get(pcode, 0)
            return (f"<b>Clients: {count}</b><br>"
                    f"Avg distance from office: {dist:.1f} km")

        m = self._build_map(
            get_ward_color=get_ward_color,
            get_client_color=get_client_color,
            get_ward_popup_extra=get_ward_popup_extra,
            title="Wards by Distance from Office"
        )

        colormap.add_to(m)
        self._save(m, filename)
        return m

    # ══════════════════════════════════════════════
    # Shared map builder
    # ══════════════════════════════════════════════

    def _build_map(self, get_ward_color, get_client_color,
                   get_ward_popup_extra, title: str = "") -> folium.Map:
        """
        Build the base Folium map with ward polygons, client dots, and office.

        Args:
            get_ward_color: Function(pcode) -> hex color for ward polygon.
            get_client_color: Function(client_idx) -> hex color for client dot.
            get_ward_popup_extra: Function(pcode) -> extra HTML for ward popup.
            title: Map title.
        """
        m = folium.Map(
            location=[self.office.lat, self.office.lon],
            zoom_start=11,
            tiles='CartoDB positron'
        )

        # Title
        if title:
            title_html = f"""
            <div style="position: fixed; top: 10px; left: 50%; 
                        transform: translateX(-50%); z-index: 1000;
                        background: white; padding: 8px 16px; 
                        border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                        font-family: Arial; font-size: 14px; font-weight: bold;">
                {title}
            </div>
            """
            m.get_root().html.add_child(folium.Element(title_html))

        # ── Ward polygons ──
        wards_active = folium.FeatureGroup(name="Wards (with clients)")
        wards_inactive = folium.FeatureGroup(name="Wards (no clients)", show=True)

        for ward in self.mapper.wards:
            pcode = ward['pcode']
            name = ward['name']
            township = ward['township']
            has_clients = pcode in self.active_pcodes

            geojson = json.loads(json.dumps(
                ward['geometry'].__geo_interface__
            ))

            if has_clients:
                fill_color = get_ward_color(pcode)
                fill_opacity = 0.4
                weight = 2
                color = fill_color
                layer = wards_active
                extra_info = get_ward_popup_extra(pcode)
            else:
                fill_color = '#d3d3d3'
                fill_opacity = 0.05
                weight = 0.5
                color = '#999999'
                layer = wards_inactive
                extra_info = "Clients: 0"

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; min-width: 160px;">
                <b>{name}</b><br>
                Township: {township}<br>
                Pcode: {pcode}<br>
                {extra_info}
            </div>
            """

            folium.GeoJson(
                geojson,
                style_function=lambda x, fc=fill_color, fo=fill_opacity,
                    c=color, w=weight: {
                    'fillColor': fc,
                    'fillOpacity': fo,
                    'color': c,
                    'weight': w,
                },
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{name} ({township})",
            ).add_to(layer)

        wards_active.add_to(m)
        wards_inactive.add_to(m)

        # ── Client dots ──
        client_layer = folium.FeatureGroup(name="Clients")

        for client_idx, loc in enumerate(self.clients):
            color = get_client_color(client_idx)
            ward_info = self.mapper.client_ward_map.get(client_idx)
            ward_name = ward_info['name'] if ward_info else 'UNMAPPED'
            pcode = ward_info['pcode'] if ward_info else 'N/A'
            was_unmapped = ward_info.get('was_unmapped', False) if ward_info else True

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Client {client_idx}</b><br>
                Ward: {ward_name}<br>
                Pcode: {pcode}<br>
                Lat: {loc.lat:.6f}, Lon: {loc.lon:.6f}
                {'<br>⚠️ Nearest ward (GPS drift)' if was_unmapped else ''}
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
                tooltip=f"Client {client_idx} → {ward_name}",
            ).add_to(client_layer)

        client_layer.add_to(m)

        # ── Office marker ──
        folium.Marker(
            location=[self.office.lat, self.office.lon],
            popup="Office / Farm",
            tooltip="Office / Farm",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
        ).add_to(m)

        # ── Layer control ──
        folium.LayerControl(collapsed=False).add_to(m)

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

    def _generate_distinct_colors(self, n: int) -> list:
        """Generate n visually distinct hex colors."""
        colors = []
        for i in range(n):
            hue = i / n
            sat = 0.7 + (i % 3) * 0.1
            val = 0.8 + (i % 2) * 0.1
            r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 1.0), min(val, 1.0))
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def _build_legend(self, title: str, items: Dict[str, str]) -> str:
        """Build an HTML legend for the map."""
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

    def _save(self, m: folium.Map, filename: str):
        """Save map and print summary."""
        m.save(filename)
        print(f"  Saved: {filename}")
        print(f"  Wards with clients: {len(self.active_pcodes)}")
        print(f"  Total clients: {len(self.clients)}")
        print(f"  Unmapped: {len(self.mapper.unmapped_clients)}")
