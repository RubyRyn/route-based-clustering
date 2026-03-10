"""
Territory Visualization with Convex Hulls

Draws convex hull polygons around each cluster so you can see:
- Whether territories are clean and separated
- Where clusters overlap
- Which clients are sitting inside another cluster's territory (violations)

Usage:
    from territory_viz import visualize_territories
    visualize_territories(labels, client_coords, office_coords, n_clusters)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull


def visualize_territories(labels: np.ndarray,
                          client_coords: np.ndarray,
                          office_coords: tuple = None,
                          n_clusters: int = None,
                          title: str = "Cluster Territories",
                          filename: str = None,
                          figsize: tuple = (14, 10),
                          show_violations: bool = True,
                          hull_alpha: float = 0.15,
                          edge_alpha: float = 0.6):
    """
    Visualize clusters with convex hull territory polygons.

    Args:
        labels: Cluster label for each client (length = n_clients).
        client_coords: Array of shape (n_clients, 2) with [lat, lon].
        office_coords: Tuple of (lat, lon) for the farm/office. Optional.
        n_clusters: Number of clusters. If None, inferred from labels.
        title: Plot title.
        filename: If provided, saves to this path.
        figsize: Figure size.
        show_violations: If True, highlight clients inside another cluster's hull.
        hull_alpha: Transparency of filled polygons (0=invisible, 1=opaque).
        edge_alpha: Transparency of polygon borders.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=figsize)

    # Use tab20 for up to 20 clusters, generate more if needed
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))

    # lon = x, lat = y
    lons = client_coords[:, 1]
    lats = client_coords[:, 0]

    hulls = {}  # Store hull objects for violation detection

    # --- Draw convex hull polygons ---
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() < 3:
            # Need at least 3 points for a convex hull
            # For 1-2 points, just skip the polygon
            continue

        cluster_lons = lons[mask]
        cluster_lats = lats[mask]
        points = np.column_stack([cluster_lons, cluster_lats])

        try:
            hull = ConvexHull(points)
            hulls[cluster_id] = (hull, points)

            # Get hull vertices in order
            hull_points = points[hull.vertices]
            # Close the polygon
            hull_points = np.vstack([hull_points, hull_points[0]])

            # Draw filled polygon
            polygon = Polygon(
                hull_points,
                closed=True,
                facecolor=colors[cluster_id],
                edgecolor=colors[cluster_id],
                alpha=hull_alpha,
                linewidth=0
            )
            ax.add_patch(polygon)

            # Draw border
            ax.plot(
                hull_points[:, 0], hull_points[:, 1],
                color=colors[cluster_id],
                alpha=edge_alpha,
                linewidth=2,
                linestyle='-'
            )

        except Exception:
            # Degenerate case (collinear points, etc.)
            pass

    # --- Detect violations (client inside another cluster's hull) ---
    violations = []
    if show_violations and len(hulls) > 0:
        for client_idx in range(len(labels)):
            client_cluster = labels[client_idx]
            client_point = np.array([lons[client_idx], lats[client_idx]])

            for cluster_id, (hull, points) in hulls.items():
                if cluster_id == client_cluster:
                    continue

                # Check if client is inside this hull
                if _point_in_hull(client_point, hull, points):
                    violations.append((client_idx, client_cluster, cluster_id))
                    break  # Only report first violation per client

    # --- Plot client dots ---
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if mask.sum() > 0:
            ax.scatter(
                lons[mask], lats[mask],
                c=[colors[cluster_id]],
                s=40,
                alpha=0.8,
                edgecolors='white',
                linewidths=0.5,
                zorder=3,
                label=f"C{cluster_id + 1} ({mask.sum()})"
            )

    # --- Highlight violations ---
    if show_violations and len(violations) > 0:
        violation_indices = [v[0] for v in violations]
        ax.scatter(
            lons[violation_indices],
            lats[violation_indices],
            c='none',
            s=150,
            edgecolors='red',
            linewidths=2.5,
            zorder=4,
            label=f"Violations ({len(violations)})"
        )

    # --- Office ---
    if office_coords is not None:
        ax.scatter(
            office_coords[1], office_coords[0],
            c='red', marker='s', s=200, zorder=5,
            edgecolors='black', linewidths=1.5,
            label='Farm'
        )

    # --- Labels and legend ---
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
    subtitle = (f"Sizes: [{sizes.min()}-{sizes.max()}] | "
                f"Std: {sizes.std():.1f}")
    if show_violations:
        subtitle += f" | Violations: {len(violations)}"

    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()

    if show_violations and len(violations) > 0:
        print(f"\n{len(violations)} territorial violations found:")
        for client_idx, own_cluster, intruding_cluster in violations[:20]:
            print(f"  Client {client_idx}: assigned to C{own_cluster + 1}, "
                  f"but inside C{intruding_cluster + 1}'s territory")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")

    return violations


def visualize_territories_comparison(labels_before: np.ndarray,
                                      labels_after: np.ndarray,
                                      client_coords: np.ndarray,
                                      office_coords: tuple = None,
                                      n_clusters: int = None,
                                      title_before: str = "Before Swapping",
                                      title_after: str = "After Swapping",
                                      filename: str = None,
                                      figsize: tuple = (24, 10)):
    """
    Side-by-side comparison of two labelings with convex hull territories.
    """
    if n_clusters is None:
        n_clusters = max(len(np.unique(labels_before)),
                         len(np.unique(labels_after)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_clusters))

    lons = client_coords[:, 1]
    lats = client_coords[:, 0]

    for ax, lbls, title in zip(axes,
                                [labels_before, labels_after],
                                [title_before, title_after]):

        violations_count = 0
        hulls = {}

        # Draw hulls
        for cluster_id in range(n_clusters):
            mask = lbls == cluster_id
            if mask.sum() < 3:
                continue

            points = np.column_stack([lons[mask], lats[mask]])
            try:
                hull = ConvexHull(points)
                hulls[cluster_id] = (hull, points)

                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])

                polygon = Polygon(
                    hull_points, closed=True,
                    facecolor=colors[cluster_id],
                    edgecolor=colors[cluster_id],
                    alpha=0.15, linewidth=0
                )
                ax.add_patch(polygon)
                ax.plot(hull_points[:, 0], hull_points[:, 1],
                        color=colors[cluster_id], alpha=0.6, linewidth=2)

            except Exception:
                pass

        # Count violations
        violation_indices = []
        for client_idx in range(len(lbls)):
            client_cluster = lbls[client_idx]
            client_point = np.array([lons[client_idx], lats[client_idx]])
            for cluster_id, (hull, points) in hulls.items():
                if cluster_id == client_cluster:
                    continue
                if _point_in_hull(client_point, hull, points):
                    violation_indices.append(client_idx)
                    violations_count += 1
                    break

        # Plot dots
        for cluster_id in range(n_clusters):
            mask = lbls == cluster_id
            if mask.sum() > 0:
                ax.scatter(lons[mask], lats[mask],
                           c=[colors[cluster_id]], s=30, alpha=0.8,
                           edgecolors='white', linewidths=0.3, zorder=3)

        # Violations
        if violation_indices:
            ax.scatter(lons[violation_indices], lats[violation_indices],
                       c='none', s=120, edgecolors='red', linewidths=2, zorder=4)

        if office_coords:
            ax.scatter(office_coords[1], office_coords[0],
                       c='red', marker='s', s=150, zorder=5,
                       edgecolors='black', linewidths=1)

        sizes = np.array([np.sum(lbls == i) for i in range(n_clusters)])
        ax.set_title(f"{title}\nSizes: [{sizes.min()}-{sizes.max()}] | "
                     f"Std: {sizes.std():.1f} | Violations: {violations_count}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()


def _point_in_hull(point: np.ndarray, hull: ConvexHull,
                    hull_points: np.ndarray) -> bool:
    """
    Check if a 2D point is inside a convex hull.
    Uses the equation-based method: a point is inside if it satisfies
    all half-plane constraints defined by the hull's equations.
    """
    # hull.equations: each row is [A, B, C] where Ax + By + C <= 0 for interior
    return np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 1e-10)