import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors


paulis = {
    'I':np.eye(2, dtype=np.complex64),
    'X':np.array([[0,1],[1,0]], dtype=np.complex64),
    'Y':np.array([[0,-1j],[1j,0]], dtype=np.complex64),
    'Z':np.array([[1,0],[0,-1]], dtype=np.complex64)
}
def draw_tn(tn, show_labels=False, title=""):
        
        # Step 1: Group nodes by their prefix
        node_groups = defaultdict(list)
        for node in tn.tensornet.nodes():
            prefix = node[0]  # Assuming the first character is the prefix
            node_groups[prefix].append(node)

        # Extract sorted prefixes for consistent color assignment
        prefixes = sorted(node_groups.keys())

        # Step 2: Assign a unique color to each prefix group
        num_prefixes = len(prefixes)
        # Use a colormap with enough distinct colors
        if num_prefixes <= 10:
            cmap = cm.get_cmap('tab10', num_prefixes)
        elif num_prefixes <= 20:
            cmap = cm.get_cmap('tab20', num_prefixes)
        else:
            # For more than 20 groups, use a continuous colormap
            cmap = cm.get_cmap('hsv', num_prefixes)

        # Create a mapping from prefix to color
        prefix_color_map = {prefix: cmap(i) for i, prefix in enumerate(prefixes)}

        # Step 3: Assign positions to each group on a distinct curved line
        pos = {}
        layer_spacing = 10    # Vertical spacing between layers (y-axis)
        y_start = 0           # Starting y-coordinate for the first group
        curvature_angle = -20 # Degrees of curvature relative to horizontal

        # Precompute rotation matrix
        theta = curvature_angle  # degrees
        rad = np.radians(theta)
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad),  np.cos(rad)]
        ])

        # Define parameters for the Bezier curves
        curve_length = 20  # Horizontal span of the curve
        control_offset = 3  # Vertical offset to define the curvature

        for idx, (prefix, nodes) in enumerate(sorted(node_groups.items())):
            N = len(nodes)
            if N == 0:
                continue

            # Define Bezier curve control points for the group
            P0 = np.array([0, 0])  # Starting point
            P1 = np.array([curve_length / 2, control_offset])  # Control point for curvature
            P2 = np.array([curve_length, 0])  # Ending point

            # Generate positions along the Bezier curve
            ts = np.linspace(0, 1, N)
            bezier_points = [(1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2 for t in ts]

            # Rotate the Bezier curve by the curvature angle
            rotated_bezier = [rotation_matrix @ point for point in bezier_points]

            # Offset each group's curve vertically to prevent overlapping curves
            vertical_offset = -idx * layer_spacing
            rotated_bezier = [(x, y + vertical_offset) for x, y in rotated_bezier]

            # Assign positions to nodes and collect colors
            for node, point in zip(sorted(nodes), rotated_bezier):
                pos[node] = (point[0], point[1])

        # Step 4: Assign positions to Measurement Nodes (Ms) on a Separate Curved Line
        measurement_groups = [prefix for prefix in node_groups.keys() if prefix != 'T']
        for prefix in measurement_groups:
            nodes = node_groups[prefix]
            N = len(nodes)
            if N == 0:
                continue

            # Define Bezier curve control points for the measurement group
            P0 = np.array([0, 0])
            P1 = np.array([curve_length / 2, -control_offset])  # Opposite curvature
            P2 = np.array([curve_length, 0])

            # Generate positions along the Bezier curve
            ts = np.linspace(0, 1, N)
            bezier_points = [(1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2 for t in ts]

            # Rotate the Bezier curve by the curvature angle
            rotated_bezier = [rotation_matrix @ point for point in bezier_points]

            # Offset each group's curve vertically to prevent overlapping curves
            vertical_offset = -(len(node_groups) + 1) * layer_spacing  # Positioned above tensor groups
            rotated_bezier = [(x, y + vertical_offset) for x, y in rotated_bezier]

            # Assign positions to measurement nodes
            for node, point in zip(sorted(nodes), rotated_bezier):
                pos[node] = (point[0], point[1])

        # Step 5: Calculate Adaptive Figure Size
        base_width = 6
        base_height = 4

        plt.figure(figsize=(base_width, base_height))
        ax = plt.gca()

        # Step 6: Draw Nodes with Group-Specific Colors
        # Prepare a list of colors corresponding to each node
        node_colors = [prefix_color_map[node[0]] for node in tn.tensornet.nodes()]
        nx.draw_networkx_nodes(
            tn.tensornet, pos,
            node_size=650,
            node_color=node_colors,
            ax=ax
        )
        nx.draw_networkx_labels(
            tn.tensornet, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # Step 7: Handle Multiple Edges Between the Same Nodes
        # Group edges by their (u, v) pairs
        edge_pairs = defaultdict(list)
        for u, v, key, metadata in tn.tensornet.edges(keys=True, data=True):
            edge_pairs[(u, v)].append((key, metadata))

        for (u, v), edges in edge_pairs.items():
            total = len(edges)
            # Define curvature angles based on the number of parallel edges
            # Assign distinct angles to prevent overlapping
            # Use predefined angles for up to 4 edges, otherwise spread evenly
            if total == 1:
                angles = [20]  # slight curvature instead of straight line
            elif total == 2:
                angles = [-20, 20]
            elif total == 3:
                angles = [-30, 0, 30]
            elif total == 4:
                angles = [-40, -10, 10, 40]
            else:
                # For more than 4 edges, spread angles from -60 to +60
                angles = np.linspace(-60, 60, total)

            for idx, (key, metadata) in enumerate(edges):
                angle_deg = angles[idx]
                curvature = angle_deg / 90  # rad parameter, scaled between -1 to +1

                # Get node positions
                x_start, y_start = pos[u]
                x_end, y_end = pos[v]

                # Create a FancyArrowPatch with the specified curvature
                arrow = FancyArrowPatch(
                    (x_start, y_start),
                    (x_end, y_end),
                    connectionstyle=f"arc3,rad={curvature}",
                    arrowstyle='-|>',
                    mutation_scale=15,
                    color='black',
                    linewidth=2,
                    zorder=1
                )
                ax.add_patch(arrow)

                # Calculate label position
                # Midpoint coordinates
                x_mid = (x_start + x_end) / 2
                y_mid = (y_start + y_end) / 2

                # Offset for label placement based on curvature
                offset = 0.4  # Adjust as needed

                # Calculate perpendicular direction for label offset
                dx = x_end - x_start
                dy = y_end - y_start
                length = np.hypot(dx, dy)
                if length == 0:
                    length = 1  # Prevent division by zero
                perp_dx = -dy / length
                perp_dy = dx / length

                # Apply curvature-based offset
                x_label = x_mid + perp_dx * curvature * offset
                y_label = y_mid + perp_dy * curvature * offset

                # Calculate rotation angle for the label
                # Adding angle_deg to align label with edge curvature
                angle = np.degrees(np.arctan2(dy, dx)) + angle_deg

                # Place the label if show_labels is True
                if show_labels:
                    ax.text(
                        x_label,
                        y_label,
                        f'{key} - {metadata["directions"]}',
                        fontsize=10,           # Increased fontsize
                        color="black",         # Black text
                        ha="center",
                        va="center",
                        rotation=angle,
                        rotation_mode='anchor',
                        bbox=dict(facecolor="white", alpha=1, edgecolor='black', pad=3)
                    )

        # Step 8: Add a Legend Mapping Colors to Prefixes
        legend_patches = [Patch(facecolor=prefix_color_map[prefix], label=prefix) for prefix in prefixes]
        plt.legend(handles=legend_patches, title="Node Groups", loc='upper right')

        # Step 9: Final Touches
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def multi_trace(tensor, directions_in, directions_out):
    
    while len(directions_in)>0:

        d_in, d_out = directions_in[0], directions_out[0]
        tensor = np.trace(tensor, axis1=d_in, axis2=d_out)
        
        if not len(directions_in): break

        # Adjust remaining indices dynamically
        directions_in = [d - (d>d_in) - (d>d_out) for d in directions_in[1:]]
        directions_out = [d - (d>d_in) - (d>d_out) for d in directions_out[1:]]

    return tensor

def _bond_dimension_cut(U, D, V, bond_dimension):
    
    if bond_dimension is None:
        bond_dimension = np.count_nonzero(D)
    else:
        bond_dimension = np.min(bond_dimension, np.count_nonzero(D))

    return U[:,:bond_dimension], D[:bond_dimension], V[:bond_dimension,:]
