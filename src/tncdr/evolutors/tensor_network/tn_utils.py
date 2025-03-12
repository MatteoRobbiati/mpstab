import re
import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from collections import defaultdict
import matplotlib.cm as cm


paulis = {
    'I':np.eye(2, dtype=np.complex64),
    'X':np.array([[0,1],[1,0]], dtype=np.complex64),
    'Y':np.array([[0,-1j],[1j,0]], dtype=np.complex64),
    'Z':np.array([[1,0],[0,-1]], dtype=np.complex64)
}

def draw_tn(tn, show_labels=False, title=""):
    # --- Step 1: Group Nodes by Label ---
    # For example, if a node’s name begins with "L" followed by digits,
    # we group them by that complete token (e.g., "L1", "L2", etc.)
    groups = defaultdict(list)
    for node in tn.tensornet.nodes():
        match = re.match(r'(L\d+)', node)
        if match:
            group_label = match.group(1)
        else:
            group_label = node[0]  # fallback: use first character
        groups[group_label].append(node)
    
    group_keys = sorted(groups.keys())
    
    # --- Step 2: Assign Curved Row Positions for Each Group ---
    # Each group is laid out along an arc (curved row). We use polar coordinates
    # so that nodes are positioned along an arc whose radius increases with the group index.
    pos = {}
    
    # Parameters for the arcs
    base_radius = 10       # Base radius for the innermost group
    radius_gap = 5         # Increase in radius for each subsequent group
    arc_width = np.radians(90)  # The angular span of the arc (here, 90°)
    center_angle = np.pi/2  # Center the arc vertically (90° = straight up)
    
    # Prepare a colormap for group colors
    cmap = cm.get_cmap('tab10', len(group_keys))
    group_color_map = {group: cmap(i) for i, group in enumerate(group_keys)}
    
    # For each group, compute positions along its arc:
    for idx, group in enumerate(group_keys):
        nodes = groups[group]
        n_nodes = len(nodes)
        
        # Separate peripheral (degree == 1) and core nodes (degree > 1)
        peripheral = sorted([node for node in nodes if tn.tensornet.degree(node) == 1])
        core = sorted([node for node in nodes if tn.tensornet.degree(node) > 1])
        
        # Determine the arc limits.
        start_angle = center_angle - arc_width/2
        end_angle = center_angle + arc_width/2
        
        # Get equally spaced angles along the arc
        all_angles = np.linspace(start_angle, end_angle, n_nodes)
        
        assigned_angles = {}
        p = len(peripheral)
        if p > 0:
            # Reserve the extreme positions for peripheral nodes.
            left_count = p // 2
            right_count = p - left_count
            peripheral_angles = list(all_angles[:left_count]) + list(all_angles[-right_count:])
            # Assign peripheral nodes to these angles (in order)
            for node, angle in zip(peripheral, peripheral_angles):
                assigned_angles[node] = angle
            # The remaining angles go to core nodes.
            remaining_angles = all_angles[left_count: n_nodes - right_count]
        else:
            remaining_angles = all_angles
        
        for node, angle in zip(core, remaining_angles):
            assigned_angles[node] = angle
        
        # Determine the arc radius for this group.
        radius = base_radius + idx * radius_gap
        
        # Convert polar coordinates (radius, angle) to Cartesian (x, y)
        for node, angle in assigned_angles.items():
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)
    
    # --- Step 3: Draw the Graph (Nodes & Labels) ---
    base_width = 8
    base_height = 8
    plt.figure(figsize=(base_width, base_height))
    ax = plt.gca()
    
    # Prepare node colors based on group membership.
    node_colors = []
    for node in tn.tensornet.nodes():
        match = re.match(r'(L\d+)', node)
        if match:
            group_label = match.group(1)
        else:
            group_label = node[0]
        node_colors.append(group_color_map[group_label])
    
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
    
    # --- Step 4: Draw Edges with Reduced Overlap ---
    # Group edges by (u,v) so that multiple edges are drawn with small, differing curvatures.
    edge_pairs = defaultdict(list)
    for u, v, key, metadata in tn.tensornet.edges(keys=True, data=True):
        edge_pairs[(u, v)].append((key, metadata))
    
    for (u, v), edges in edge_pairs.items():
        total = len(edges)
        # Choose curvature angles (in radians) to reduce overlap.
        if total == 1:
            angles = [0]
        elif total == 2:
            angles = [-0.1, 0.1]
        elif total == 3:
            angles = [-0.2, 0, 0.2]
        elif total == 4:
            angles = [-0.3, -0.1, 0.1, 0.3]
        else:
            angles = np.linspace(-0.4, 0.4, total)
        
        for idx, (key, metadata) in enumerate(edges):
            curvature = angles[idx]
            x_start, y_start = pos[u]
            x_end, y_end = pos[v]
            
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
            
            if show_labels:
                # Compute a midpoint and offset perpendicular to the edge.
                x_mid = (x_start + x_end) / 2
                y_mid = (y_start + y_end) / 2
                dx = x_end - x_start
                dy = y_end - y_start
                length = math.hypot(dx, dy) or 1
                perp_dx = -dy / length
                perp_dy = dx / length
                offset = 0.3
                x_label = x_mid + perp_dx * curvature * offset
                y_label = y_mid + perp_dy * curvature * offset
                angle = np.degrees(np.arctan2(dy, dx))
                ax.text(
                    x_label,
                    y_label,
                    f'{key} - {metadata["directions"]}',
                    fontsize=10,
                    color="black",
                    ha="center",
                    va="center",
                    rotation=angle,
                    rotation_mode='anchor',
                    bbox=dict(facecolor="white", alpha=1, edgecolor='black', pad=3)
                )
    
    # --- Step 5: Add Legend and Final Touches ---
    legend_patches = [Patch(facecolor=group_color_map[group], label=group) for group in group_keys]
    plt.legend(handles=legend_patches, title="Node Groups", loc='upper right')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if title is not None:
        plt.savefig(f"../plots/{title}.png", bbox_inches="tight")


def multi_trace(tensor, directions_in:list[int], directions_out:list[int]):
    """
    Trace out the indices correpoding to the directios specified.
    
    Uses np.trace iteratively to achieve it, in addition to dynamical handling of the directions
    to be conrtracted 
    """
    
    while len(directions_in)>0:

        d_in, d_out = directions_in[0], directions_out[0]
        tensor = np.trace(tensor, axis1=d_in, axis2=d_out)
        
        if not len(directions_in): break

        # Adjust remaining indices dynamically
        directions_in = [d - (d>d_in) - (d>d_out) for d in directions_in[1:]]
        directions_out = [d - (d>d_in) - (d>d_out) for d in directions_out[1:]]

    return tensor

def _complex_conjugate(tensornet:nx.MultiDiGraph):
    """
    Given a graph corresponding to a TensorNetwork, take the complex conjugate of each tensor.
    """

    for t in list(tensornet.nodes):
        tensornet.nodes[t]['tensor'] = np.conj(tensornet.nodes[t]['tensor'])
        nx.relabel_nodes(tensornet, {t:f'{t}_dg'}, copy=False)

def _bond_dimension_cut(U, D, V, max_bond_dimension):
    """
    Given the output of an SVD procedure, remove the smallest singular values that exceed the 
    maximum allowed number, max_bond_dimension.

    Assumes that the singular values are stored in descending order in D.
    """
    
    if max_bond_dimension is None:
        bond_dimension = np.count_nonzero(D)
    else:
        bond_dimension = np.min(max_bond_dimension, np.count_nonzero(D))

    return U[:,:bond_dimension], D[:bond_dimension], V[:bond_dimension,:]
