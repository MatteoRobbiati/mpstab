import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from typing import Union
from tncdr.stabilizer_mps.tn_utils import paulis

from dataclasses import dataclass

@dataclass
class TensorNetwork:

    #TODO Add key tensors:
    # - Measurement: shape -> (2,)
    # - Copy: shape -> (n,n,n), n given in input
    # - Pauli: shape -> (2,2)
    # - PauliRot: shape -> (2,2,2)

    def __post_init__(self):
        self.tensornet = nx.MultiDiGraph()

    @property
    def n_tensors(self):
        return len(self.tensornet.nodes)
    
    def add_tensor(self, id:str, tensor:np.ndarray):
        self.tensornet.add_node(
            id, 
            tensor=tensor, 
            shape=tensor.shape, 
            free_directions=[True]*len(tensor.shape)
        )

    def add_measurement(self, id:str, alpha:float=1.0, beta:float=0.0):
        self.add_tensor(
            id=id,
            tensor=np.array([alpha, beta])
        )

    def add_pauli_pair(self, p0:str, p1:str):
        tensor = np.array([paulis[p0], paulis[p1]])
        self.add_tensor(
            id=f'{p0}{p1} pair',
            tensor=tensor,
        )

    def add_edge(self, node_in:str, node_out:str, edge_id:str, directions:tuple[int, int], **edge_metadata):

        d_in, d_out = directions
        if self.tensornet.nodes[node_in]['shape'][d_in] != self.tensornet.nodes[node_out]['shape'][d_out]:
            raise ValueError(
                f"Incompatible connected tensor directions {directions}: dim(T_{node_in}[{d_in}]) != dim(T_{node_out}[{d_out}]) ({self.tensornet.nodes[node_in]['shape'][d_in]} != {self.tensornet.nodes[node_out]['shape'][d_out]})"
            )
        
        if not (self.tensornet.nodes[node_in]['free_directions'][d_in] and self.tensornet.nodes[node_out]['free_directions'][d_out]):
            raise ValueError(
                'Node directions already in use.'
            )

        self.tensornet.add_edge(
            node_in,
            node_out,
            key=edge_id,
            directions=directions,
            **edge_metadata,
        )

        self.tensornet.nodes[node_in]['free_directions'][d_in] = False
        self.tensornet.nodes[node_out]['free_directions'][d_out] = False

    def remove_edge(self, node_in, node_out, edge_id):
        
        d_in, d_out = self.tensornet.edges[node_in,node_out,edge_id]['directions']

        self.tensornet.remove_edge(node_in, node_out, edge_id)

        self.tensornet.nodes[node_in]['free_directions'][d_in] = True
        self.tensornet.nodes[node_out]['free_directions'][d_out] = True

    def draw(self, show_labels=False, title=""):
        # Step 1: Group nodes by their prefix
        node_groups = defaultdict(list)
        for node in self.tensornet.nodes():
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
        node_colors = [prefix_color_map[node[0]] for node in self.tensornet.nodes()]
        nx.draw_networkx_nodes(
            self.tensornet, pos,
            node_size=650,
            node_color=node_colors,
            ax=ax
        )
        nx.draw_networkx_labels(
            self.tensornet, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        # Step 7: Handle Multiple Edges Between the Same Nodes
        # Group edges by their (u, v) pairs
        edge_pairs = defaultdict(list)
        for u, v, key, metadata in self.tensornet.edges(keys=True, data=True):
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

    def contract(self, node_in:str, node_out:str, edge_ids:Union[str, list[str]], new_node_id:str):
        
        # Constructing a list of edges ids, useful later
        if type(edge_ids) is str: 
            edge_ids = [edge_ids]

        # Collect all edges metadata in a second list
        edge_metadatas = [self.tensornet.edges[node_in,node_out,id] for id in edge_ids]

        # Constructing two lists: 
        # -- 1: directions_in contains the physical directions of input nodes of the edges
        # -- 2: directions_out contains the physical directions of output nodes of the edges
        directions_in, directions_out = [], []
        for metadata in edge_metadatas:
            directions_in.append(metadata['directions'][0])
            directions_out.append(metadata['directions'][1])

        # Collecting all non-contracted index (in and out)
        non_contracted_index_in = [
            i for i in range(len(self.tensornet.nodes[node_in]['shape'])) if i not in directions_in
        ]
        non_contracted_index_out = [
            i for i in range(len(self.tensornet.nodes[node_out]['shape'])) if i not in directions_out
        ]
        
        # Construct the contracted tensor (the future new node in the graph)
        new_tensor = np.tensordot(
            a=self.tensornet.nodes[node_in]['tensor'],
            b=self.tensornet.nodes[node_out]['tensor'],
            axes=(directions_in, directions_out),
        )

        # Remove the edges from the graph
        for edge_id in edge_ids:
            self.remove_edge(node_in=node_in, node_out=node_out, edge_id=edge_id)
            
        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)
        
        # Transfer the edge connections from the old to the new node and delete it
        self._reconnect_edges(
            node=node_in, 
            new_node_id=new_node_id, 
            survived_directions=non_contracted_index_in,
        )
        self._reconnect_edges(
            node=node_out, 
            new_node_id=new_node_id, 
            survived_directions=non_contracted_index_out, 
            shift=len(non_contracted_index_in), # Comply with indexing convention of numpy tensordot
        )
    
    def _partial_trace(self, *args):
        pass

    def _reconnect_edges(self, node:str, new_node_id:str, survived_directions:list, shift:int=0):

        # First we take all the edges entering the node
        for u,v,edge_id,metadata in list(self.tensornet.out_edges(nbunch=node, keys=True, data=True)):

            directions = (
                # Updated tensor direction in the new node corresponding to the edge
                survived_directions.index(metadata['directions'][0]) + shift,
                # Kept direction on the connected node
                metadata['directions'][1],
            )

            self.remove_edge(node_in=u, node_out=v, edge_id=edge_id)
            self.add_edge(node_in=new_node_id, node_out=v, edge_id=edge_id, directions=directions)
        
        # Second we take all the edges exiting the node
        for u,v,edge_id,metadata in list(self.tensornet.in_edges(nbunch=node, keys=True, data=True)):

            directions = (
                # Kept direction on the connected node
                metadata['directions'][0],
                # Updated tensor direction in the new node corresponding to the edge
                survived_directions.index(metadata['directions'][1]) + shift,
            )
            self.remove_edge(node_in=u, node_out=v, edge_id=edge_id)
            self.add_edge(node_in=u, node_out=new_node_id, edge_id=edge_id, directions=directions)

        # Remove node
        self.tensornet.remove_node(node)
     