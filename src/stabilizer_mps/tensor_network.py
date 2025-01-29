import networkx as nx
import numpy as np

from typing import Union

from dataclasses import dataclass

@dataclass
class TensorNetwork:

    #TODO Add measurements

    def __post_init__(self):
        self.tensornet = nx.MultiDiGraph()
    
    def add_tensor(self, id:str, tensor:np.ndarray):
        self.tensornet.add_node(
            id, 
            tensor=tensor, 
            shape=tensor.shape, 
            free_directions=[True]*len(tensor.shape)
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

    def draw(self):
        
        from matplotlib import pyplot as plt
        pos = nx.spring_layout(self.tensornet)

        # Draw nodes and labels
        nx.draw_networkx_nodes(self.tensornet, pos, node_size=450)
        nx.draw_networkx_labels(self.tensornet, pos)

        for u, v, key in self.tensornet.edges(keys=True):
            nx.draw_networkx_edges(
                self.tensornet,
                pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3",
            )

        # Manually add edge labels for multi-edges
        for u, v, key, metadata in self.tensornet.edges(keys=True, data=True):
            x, y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2  # Midpoint of edge
            plt.text(
                x, y, f'{key} - {metadata["directions"]}', fontsize=7, color="red", ha="center", bbox=dict(facecolor="white", alpha=0.7)
            )

        # Display the graph
        plt.title("TensorNetwork")
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
    