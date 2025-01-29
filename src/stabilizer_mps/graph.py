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

        # Construct physical map linking new dimensions with old index
        _, reverse_lookup_map = _construct_physical_map(
            non_contracted_index_in=non_contracted_index_in,
            non_contracted_index_out=non_contracted_index_out,
            new_physical_dimensions=list(new_tensor.shape)
        )


        # Remove the edges from the graph
        for edge_id in edge_ids:
            self.remove_edge(node_in=node_in, node_out=node_out, edge_id=edge_id)

        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)

        self.reconnect_edges(
            node=node_in, new_node_id=new_node_id, direction_label="in", reverse_lookup_map=reverse_lookup_map
        )
        self.reconnect_edges(
            node=node_out, new_node_id=new_node_id, direction_label="out", reverse_lookup_map=reverse_lookup_map
        )

    
    def reconnect_edges(self, node, new_node_id, direction_label, reverse_lookup_map):

        # first we take all the edges entering the node
        for edge in list(self.tensornet.in_edges(nbunch=node, keys=True, data=True)):
            u, edge_id, original_directions = edge[0], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (
                original_directions[0], 
                reverse_lookup_map[(direction_label, original_directions[1])]
            )
            self.add_edge(node_in=u, node_out=new_node_id, edge_id=edge_id, directions=directions)
        
        # second we take all the edges exiting the node
        for edge in list(self.tensornet.out_edges(nbunch=node, keys=True, data=True)):
            v, edge_id, original_directions = edge[1], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (
                reverse_lookup_map[(direction_label, original_directions[0])],
                original_directions[1],
            )
            self.add_edge(node_in=new_node_id, node_out=v, edge_id=edge_id, directions=directions)

        # Remove node
        self.tensornet.remove_node(node)

def _construct_physical_map(
        non_contracted_index_in: list[int],
        non_contracted_index_out: list[int],
        new_physical_dimensions: tuple[int],
):
    if (len(non_contracted_index_in) + len(non_contracted_index_out) != len(new_physical_dimensions)):
        raise ValueError(
            f"Number of physical dimensions has to match the sum of the lengths of non contracted index lists."
        )  
    
    index_map, reverse_lookup = {}, {}

    # Concatenate lists
    non_contracted_index_list = non_contracted_index_in + non_contracted_index_out

    for i, phys_dim in enumerate(new_physical_dimensions):
        if i < len(non_contracted_index_in):
            label = "in"
        else:
            label = "out"
        index_map.update(
            {
                f"{i}": {
                    "direction_type": label,
                    "physical_dim": phys_dim,
                    "old_index": non_contracted_index_list[i]
                }
            }
        )

        # Store in reverse lookup
        reverse_lookup[(label, non_contracted_index_list[i])] = i  # Store as integer

    return index_map, reverse_lookup
    