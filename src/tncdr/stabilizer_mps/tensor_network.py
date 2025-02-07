import networkx as nx
import numpy as np

from typing import Union
from tncdr.stabilizer_mps.tn_utils import paulis, draw_tn, multi_trace

from dataclasses import dataclass

@dataclass
class TensorNetwork:

    #TODO Add key tensors:
    # - Copy: shape -> (n,n,n), n given in input

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

    def add_pauli_pair(self, id:str, p0:str, p1:str):
        tensor = np.array([paulis[p0], paulis[p1]])
        self.add_tensor(
            id=id,
            tensor=tensor,
        )

    def add_copy_tensor(self, id:str, n:int):
        tensor = np.zeros(shape=(n,n,n))
        np.fill_diagonal(tensor, 1)
        self.add_tensor(
            id=id,
            tensor=tensor
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
        return draw_tn(tn=self, show_labels=show_labels, title=title)

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
        
        # Remove the edges from the graph
        for edge_id in edge_ids:
            self.remove_edge(node_in=node_in, node_out=node_out, edge_id=edge_id)

        if node_in==node_out:
            return self._partial_trace(
                node=node_in,
                new_node_id=new_node_id,
                directions_in=directions_in,
                directions_out=directions_out
            )
        else:
            return self._contract_separate_nodes(
                node_in=node_in,
                node_out=node_out,
                new_node_id=new_node_id,
                directions_in=directions_in,
                directions_out=directions_out
            )

    def _contract_separate_nodes(self, node_in:str, node_out:str, new_node_id:str, directions_in:list, directions_out:list):
        
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
    
    def _partial_trace(self, node:str, new_node_id:str, directions_in:list, directions_out:list):

        non_contracted_index = [
            i for i in range(len(self.tensornet.nodes[node]['shape'])) if i not in (directions_in+directions_out)
        ]

        new_tensor = multi_trace(
            tensor=self.tensornet.nodes[node]['tensor'],
            directions_in=directions_in,
            directions_out=directions_out,
        )
    
        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)
        
        # Transfer the edge connections from the old to the new node and delete it
        self._reconnect_edges(
            node=node, 
            new_node_id=new_node_id, 
            survived_directions=non_contracted_index,
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
     