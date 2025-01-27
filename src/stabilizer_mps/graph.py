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
                f'Incompatible connected tensor directions {directions}: dim(T_{node_in}[{d_in}]) != dim(T_{node_out}[{d_out}]) ({self.tensornet.nodes[node_in]['shape'][d_in]} != {self.tensornet.nodes[node_out]['shape'][d_out]})'
            )
        
        if not (self.tensornet.nodes[node_in]['free_directions'][d_in] and self.tensornet.nodes[node_out]['free_directions'][d_out]):
            raise ValueError(
                'Node directions already in use.'
            )

        self.tensornet.add_edge(
            u_for_edge=node_in,
            v_for_edge=node_out,
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
        
        if type(edge_ids) is str: 
            edge_ids = [edge_ids]

        edge_metadatas = [self.tensornet.edges[node_in,node_out,id] for id in edge_ids]
        
        directions_in, directions_out = [], []
        for metadata in edge_metadatas:
            directions_in.append(metadata['directions'][0])
            directions_out.append(metadata['directions'][1])

        new_tensor = np.tensordot(
            a=self.tensornet.nodes[node_in]['tensor'],
            b=self.tensornet.nodes[node_out]['tensor'],
            axes=(directions_in, directions_out),
        )

        for id in edge_ids:
            self.remove_edge(node_in=node_in, node_out=node_out, edge_id=id)

        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)

        survived_node_in = []
        survived_node_out = []
        for i,(m_in,m_out) in enumerate(zip(self.tensornet.nodes[node_in]['free_directions'], self.tensornet.nodes[node_out]['free_directions'])):
            if not m_in:
                survived_node_in.append(i)
            if not m_out:
                survived_node_out.append(i)

        direction_map = lambda node, direction: list(range(len(self.tensornet.nodes[node_in]['shape']))).index(direction) if node is node_in else len(self.tensornet.nodes[node_in]['shape'])+list(range(len(self.tensornet.nodes[node_out]['shape']))).index(direction)
        print([direction_map(node_out, d) for d in survived_node_out])
        # Add edges to the new node according to non-contracted parent tensors indexes

        for edge in list(self.tensornet.in_edges(nbunch=node_in, keys=True, data=True)):
            print(edge)
            u,edge_id, directions = edge[0], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (directions[0], direction_map(node_in, directions[1]))
            self.add_edge(node_in=u, node_out=new_node_id, edge_id=edge_id, directions=directions)

        for edge in list(self.tensornet.out_edges(nbunch=node_in, keys=True, data=True)):
            
            v,edge_id, directions = edge[1], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (direction_map(node_in, directions[0]), directions[1])
            self.add_edge(node_in=new_node_id, node_out=v, edge_id=edge_id, directions=directions)

        for edge in list(self.tensornet.in_edges(nbunch=node_out, keys=True, data=True)):

            u,edge_id, directions = edge[0], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (directions[0], direction_map(node_out, directions[1]))
            self.add_edge(node_in=u, node_out=new_node_id, edge_id=edge_id, directions=directions)

        for edge in list(self.tensornet.out_edges(nbunch=node_out, keys=True, data=True)):
            
            v,edge_id, directions = edge[1], edge[2], edge[3]["directions"]
            self.remove_edge(*edge[:3])
            directions = (direction_map(node_out, directions[0]), directions[1])
            self.add_edge(node_in=new_node_id, node_out=v, edge_id=edge_id, directions=directions)

        self.tensornet.remove_node(node_in)
        self.tensornet.remove_node(node_out)

        