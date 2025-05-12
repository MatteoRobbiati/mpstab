from typing import Optional

import numpy as np

from tncdr.evolutors.tensor_network import TensorNetwork

class MPO(TensorNetwork):
    """
    Simple implementation of a MPO TensorNetwork, able to represent both quantum gates and observables
    """

    def __init__(
            self, 
            tensors:list[np.ndarray], 
            link_directions:Optional[list[tuple[int, int]]]=None,
            physical_directions:Optional[list[tuple[int, int]]]=None,
            tensor_prefix:str='O', 
            link_name:str='h_link',
        ):

        n_tensors = len(tensors)
 
        if link_directions is None: 
            link_directions = [(2,3) if i < n_tensors-2 else (2,2) for i in range(n_tensors-1)] if n_tensors > 1 else []
        
        if physical_directions is None: 
            physical_directions = [(1,0) for i in range(n_tensors)]
        
        assert n_tensors == len(physical_directions), f'Mismatch in the number of tensors and physical legs, {n_tensors}!={len(physical_directions)}.'
        assert n_tensors == len(link_directions) + 1, f'Mismatch in the number of tensors ({n_tensors}) and link directions ({len(link_directions)}, should be {n_tensors-1}).' 

        self.physical_directions = physical_directions
        self.prefix = tensor_prefix
        self.link = link_name

        super().__init__()

        self.add_tensor(self.prefix+'0', tensor=tensors[0])
        for q, (t, link_dir) in enumerate(zip(tensors[1:], link_directions), start=1):
            self.add_tensor(self.prefix+f'{q}', tensor=t)
            self.add_edge(self.prefix+f'{q-1}', self.prefix+f'{q}', link_name, link_dir)