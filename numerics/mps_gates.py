from dataclasses import dataclass

import numpy as np

@dataclass
class MPSGateTensor():

    tensor:np.ndarray
    in_directions:list[int]
    out_directions:list[int]

    def __post_init__(self):

        assert len(self.in_directions) == len(self.out_directions), 'Tensors must preserve the number of physical gates.'
        self.n = len(self.in_directions)

        tshape = self.tensor.shape
        for din, dout in zip(self.in_directions, self.out_directions):
            assert din < len(tshape) and dout < len(tshape), 'Assigned directions out of bounds for given tensor. Check dimensions.'
            assert din not in self.out_directions, 'In-directions and out-dirctions must be disjoint.'
            assert tshape[din] == tshape[dout], 'Tensor must preserve the dimension of the pysical legs.'
        
        assert len(self.in_directions + self.out_directions) == len(tshape), 'Mismatch between the number of assigned directions and tensor shape.'

CNOT = MPSGateTensor(
    tensor=np.array([[[[1,0],[0,1]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,1],[1,0]]]]),
    in_directions=[0,2],
    out_directions=[1,3]
)

CZ = MPSGateTensor(
    tensor=np.array([[[[1,0],[0,1]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[-1,0],[-1,0]]]]),
    in_directions=[0,2],
    out_directions=[1,3]
)

H = MPSGateTensor(
    tensor=np.array([[1,1],[1,-1]])/np.sqrt(2),
    in_directions=[0],
    out_directions=[1]
)

X = MPSGateTensor(
    tensor=np.array([[0,1],[1,0]]),
    in_directions=[0],
    out_directions=[1]
)

Y = MPSGateTensor(
    tensor=np.array([[0,-1j],[1j,0]]),
    in_directions=[0],
    out_directions=[1]
)

Z = MPSGateTensor(
    tensor=np.array([[1,0],[0,-1]]),
    in_directions=[0],
    out_directions=[1]
)

S = MPSGateTensor(
    tensor=np.array([[1,0],[0,1j]]),
    in_directions=[0],
    out_directions=[1]
)

T = MPSGateTensor(
    tensor=np.array([[1,0],[0,1.0/np.sqrt(2)+1j/np.sqrt(2)]]),
    in_directions=[0],
    out_directions=[1]
)