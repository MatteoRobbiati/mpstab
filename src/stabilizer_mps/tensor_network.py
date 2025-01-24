from abc import ABC, abstractmethod
import numpy as np

class TensorIndex():
    
    def __init__(self, name:str='')->None:
        self.name = name
        self.tensor_info = []

    def add_tensor(self, tensor:'Tensor', t_index:int)->None:

        if len(self.tensor_info)>=2:
            raise ValueError('Cannot assign the same index to more than 2 tensors.')
        
        self.tensor_info.append((tensor, t_index))
    
    def __repr__(self)->str:
        return self.name
    
    def remove(self, tensor:'Tensor')->None:
        for i, (_,ti) in self.tensor_info:
            if ti is tensor:
                self.tensor_info.pop(i)

    def reset(self)->None:
        self.name=''
        self.tensor_info = []

class Tensor(ABC):
    """
    Implements a Tensor for Tensor Network computations.
    """

    def __init__(self, *index:tuple[TensorIndex])->None:
        
        assert len(self.matrix.shape) == len(index), f'Wrong number of indices. Expected {len(self.matrix.shape)}, given {len(index)}.'
        self.ids = list(index)
        for i_tensor, i in enumerate(index):
            i.add_tensor(self, i_tensor)
    
    def __repr__(self)->str:
        return 'T_'+','.join(str(i) for i in self.ids)

    def __del__(self):
        for i in self.ids:
            i.remove(self)
        del self

    def rank(self)->None:
        return len(self.matrix.shape)

    @property
    @abstractmethod
    def matrix(self)->np.ndarray:
        pass

def contract(index:TensorIndex):

    a = 'abcdefghijklmnopqrtstuvwyz'
    assert len(index.tensor_info) == 2,'Cannot contract index if it is not associated to 2 tensors.'
    
    t1,i1 = index.tensor_info[0]
    t2,i2 = index.tensor_info[1]

    eini1 = a[:i1]+'x'+a[i1:t1.rank()-1]
    eini2 = a[t1.rank()-1:i2+t1.rank()-1]+'x'+a[i2+t1.rank()-1:t1.rank()+t2.rank()-2]
    einir = a[:t1.rank()+t2.rank()-2]
    
    contracted_matrix = np.einsum(f'{eini1},{eini2}->{einir}', t1.matrix, t2.matrix)

    idx1 = t1.ids
    idx1.remove(index)
    del t1

    idx2 = t2.ids
    idx2.remove(index)
    del t2
    del index

    new_index = idx1+idx2
    return MatrixTensor(contracted_matrix, new_index)

class MatrixTensor(Tensor):

    def __init__(self, mat, *index):

        self.matrix = mat
        super().__init__(*index)

    @property
    def matrix(self)->np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, mat:np.ndarray)->None:
        self._matrix = mat

    @matrix.deleter
    def matrix(self)->None:
        del self._matrix