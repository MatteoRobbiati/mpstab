from tableaus import CNOT, H, S
from pauli_string import Pauli

c = Pauli('ZXZZ')
d = Pauli('YXZXIYY')

print('Pauli main features:')
print(f'c = {c}')
print(f'd = {d}')
print(f'c@d = {c@d}')

ops=[S(0), H(2), CNOT(3,1), CNOT(2,4)]
print('\nApply Clifford operations to update the Pauli instance.')
print(f'Initial state: {d}')

for op in ops:
    d.apply(op)

print(f'Updated string = {d}')