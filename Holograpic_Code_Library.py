import tensornetwork as tn
import numpy as np
from math import gcd
from fractions import Fraction

class TN:
    def __init__(self):
        self.edges = []
        
    def connect(self, e1, e2):
        self.edges.append(tn.connect(e1, e2))
        
    def get_result(self):
        for edge in self.edges:    
            result = tn.contract(edge)
        return result

def operator_to_tensor(operator):
    rows, columns = operator.shape
    n = int(np.log2(rows))
    return operator.transpose().reshape([2]*2*n)



def tensor_to_operator(tensor):
    shape = tensor.shape
    half_idx = int(len(shape)/2)
    n = np.prod(shape[:half_idx])
    operator = tensor.reshape(n, n).transpose() 
    return operator

def get_pushed_operator(operator):
    return np.matmul(pentagon_operator, np.matmul(operator, pentagon_operator.transpose()))

def greatest_common_divisor(arr):
    # Convert elements to Fractions
    fractions = [Fraction(x).limit_denominator() for x in arr]

    # Initialize gcd as the first element of the array
    result_gcd = fractions[0]

    # Find gcd for each element in the array
    for i in range(1, len(fractions)):
        result_gcd = gcd(result_gcd.numerator, fractions[i].numerator) // gcd(result_gcd.denominator, fractions[i].denominator)
    
    return result_gcd

'''
Returns (np.array): 
    returns a matrix with 4 elements: Splits a matrix into 4 square blocks [[A, B], [C, D]]
'''
def block_matrix(matrix):
    # Get the shape of the input matrix
    rows, cols = matrix.shape
    
    # Ensure the input matrix is square
    if rows != cols:
        raise ValueError("Input matrix should be square")

    # Split the matrix into 2x2 block matrices
    block0 = matrix[:rows//2, :cols//2]
    block1 = matrix[:rows//2, cols//2:]
    block2 = matrix[rows//2:, :cols//2]
    block3 = matrix[rows//2:, cols//2:]

    # Create a 2x2 matrix with block matrices as elements
    result_matrix = np.array([[block0, block1], [block2, block3]])

    return result_matrix

'''
Args:
    operator (np.array): A square  matrix of any dimension
Returns:
    A (np.array) and B (np.array), such that A ⊗ B = operator. A and B are both square matrices.
'''
def factor_matrix(operator):
    blocks = block_matrix(operator)
    rows, columns = blocks[0, 0].shape
    blocks = blocks.reshape(4, rows, columns) # [[A, B], [C, D]] -> [A, B, C, D]

    gcd0 = greatest_common_divisor(blocks[0].flatten())
    gcd1 = greatest_common_divisor(blocks[1].flatten())
    gcd2 = greatest_common_divisor(blocks[2].flatten())
    gcd3 = greatest_common_divisor(blocks[3].flatten())
    A = np.array([gcd0, gcd1, gcd2, gcd3])

    # Select any non zero block
    for block in blocks:
        if block.any():
            B = block
            break
    
    # Check that if we need to muliply times -1
    for i in range(4):
        if np.array_equal(A[i] * B,  -blocks[i]):
            A[i] = -A[i]

    A = A.reshape(2,2)
    
    return A, B

'''
Args:
    operator (numpy.array): an 8x8 matrix
Returns:
    An array of three 2x2 matrices
'''
def factorize_operator(operator):
    gate1, gate2 = factor_matrix(operator)
    gate2, gate3 = factor_matrix(gate2) 
    return gate1, gate2, gate3

def tensor_product(matrices):
    ret_mat = np.kron(matrices[0], matrices[1])
    for matrix in matrices[2:]:
        ret_mat = np.kron(ret_mat, matrix)
    return ret_mat

def get_pushed_pentagon(operator):
    
    pentagon = tn.Node(pentagon_tensor)
    
    pushed_operator = get_pushed_operator(operator)
    pushed_operator = tn.Node(operator_to_tensor(pushed_operator))
    
    network = TN()
    network.connect(pentagon[3], pushed_operator[0])
    network.connect(pentagon[4], pushed_operator[1])
    network.connect(pentagon[5], pushed_operator[2])

    pushed_pentagon = network.get_result()
    return pushed_pentagon

def network_equal(nw1, nw2):
    return np.array_equal(nw1.tensor, nw2.tensor)

def operator_to_node(operator):
    return tn.Node(operator_to_tensor(operator))

'''
Connect outcoming legs of gate1 to incoming legs of gate2
'''
def connect_two_gates(gate1, gate2):
    network = TN()
    network.connect(gate1[3], gate2[0])
    network.connect(gate1[4], gate2[1])
    network.connect(gate1[5], gate2[2])
    return network.get_result()

def new_pentagon_with_operator_to_the_left(operator):
    pentagon = tn.Node(pentagon_tensor)
    operator = operator_to_node(operator)
    new_pentagon = connect_two_gates(operator, pentagon)
    return new_pentagon

def new_pentagon_with_operator_to_the_right(operator):
    pentagon = tn.Node(pentagon_tensor)
    operator = operator_to_node(operator)
    new_pentagon = connect_two_gates(pentagon, operator)
    return new_pentagon

def new_pentagon():
    return tn.Node(pentagon_tensor)

def new_qubit(vector):
    qubit_node = tn.Node(np.array(vector))
    qubit = qubit_node[0]
    return qubit

# Defining some CONSTANTS

'''
The pentagon_operator defines the pentagon gate, which is an interpretation of the 5 qubit code,
where the first 3 bits are interpreted as input and the last 3 as output.
For example:
    Column 0 corresponds to the state U_{pengagon_gate} |000> (0 = 000 in binary), and respectively
    column 3 corresponds to the state U_{pengagon_gate} |011> (3 = 011 in binary).
'''
pentagon_operator = np.array(
    [[ 1.,  0.,  0., -1.,  0., -1., -1.,  0.],
       [ 0.,  1., -1.,  0., -1.,  0.,  0., -1.],
       [ 0.,  1.,  1.,  0., -1.,  0.,  0.,  1.],
       [-1.,  0.,  0., -1.,  0.,  1., -1.,  0.],
       [ 0., -1.,  1.,  0., -1.,  0.,  0., -1.],
       [ 1.,  0.,  0., -1.,  0.,  1.,  1.,  0.],
       [-1.,  0.,  0., -1.,  0., -1.,  1.,  0.],
       [ 0., -1., -1.,  0., -1.,  0.,  0.,  1.]])/2

pentagon_tensor = operator_to_tensor(pentagon_operator)
