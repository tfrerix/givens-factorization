import math
import numpy as np
from lapsolver import solve_dense
from givens_gpu import build_cost_matrix

class Rotation():
    """
    Givens rotation in 2D subspace.
    """
    def __init__(self):
        self.R = np.zeros((2,2))

    def rotate(self, alpha, v):
        c = math.cos(alpha)
        s = math.sin(alpha)
        self.R[0,0] = c
        self.R[0,1] = -s
        self.R[1,0] = s
        self.R[1,1] = c
        return self.R.dot(v)

def symmetrized_norm(U, V):
    """
    Computes the Frobenius error between U and V while minimizing over all
    possible signed permutations of the columns of V by solving a linear assignment problem.

    Solves a linear assigment problem to find the best matching between columns of U and V.
    The cost for the problem is the squared 2-norm of the difference of the two respective columns.
    """
    C = build_cost_matrix(U, V);
    row_idx, col_idx = solve_dense(C)
    best_frobenius_norm = np.sqrt(C[row_idx, col_idx].sum())
    return best_frobenius_norm

