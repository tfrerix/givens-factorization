import pytest
import numpy as np
from scipy.stats import ortho_group

TOL = 1e-3


def l1_norm(X):
    return np.sum(np.abs(X))

from givens_gpu import coordinate_descent_l1 as cd_gpu
from coordinate_descent import coordinate_descent as cd_np

def test_coordinate_descent():
    d = 50
    n = 100
    U = ortho_group.rvs(d).astype(np.float32)
    t_gpu = cd_gpu(U.copy(), n)
    t_np = cd_np(U.copy(), l1_norm, n)

    for k in range(n):
        val_np = t_np[k]
        val_gpu = t_gpu[k]
        assert np.isclose(val_np[0], val_gpu[0], atol=TOL)
        assert np.isclose(val_np[1], val_gpu[1], atol=TOL)
        assert np.isclose(val_np[2], val_gpu[2], atol=TOL)



from givens_gpu import build_cost_matrix

def test_build_cost_matrix():
    d = 256
    U = np.random.randn(d,d)
    V = np.random.randn(d,d)
    C_gpu = build_cost_matrix(U, V)
    C_np = np.zeros((d, 2*d))
    signed_V = np.concatenate([V, -V], axis=1)
    for i in range(d):
        for j in range(2*d):
            u = U[:,i]
            v = signed_V[:,j]
            temp = u - v
            C_np[i, j] = np.sum(temp * temp)
    assert np.allclose(C_gpu, C_np)

