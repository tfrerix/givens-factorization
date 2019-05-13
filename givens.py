import math
import numpy as np
import numpy.linalg as la
import functools

from util import Rotation

"""
Basic Givens transformation, cf. Section 5.1.9 in Golub/van Loan.
"""
def left_givensT(c, s, A, r1, r2):
    ''' update A <- G.T.dot(A) ... affects rows r1 and r2 '''
    givensT = np.array([[c, -s],   # manually transposed
                        [s,  c]])
    A[[r1, r2], :] = np.dot(givensT, A[[r1, r2], :])
    return A


def right_givens(c, s, A, c1, c2):
    ''' update A <- A.dot(G) ... affects cols c1 and c2 '''
    givens = np.array([[c, s],
                       [-s, c]])
    A[:, [c1, c2]] = np.dot(A[:, [c1, c2]], givens)
    return A


def max_abs_idx(A):
    """
    Methods for Jacobi eigenvalue algorithm.
    """
    return np.unravel_index(np.argmax(np.abs(A)), A.shape)


def compute_cs_jacobi(A, p, q):
    a_qq = A[q, q]
    a_pq = A[p, q]
    a_pp = A[p, p]
    theta = (a_qq - a_pp) / (2 * a_pq)
    if theta >= 0:
        t = 1 / (theta + np.sqrt(1 + theta**2))
    else:
        t = 1 / (theta - np.sqrt(1 + theta**2))
    c = 1 / np.sqrt(1 + t**2)
    s = t * c
    return c, s


def jacobi_diagonalization(M, max_iter=1000, verbose=False):
    """
    Run Jacobi algorithm to diagonalize symmetric matrix.
    """
    A = np.copy(M)
    n = A.shape[0]
    trajectory = []
    for it in range(max_iter):
        diag_A = np.diag(A)
        A[np.arange(n), np.arange(n)] = 0
        p, q = max_abs_idx(A)
        A[np.arange(n), np.arange(n)] = diag_A

        c, s = compute_cs_jacobi(A, p, q)
        A = right_givens(c, s, A, p, q)
        A = left_givensT(c, s, A, p, q)

        alpha = np.arctan2(s, c)
        trajectory += [(alpha, p, q)]
    return trajectory


def compute_angle_and_factor(U, i, j):
    d = U.shape[0]
    I = np.eye(d)
    c, s = compute_cs_qr(U, i, j)
    alpha = np.arctan2(s, c)
    U = left_givensT(c, s, U.copy(), i, j)
    G = left_givensT(c, s, I.copy(), i, j)
    return alpha, G, U


def compute_cs_qr(A, p, q):
    a_qq = A[q, q]
    a_pq = A[p, q]
    if np.abs(a_qq) < 1e-5 and np.abs(a_pq) < 1e-5:
        c = 1
        s = 0
    else:
        c = a_qq / np.sqrt(a_qq**2 + a_pq**2)
        s = a_pq / np.sqrt(a_qq**2 + a_pq**2)
    return c, s


def elimination_factorization(U, max_iter=None):
    """
    Structured elimination algorithm.
    """
    U = U.copy()
    d = U.shape[0]
    trajectory = []
    iter_counter = 0
    done = False
    for j in range(d):
        if done:
            break
        for i in reversed(range(j+1, d)):
            if max_iter is not None and iter_counter > max_iter:
                done = True
                break
            alpha, _, U = compute_angle_and_factor(U, i, j)
            trajectory.append((alpha, i, j))
            iter_counter += 1
    return trajectory


def minimize_frobenius_subspace(U, i, j):
    """
    Finds angle that minimizes Givens rotation in subspace (i,j).
    """
    u_ii = U[i, i]
    u_ij = U[i, j]
    u_ji = U[j, i]
    u_jj = U[j, j]

    g = lambda alpha : -2 * ((u_ii + u_jj) * np.cos(alpha) + (u_ij - u_ji) * np.sin(alpha))

    alpha = np.arctan2(u_ij - u_ji, u_ii + u_jj)
    alpha_shifted = alpha - np.sign(alpha) * np.pi
    if g(alpha) > g(alpha_shifted):
        alpha = alpha_shifted

    diff = g(0) - g(alpha)
    return diff, alpha


def greedy_baseline(U, max_iter=None):
    """
    Greedily chooses Givens factor that minimizes Frobenius norm to the target U
    """
    U = U.copy()
    d = U.shape[0]

    if max_iter is None:
        max_iter = d * (d - 1) // 2

    trajectory = []
    idx_to_check = list(range(d))
    rotation = Rotation()
    for it in range(max_iter):
        subspace_results = []
        for i in range(d):
            for j in range(i + 1, d):
                if i not in idx_to_check and j not in idx_to_check:
                    continue
                else:
                    f_diff, theta = minimize_frobenius_subspace(U, i, j)
                    subspace_results.append((f_diff, theta, (i, j)))
        _, theta, idx = max(subspace_results, key=lambda t: t[0])
        idx_to_check = list(idx)

        U[idx, :] = rotation.rotate(theta, U[idx, :])
        trajectory.append((theta, idx[0], idx[1]))
    return trajectory


def random_planted_matrix(d, n, replace='True'):
    """
    Generates a random sequence of n Givens factors in d dimensions, where the Givens angles are drawn
    uniformly from [-pi,pi].
    If replace is True the subspaces are sampled with replacement otherwise not.
    """
    all_idx = np.asarray(list(zip(*np.tril_indices(d,-1))))
    chosen_idx_positions = np.random.choice(len(all_idx), size=n, replace=replace)
    subspaces = all_idx[chosen_idx_positions]
    angles = 2*np.pi * (np.random.rand(len(subspaces)) - 0.5)
    U = np.eye(d)
    for s, alpha in zip(subspaces, angles):
        U = right_givens(math.cos(alpha), math.sin(alpha), U, s[0], s[1])
    return U
