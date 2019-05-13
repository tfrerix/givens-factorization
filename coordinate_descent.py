"""
These methods implement a Python version of the main manifold coordinate descent algorithm.
They are only used for testing purposes and are easier to read than the CUDA code.
"""
import numpy as np
from util import Rotation
import operator


def coordinate_descent(U, f, max_iter=None):
    """
    Givens coordinate descent algorithm on the orthogonal group.

    f is a function from O(d) -> R
    """
    d = U.shape[0]
    if max_iter is None:
        max_iter = d * (d - 1) // 2

    trajectory = []

    idx_to_check = list(range(d))
    subspace_results = {}
    rotation = Rotation()

    for it in range(max_iter):
        for i in range(d):
            for j in range(i+1, d):
                if i not in idx_to_check and j not in idx_to_check:
                    continue
                else:
                    f_diff, theta = minimize_subspace(U, i, j, f)
                    subspace_results[(i,j)] = (f_diff, theta)
        idx, max_result = max(subspace_results.items(), key=lambda x : operator.itemgetter(1)(x)[0])
        theta = max_result[1]
        idx_to_check = list(idx)

        U[idx, :] = rotation.rotate(theta, U[idx, :].copy())
        trajectory.append((theta, idx[0], idx[1]))

    return trajectory


def compute_angles(X):
    return np.arctan2(X[1, :], X[0, :])


def map_angles_to_first_quadrant(angles_X):
    """
    Assumes the unit circle to parameterized in [-pi, pi] (output of arctan2) and maps all points to
    the interval [0, pi/2] such that they have the same angle with the axis as in their original quadrant
    when following them counterclockwise. This means for the quadrants to be mapped as:
    I   -> I
    II  -> II  - pi/2
    III -> III + pi
    IV  -> IV  + pi/2
    """
    first_quadrant = (angles_X >= 0) & (angles_X <= np.pi / 2)
    second_quadrant = (angles_X > np.pi / 2) & (angles_X <= np.pi)
    third_quadrant = (angles_X < -np.pi / 2) & (angles_X > -np.pi)
    fourth_quadrant = (angles_X < 0) & (angles_X > -np.pi / 2)

    transformed_angles = np.zeros_like(angles_X)
    transformed_angles[first_quadrant] = angles_X[first_quadrant]
    transformed_angles[second_quadrant] = angles_X[second_quadrant] - np.pi / 2
    transformed_angles[third_quadrant] = angles_X[third_quadrant] + np.pi
    transformed_angles[fourth_quadrant] = angles_X[fourth_quadrant] + np.pi / 2
    return transformed_angles


def minimize_subspace(X, i, j, distance_fn):
    """
    For an input X, get a vector v = X[[i,j],:] of size (2 x N) with N points in 2D, find the rotation that minimizes
    the entry-wise l1 norm / smoothed l1 norm of v.
    The specific distance function is specified by distance_fn and can be a l1 norm or smoothed l1 norm.
    """
    v = X[[i, j], :]
    v_distance = distance_fn(v)

    angles = map_angles_to_first_quadrant(compute_angles(v))
    assert np.all(angles > -1e-6) and np.all(angles < np.pi / 2 + 1e-6), angles

    # use apriori knowledge about smallest rotation angle to a coordinate axis
    below_diag_ind = (angles < np.pi / 4)
    above_diag_ind = (angles >= np.pi / 4)
    angles[below_diag_ind] = -angles[below_diag_ind]
    angles[above_diag_ind] = np.pi / 2 - angles[above_diag_ind]

    # simulate a rotation with all possible angles
    rotated_norms = np.zeros_like(angles)
    rotation = Rotation()
    for k, alpha in enumerate(angles):
        rotated_v = rotation.rotate(alpha, v)
        rotated_norms[k] = distance_fn(rotated_v)

    # choose the rotation that yields the overall minimal rotated norm
    k_argmin = np.argmin(rotated_norms)
    min_distance = rotated_norms[k_argmin]
    min_rotation_angle = angles[k_argmin]

    # compute absolute progress on the distance function for the two subselected rows
    distance_diff = v_distance - min_distance

    return distance_diff, min_rotation_angle
