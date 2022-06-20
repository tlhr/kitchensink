"""Various math routines"""

from typing import List, Tuple, Dict, Optional

import networkx as nx
import numba
import numpy as np
from scipy.linalg import eig
from scipy.special import i0  # pylint: disable=no-name-in-module
from sklearn.neighbors import NearestNeighbors

EPSILON = 1e-10


def intrinsic_dimension(X: np.ndarray, max_neighbors: int=1000, n_jobs: int=-1) -> float:
    """
    Finds the intrinsic dimension of the corresponding dataset.

    Uses the TWO-NN algorithm developed by Facco et al. [4]_

    Parameters
    ----------
    X
        Dataset of shape (n_samples, n_features)
    max_neighbors
        The maximum amount of neighbors to use for the nearest-neighbor estimation
    n_jobs
        Number of jobs / threads to run

    Returns
    -------
    float
        Intrinsic dimension of the dataset

    References
    ----------
    .. [4] Facco, E., Errico, M. D. X., Rodriguez, A. & Laio, A.
           Estimating the intrinsic dimension of datasets by a minimal
           neighborhood information. Sci Rep 1-8 (2017)

    """
    n_points = X.shape[0]
    nn = NearestNeighbors(n_neighbors=max_neighbors, n_jobs=n_jobs)
    shortest = nn.fit(X).kneighbors(X)[0][:, 1:3]
    mu = shortest[:, 1] / shortest[:, 0]
    mu.sort()
    x, y = np.log(mu), -np.log(1 - np.arange(n_points) / n_points)
    # Find the slope with intercept 0, use lower 90 % of points
    ind, *_ = np.linalg.lstsq(x[:-(n_points // 10), np.newaxis],
                              y[:-(n_points // 10)])
    return ind[0]


def estimate_koopman(data: List[np.ndarray], lag: int) -> np.ndarray:
    """
    Estimate the Koopman matrix.

    Parameters
    ----------
    data
        List of state vector trajectories
    lag
        Lag time for estimating the matrix

    Returns
    -------
    ndarray
        The Koopman matrix

    """
    try:
        import pyemma as pe
    except ImportError as err:
        raise ImportError("Koopman matrix estimation requires PyEMMA to be installed!") from err

    cl = pe.coordinates.covariance_lagged(
        data=data, lag=lag, weights="empirical",
        reversible=True, bessel=True)
    return np.linalg.pinv(cl.C00_) @ cl.C0t_


def stationary_distribution(X: np.ndarray) -> np.ndarray:
    """
    Calculate the equilibrium distribution of a transition matrix.

    Parameters
    ----------
    X
        Row-stochastic transition matrix

    Returns
    -------
    ndarray
        Stationary distribution, i.e. the left eigenvector associated with eigenvalue 1.

    """
    ev, evec = eig(X, left=True, right=False)
    mu = evec.T[ev.argmax()]
    mu /= mu.sum()
    return mu


# From: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def find_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find a rotation matrix encoding the rotation between two vectors.

    Parameters
    ----------
    a, b
        Vectors to find the rotation between

    Returns
    -------
    ndarray
        Rotation matrix

    """
    a, b = normed(a), normed(b)
    v = np.cross(a, b)
    skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + skew + skew @ skew * 1 / (1 + a @ b)


def smallest_circle_with_edge(adj_matrix: np.ndarray) -> Dict[Tuple[int, int], int]:
    """
    Given a graph as an adjacency matrix, finds the smallest circle including an edge.

    Parameters
    ----------
    adj_matrix
        2D square boolean matrix describing the connectivity

    Returns
    -------
    Dict[Tuple[int, int], int]
        The circle size for each edge, if there is no circle, returns -1 as the size.

    """
    edges = {}
    graph = nx.Graph(adj_matrix)
    for u, v in graph.edges:
        g = graph.copy()

        # Removing the edge temporarily lets us find the other
        # shortest path, which is equal to the smallest circle.
        g.remove_edge(u=u, v=v)
        edges[(u, v)] = nx.shortest_path_length(
            g, source=u, target=v) if nx.is_connected(g) else -1
    return edges


def center_of_mass(coordinates: np.ndarray, masses: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Calculate the center-of-mass of a molecule.

    Parameters
    ----------
    coordinates
        Nx3 array of coordinates
    masses
        N array of atomic masses, assumes uniform masses if not given

    Returns
    -------
    ndarray
        Center-of-mass of the molecule

    """
    if masses is None:
        masses = np.ones(coordinates.shape[0])
    return (masses.reshape(-1, 1) * coordinates).sum(axis=0) / masses.sum()


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Create a 3D rotation matrix about an axis.

    Parameters
    ----------
    axis
        Vector representing the axis
    theta
        Angle in radians

    Returns
    -------
    ndarray
        3x3 rotation matrix

    """
    return (np.cos(theta) * np.identity(3) +
            np.sin(theta) * np.cross(np.eye(3), axis) +
            (1 - np.cos(theta)) * np.tensordot(axis, axis, axes=0))


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle between 3 points in space.

    Parameters
    ----------
    a
        Vector for point a
    b
        Vector for point b
    c
        Vector for point c

    Returns
    -------
    float
        Angle between all three points in degrees

    """
    ab, cb = a - b, c - b
    ab /= np.linalg.norm(ab)
    cb /= np.linalg.norm(cb)
    dot = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))


def normal_to_plane(a: np.ndarray, b: np.ndarray, c: np.ndarray, rev: bool=False) -> np.ndarray:
    """
    Calculate the normal vector to a plane spanned by three points.

    Parameters
    ----------
    a
        Vector for point a
    b
        Vector for point b
    c
        Vector for point c
    rev
        Return the reverse cross-product

    Returns
    -------
    ndarray
        The normal of the plane

    """
    ab, bc = a - b, b - c
    ab /= np.linalg.norm(ab)
    bc /= np.linalg.norm(bc)
    if rev:
        return np.cross(bc, ab)
    return np.cross(ab, bc)


def dihedral_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """
    Calculate the dihedral angle given by 4 points.

    Parameters
    ----------
    a
        Vector for point a
    b
        Vector for point b
    c
        Vector for point c
    d
        Vector for point d

    Returns
    -------
    float
        The b-c dihedral angle in degrees

    """
    v1 = b - a
    v2 = c - b
    v3 = d - c
    c1 = np.cross(v1, v2)
    c2 = np.cross(v2, v3)
    psin = np.dot(np.linalg.norm(v2) * v1, c2)
    pcos = np.dot(c1, c2)
    return np.rad2deg(np.arctan2(psin, pcos))


def grid_torus(size: int, d: int=2) -> np.ndarray:
    """
    Create a grid of indices on a (hyper-)torus.

    Parameters
    ----------
    size
        Number of points for each dimension
    d
        Number of dimensions

    Returns
    -------
    ndarray
        Toroidal grid of indices, equivalent to meshgrid

    """
    return np.stack(np.meshgrid(
        *(np.linspace(0, 2 * np.pi, size)
          for _ in range(d)),
        indexing="ij")).T


@numba.jit(nopython=True)
def _toroidal_kde_kernel(data: np.ndarray, grid: np.ndarray, conc: int=25) -> np.ndarray:
    n_points, _ = data.shape
    n_grid = grid.shape[0]
    summand = np.zeros((n_grid, n_grid))
    # https://github.com/PyCQA/pylint/issues/2910
    for i in numba.prange(n_points):  # pylint: disable=not-an-iterable
        summand += np.exp(conc * np.cos(grid - data[i]).sum(axis=2))
    return summand


def toroidal_kde(data: np.ndarray, size: Optional[int]=None, conc: int=25) -> np.ndarray:
    """
    Compute the kernel density estimate on a (hyper-)torus.

    Parameters
    ----------
    data
        Dataset of (n_points, n_dim)
    size
        Number of points to evaluate on
    conc
        Concentration parameter for the KDE

    Returns
    -------
    ndarray
        Grid of evaluated points

    """
    n_points, n_dim = data.shape
    if size is None:
        size = 100 if n_dim == 2 else np.floor(10 ** (6 / n_dim))
    grid = grid_torus(size, n_dim)
    summand = _toroidal_kde_kernel(data=data, grid=grid, conc=conc)
    return summand / n_points / (2 * np.pi * i0(conc)) ** n_dim


def spherical_transform(vec: np.ndarray) -> np.ndarray:
    """
    Transform a vector from cartesian to spherical coordinates.

    Parameters
    ----------
    vec
        Cartesian vector

    Returns
    -------
    ndarray
        Spherical vector (r, phi, theta)

    """
    return np.array([
        np.linalg.norm(vec),
        np.arctan(np.linalg.norm(vec[:-1]) / vec[-1]),
        np.arctan(vec[1] / vec[0])
    ])


def cartesian_transform(vec: np.ndarray) -> np.ndarray:
    """
    Transform a vector from spherical to cartesian coordinates.

    Parameters
    ----------
    vec
        Spherical vector (r, phi, theta)

    Returns
    -------
    ndarray
        Cartesian vector

    """
    return np.array([
        vec[0] * np.sin(vec[1]) * np.cos(vec[2]),
        vec[0] * np.sin(vec[1]) * np.sin(vec[2]),
        vec[0] * np.cos(vec[1])
    ])


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    a, b
        Vectors to find the angle between

    Returns
    -------
    float
        Angle

    """
    return abs(np.arccos(a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))


def normed(a: np.ndarray) -> np.ndarray:
    """
    Compute the normed form of a vector.

    Parameters
    ----------
    a
        Vector to norm

    Returns
    -------
    ndarray
        Vector normalised to length 1

    """
    return a / np.linalg.norm(a)


def q_to_bw(q: float) -> float:
    """
    Converts a filter q value to the bandwidth.

    Parameters
    ----------
    q
        filter q value

    Returns
    -------
    float
        Bandwidth

    """
    return (np.log(1 + 1 / (2 * q ** 2) + np.sqrt(((2 * q ** 2 + 1) / (q ** 2)) ** 2 / 4 - 1))) / np.log(2)


def levi_civita(i: int, j: int, k: int) -> int:
    """
    Calculate the Levi-Civita symbol for indices i, j and k.

    Parameters
    ----------
    i, j, k
        Indices into tensor

    Returns
    -------
    int
        Levi-Civita symbol value

    """
    return (i - j) * (j - k) * (k - i) // 2


def triu_inverse(x: np.ndarray, n: int, offset: int=0) -> np.ndarray:
    """
    Converts flattened upper-triangular matrices into full symmetric matrices.

    Parameters
    ----------
    x
        Flattened matrices
    n
        Size of the n * n matrix

    Returns
    -------
    ndarray
        Array of shape (length, n, n)

    """
    length = x.shape[0]
    mat = np.zeros((length, n, n))
    a, b = np.triu_indices(n, k=offset)
    mat[:, a, b] = x
    mat += mat.swapaxes(1, 2)
    return mat


def matrix_inverse(mat: np.ndarray) -> np.ndarray:
    """
    Calculates the inverse of a square matrix.

    Parameters
    ----------
    mat
        Square real matrix

    Returns
    -------
    ndarray
        Inverse of the matrix

    """
    eigva, eigveca = np.linalg.eigh(mat)
    inc = eigva > EPSILON
    eigv, eigvec = eigva[inc], eigveca[:, inc]
    return eigvec @ np.diag(1. / eigv) @ eigvec.T


def covariances(data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates (lagged) covariances.

    Parameters
    ----------
    data
        Data at time t and t + tau

    Returns
    -------
    ndarray
        Inverse covariance
    ndarray
        Lagged covariance

    """
    chil, chir = data
    norm = 1. / chil.shape[0]
    C0, Ctau = norm * chil.T @ chil, norm * chil.T @ chir
    C0inv = matrix_inverse(C0)
    return C0inv, Ctau


def closest_reversible(K: np.ndarray, pi: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Find the closest reversible transition matrix with a given equilibrium distribution. [5]_

    Parameters
    ----------
    K
        Initial transition matrix
    pi
        Desired equilibrium distribution

    Returns
    -------
    ndarray
        New transition matrix with the given equilibrium distribution

    References
    ----------
    .. [5] Nielsen, A. J. N. et al. Computing the nearest reversible Markov chain.
           Numerical Linear Algebra with Applications 22 (3), 483-499 (2015)

    """
    from cvxopt import matrix, solvers  # pylint: disable=import-error,import-outside-toplevel
    def quadprog(Q: np.ndarray, f: np.ndarray, C: np.ndarray) -> np.ndarray:
        m, n = C.shape
        Q, f, C = matrix(Q), matrix(f), matrix(C)
        return solvers.qp(Q, f, G=C, h=matrix(np.zeros(m)),
                        A=matrix(np.atleast_2d(np.ones(n))),
                        b=matrix(np.array([1.])))
    if pi is None:
        pi = stationary_distribution(K)

    # We might not need to do anything...
    D = np.diag(pi)
    if (D @ K == K.T @ D).all():
        return K

    n = K.shape[0]
    m = (n - 1) * n // 2 + 1
    Dinv = np.linalg.inv(D)

    # Construct basis
    basis = np.empty((m, n, n))
    i = 0
    for r in range(n - 1):
        for s in range(r + 1, n):
            b = np.eye(n)
            b[r, s], b[s, r] = pi[s], pi[r]
            b[r, r], b[s, s] = 1 - pi[s], 1 - pi[r]
            basis[i] = b
            i += 1
    basis[-1] = np.eye(n)

    # Construct f, Q
    f, Q = np.empty(m), np.empty((m, m))
    for i in range(m):
        f[i] = -2. * np.trace(D @ basis[i] @ Dinv @ K.T)

    for i in range(m):
        Z = D @ basis[i] @ Dinv
        for j in range(m):
            t = 2 * np.trace(basis[j].T @ Z)
            Q[i, j] = Q[j, i] = t

    # Construct constraints
    C = -np.eye((m - 1 + n), m)
    C[-1, -1] = 0

    for i in range(n):
        idx = 0
        for r in range(n - 1):
            for s in range(r + 1, n):
                if s == i:
                    C[m - 2 + i, idx] = -1 + pi[r]
                elif r == i:
                    C[m - 2 + i, idx] = -1 + pi[s]
                else:
                    C[m - 2 + i, idx] = -1
                idx += 1
        C[m - 2 + i, m - 1] = -1

    # Actual optimization
    sol = quadprog(Q, f, C)
    x = np.array(sol["x"]).flatten()

    # Construct new matrix
    U = np.zeros((n, n))
    for i in range(m):
        U += x[i] * basis[i]

    return U
