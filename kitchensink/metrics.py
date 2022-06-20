"""Metrics for evaluating distances of different kind"""

from typing import Tuple, Sequence

import mdtraj as md
import numpy as np
import numba


def mean_square_displacement(trajs: Sequence[md.Trajectory], dtmax: int, nres: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean-squared displacement (MSD) for a set of MD trajectories.

    Parameters
    ----------
    trajs
        MD trajectories to calculate the MSD on
    dtmax
        Maximum timestep to use
    nres
        Number of residues of the protein

    Returns
    -------
    ndarray
        Timesteps used for the MSD
    ndarray
        Mean-square displacements

    """
    dts = np.arange(1, dtmax)
    msds = np.empty((dts.shape[0], nres))
    for i, dt in enumerate(dts):
        n = 0
        msd = np.empty(nres)
        for j, traj in enumerate(trajs):
            # There are a few trajectories with very few frames heavily
            # limiting the maximum timestep, so we skip those.
            if traj.shape[0] < dtmax:
                continue
            print(f"{i + 1}/{dts.shape[0]} :: {j + 1}/{len(trajs)}", end="\r")
            trajr = traj.reshape(-1, nres, 3)
            trajlen = trajr.shape[0]

            for t in range(dt, trajlen, dt):
                msd += ((trajr[t] - trajr[t - dt]) ** 2).sum(axis=1)
                n += 1
        msds[i] = msd / n
    return dts, msds


def pairwise_rmsd(traj: md.Trajectory, verbose: bool = False) -> np.ndarray:
    """
    Computes the pairwise root-mean-square deviation for all frames in a trajectory.

    Parameters
    ----------
    traj
        mdtraj trajectory

    Returns
    -------
    ndarray
        Distance matrix of shape (n_frames, n_frames)

    """
    rmsds = np.zeros((traj.n_frames, traj.n_frames), dtype=float)
    for i in range(traj.n_frames):
        if i % 10 == 0 and verbose:
            print(f"{i:5d}/{traj.n_frames}", end="\r")
        rmsds[i, i:] = md.rmsd(traj[i:], traj, frame=i)
    rmsds += rmsds.T
    np.fill_diagonal(rmsds, 0)
    return rmsds


@numba.jit(nopython=True)
def jensen_shannon(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon divergence,
    a symmetrized form of the Kullback-Leibler divergence.

    Parameters
    ----------
    a, b
        Arrays of distributions

    Returns
    -------
    float
        Jensen-Shannon divergence

    """
    m = 0.5 * (a + b)
    dam = np.nansum(a * np.log(a / m))
    dbm = np.nansum(b * np.log(b / m))
    return np.sqrt(0.5 * (dam + dbm))


@numba.jit(nopython=True)
def hungarian(a: np.ndarray, b: np.ndarray, alpha: float=0.5) -> float:
    """
    Computes a crude earth-movers distance using the hungarian algorithm.

    Parameters
    ----------
    a, b
        Arrays of distributions

    Returns
    -------
    float
        Hungarian distance

    """
    nbins = a.shape[0]
    emd = [0]
    for i in range(nbins):
        emd.append(a[i] + alpha * emd[i] - b[i])
    return abs(np.array(emd).sum())


INV_TWOPI = 1 / (2 * np.pi)
TWOPI = 2 * np.pi

@numba.jit(numba.float64[:](numba.float64[:], numba.int64, numba.float64[:]), nopython=True)
def _nround(x, decimals, out):
    return np.round_(x, decimals, out)


@numba.jit(numba.float64[:](numba.float64[:]), nopython=True)
def _round(x):
    out = np.empty_like(x)
    _nround(x, decimals=0, out=out)
    return out


@numba.jit(nopython=True)
def periodic(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the periodic distance of two (high-dimensional) angles.

    Parameters
    ----------
    a, b
        Arrays of angles

    Returns
    -------
    float
        Distance

    """
    diff = a - b
    dx = diff * INV_TWOPI
    dx -= _round(dx)
    dx *= TWOPI
    return np.sqrt((dx ** 2).sum())


@numba.jit(nopython=True)
def dist_pbc(a: np.ndarray, b: np.ndarray, bvs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance between two points with periodic boundary conditions,
    utilising the minimum image convention.

    Parameters
    ----------
    a, b
        Two list of points in 3D
    bvs
        List of 3x3 matrices representing the box vectors

    Returns
    -------
    ndarray
        Squared distances
    ndarray
        Distance vectors

    """
    dx = a - b
    bvdiag = np.array([bvs[0, i, i] for i in range(3)])
    dx -= (bvs * np.atleast_3d(_round(dx / bvdiag))).sum(axis=1)
    r2 = (dx ** 2).sum(axis=1)
    for ii in range(-1, 2):
        v1 = bvs[:, 0] * ii
        for jj in range(-1, 2):
            v12 = bvs[:, 1] * jj + v1
            for kk in range(-1, 2):
                ndx = dx + v12 + bvs[:, 2] * kk
                nr2 = (ndx ** 2).sum(axis=1)
                amin = nr2 < r2
                r2 = np.where(amin, nr2, r2)
                amin3d = np.column_stack((amin, amin, amin))
                dx = np.where(amin3d, ndx, dx)
    return r2, dx


@numba.njit()
def circular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate a circular distance between two angular vectors.

    Parameters
    ----------
    a, b
        Angular vectors

    Returns
    -------
    float
        Circular distance

    """
    dab = np.abs(a - b)
    return np.sqrt((np.min(((2 * np.pi) - dab, dab)) ** 2).sum())
