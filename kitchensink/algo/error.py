"""Error estimation routines based on resampling"""

from typing import Union, List, Callable, Sequence, Optional

import numpy as np
from scipy.optimize import curve_fit


def bootstrap(
        data: Union[List[np.ndarray], np.ndarray],
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        func_dim: int = 1,
        weights: Optional[Union[List[np.ndarray], np.ndarray]]=None,
        n_samples: int = 100) -> np.ndarray:
    """
    Compute a bootstrap estimate of a given dataset.

    Parameters
    ----------
    data
        One- or two-dimensional dataset of observations
        with shape (observations, features).
    func
        Function to use to compute the estimate. It should accept
        the data and a set of weights of the same length.
    func_dim
        Dimension of the array returned by `func`.
    weights
        Optional weights, either per-frame or additionally
        per-cluster with shape (observations, clusters).
    n_samples
        The number of samples to draw.

    Returns
    -------
    ndarray
        Bootstrap samples with shape (samples, clusters, features).

    """

    # We want list of arrays
    if not isinstance(data, list):
        data = [data]
    n_traj = len(data)

    # We assume multiple observables
    if data[0].ndim < 2:
        data = [np.atleast_2d(d).T for d in data]

    # Dummy weights if none are given
    if weights is None:
        weights = [np.ones(len(d)) / len(d) for d in data]

    # Same procedure as above
    if not isinstance(weights, list):
        weights = [weights]
    assert n_traj == len(weights)

    # We may have cluster weights instead of probabilities
    if weights[0].ndim < 2:
        weights = [np.atleast_2d(w).T for w in weights]
    n_clusters, n_obs = weights[0].shape[1], data[0].shape[1]

    samples = np.empty((n_traj * n_samples, n_clusters, n_obs, func_dim))
    count = 0
    for d, w in zip(data, weights):
        n_data = d.shape[0]
        for _ in range(n_samples):
            indices = np.random.randint(n_data, size=n_data)

            # Note that if n_clusters and n_obs > 1, func will need to perform an outer product!
            samples[count] = func(
                d[indices], w[indices] / w[indices].sum(axis=0))
            count += 1

    return np.squeeze(samples)


def circular_block_bootstrap(
        data: Union[List[np.ndarray], np.ndarray],
        corr: int,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        weights: Optional[Union[List[np.ndarray], np.ndarray]]=None,
        n_samples: int=100) -> np.ndarray:
    """
    Computes the circular block bootstrap sample for a time-correlated dataset. [3]_

    Parameters
    ----------
    data
        Trajectory or list of trajectories of shape (n_points, n_features)
    corr
        Time correlation constant in frames
    func
        The aggregation function taking a set of samples (n_points, n_features)
        with associated weights (n_points, n_clusters) and returning an aggregated
        array (n_clusters, n_features)
    weights
        Weights or list of weights (n_points, n_clusters) associated with each trajectory
    n_samples
        Number of bootstrap samples

    Returns
    -------
    ndarray
        Bootstrap samples (n_samples, n_clusters, n_features)

    References
    ----------
    .. [3] Lahiri, S. N. Resampling Methods for Dependent Data.
           Springer Series in Statistics, ISBN 978-1-4419-1848-2

    """

    # We want list of arrays
    if not isinstance(data, list):
        data = [data]
    n_traj = len(data)

    # We assume multiple observables
    if data[0].ndim < 2:
        data = [np.atleast_2d(d).T for d in data]

    # Dummy weights if none are given
    if weights is None:
        weights = [np.ones(len(d)) / len(d) for d in data]

    # Same procedure as above
    if not isinstance(weights, list):
        weights = [weights]
    assert n_traj == len(weights)

    # We may have cluster weights instead of probabilities
    if weights[0].ndim < 2:
        weights = [np.atleast_2d(w).T for w in weights]
    n_clusters, n_obs = weights[0].shape[1], data[0].shape[1]

    samples = np.empty((n_traj * n_samples, n_clusters, n_obs))
    count = 0
    for d, w in zip(data, weights):
        n_data = d.shape[0]
        n_blocks = np.ceil(n_data / corr).astype(int)

        for _ in range(n_samples):
            indices = np.random.randint(n_data, size=n_blocks)
            indices = (indices.reshape(-1, 1) +
                       np.arange(corr)).flatten() % n_data

            # Note that if n_clusters and n_obs > 1, func will need to perform an outer product!
            samples[count] = func(
                d[indices], w[indices] / w[indices].sum(axis=0))
            count += 1

    return np.squeeze(samples)


class Error:
    """
    Block Standard Error (BSE).

    Calculates the Block Standard Error by computing the weighted sample
    variance among blocks of the data with increasing size. An analytical
    curve is then fitted to this error data, the stationary value of which
    is the error estimate. With increasing block size, the effects of
    autocorrelation (causing an underestimation of the error) are reduced.
    Once autocorrelation is negligible, the error will reach a temporary
    stable value, before increasing again.

    Parameters
    ----------
    blocks
        A sequence of integers to use as blocks, or an integer. In the latter
        case, the block sizes will be chosen to be logarithmically increasing.
    verbose
        If true, will print out status messages.

    Attributes
    ----------
    all_errors_
        The computed errors.
    analytical_errors_
        The fitted analytical error function.

    """

    def __init__(self,
                 blocks: Union[int, Sequence[int]]=100,
                 verbose: bool=False):
        self.blocks = blocks
        self.verbose = verbose
        self._blocks: np.ndarray
        self.n_points: int
        self.n_data: int
        self.all_errors_: np.ndarray
        self.analytical_errors_: np.ndarray

    def _bse(self, data: np.ndarray, weights: Optional[np.ndarray]=None):
        """Run the block standard error calculation."""

        # We're working in 2D by default
        self.n_points = data.shape[0]
        if len(data.shape) < 2:
            data = data.reshape(-1, 1)
        self.n_data = data.shape[1]

        # Create weights if none are given
        if weights is None:
            weights = np.ones(self.n_points)
            weights /= weights.sum()

        # Logarithmically partition the block lengths
        if isinstance(self.blocks, int):
            self._blocks = np.unique(np.logspace(
                0, np.log10(self.n_points / 2), self.blocks, dtype=int))

        # Prepare constants, we're working with lists as numpy arrays
        # have to be uniform (i.e. equal sizes for all dimensions)
        errors = []
        wav = (data * weights.reshape(-1, 1)).sum(axis=0)

        # Increase block length beyond autocorrelation
        for j, length in enumerate(self._blocks):

            # We split the dataset into blocks
            n_blocks = self.n_points // length
            chunks = np.array_split(data, n_blocks)
            wchunks = np.array_split(weights, n_blocks)

            # Calculate the averages and weights for each block
            wavs = np.empty((n_blocks, self.n_data))
            chunk_weights = np.empty(n_blocks)
            for i, (cc, wc) in enumerate(zip(chunks, wchunks)):
                wavs[i] = (cc * wc.reshape(-1, 1)).sum(axis=0) / wc.sum()
                chunk_weights[i] = wc.sum()

            # Calculate the sample variance among the blocks
            sample_var = (
                (chunk_weights.reshape(-1, 1) * (wavs - wav) ** 2).sum(axis=0) /
                (1.0 - (chunk_weights ** 2).sum())
            )

            if self.verbose:
                print(f"{j+1}/{self._blocks.shape[0]} Length: {length}", end="\r")

            # The actual error is not the sample variance itself!
            errors.append(np.sqrt(sample_var / n_blocks))

        self.all_errors_ = np.array(errors)

    def _func_analytical(self, t: float, sigma: float, tau1: float,
                         tau2: float, alpha: float) -> float:
        """Analytical function to be fitted to the original errors."""
        e1 = alpha * (tau1 * (np.expm1(-t / tau1) * (tau1 / t) + 1))
        e2 = (1 - alpha) * (tau2 * (np.expm1(-t / tau2) * (tau2 / t) + 1))
        return sigma * np.sqrt((2 / self.n_points) * (e1 + e2))

    def _analytical_errors(self):
        """Fits an analytical curve to the original errors."""

        # Check we have errors to fit to
        if not hasattr(self, "all_errors_"):
            raise ValueError("You need to predict the errors before you can "
                             "get an analytical estimate! Run `_bse()` first.")

        self.analytical_errors_ = np.empty_like(self.all_errors_)

        # Calculate analytical errors for each column
        for i in range(self.n_data):
            if self.verbose:
                print(f"Calculating analytical error: {i}/{self.n_data}", end="\r")

            # Curve fit can fail, in which case we
            # just return the original errors
            try:
                paras, *_ = curve_fit(
                    self._func_analytical, self.blocks, self.all_errors_[:, i])
                self.analytical_errors_[:, i] = self._func_analytical(
                    self.blocks, *paras)
            except RuntimeError:
                self.analytical_errors_[:, i] = self.all_errors_[:, i]

    def fit_predict(self, data: np.ndarray,
                    weights: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Predict the errors for time-series data.

        Parameters
        ----------
        data
            Array of shape (n_samples, n_data) containing the time series data.
        weights
            1D Array of weights of length (n_samples) for each data point.
            If not given will assume uniform weight distribution.

        Returns
        -------
        ndarray
            1D array of errors for each column of the input data.

        """
        self._bse(data, weights)
        self._analytical_errors()
        return self.analytical_errors_[-1]
