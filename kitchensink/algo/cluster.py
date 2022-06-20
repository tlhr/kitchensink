"""Clustering algorithms"""

from typing import Optional, Tuple, Literal

import mdtraj as md
import numba as nb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable


@nb.njit()
def cluster_multiplets(data: np.ndarray, weights: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merges multiplets of numbers or indices based on a distance cutoff.

    Parameters
    ----------
    data
        Data to be clustered with shape (n_points, n_features)
    weights
        Associated weights (n_points)
    cutoff
        Associated cutoff for cluster assignment

    Returns
    -------
    ndarray
        Clustered data
    ndarray
        Associated cluster weights

    """
    ndata = len(data)

    # We loop until our list of indices stops shrinking
    changed = True
    while changed:

        # We compare each pair of index lists, if their distance is
        # below the cutoff we take their average and insert it into
        # the first elements position. We only loop once, because the
        # array will change its size upon change.
        for i in range(ndata):
            for j in range(i + 1, ndata):
                changed = (np.abs(data[i] - data[j]) < cutoff).all()
                if changed:
                    # New point is a weighted average
                    data[i] = ((weights[i] * data[i] + weights[j] * data[j]) /
                               (weights[i] + weights[j]))

                    # Instead of explicitly deleting the point we
                    # just take a view of the array, this is a lot faster.
                    index = np.ones(ndata, dtype=np.bool8)
                    index[j] = 0
                    data = data[index]
                    weights[i] = weights[i] + weights[j]
                    weights = weights[index]

                    # The length has changed, which is why we need
                    # to exit both loops and start over.
                    ndata = len(data)
                    break
            if changed:
                break

    return data, weights


def gromos_clustering_traj(X: md.Trajectory, cutoff: float, atom_indices: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster structures using the GROMOS [1]_ clustering algorithm.

    Parameters
    ----------
    X
        Trajectory in mdtraj format
    cutoff
        Cutoff value in nm
    atom_indices
        Indices of the system to use for the RMSD calculation,
        uses all atoms by default.

    Returns
    -------
    ndarray
        The indices of all cluster centers
    ndarray
        The cluster assigned to each frame of the trajectory

    """
    n_points = X.n_frames

    # We only use the radius, so no need for the actual number of neighbors
    inds = np.arange(n_points, dtype=int)
    points = np.empty(n_points, dtype=object)
    for i in tqdm(inds):
        points[i] = inds[md.rmsd(X, X[i], atom_indices=atom_indices) < cutoff]

    # Prepare the indices, labels and the mask
    n_neighbors = np.array([len(p) for p in points])
    mask = np.ones(n_points, dtype=bool)
    labels = np.empty(n_points, dtype=int)

    cluster_centers = []
    i = 0
    while mask.sum() > 0:
        # We continuously update the mask and stop the process once
        # all points have been masked out (i.e. assigned)
        cc = inds[mask][n_neighbors[mask].argmax()]

        # The point with the most neighbors is the cluster center
        cluster_centers.append(cc)

        # All neighbors of that point form the cluster
        labels[points[cc]] = labels[cc] = i

        # We remove these points by masking them out
        mask[points[cc]] = mask[cc] = False
        i += 1

    return np.array(cluster_centers), labels


def gromos_clustering(X: np.ndarray, cutoff: float, metric: str="minkowski") -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster structures using the GROMOS [1]_ clustering algorithm.

    Parameters
    ----------
    X
        Points to cluster
    cutoff
        Cutoff value in nm
    metric
        Metric to use for distance calculation

    Returns
    -------
    ndarray
        The indices of all cluster centers
    ndarray
        The cluster assigned to each frame of the trajectory

    References
    ----------
    .. [1] Daura, X. et al. Peptide Folding: When Simulation Meets Experiment.
           Angewandte Chemie International Edition 38, 236-240 (1999).

    """
    n_points = X.shape[0]

    # We only use the radius, so no need for the actual number of neighbors
    nn = NearestNeighbors(radius=cutoff, metric=metric).fit(X)
    points = nn.radius_neighbors(return_distance=False)

    # Prepare the indices, labels and the mask
    inds = np.arange(n_points, dtype=int)
    n_neighbors = np.array([len(p) for p in points])
    mask = np.ones(n_points, dtype=bool)
    labels = np.empty(n_points, dtype=int)

    cluster_centers = []
    i = 0
    while mask.sum() > 0:
        # We continuously update the mask and stop the process once
        # all points have been masked out (i.e. assigned)
        cc = inds[mask][n_neighbors[mask].argmax()]

        # The point with the most neighbors is the cluster center
        cluster_centers.append(cc)

        # All neighbors of that point form the cluster
        labels[points[cc]] = labels[cc] = i

        # We remove these points by masking them out
        mask[points[cc]] = mask[cc] = False
        i += 1

    return np.array(cluster_centers), labels


class SOM:
    """
    SOM - Self-Organising-Map. [2]_

    A 2D neural network that clusters high-dimensional data iteratively.

    Parameters
    ----------
    nx
        Number of neurons on x-axis.
    ny
        Number of neurons on y-axis.
    ndims
        Dimension of input data.
    iterations
        Total number of iterations to perform.
        Should be at least 10 times the number of neurons.
    learning_rate
        The learning rate specifies the tradeoff between speed and accuracy of the SOM.
    distance
        The distance metric to use.
    init
        Initialization method. "pca" uses a grid spanned by the first two
        eigenvectors of the principal component analysis of the input data.
    grid
        Layout of the SOM, can be either rectangular or hexagonal with
        equidistant nodes. The latter can provide smoother visualization.
    train
        Training algorithm to use. Sequential picks random feature vectors one
        at a time, while batch mode trains using all features per iteration.
        This can significantly speed up convergence.
    neighbour
        Type of neighbourhood decay function to use. "bubble" uses a hard
        cutoff, "gaussian" falls off smoothly, and "epanechnikov" starts
        smoothly and ends with a hard cutoff.
    learning
        Type of decay for the learning rate. A linear decay can
        improve results in certain cases.
    seed
        Seed for the random number generator.

    Attributes
    ----------
    grid
        Grid with all x, y positiions of the nodes. Useful for visualization.
    weights
        Weight vectors of the SOM in shape = (nx, ny, ndims).

    Examples
    --------
    Here we train a 20 by 30 SOM on some colors:

    >>> som = SOM(20, 30, 3, iterations=400, learning_rate=0.2)
    >>> colors = np.array(
    ...     [[0., 0., 0.],
    ...      [0., 0., 1.],
    ...      [0., 1., 1.],
    ...      [1., 0., 1.],
    ...      [1., 1., 0.],
    ...      [1., 1., 1.],
    ...      [.33, .33, .33],
    ...      [.5, .5, .5],
    ...      [.66, .66, .66]]
    ... )
    >>> som.fit(colors)

    References
    ----------
    .. [2] Kohonen, T. Self-Organized Formation of Topologically Correct
           Feature Maps. Biological Cybernetics 43 (1), 59-69 (1982).

    """

    def __init__(
            self,
            nx: int,
            ny: int,
            ndims: int,
            iterations: int,
            learning_rate: float=0.5,
            distance: Literal["euclidean", "periodic"]="euclidean",
            init: Literal["random", "pca"]="random",
            grid: Literal["rect", "hex"]="rect",
            train: Literal["seq", "batch"]="seq",
            neighbour: Literal["gaussian", "bubble", "epanechnikov"]="gaussian",
            learning: Literal["exponential", "linear"]="exponential",
            seed: Optional[int]=None
    ):

        self._iterations = iterations
        self._init_learning_rate = learning_rate
        self._learning_rate = self._init_learning_rate
        self._ndims = ndims
        self._map_radius = max(nx, ny) / 2
        self._dlambda = self._iterations / np.log(self._map_radius)
        self._shape = (nx, ny)
        self._trained = False

        if seed is not None:
            np.random.seed(seed)

        # Establish training algorithm
        if train.startswith("seq"):
            self._type = "s"
        elif train.startswith("batch"):
            self._type = "b"
        else:
            e = "Invalid training type! Valid types: sequential, batch"
            raise ValueError(e)

        # Init distance type
        if distance.startswith("euclid"):
            self._dist = self._euclid_dist
        elif distance.startswith("per"):
            self._dist = self._periodic_dist
        else:
            e = "Invalid distance type! Valid types: euclidean, periodic"
            raise ValueError(e)

        # Init weights
        if init.startswith("r"):
            self.weights = np.random.rand(nx, ny, ndims)
        elif not init.startswith("p"):
            e = "Invalid initialization type! Valid types: random, pca"
            raise ValueError(e)

        # Init grid
        self._X, self._Y = np.meshgrid(np.arange(ny), np.arange(nx))
        if grid.startswith("r"):
            self._locX = self._X
            self._locY = self._Y
        elif grid.startswith("h"):
            self._locX = np.asarray([
                x + 0.5 if i % 2 == 0 else x
                for i, x in enumerate(self._X.astype(float))
            ])
            self._locY = self._Y * 0.33333
        else:
            e = "Invalid grid type! Valid types: rect, hex"
            raise ValueError(e)

        # Init neighbourhood function
        if neighbour.startswith("gauss"):
            self._nb = self._nb_gaussian
        elif neighbour.startswith("bub"):
            self._nb = self._nb_bubble
        elif neighbour.startswith("epa"):
            self._nb = self._nb_epanechnikov
        else:
            e = ("Invalid neighbourhood function!" +
                 "Valid types: gaussian, bubble, epanechnikov")
            raise ValueError(e)

        # Init learning-rate function
        if learning.startswith("exp"):
            self._lr = self._lr_exp
        elif learning.startswith("pow"):
            self._final_lr = self._init_learning_rate * np.exp(-1)
            self._lr = self._lr_pow
        elif learning.startswith("lin"):
            self._lr = self._lr_lin
        else:
            e = ("Invalid learning rate function!" +
                 "Valid types: exp, power, linear")
            raise ValueError(e)

        # Create empty index grid
        self.index = np.zeros(self._shape, dtype=np.int32)

        # Output grid for easier plotting
        self.grid = np.asarray(list(zip(self._locX.flatten(),
                                        self._locY.flatten())))

    def _init_weights(self, X: np.ndarray) -> None:
        """Initialize weights from PCA eigenvectors"""
        if not hasattr(self, "weights"):
            pca = PCA(n_components=self._ndims)
            comp = pca.fit(X).components_[:2]
            coeff = X.mean(0) + 5 * X.std(0) / self._shape[0]

            # Create grid based on PCA eigenvectors and std dev of features
            raw_weights = np.asarray([
                (coeff * (comp[0] * (x - 0.5 / self._shape[0]) +
                          comp[1] * (y - 0.5 / self._shape[1])))
                for x, y in zip(np.nditer(self._X.flatten()),
                                np.nditer(self._Y.flatten()))
            ]).reshape(self._shape + (self._ndims,))

            # Scale to (0, 1)
            full_shape = self._shape + (1,)
            self.weights = (
                (raw_weights - raw_weights.min(2).reshape(full_shape)) /
                raw_weights.ptp(2).reshape(full_shape)
            )

    @staticmethod
    def _nb_gaussian(dist: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-dist ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _nb_bubble(dist: np.ndarray, sigma: float) -> np.ndarray:
        # pylint: disable=unused-argument
        return dist

    @staticmethod
    def _nb_epanechnikov(dist: np.ndarray, sigma: float) -> np.ndarray:
        # pylint: disable=unused-argument
        return np.maximum(np.zeros_like(dist), 1 - dist ** 2)

    def _lr_exp(self, t: int) -> float:
        return self._init_learning_rate * np.exp(-t / self._iterations)

    def _lr_pow(self, t: int) -> float:
        return (self._init_learning_rate *
                (self._final_lr / self._init_learning_rate) **
                (t / self._iterations))

    def _lr_lin(self, t: int) -> float:
        return (self._init_learning_rate -
                (self._init_learning_rate * t * (np.exp(1) - 1) /
                 (self._iterations * np.exp(1))))

    def _euclid_dist(
            self,
            xmat: np.ndarray,
            index: Tuple[Optional[int], Optional[int]] = (None, None),
            axis: int = 2
    ) -> np.ndarray:
        return np.sqrt(((xmat - self.weights[index]) ** 2).sum(axis=axis))

    def _periodic_dist(
            self,
            xmat: np.ndarray,
            index: Tuple[Optional[int], Optional[int]] = (None, None),
            axis: int = 2
    ) -> np.ndarray:
        pi2 = np.pi * 2
        dx = (xmat - self.weights[index]) / pi2
        return np.sqrt((((dx - round(dx)) * pi2) ** 2).sum(axis=axis))

    def _train(self, X: np.ndarray) -> None:
        for t in range(self._iterations):
            # Update learning rate, reduce radius
            lr = self._lr(t)
            neigh_radius = self._map_radius * np.exp(-t / self._dlambda)

            # Choose random feature vector
            f = X[np.random.choice(len(X))]

            # Calc euclidean distance
            xmat = np.broadcast_to(f, self._shape + (self._ndims,))
            index = self._dist(xmat).argmin()
            bmu = np.unravel_index(index, self._shape)

            # Create distance matrix
            distmat = (
                (self._locX - self._locX[bmu]) ** 2 +
                (self._locY - self._locY[bmu]) ** 2
            ).reshape(self._shape + (1,))

            # Mask out unaffected nodes
            mask = (distmat < neigh_radius).astype(int)
            theta = self._nb(distmat * mask, neigh_radius)
            self.weights += mask * theta * lr * (f - self.weights)

    def _batch_train(self, X: np.ndarray) -> None:
        for t in range(self._iterations):
            # Update learning rate, reduce radius
            lr = self._lr(t)
            neigh_radius = self._map_radius * np.exp(-t / self._dlambda)

            for f in X:
                # Calc euclidean distance
                xmat = np.broadcast_to(f, self._shape + (self._ndims,))
                index = self._dist(xmat).argmin()
                bmu = np.unravel_index(index, self._shape)

                # Create distance matrix
                distmat = (
                    (self._locX - self._locX[bmu]) ** 2 +
                    (self._locY - self._locY[bmu]) ** 2
                ).reshape(self._shape + (1,))

                # Mask out unaffected nodes
                mask = (distmat < neigh_radius).astype(int)
                theta = self._nb(distmat * mask, neigh_radius)
                self.weights += mask * theta * lr * (f - self.weights)

    def fit(self, X: np.ndarray):
        """
        Run the SOM.

        Parameters
        ----------
        X
            Input data as array of vectors.

        """
        self._init_weights(X)
        if self._type == "s":
            self._train(X)
        else:
            self._batch_train(X)
        self._trained = True

    def create_index(self, X: np.ndarray):
        """
        Create an index grid, allowing the coloring of the map with arbitrary
        feature data. For instance, one could train the SOM on a subset of the
        data, and then create an index using the full dataset. The transform()
        method will only need to check the created index grid featuring the
        best matching datapoint index per node.

        Parameters
        ----------
        X
            Input data as used to train the SOM, can be significantly larger.

        """
        if not self._trained:
            raise ValueError("You need to train the SOM first!")

        # For each node we calculate the distance to each datapoint
        for index in np.ndindex(self._shape):
            self.index[index] = self._dist(X, index=index, axis=1).argmin()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform a dataset based on the index grid created by index().
        This method will return a subset of the dataset in the shape of
        the node matrix.

        Parameters
        ----------
        X
            Input data

        Returns
        -------
        ndarray
            Subset of the input data assigned to the best nodes

        """
        if not self._trained:
            raise ValueError("You need to train the SOM first!")
        if not hasattr(self, "index"):
            raise ValueError("You need to index the SOM first!")

        grid = np.zeros(self._shape)
        for index in np.ndindex(self.index.shape):
            grid[index] = X[self.index[index]]

        return grid
