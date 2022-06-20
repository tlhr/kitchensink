"""Various miscellaneous utilities"""

from collections import UserList
from functools import wraps, update_wrapper
from inspect import signature
import itertools
import re
from typing import (Sequence, Union, Any, Iterator, AnyStr, Type,
                    Callable, Tuple, Optional, Dict, List, TypeVar)
import warnings

import numpy as np

T = TypeVar("T")
MaybeListType = Union[List[T], T]


def typecheck(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Checks type annotated function arguments."""
    @wraps(func)
    def decorator(*args, **kwargs):
        if hasattr(func, '__annotations__'):
            hints = func.__annotations__
            sig = signature(func)
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in hints and not isinstance(value, hints[name]):
                    raise TypeError(f"Type mismatch: {name} != {hints[name]}")

        return func(*args, **kwargs)
    return decorator


def deprecated(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator for deprecating functions and methods. Raises a DeprecationWarning."""
    @wraps
    def inner(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(f"{func.__name__} is deprecated.",
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter("default", DeprecationWarning)
        return func(*args, **kwargs)
    return inner


class lazy_property:
    """Decorator to enable lazy property evaluation."""

    def __init__(self, fget):
        self.fget = fget
        update_wrapper(self, fget)

    def __get__(self, instance, owner):
        if instance is None:
            return self

        value = self.fget(instance)
        setattr(instance, self.fget.__name__, value)
        return value


def get_serializable_attributes(obj: object) -> Dict[str, Any]:
    """
    Finds all object attributes that are serializable with HDF5.

    Parameters
    ----------
    obj
        Object to serialize

    Returns
    -------
    Dict[str, Any]
        All serializable public attributes

    """
    valids = {int, float, str, list}
    return {k: v for k, v in obj.__dict__.items()
            if any(isinstance(v, valid) for valid in valids)
            and not k.startswith("_")}


def make_list(item: MaybeListType[T], cls: Type=list) -> List[T]:
    """
    Turn an object into a list, if it isn't already.

    Parameters
    ----------
    item
        Item to contain in a list

    Returns
    -------
    list
        List with item as only element

    """
    if not isinstance(item, list):
        item = [item]
    return cls(item)


class ReversibleList(UserList):
    """
    List object that is identical to its reversed counterpart.

    The only implemented operations are the equality operator
    and hashing, other comparisons will NOT work!

    """

    def __eq__(self, other):
        return super().__eq__(other) or super().__eq__(other[::-1])

    def __hash__(self):
        return hash(tuple(self.data)) + hash(tuple(self.data[::-1]))


class KeepLast(UserList):
    """
    A list that will always keep the first item.

    Parameters
    ----------
    data
        Data to construct the list from

    """
    def __init__(self, data: Sequence[T]):
        super().__init__()
        self.data = list(reversed(data))

    def pop_first(self) -> T:
        """
        Returns the first item from the list, but only deletes
        it if there's at least one more item in the list.

        Returns
        -------
        Any
            First item

        """
        if len(self) < 2:
            return self.data[-1]
        return self.data.pop(-1)


def concat(data: Sequence[Sequence[T]]) -> List[T]:
    """
    Concatenate multiple sequences together.

    """
    return list(itertools.chain(*data))


def dejitter(arr: Sequence[T]) -> List[T]:
    """
    Remove consecutive elements from list.

    """
    new = [arr[0]]
    prev = arr[0]
    for ele in arr[1:]:
        if ele != prev:
            prev = ele
            new.append(ele)
    return new


def transpose(data: List[Tuple[Any]]) -> Tuple[List[Any], ...]:
    """
    Transpose a list of tuples to a tuple of lists.

    Parameters
    ----------
    data
        List of tuples with arbitrary data

    Returns
    -------
    Tuple[List[Any], ...]
        Transposed data

    """
    tup_len, dat_len = len(data[0]), len(data)
    return tuple([data[j][i] for j in range(dat_len)] for i in range(tup_len))


def find_blocks(data: List[bool]) -> List[Tuple[int, int]]:
    """
    Finds the starting and end point of blocks of booleans sequences.

    Parameters
    ----------
    data
        list of boolean values

    Returns
    -------
    List[Tuple[int, int]]
        list of tuples with start and end indices

    """
    # Special initialisation because we could begin at 0
    prev, begin, end = False, -1, -1
    blocks = []
    for i, v in enumerate(data):
        if v and not prev:
            begin = i
        elif prev and not v:
            end = i - 1

        # Special case: ending on True
        elif i == len(data) - 1:
            end = i
        if begin >= 0 and end >= 0:
            blocks.append((begin, end))
            begin, end = -1, -1
        prev = v
    return blocks


def filter_glob(items: Sequence[AnyStr], patterns: Sequence[AnyStr]) -> Optional[Iterator[AnyStr]]:
    """
    Filter a sequence of objects according to a sequence of patterns.

    Parameters
    ----------
    items
        Sequence of arbitrary items
    patterns
        Sequence of arbitrary regular expression patterns

    Returns
    -------
    Iterator[AnyStr]
        Filtered item

    """
    for item in items:
        for pattern in patterns:
            if re.search(pattern, item):
                yield item


def unflatten(source: np.ndarray, lengths: Sequence[int], axis: int=0) -> List[np.ndarray]:
    """
    Converts a flat array into a list of arrays, according to specified lengths.

    Parameters
    ----------
    source
        Flat array to be split, can be any dimension
    lengths
        Single nested list of lengths of individual arrays
    axis
        The axis along which to split the array

    Returns
    -------
    List[ndarray]
        List of arrays

    """
    assert np.array(lengths).sum() == source.shape[axis], "The source array must be divisible by lengths!"
    return np.array_split(source, np.array(lengths).cumsum()[:-1], axis=axis)


# https://github.com/python/mypy/issues/731
# _RecList = Union[int, List["_RecList"]]
def build(size: Union[int, Any]) -> Union[int, Any]:
    """
    Build an arbitrarily nested list of arrays.

    Parameters
    ----------
    size
        (Nested) list of integers representing the sizes of the subarrays

    Returns
    -------
    ndarray
        Nested arrays

    """
    if isinstance(size, int):
        return np.arange(size)
    return [build(arr) for arr in size]


def extend_array(arr: np.ndarray, modulo: int) -> np.ndarray:
    """
    Extend a 2D array with zeros along axis 1, so that it's width is divisible
    by modulo.

    Parameters
    ----------
    arr
        Numpy array to be extended.
    modulo
        Divisor, so that arr.shape[1] % modulo == 0

    Returns
    -------
    ndarray
        extended numpy array

    """
    if arr.shape[1] % modulo == 0:
        return arr
    else:
        return np.column_stack(
            (arr, np.zeros((arr.shape[0], modulo - arr.shape[1] % modulo),
                           dtype=arr.dtype))
        )


def hypercube(n: int) -> np.ndarray:
    """
    Create hypercube coordinates.

    Parameters
    ----------
    n
        Dimensionality

    Returns
    -------
    ndarray
        2D numpy array with all cube vertices.

    """
    return np.asarray(list(itertools.product((0, 1), repeat=n)))


def make_blobs(dim: int, npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create blobs of points in higher dimensions on corners of a hypercube.
    Good for testing clustering algorithms.

    Parameters
    ----------
    dim
        Dimensionality of the dataset
    npoints
        Number of points to generate

    Returns
    -------
    ndarray
        2D array containing the positions of all points
    ndarray
        1D array containing the cluster identity

    """
    hc = hypercube(dim)
    highd = np.empty((npoints, dim))
    clusters = np.empty((npoints,), dtype=int)
    for i in range(npoints):
        index = np.random.randint(0, hc.shape[0])
        highd[i] = hc[index] + np.random.randn(dim) / 10
        clusters[i] = index
    return highd, clusters


def array_to_pointer(arr: np.ndarray) -> np.ndarray:
    """
    Convert 2D numpy arrays to array of pointers.

    Parameters
    ----------
    arr
        Array to be converted, should be C-contiguous

    Returns
    -------
    ndarray
        Array of pointers

    """
    return (arr.__array_interface__['data'][0] +
            np.arange(arr.shape[0]) * arr.strides[0]).astype(np.uintp)


def align_array(arr: np.ndarray, alignment: int=32) -> np.ndarray:
    """
    Align numpy array to a specific boundary.

    Parameters
    ----------
    arr
        Numpy array
    alignment
        Alignment in bytes

    Returns
    -------
    ndarray
        Aligned array

    """
    # Already aligned
    if (arr.ctypes.data % alignment) == 0:
        return arr

    extra = alignment // arr.itemsize
    buffer = np.empty(arr.size + extra, dtype=arr.dtype)
    offset = (-buffer.ctypes.data % alignment) // arr.itemsize
    newarr = buffer[offset:offset + arr.size].reshape(arr.shape)
    np.copyto(newarr, arr)
    assert (newarr.ctypes.data % alignment) == 0
    return newarr
