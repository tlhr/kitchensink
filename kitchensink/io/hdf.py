from pathlib import Path
from typing import Union

import h5py


def h5tree(file: Union[Path, str], depth: int=-1):
    """
    Show the contents of an HDF5 file as a tree.

    Parameters
    ----------
    file
        Path to the HDF5 file
    depth
        Maximum depth to show (-1 for unlimited)

    """
    def _recurse(obj: Union[h5py.Group, h5py.Dataset], counter: int=0):
        # Stop recursing when hitting the maximum depth or a dataset
        if not isinstance(obj, h5py.Group) or counter == depth:
            return

        for k, o in obj.items():
            print(f"{'  ' * counter} - {k}")
            _recurse(o, counter + 1)

    with h5py.File(file, "r") as read:
        _recurse(read)
