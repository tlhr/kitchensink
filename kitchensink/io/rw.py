import binascii
from collections import OrderedDict
from contextlib import contextmanager
import glob
import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence, Union, Tuple, List, Iterator, Optional, Dict, Any, Literal

MaybePathType = Union[Path, str]

import numpy as np
import pandas as pd


def hex_to_bin(data: str) -> bytes:
    """
    Convert a string of hex values to binary data.

    Parameters
    ----------
    data
        String of space-separated hex values
    
    Returns
    -------
    bytes
        Resulting bytes object
    
    """
    return b"".join(binascii.unhexlify(char) for char in data.split())


def handle_path(path: MaybePathType, non_existent: bool=False) -> Path:
    """
    Check path validity and return `Path` object.

    Parameters
    ----------
    path
        Filepath to be checked.
    non_existent
        If false, will raise an error if the path does not exist.

    Returns
    -------
    Path
        The converted and existing path.

    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except Exception as err:
            message = "Couldn't read path {0}! Original message: {1}"
            raise ValueError(message.format(path, err))
    if not path.exists() and not non_existent:
        raise IOError("File {0} does not exist!".format(path))
    if not path.parent.exists():
        path.parent.mkdir()
    return path


def patch(file: Path, diff: Path):
    """
    Patch a file with a diff file.

    Parameters
    ----------
    file
        File to be patched
    diff
        Diff file
    
    """
    cmd = f"patch {file.as_posix()} {diff.as_posix()}"
    subprocess.run(cmd.split(), check=True)


@contextmanager
def working_dir(path: Path):
    """
    Ensure a function maintains a working directory after returning.

    Parameters
    ----------
    path
        Working directory
    
    """
    old = Path(".").absolute()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def is_plumed(file: str) -> bool:
    """
    Checks if the file is of plumed format.

    Parameters
    ----------
    file
        Path to plumed file.

    Returns
    -------
    bool
        Returns true if file is valid plumed format,
        raises ValueError otherwise.

    """
    with open(file, 'r') as f:
        head = f.readlines(0)[0]
        if head.startswith('#!'):
            return True
        else:
            raise ValueError('Not a valid plumed file')


def is_same_shape(data: List[np.ndarray]) -> bool:
    """
    Checks if a list of ndarrays all have the same shape.

    Parameters
    ----------
    data
        List of arrays

    Returns
    -------
    bool
        True if same shape, False if not

    """
    return len(set(d.shape for d in data)) == 1


def _offbyone_check(num1: int, num2: int) -> bool:
    """
    Check if two integers are the same by a margin of one.

    Parameters
    ----------
    num1
        First number
    num2
        Second number

    Returns
    -------
    bool
        True if numbers are the same with offset of one, False otherwise.

    """
    return num1 == num2 or num1 + 1 == num2 or num1 - 1 == num2


def read_plumed_fields(file: str) -> List[str]:
    """
    Reads the fields specified in the plumed file.

    Parameters
    ----------
    file
        Path to plumed file.

    Returns
    -------
    List[str]
        List of field names.

    """
    is_plumed(file)
    with open(file, 'br') as f:
        head = f.readlines(0)[0].split()[2:]
        fields = [x.decode('utf-8') for x in head]
    return fields


def plumed_iterator(file: str) -> Iterator[List[float]]:
    """
    Creates an iterator over a plumed file.

    Parameters
    ----------
    file
        Path to plumed file.

    Yields
    ------
    Iterator[List[float]]   
        List of floats for each line read.

    """
    is_plumed(file)
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            yield [float(n) for n in line.split()]


def file_length(file: str, skip_comments: bool=False) -> int:
    """
    Counts number of lines in file.

    Parameters
    ----------
    file
        Path to file.
    skip_comments
        Skipping comments is slightly slower,
        because we have to check each line.

    Returns
    -------
    int
        Length of the file.

    """
    with open(file, 'r') as f:
        i = -1
        if skip_comments:
            for line in f:
                if line.startswith('#'):
                    continue
                i += 1
        else:
            for i, _ in enumerate(f):
                pass
    return i + 1


def field_glob(
        fields: Union[str, Sequence[str]],
        full_fields: Sequence[str]
) -> List[str]:
    """
    Gets a list of matching fields from valid regular expressions.

    Parameters
    ----------
    fields
        Regular expression(s) to be used to find matches.
    full_fields
        Full list of fields to match from.

    Returns
    -------
    List[str]
        List of matching fields.

    """
    if isinstance(fields, str):
        fields = [fields]

    globbed = set()
    for field in fields:
        if field in full_fields:
            globbed.add(field)

        for f_target in full_fields:
            if re.search(field, f_target):
                globbed.add(f_target)

    return list(globbed)


def read_plumed(
        file: str,
        column_names: Optional[Union[List[str], str]]=None,
        column_inds: Optional[List[int]]=None,
        step: int=1,
        start: int=0,
        stop: int=sys.maxsize,
        replicas: bool=False,
        high_mem: bool=True,
        raise_error: bool=False,
        drop_nan: bool=True,
) -> Tuple[List[str], np.ndarray]:
    """
    Read a plumed file and return its contents as a 2D ndarray.

    Parameters
    ----------
    file
        Path to plumed file.
    columns
        Column numbers or field names to read from file.
    step
        Stepsize to use. Will skip every [step] rows.
    start
        Starting point in lines from beginning of file, including commented lines.
    stop
        Stopping point in lines from beginning of file, including commented lines.
    replicas
        Enable chunked reading (multiple datapoints per timestep).
    high_mem
        Use high memory version, which might be faster.
        Reads in the whole array and then slices it.
    raise_error
        Raise error in case of length mismatch.
    drop_nan
        Drop missing values.

    Returns
    -------
    List[str]
        List of field names.
    ndarray
        2D numpy ndarray with the contents from file.

    """
    return _read_plumed(file, dataframe=False, column_names=column_names, column_inds=column_inds,
                        step=step, start=start, stop=stop, replicas=replicas,
                        high_mem=high_mem, raise_error=raise_error, drop_nan=drop_nan)


def read_plumed_df(
        file: str,
        column_names: Optional[Union[List[str], str]]=None,
        column_inds: Optional[List[int]]=None,
        step: int=1,
        start: int=0,
        stop: int=sys.maxsize,
        replicas: bool=False,
        high_mem: bool=True,
        raise_error: bool=False,
        drop_nan: bool=True,
) -> pd.DataFrame:
    """
    Read a plumed file and return its contents as a dataframe.

    Parameters
    ----------
    file
        Path to plumed file.
    columns
        Column numbers or field names to read from file.
    step
        Stepsize to use. Will skip every [step] rows.
    start
        Starting point in lines from beginning of file, including commented lines.
    stop
        Stopping point in lines from beginning of file, including commented lines.
    replicas
        Enable chunked reading (multiple datapoints per timestep).
    high_mem
        Use high memory version, which might be faster.
        Reads in the whole array and then slices it.
    raise_error
        Raise error in case of length mismatch.
    drop_nan
        Drop missing values.

    Returns
    -------
    DataFrame
        Read dataframe

    """
    return _read_plumed(file, dataframe=True, column_names=column_names, column_inds=column_inds,
                        step=step, start=start, stop=stop, replicas=replicas,
                        high_mem=high_mem, raise_error=raise_error, drop_nan=drop_nan)


def _read_plumed(
        file: str,
        column_names: Optional[Union[List[str], str]]=None,
        column_inds: Optional[List[int]]=None,
        step: int=1,
        start: int=0,
        stop: int=sys.maxsize,
        replicas: bool=False,
        high_mem: bool=True,
        raise_error: bool=False,
        drop_nan: bool=True,
        dataframe: bool=True
) -> Union[Tuple[List[str], np.ndarray], pd.DataFrame]:
    """
    Read a plumed file and return its contents as a 2D ndarray.

    Parameters
    ----------
    file
        Path to plumed file.
    columns
        Column numbers or field names to read from file.
    step
        Stepsize to use. Will skip every [step] rows.
    start
        Starting point in lines from beginning of file, including commented lines.
    stop
        Stopping point in lines from beginning of file, including commented lines.
    replicas
        Enable chunked reading (multiple datapoints per timestep).
    high_mem
        Use high memory version, which might be faster.
        Reads in the whole array and then slices it.
    raise_error
        Raise error in case of length mismatch.
    drop_nan
        Drop missing values.

    Returns
    -------
    List[str]
        List of field names.
    ndarray
        2D numpy ndarray with the contents from file.

    """
    is_plumed(file)
    length = file_length(file)
    if stop != sys.maxsize and length < stop and raise_error:
        raise ValueError('Value for [stop] is larger than number of lines')

    all_fields, all_columns = set(), set()
    full_fields = read_plumed_fields(file)
    if column_names is not None:
        selected_column_names = field_glob(column_names, full_fields)
        all_fields |= set(selected_column_names)
        all_columns |= set(full_fields.index(f) for f in all_fields if f in full_fields)
    if column_inds is not None:
        all_fields |= set(full_fields[f] for f in column_inds)
        all_columns |= set(column_inds)
    fields = list(all_fields)
    columns = list(all_columns)

    full_array = (step == 1 and start == 0 and
                  stop == sys.maxsize and len(columns) == 0)

    if full_array or high_mem:
        nrows = stop - start if stop != sys.maxsize else None

        df = pd.read_csv(
            file,
            sep=r'\s+',
            header=None,
            comment='#',
            names=full_fields,
            dtype=np.float64,
            skiprows=start,
            nrows=nrows,
            usecols=columns,
        )

        # If several replicas write to the same file, we shouldn't use
        # the normal step, since we would only be reading a subset of
        # the replica data (worst case only one!). So we read in chunks
        # the size of the number of replicas.
        if replicas:
            data = pd.concat([
                dfg for i, (_, dfg) in enumerate(df.groupby('time'))
                if i % step == 0
            ])
        else:
            data = df[::step]

        if drop_nan:
            data.dropna(axis=0)

        if not dataframe:
            data = data.values

    else:
        with open(file, 'br') as f:
            data = np.genfromtxt(itertools.islice(f, start, stop, step),
                                 skip_header=1, invalid_raise=False,
                                 usecols=columns)
        if dataframe:
            data = pd.DataFrame(OrderedDict(zip(fields, data.T)))

    if not dataframe:
        return fields, data
    else:
        return data


def read_multi(
        files: Union[Sequence[str], str],
        ret: Literal["horizontal", "vertical", "mean"]="horizontal",
        **kwargs
) -> pd.DataFrame:
    """
    Read multiple Plumed files and return as concatenated dataframe.

    Parameters
    ----------
    files
        Sequence of (globbed) files to be read.
    ret
        Type of dataframe to return. With default horizontal concatenation,
        columns get sequentially modified names. 'vertical' simply elongates
        the dataframe, 'list' just returns a list of individual dataframes.
        'mean' computes the per cell average of all read files.
    kwargs
        Arguments passed to read_plumed().

    Returns
    -------
    DataFrame
        Dataframe with concatenated data.

    """

    if isinstance(files, str):
        files = [files]

    filelist: List[str] = []
    for file in files:
        if any(char in file for char in '*?[]'):
            filelist.extend(glob.iglob(file))
        else:
            filelist.append(file)

    dflist: List[str] = []
    for i, file in enumerate(filelist):
        df = read_plumed_df(file, **kwargs)

        # Horizontal concatenation requires unique column names
        if ret.startswith('h'):
            dflist.append(df.rename(columns={
                k: '{0}_{1}'.format(i, k) for k in df.columns
                if 'time' not in k
            }))
        else:
            dflist.append(df)

    # Horizontal
    if ret == "horizontal":
        data = pd.concat(dflist, axis=1)

    # Vertical
    elif ret == "vertical":
        data = pd.concat(dflist, axis=0)

    # Average over all dataframes
    elif ret == "mean":
        data = pd.concat(dflist).groupby(level=0).mean()

    else:
        raise ValueError('{0} is not a valid return type!'.format(ret))

    return data
