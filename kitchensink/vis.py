"""Various plotting routines"""

import sys
from typing import Union, Sequence, List, Optional

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns

from .io import read_multi


def fast(filename: str,
         step: int=1,
         column_names: Optional[List[str]]=None,
         column_inds: Optional[List[int]]=None,
         start: int=0,
         stop: int=sys.maxsize,
         stat: bool=True,
         plot: bool=True) -> None:
    """
    Plot first column with every other column and show statistical information.

    Parameters
    ----------
    filename
        Plumed file to read.
    step
        Reads every step-th line instead of the whole file.
    columns
        Column numbers or field names to read from file.
    start
        Starting point in lines from beginning of file, including commented lines.
    stop
        Stopping point in lines from beginning of file, including commented lines.
    stat
        Show statistical information.
    plot
        Plot Information.

    """
    if column_names is not None:
        if "time" not in column_names:
            column_names.insert(0, "time")

    data = read_multi(
        filename,
        column_names=column_names,
        column_inds=column_inds,
        step=step,
        start=start,
        stop=stop,
        ret="horizontal"
    )

    if len(data["time"].values.shape) > 1:
        time = data["time"].values[:, 0]
        data = data.drop(["time"], axis=1)
        data["time"] = time

    fig = plt.figure(figsize=(16, 3 * len(data.columns)))

    i = 0
    for col in data.columns:
        if col == "time":
            continue
        i += 1
        ax = fig.add_subplot(len(data.columns) // 2 + 1, 2, i)
        ax.plot(data["time"], data[col])
        ax.set_xlabel("time")
        ax.set_ylabel(col)


def plot_its(its: np.ndarray, lags: np.ndarray, dt: float=1.0,
             colors: Optional[np.ndarray]=None):
    """
    Plot implied timescales.

    Parameters
    ----------
    its
        Implied timescales (n_its, n_lagtimes)
    lags
        Lagtimes used for the calculation
    dt
        Timestep of the underlying data
    colors
        Colors to be used for plotting

    """
    if colors is None:
        colors = sns.color_palette("husl", 8)
    multi = its.ndim == 3
    nits, nlags = its.shape[-2], its.shape[-1]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    if multi:
        itsm = its.mean(axis=0)
        cfl, cfu = np.percentile(its, q=(2.5, 97.5), axis=0)
    else:
        itsm = its

    ax.semilogy(lags * dt, lags * dt, color="k")
    ax.fill_between(lags * dt, ax.get_ylim()[0] * np.ones(len(lags)),
                    lags * dt, color="k", alpha=0.2)
    for i in range(nits):
        ax.plot(lags * dt, itsm[i], marker="o",
                linestyle="dashed", linewidth=1.5, color=colors[-(i + 2)])
        ax.plot(lags * dt, itsm[i], marker="o",
                linewidth=1.5, color=colors[-(i + 2)])
        if multi:
            ax.fill_between(lags * dt, cfl[i], cfu[i],
                            interpolate=True, color=colors[-(i + 2)], alpha=0.2)
    loc = ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax.set_ylim(0, 50000)
    ax.set_yticks(10 ** np.arange(6))
    ax.yaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"$\tau$ [ns]", fontsize=24)
    ax.set_ylabel(r"$t_i$ [ns]", fontsize=24)
    ax.tick_params(labelsize=24)
    sns.despine(ax=ax)
    return fig


def plot_ck(cke: np.ndarray, ckp: np.ndarray, lag: int, dt: float=1.0,
            ranges: float=0.2, colors: Optional[np.ndarray]=None):
    """
    Plot results of Chapman-Kolmogorov test.

    Parameters
    ----------
    cke
        Expected CK-test results
    ckp
        Predicted CK-test results
    lag
        Lagtime used for the test
    dt
        Timestep used for the transition matrix estimation
    ranges
        Subset of probability to show
    colors
        Colors to be used for plotting

    """
    if colors is None:
        colors = sns.color_palette("husl", 8)

    multi = cke.ndim == 4
    n = cke.shape[-2]
    steps = cke.shape[-1]

    if multi:
        ckem = cke.mean(axis=0)
        ckpm = ckp.mean(axis=0)
        ckep = np.percentile(cke, q=(2.5, 97.5), axis=0)
        ckpp = np.percentile(ckp, q=(2.5, 97.5), axis=0)
    else:
        ckem = cke
        ckpm = ckp

    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n), sharex=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x = np.arange(0, steps * lag, lag)
            if multi:
                ax.errorbar(x, ckpm[i, j], yerr=[ckpm[i, j] - ckpp[0, i, j], ckpp[1, i, j] - ckpm[i, j]],
                            linewidth=2, elinewidth=2)
                ax.fill_between(x, ckep[0, i, j], ckep[1, i, j],
                                alpha=0.2, interpolate=True, color=colors[1])
            else:
                ax.plot(x, ckpm[i, j], linestyle="-",
                        color=colors[0], linewidth=2)
            ax.plot(x, ckem[i, j], linestyle="--",
                    color=colors[1], linewidth=2)

            if i == j:
                ax.set_ylim(1 - ranges - 0.02, 1.02)
                ax.text(0, 1 - ranges, r"{0} $\to$ {1}".format(i, j),
                        fontsize=24, verticalalignment="center")
            else:
                ax.set_ylim(-0.02, ranges + 0.02)
                ax.text(0, 0.2, r"{0} $\to$ {1}".format(i, j),
                        fontsize=24, verticalalignment="center")
            ax.set_xticks(np.arange(0, steps * lag, lag), minor=True)
            ax.set_xticks(np.arange(0, steps * lag, 2 * lag))
            ax.set_xticklabels(
                (np.arange(0, steps * lag, 2 * lag) * dt).astype(int))
            ax.tick_params(labelsize=24)
    fig.text(0.5, 0.01 * 1.5 * n, r"$\tau$ [ns]", ha="center", fontsize=24)
    fig.text(0.01 * 1.5 * n, 0.5, r"$P$", va="center",
             rotation="vertical", fontsize=24)
    fig.subplots_adjust(wspace=0.25)
    return fig
