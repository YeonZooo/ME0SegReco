import torch
from torch import Tensor
import numpy as np

from torchmetrics.utilities.compute import _safe_divide
from hist.intervals import clopper_pearson_interval
from matplotlib.axes._axes import Axes
from typing import cast, Any

from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import mplhep as mh

def save_fig(fig: Figure,
             output_name: str,
             log_dir: str,
             suffix_list: list[str] = ['.png', '.pdf'],

) -> None:
    log_dir = Path(log_dir)
    output_path = log_dir / output_name 

    for suffix in suffix_list:
        fig.savefig(output_path.with_suffix(suffix), bbox_inches="tight")
        plt.close(fig)

def compute_error(y: Tensor,
                  num: Tensor,
                  denom: Tensor,
                  coverage=0.68,
) -> tuple[Tensor, Tensor]:
    y = y.cpu().numpy()
    num = num.cpu().numpy()
    denom = denom.cpu().numpy()

    ylow, yup = clopper_pearson_interval(num, denom, coverage=coverage)

    yerr_low = y - ylow
    yerr_up = yup - y

    yerr_low = torch.from_numpy(yerr_low)
    yerr_up = torch.from_numpy(yerr_up)
    return yerr_low, yerr_up

def bin_centers(edges) -> Tensor:
    return (edges[:-1] + edges[1:]) / 2

def bin_widths(edges) -> Tensor:
    return edges[1:] - edges[:-1]
    
def plot(edges: Tensor,
         num: Tensor | None = None,
         denom: Tensor | None = None,
         ax: Axes | None = None,
         xlabel: str | None = None,
         ylabel: str | None = None,
         xlim: tuple[float, float] | None = None,
         ylim: tuple[float, float] | None = None,
         label: str | None = None,
         **kwargs: Any | None,
) -> tuple[Figure, Axes]:
    
    y = _safe_divide(num=num, denom=denom).cpu()
    yerr = [each.cpu() for each in compute_error(y, num, denom)]

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.get_figure()
    fig = cast(Figure, fig)
    ax = cast(Axes, ax)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    mh.histplot(H=y.tolist(),
                bins=edges.tolist(),
                histtype='errorbar', 
                yerr=yerr,
                xerr=True,
                ax=ax,
                elinewidth=3,
                linewidth=3,
                label=label,
                **kwargs)
    return fig, ax
#    mh.cms.label('Preliminary', data=False)


class Plot:
    def __init__(self, edges, ylabel, xlabel):
        self.edges = edges
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.fig, self.ax = plt.subplots(constrained_layout=True)
        self.ax.set_xlabel(xlabel, fontsize=15, labelpad=10, loc='right')
        self.ax.set_ylabel(ylabel, fontsize=15, labelpad=10, loc='top')

    def add(
        self,
        num,
        denom,
        label,
        **kwargs: Any,
    ):
        num = num.cpu()
        denom = denom.cpu()
    
        self.fig, self.ax = plot(
            edges=self.edges,
            num=num,
            denom=denom,
            label=label,
            ax=self.ax,
            **kwargs)

    def get_fig(self):
        self.ax.legend(loc='lower left', fontsize=15)
        self.ax.grid()
        self.fig.set_size_inches(6,6)
        return self.fig, self.ax
