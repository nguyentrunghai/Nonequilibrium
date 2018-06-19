
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# see http://matplotlib.org/api/markers_api.html
MARKERS = ["v", "^", "<", ">", "o", ".", ",", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_" ] 


def plot_lines(xs, ys, xerrs=None, yerrs=None, xlabel=None, ylabel=None, out="out.pdf",
               legends=None, legend_pos="upper left", legend_fontsize=8,
               x_logscale=False,
               y_logscale=False,
               figure_size=(3.2, 3.2*6/8),
               dpi=300,
               fontsize=8,
               lw=1.5,
               alpha=1.,
               line_styles=None,
               markers=None,
               markersize=4,
               colors=None,
               xlimits=None, ylimits=None,
               n_xtics=5,
               n_ytics=5):
    """
    """
    font = {"fontname": "Arial"}

    assert type(xs) == list, "xs must be a list of 1D array"
    assert type(ys) == list, "ys must be a list of 1D array"
    assert len(xs) == len(ys), "xs and ys must have the same len"
    if xerrs is not None:
        assert type(xerrs) == list and len(xerrs) == len(xs), "xerrs must be a list of same len as xs"
    if yerrs is not None:
        assert type(yerrs) == list and len(yerrs) == len(ys), "yerrs must be a list of same len as ys"

    if legends is not None:
        assert len(legends) == len(xs), "legends has wrong len"


    if markers is None:
        markers = [None for _ in range(len(xs))]

    if line_styles is None:
        line_styles = ["-" for i in range(len(xs))]

    plt.figure(figsize=figure_size)
    ax = plt.axes()

    for i in range(len(xs)):

        if colors is None:

            if (xerrs is None) and (yerrs is None):
                plt.plot(xs[i], ys[i], linestyle=line_styles[i], marker=markers[i], ms=markersize, lw=lw, alpha=alpha)

            elif (xerrs is not None) and (yerrs is None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize,
                             lw=lw, alpha=alpha)

            elif (xerrs is None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i], ms=markersize,
                             lw=lw, alpha=alpha)

            elif (xerrs is not None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], marker=markers[i],
                             ms=markersize, lw=lw, alpha=alpha)

        else:

            if (xerrs is None) and (yerrs is None):
                plt.plot(xs[i], ys[i], linestyle=line_styles[i], color=colors[i], marker=markers[i],
                         ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], linestyle=line_styles[i], color=colors[i],
                             marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i],
                             marker=markers[i], ms=markersize, lw=lw)

            elif (xerrs is not None) and (yerrs is not None):
                plt.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], linestyle=line_styles[i], color=colors[i],
                             marker=markers[i], ms=markersize, lw=lw)

    ax.locator_params(axis='x', nbins=n_xtics)
    ax.locator_params(axis='y', nbins=n_ytics)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize, **font)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize, **font)

    axes = plt.gca()
    if xlimits is not None:
        axes.set_xlim(xlimits)

    if ylimits is not None:
        axes.set_ylim(ylimits)

    if x_logscale:
        ax.set_xscale("log")
    if y_logscale:
        ax.set_yscale("log")

    if legends is not None:
        plt.legend(legends, frameon=False, loc=legend_pos, fancybox=False, fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None

