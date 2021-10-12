"""
Module: LibFMP.C3.C3S2_DTW_plot
Author: Frank Zalkow, Meinard Mueller
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt
import LibFMP.B


def plot_matrix_with_points(C, P=np.empty((0, 2)), color='r', marker='o', linestyle='', **kwargs):
    """Compute the cost matrix of two feature sequences

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C: Matrix to be plotted
        P: List of index pairs, to be visualized on the matrix
        color: The color of the line plot
            See https://matplotlib.org/users/colors.html
        marker: The marker of the line plot
            See https://matplotlib.org/3.1.0/api/markers_api.html
        linestyle: The line-style of the line plot
            See https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
        **Kwargs: Arguments for `LibFMP.B.plot_matrix`

    Returns:
        im: The image plot
        line: The line plot
    """

    fig, ax, im = LibFMP.B.plot_matrix(C, **kwargs)
    line = ax[0].plot(P[:, 1], P[:, 0], marker=marker, color=color, linestyle=linestyle)

    if fig is not None:
        plt.tight_layout()

    return fig, im, line
