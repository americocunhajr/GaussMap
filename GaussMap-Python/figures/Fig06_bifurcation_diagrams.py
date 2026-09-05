#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig06_bifurcation_diagrams.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Bifurcation diagrams for fixed-A slices
#
#  Computes four critical-orbit bifurcation diagrams for A=-3,-2,-1,0 and overlays the analytical contraction, flip, and saddle-node thresholds.
#
#  Mathematical model:
#
#      T(y) = A + B exp(-y^2),    B > 0.
#
#  The numerical and plotting parameters below are the finalized values
#  used to reproduce the corresponding figure in Ref. [1].  Each script
#  is self-contained and writes both PNG and PDF versions to ../output/.
#
#  Software requirements:
#      Python 3.10+; NumPy; Matplotlib
#      SciPy is additionally required by Figure 9.
#
#  Reference:
#
#  [1] A. Cunha Jr
#      The Generalized Gaussian Iterated Map
#      Manuscript prepared for submission to the SIAM Journal on
#      Applied Dynamical Systems (SIADS), 2026.
# -----------------------------------------------------------------


"""
Bifurcation diagrams for the normalized Gaussian map
====================================================

This script generates a publication-quality four-panel bifurcation diagram
for

    T(y) = A + B exp(-y^2),

using four fixed values of A and varying B. For each parameter value, the
critical orbit y_0=0 is evolved, a transient is discarded, and the final
iterates are plotted.

Analytical thresholds are superimposed without in-panel labels:
    - global contraction threshold B=sqrt(e/2);
    - fixed-point period-doubling threshold;
    - saddle-node thresholds when present.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig06.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig06.pdf"
DPI = 360

A_VALUES = (-3.0, -2.0, -1.0, 0.0)

SLICE_COLORS = {
    -3.0: "#0072B2",
    -2.0: "#009E73",
    -1.0: "#7B3294",
     0.0: "#D55E00",
}

B_MIN = 0.05
B_MAX = 8.00
N_B = 1800

N_TRANSIENT = 2500
N_KEEP = 260

POINT_SIZE = 0.22
POINT_ALPHA = 0.48

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.2,
    "figure.titlesize": 18,
})

def T(y, A, B):
    return A + B*np.exp(-y*y)

def period_doubling_threshold(A):
    y_pd = 0.5*(A + np.sqrt(A*A + 2.0))
    B_pd = np.exp(y_pd*y_pd)/(2.0*y_pd)
    return y_pd, B_pd

def saddle_node_thresholds(A):
    if A > -np.sqrt(2.0):
        return []

    discriminant = A*A - 2.0
    if discriminant < 0.0:
        return []

    root = np.sqrt(max(discriminant, 0.0))
    y_values = [
        0.5*(A + root),
        0.5*(A - root),
    ]

    values = []
    for y in y_values:
        B = -np.exp(y*y)/(2.0*y)
        values.append((y, B))

    values.sort(key=lambda item: item[1])
    return values

def compute_bifurcation_data(A, B_values):
    y = np.zeros_like(B_values)

    for _ in range(N_TRANSIENT):
        y = T(y, A, B_values)

    tail = np.empty((N_KEEP, B_values.size), dtype=float)

    for k in range(N_KEEP):
        y = T(y, A, B_values)
        tail[k, :] = y

    return tail

def robust_vertical_limits(A, B_values, tail):
    data = tail[np.isfinite(tail)]
    low, high = np.percentile(data, [0.15, 99.85])

    theoretical_low = A
    theoretical_high = A + np.max(B_values)

    low = max(low, theoretical_low)
    high = min(high, theoretical_high)

    span = max(high-low, 0.5)
    return low-0.035*span, high+0.035*span

def draw_panel(ax, A, panel_letter, B_values, tail):
    B_cloud = np.broadcast_to(B_values, tail.shape).ravel()
    y_cloud = tail.ravel()

    ax.scatter(
        B_cloud,
        y_cloud,
        s=POINT_SIZE,
        color=SLICE_COLORS[A],
        alpha=POINT_ALPHA,
        linewidths=0.0,
        rasterized=True,
        zorder=2
    )

    B_contraction = np.sqrt(np.e/2.0)
    ax.axvline(
        B_contraction,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.55,
        alpha=0.95,
        zorder=5
    )

    _, B_pd = period_doubling_threshold(A)
    if B_MIN <= B_pd <= B_MAX:
        ax.axvline(
            B_pd,
            color="#E69F00",
            linestyle="-.",
            linewidth=1.85,
            alpha=0.98,
            zorder=6
        )

    for _, B_sn in saddle_node_thresholds(A):
        if B_MIN <= B_sn <= B_MAX:
            ax.axvline(
                B_sn,
                color="#56B4E9",
                linestyle=":",
                linewidth=1.85,
                alpha=0.98,
                zorder=6
            )

    y_min, y_max = robust_vertical_limits(A, B_values, tail)

    ax.set_xlim(B_MIN, B_MAX)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$B$")
    ax.set_ylabel(r"post-transient state $y_n$")
    ax.set_title(
        rf"({panel_letter}) Fixed slice $A={A:g}$",
        pad=9
    )
    ax.grid(True, alpha=0.13)

def main():
    B_values = np.linspace(B_MIN, B_MAX, N_B)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.8, 9.9),
        constrained_layout=True
    )

    for ax, A, letter in zip(
        axes.ravel(),
        A_VALUES,
        ("a", "b", "c", "d")
    ):
        tail = compute_bifurcation_data(A, B_values)
        draw_panel(ax, A, letter, B_values, tail)

    legend_handles = [
        Line2D([0], [0], marker=".", linestyle="",
               color="#0072B2", markersize=7, label=r"slice $A=-3$"),
        Line2D([0], [0], marker=".", linestyle="",
               color="#009E73", markersize=7, label=r"slice $A=-2$"),
        Line2D([0], [0], marker=".", linestyle="",
               color="#7B3294", markersize=7, label=r"slice $A=-1$"),
        Line2D([0], [0], marker=".", linestyle="",
               color="#D55E00", markersize=7, label=r"slice $A=0$"),
        Line2D([0], [0], color="#4D4D4D", linestyle="--",
               linewidth=1.7, label=r"contraction threshold $B=\sqrt{e/2}$"),
        Line2D([0], [0], color="#E69F00", linestyle="-.",
               linewidth=1.8, label="fixed-point period doubling"),
        Line2D([0], [0], color="#56B4E9", linestyle=":",
               linewidth=1.8, label="saddle-node threshold"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.8
    )

    fig.suptitle(
        r"Bifurcation diagrams of the normalized Gaussian map "
        r"$T(y)=A+B e^{-y^2}$"
        + "\n"
        + r"Post-transient states of the critical orbit $y_0=0$"
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=DPI,
        bbox_inches="tight"
    )
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")

    plt.close(fig)

if __name__ == "__main__":
    main()
