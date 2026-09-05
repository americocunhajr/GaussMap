#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig01_representative_geometries.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Jul--Aug 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Representative geometries of the normalized Gaussian map
#
#  Generates the four representative map geometries used in Figure 1: global contraction, a three-fixed-point case, the exact fixed-point flip threshold, and a member of the explicit Collet--Eckmann family.
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
Final repository version for Figure 1.

The script uses the same parameter values and visual encoding as the final
manuscript figure.  Fixed points are obtained numerically by sign-change
bracketing and bisection, avoiding an additional root-finding dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig01.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig01.pdf"
DPI = 320

# Shared plotting interval used in all four panels of the published figure.
X_MIN, X_MAX = -3.0, 2.0
Y_MIN, Y_MAX = -3.0, 2.0

plt.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 14.0,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 17.0,
})

MAP_COLOR = "#0072B2"
IDENTITY_COLOR = "#D55E00"
STABLE_COLOR = "#2CA02C"
UNSTABLE_COLOR = "#D62728"
TANGENT_COLOR = "#2CA02C"
CE_COLOR = "#FF7F0E"
SHADE_COLOR = "#C6DBEF"


def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def dT(y, B):
    """Derivative of the normalized Gaussian map."""
    return -2.0*B*y*np.exp(-y*y)


def fixed_points(A, B, n_grid=30000):
    """Locate every fixed point in the plotting window using bisection."""
    grid = np.linspace(X_MIN, X_MAX, n_grid)
    values = T(grid, A, B) - grid
    roots = []
    for j in range(n_grid-1):
        if values[j] == 0.0:
            roots.append(grid[j])
        elif values[j]*values[j+1] < 0.0:
            left, right = grid[j], grid[j+1]
            fleft = values[j]
            for _ in range(70):
                mid = 0.5*(left+right)
                fmid = T(mid, A, B)-mid
                if fleft*fmid <= 0.0:
                    right = mid
                else:
                    left = mid
                    fleft = fmid
            roots.append(0.5*(left+right))
    unique = []
    for root in roots:
        if not unique or abs(root-unique[-1]) > 1.0e-8:
            unique.append(root)
    return np.asarray(unique)


def annotate_fixed_points(ax, A, B, roots, panel):
    """Draw stability-coded fixed points and short stability labels."""
    offsets = {
        "a": [(10, -24)],
        "b": [(-2, -26), (-48, 18), (8, 14)],
        "c": [(8, 13)],
        "d": [(8, 10), (8, -27), (8, -24)],
    }
    for j, root in enumerate(roots):
        eta = dT(root, B)
        stable = abs(eta) < 1.0
        if stable:
            ax.scatter([root], [root], s=68, facecolor=STABLE_COLOR,
                       edgecolor="black", linewidth=1.0, zorder=8)
        else:
            ax.scatter([root], [root], s=66, facecolor="none",
                       edgecolor=UNSTABLE_COLOR, linewidth=1.8, zorder=8)
        dx, dy = offsets.get(panel, [(7, 8)]*len(roots))[min(j, len(offsets.get(panel, [(7,8)]))-1)]
        ax.annotate("stable" if stable else "unstable", xy=(root, root),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9.7,
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                              edgecolor="none", alpha=0.84), zorder=9)


def draw_panel(ax, A, B, title, panel, pd_tangent=False, ce_orbit=False):
    """Draw one representative geometry."""
    x = np.linspace(X_MIN, X_MAX, 5000)
    ax.axvspan(A, A+B, color=SHADE_COLOR, alpha=0.36, zorder=0)
    ax.plot(x, T(x, A, B), color=MAP_COLOR, lw=2.4, zorder=3)
    ax.plot(x, x, color=IDENTITY_COLOR, ls="--", lw=1.5, zorder=2)

    roots = fixed_points(A, B)
    annotate_fixed_points(ax, A, B, roots, panel)

    # Critical value T(0)=A+B.  In the CE panel the critical point is part
    # of the highlighted preperiodic orbit and is therefore shown in orange.
    ccolor = CE_COLOR if ce_orbit else MAP_COLOR
    ax.scatter([0.0], [A+B], s=62, facecolor=ccolor, edgecolor="white",
               linewidth=0.8, zorder=10)
    ax.annotate(r"$T(0)=A+B$", xy=(0.0, A+B), xytext=(8, 12),
                textcoords="offset points", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="0.2", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                          edgecolor="none", alpha=0.84), zorder=11)

    if pd_tangent:
        ystar = 1.0/np.sqrt(2.0)
        xx = np.linspace(-0.05, 1.40, 250)
        yy = ystar - (xx-ystar)
        ax.plot(xx, yy, color=TANGENT_COLOR, ls=":", lw=1.8, zorder=5)
        # Inset used in the final figure to emphasize tangency with slope -1.
        iax = inset_axes(ax, width="35%", height="31%", loc="lower right", borderpad=1.9)
        xi = np.linspace(0.35, 1.30, 300)
        iax.plot(xi, T(xi, A, B), color=MAP_COLOR, lw=1.7)
        iax.plot(xi, xi, color=IDENTITY_COLOR, ls="--", lw=1.15)
        iax.plot(xi, ystar-(xi-ystar), color=TANGENT_COLOR, ls=":", lw=1.5)
        iax.scatter([ystar], [ystar], s=32, color=MAP_COLOR, zorder=5)
        iax.set_xticks([]); iax.set_yticks([])
        iax.set_title(r"$T'(y^*)=-1$", fontsize=8.8, pad=2)
        iax.grid(True, alpha=0.12)

    if ce_orbit:
        q = 1.0
        # Highlight 0 -> q -> -q -> -q as points on the graph (x,T(x)).
        xs = np.array([0.0, q, -q])
        ys = T(xs, A, B)
        ax.scatter(xs, ys, s=62, facecolor=CE_COLOR, edgecolor=IDENTITY_COLOR,
                   linewidth=0.9, zorder=12)
        labels = [r"$0\mapsto q$", r"$q\mapsto -q$", r"$-q\mapsto -q$"]
        offsets = [(6, -28), (8, 13), (-78, 8)]
        for xx, yy, label, ofs in zip(xs, ys, labels, offsets):
            ax.annotate(label, xy=(xx, yy), xytext=ofs,
                        textcoords="offset points", fontsize=9.2,
                        arrowprops=dict(arrowstyle="->", color="0.35", lw=0.7),
                        bbox=dict(boxstyle="round,pad=0.06", facecolor="white",
                                  edgecolor="none", alpha=0.82), zorder=13)

    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"$y_n$"); ax.set_ylabel(r"$y_{n+1}$")
    ax.set_title(title, pad=8)
    ax.grid(True, alpha=0.16)


def main():
    A1, B1 = -1.0, 0.80
    A2, B2 = -2.0, 3.00
    A3, B3 = 0.0, np.sqrt(np.e/2.0)
    q = 1.0
    A4 = -q*(1.0+np.exp(-q*q))/(1.0-np.exp(-q*q))
    B4 = 2.0*q/(1.0-np.exp(-q*q))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.6), constrained_layout=True)
    draw_panel(axes[0,0], A1, B1, "(a) Global contraction", "a")
    draw_panel(axes[0,1], A2, B2, "(b) Three fixed points", "b")
    draw_panel(axes[1,0], A3, B3, "(c) Period-doubling threshold", "c", pd_tangent=True)
    draw_panel(axes[1,1], A4, B4, "(d) Explicit Collet--Eckmann geometry", "d", ce_orbit=True)

    handles = [
        Line2D([0],[0], color=MAP_COLOR, lw=2.4, label=r"$T(y)$"),
        Line2D([0],[0], color=IDENTITY_COLOR, ls="--", lw=1.5, label=r"$y_{n+1}=y_n$"),
        Line2D([0],[0], color=TANGENT_COLOR, ls=":", lw=1.8, label="tangent at PD"),
        Line2D([0],[0], marker="o", ls="", markerfacecolor=STABLE_COLOR,
               markeredgecolor="black", markersize=7.5, label="stable fixed point"),
        Line2D([0],[0], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=UNSTABLE_COLOR, markeredgewidth=1.6,
               markersize=7.5, label="unstable fixed point"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, columnspacing=1.5, handlelength=2.4)
    fig.suptitle(r"Representative geometries of the normalized Gaussian map "
                 r"$T(y)=A+B e^{-y^2}$")
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
