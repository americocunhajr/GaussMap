#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig11_singularity_propagation.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Propagation of post-critical singularities and amplitudes
#
#  Combines a long-orbit empirical density, the post-critical sequence, and the theoretical singularity-amplitude law based on cumulative derivative growth.
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
Propagation of invariant-density singularities
==============================================

This script generates a publication-quality three-panel figure for the
normalized Gaussian map

    T(y) = A + B exp(-y^2),

illustrating the propagation law for square-root singularities along the
post-critical orbit

    c_0 = 0,
    c_{n+1} = T(c_n).

The theoretical amplitudes satisfy

    a_1 = rho(0)/sqrt(B),

and, for n >= 2,

    a_n/a_1
      = [ product_{k=1}^{n-1} |T'(c_k)| ]^(-1/2).

Panels:
    (a) numerical invariant-density histogram with post-critical singularity
        locations and marker heights proportional to a_n/a_1;
    (b) forward post-critical orbit c_n;
    (c) normalized theoretical amplitudes a_n/a_1 and cumulative derivative
        growth.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# OUTPUT AND NUMERICAL CONTROLS
# ================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig11.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig11.pdf"
DPI = 360

A = -1.0
B = 2.25

N_TRANSIENT = 50_000
N_ORBIT = 2_000_000
N_HIST_BINS = 360

N_CRITICAL_IMAGES = 18
N_LABELLED_IMAGES = 10

# Display interval chosen from the empirically occupied support.
DISPLAY_LEFT = -0.56
DISPLAY_RIGHT = 1.295


# ================================================================
# TYPOGRAPHY
# ================================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.2,
    "figure.titlesize": 18,
})


# ================================================================
# MAP AND POST-CRITICAL QUANTITIES
# ================================================================

def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def dT(y, B):
    """Derivative of the normalized Gaussian map."""
    return -2.0*B*y*np.exp(-y*y)


def postcritical_orbit(A, B, n_images):
    """Return c_1,...,c_n."""
    values = np.empty(n_images)
    y = 0.0

    for k in range(n_images):
        y = T(y, A, B)
        values[k] = y

    return values


def normalized_singularity_amplitudes(c_values, B):
    """
    Compute a_n/a_1 from the propagation formula.

    For n >= 2:
        a_n/a_1 = [prod_{k=1}^{n-1}|T'(c_k)|]^(-1/2).
    """
    n = len(c_values)
    amplitudes = np.ones(n)
    cumulative_derivative = np.ones(n)

    product = 1.0

    for j in range(1, n):
        product *= abs(dT(c_values[j-1], B))
        cumulative_derivative[j] = product
        amplitudes[j] = product**(-0.5)

    return amplitudes, cumulative_derivative


# ================================================================
# LONG-ORBIT HISTOGRAM
# ================================================================

def orbit_histogram(A, B, edges):
    """Estimate the physical invariant density from a long critical orbit."""
    y = 0.0

    for _ in range(N_TRANSIENT):
        y = T(y, A, B)

    counts = np.zeros(len(edges)-1, dtype=np.int64)
    remaining = N_ORBIT
    chunk = 200_000

    while remaining > 0:
        m = min(chunk, remaining)
        values = np.empty(m)

        for j in range(m):
            y = T(y, A, B)
            values[j] = y

        counts += np.histogram(values, bins=edges)[0]
        remaining -= m

    widths = np.diff(edges)
    density = counts/(counts.sum()*widths)
    centers = 0.5*(edges[:-1] + edges[1:])

    return centers, density, widths


# ================================================================
# MAIN FIGURE
# ================================================================

def main():
    c_values = postcritical_orbit(
        A,
        B,
        N_CRITICAL_IMAGES
    )

    amplitudes, cumulative_derivative = (
        normalized_singularity_amplitudes(c_values, B)
    )

    hist_edges = np.linspace(
        A,
        A+B,
        N_HIST_BINS+1
    )
    hist_x, hist_rho, hist_widths = orbit_histogram(
        A,
        B,
        hist_edges
    )

    density_color = "#9ECAE1"
    density_edge = "#3182BD"
    orbit_color = "#7B3294"
    amplitude_color = "#D55E00"
    derivative_color = "#009E73"
    marker_color = "#E69F00"
    neutral_color = "#4D4D4D"

    fig = plt.figure(
        figsize=(13.6, 10.0),
        constrained_layout=True
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.1, 1.0]
    )

    # ------------------------------------------------------------
    # Panel (a): density and singularity locations
    # ------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])

    ax.bar(
        hist_x,
        hist_rho,
        width=0.92*hist_widths,
        color=density_color,
        edgecolor=density_edge,
        linewidth=0.28,
        alpha=0.70,
        label=rf"long-orbit histogram ({N_HIST_BINS} bins)",
        zorder=1
    )

    density_scale = 0.92*np.max(hist_rho)
    relative_display = amplitudes/np.max(amplitudes)
    stem_heights = density_scale*relative_display

    for n, (c_n, height) in enumerate(
        zip(c_values, stem_heights),
        start=1
    ):
        if DISPLAY_LEFT <= c_n <= DISPLAY_RIGHT:
            ax.vlines(
                c_n,
                0.0,
                height,
                color=amplitude_color,
                linewidth=1.2 if n > 1 else 2.0,
                alpha=max(0.38, 0.95-0.035*n),
                zorder=4
            )

            ax.scatter(
                [c_n],
                [height],
                s=44 if n <= N_LABELLED_IMAGES else 25,
                facecolor=marker_color,
                edgecolor="white",
                linewidth=0.7,
                zorder=6
            )

            if n <= N_LABELLED_IMAGES:
                offset = 8 if n % 2 else -20
                ax.annotate(
                    rf"$c_{{{n}}}$",
                    xy=(c_n, height),
                    xytext=(4, offset),
                    textcoords="offset points",
                    ha="left",
                    va="bottom" if offset > 0 else "top",
                    fontsize=9.2,
                    color="#8A4100",
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.84
                    }
                )

    ax.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"invariant density $\rho(y)$")
    ax.set_title(
        "(a) Numerical density and propagated singularity locations",
        pad=9
    )
    ax.grid(True, alpha=0.16)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.97
    )

    ax.text(
        0.985,
        0.965,
        (
            r"orange stems: relative theoretical amplitudes "
            r"$a_n/a_1$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": amplitude_color,
            "alpha": 0.93
        }
    )

    # ------------------------------------------------------------
    # Panel (b): post-critical orbit
    # ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])

    n_values = np.arange(1, N_CRITICAL_IMAGES+1)

    ax.plot(
        n_values,
        c_values,
        color=orbit_color,
        marker="o",
        markersize=5.8,
        linewidth=1.8,
        markerfacecolor="white",
        markeredgewidth=1.5
    )

    ax.axhline(
        A,
        color=neutral_color,
        linestyle=":",
        linewidth=1.0,
        alpha=0.70
    )
    ax.axhline(
        A+B,
        color=neutral_color,
        linestyle=":",
        linewidth=1.0,
        alpha=0.70
    )

    for n, c_n in zip(n_values[:8], c_values[:8]):
        offset = (4, 7) if n % 2 else (4, -17)
        ax.annotate(
            rf"$c_{{{n}}}$",
            xy=(n, c_n),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.0,
            color=orbit_color
        )

    ax.set_xlabel(r"post-critical index $n$")
    ax.set_ylabel(r"$c_n=T^n(0)$")
    ax.set_title(
        "(b) Forward post-critical orbit",
        pad=9
    )
    ax.grid(True, alpha=0.17)

    # ------------------------------------------------------------
    # Panel (c): amplitudes and derivative growth
    # ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])

    ax.semilogy(
        n_values,
        amplitudes,
        color=amplitude_color,
        marker="o",
        markersize=5.4,
        linewidth=2.0,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label=r"$a_n/a_1$"
    )

    ax.semilogy(
        n_values,
        np.sqrt(cumulative_derivative),
        color=derivative_color,
        linestyle="--",
        linewidth=2.0,
        label=r"$\sqrt{\prod_{k=1}^{n-1}|T'(c_k)|}$"
    )

    ax.set_xlabel(r"singularity index $n$")
    ax.set_ylabel(r"normalized magnitude")
    ax.set_title(
        "(c) Propagation amplitudes and derivative growth",
        pad=9
    )
    ax.grid(True, which="both", alpha=0.17)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.97
    )

    ax.text(
        0.97,
        0.06,
        (
            r"$\frac{a_n}{a_1}"
            r"=\left(\prod_{k=1}^{n-1}|T'(c_k)|\right)^{-1/2}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": amplitude_color,
            "alpha": 0.93
        }
    )

    fig.suptitle(
        r"Propagation of invariant-density singularities for "
        r"$T(y)=A+B e^{-y^2}$ "
        + rf"with $A={A:.2f}$ and $B={B:.2f}$"
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
