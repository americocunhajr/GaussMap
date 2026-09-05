#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig10_perron_frobenius_evolution.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Jul 29, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Perron--Frobenius evolution of probability densities
#
#  Iterates the closed-form Perron--Frobenius operator, illustrating formation and transport of fold singularities and comparison with a long-orbit histogram.
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

A = -1.0
B = 2.25

I_LEFT = A
I_RIGHT = A + B

DISPLAY_LEFT = -0.56
DISPLAY_RIGHT = 1.29

N_GRID = 10_000
SNAPSHOTS = (0, 1, 2, 4, 8, 16, 40)

N_TRANSIENT = 50_000
N_ORBIT = 1_000_000
N_HIST_BINS = 360
N_CRITICAL_IMAGES = 10

DPI = 360

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig10.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig10.pdf"

plt.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 14,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.0,
    "figure.titlesize": 17.0,
})


def T(x):
    return A + B*np.exp(-x*x)


def initial_density(x):
    s = (x-I_LEFT)/(I_RIGHT-I_LEFT)
    rho = np.zeros_like(x)
    mask = (s >= 0.0) & (s <= 1.0)
    rho[mask] = 1.0 - np.cos(2.0*np.pi*s[mask])
    rho /= np.trapezoid(rho, x)
    return rho


def perron_frobenius_step(rho, x):
    y = x
    ratio = (y-A)/B

    new_rho = np.zeros_like(rho)
    valid = (ratio > 0.0) & (ratio < 1.0)

    r = np.sqrt(-np.log(ratio[valid]))
    denominator = 2.0*r*(y[valid]-A)

    x_plus = r
    x_minus = -r

    rho_plus = np.interp(
        x_plus, x, rho, left=0.0, right=0.0
    )
    rho_minus = np.interp(
        x_minus, x, rho, left=0.0, right=0.0
    )

    new_rho[valid] = (rho_plus+rho_minus)/denominator
    new_rho = np.maximum(new_rho, 0.0)

    mass = np.trapezoid(new_rho, x)

    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError(
            "Invalid mass during transfer-operator iteration."
        )

    new_rho /= mass
    return new_rho


def long_orbit_histogram(edges):
    y = 0.0

    for _ in range(N_TRANSIENT):
        y = T(y)

    values = np.empty(N_ORBIT)

    for j in range(N_ORBIT):
        y = T(y)
        values[j] = y

    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts/(counts.sum()*widths)
    centers = 0.5*(edges[:-1]+edges[1:])

    return centers, density, widths


def postcritical_images(n):
    values = np.empty(n)
    y = 0.0

    for k in range(n):
        y = T(y)
        values[k] = y

    return values


def readable_upper_limit(*arrays):
    values = np.concatenate([
        np.asarray(a)[np.isfinite(a)] for a in arrays
    ])
    return 1.15*np.percentile(values, 99.8)


def main():
    dx = (I_RIGHT-I_LEFT)/N_GRID
    x = I_LEFT + (np.arange(N_GRID)+0.5)*dx

    rho = initial_density(x)
    snapshots = {0: rho.copy()}

    for n in range(1, max(SNAPSHOTS)+1):
        rho = perron_frobenius_step(rho, x)

        if n in SNAPSHOTS:
            snapshots[n] = rho.copy()

    hist_edges = np.linspace(
        I_LEFT, I_RIGHT, N_HIST_BINS+1
    )
    hist_x, hist_rho, hist_widths = long_orbit_histogram(
        hist_edges
    )

    critical_values = postcritical_images(
        N_CRITICAL_IMAGES
    )

    fig = plt.figure(
        figsize=(13.2, 9.7),
        constrained_layout=True
    )
    gs = fig.add_gridspec(2, 2)

    # Okabe--Ito color-blind-safe palette combined with distinct line styles.
    styles = {
        0: {
            "color": "#000000",
            "linestyle": "-",
            "linewidth": 2.6,
        },
        1: {
            "color": "#56B4E9",
            "linestyle": "--",
            "linewidth": 2.3,
        },
        2: {
            "color": "#E69F00",
            "linestyle": "-.",
            "linewidth": 2.2,
        },
        4: {
            "color": "#009E73",
            "linestyle": ":",
            "linewidth": 2.4,
        },
        8: {
            "color": "#CC79A7",
            "linestyle": (0, (7, 2)),
            "linewidth": 2.2,
        },
        16: {
            "color": "#D55E00",
            "linestyle": (0, (3, 1, 1, 1)),
            "linewidth": 2.2,
        },
        40: {
            "color": "#0072B2",
            "linestyle": "-",
            "linewidth": 2.8,
        },
    }

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        x, snapshots[0],
        **styles[0],
        label=r"$\rho_0$"
    )
    ax.plot(
        x, snapshots[1],
        **styles[1],
        label=r"$\rho_1=\mathcal{P}\rho_0$"
    )
    ax.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax.set_ylim(
        0.0,
        readable_upper_limit(
            snapshots[0], snapshots[1]
        )
    )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("density")
    ax.set_title(
        r"(a) Formation of the first fold singularity"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True)

    ax = fig.add_subplot(gs[0, 1])
    for n in (1, 2, 4):
        ax.plot(
            x, snapshots[n],
            **styles[n],
            label=rf"$\rho_{{{n}}}$"
        )
    ax.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax.set_ylim(
        0.0,
        readable_upper_limit(
            snapshots[1],
            snapshots[2],
            snapshots[4]
        )
    )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("density")
    ax.set_title(
        r"(b) Early propagation under $\mathcal{P}$"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, ncol=3)

    ax = fig.add_subplot(gs[1, 0])
    for n in (4, 8, 16):
        ax.plot(
            x, snapshots[n],
            **styles[n],
            label=rf"$\rho_{{{n}}}$"
        )
    ax.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax.set_ylim(
        0.0,
        readable_upper_limit(
            snapshots[4],
            snapshots[8],
            snapshots[16]
        )
    )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("density")
    ax.set_title(
        r"(c) Development of the singularity cascade"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, ncol=3)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(
        hist_x,
        hist_rho,
        width=0.92*hist_widths,
        color="#bdbdbd",
        edgecolor="#737373",
        linewidth=0.25,
        alpha=0.58,
        label=rf"critical-orbit histogram ({N_HIST_BINS} bins)",
        zorder=1
    )
    ax.plot(
        x,
        snapshots[40],
        **styles[40],
        label=r"$\rho_{40}=\mathcal{P}^{40}\rho_0$",
        zorder=4
    )

    for k, value in enumerate(
        critical_values,
        start=1
    ):
        ax.axvline(
            value,
            color="#009E73",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            zorder=2
        )

        if k <= 8:
            ax.text(
                value,
                0.97-0.055*((k-1) % 2),
                rf"$c_{{{k}}}$",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="top",
                fontsize=8.7,
                color="#007A5E"
            )

    ax.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax.set_ylim(
        0.0,
        readable_upper_limit(
            snapshots[40],
            hist_rho
        )
    )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel("density")
    ax.set_title(
        r"(d) Late iterate and physical invariant density"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=True
    )

    fig.suptitle(
        r"Perron--Frobenius evolution for "
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
