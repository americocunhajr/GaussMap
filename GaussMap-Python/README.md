# GaussMap — figure-generation codes

This archive contains the finalized Python codes organized to reproduce **Figures 1–11** of:

> **A. Cunha Jr, _The Generalized Gaussian Iterated Map_, manuscript prepared for submission to the SIAM Journal on Applied Dynamical Systems (SIADS), 2026.**

The scripts implement the normalized Gaussian map

\[
T(y)=A+B e^{-y^2},\qquad B>0,
\]

and the analytical/numerical constructions used in the manuscript.  The numerical parameters and plotting choices are the final repository versions consolidated on **September 4, 2026**.

## Directory organization

```text
GaussMap_SIADS_Figure_Codes/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── run_all_figures.py
├── figures/
│   ├── Fig01_representative_geometries.py
│   ├── Fig02_quadratic_critical_point_geometry.py
│   ├── Fig03_parameter_space_geometry.py
│   ├── Fig04_critical_orbit_evolution.py
│   ├── Fig05_cobweb_evolution.py
│   ├── Fig06_bifurcation_diagrams.py
│   ├── Fig07_orbit_averaged_expansion_map.py
│   ├── Fig08_inverse_branch_geometry.py
│   ├── Fig09_empirical_density_postcritical_peaks.py
│   ├── Fig10_perron_frobenius_evolution.py
│   └── Fig11_singularity_propagation.py
└── output/
```

Each script writes both `FigXX.png` and `FigXX.pdf` to `output/`.  The PDF names are directly compatible with the manuscript figure calls `Figures/FigXX.pdf` after copying them to the manuscript's `Figures/` directory.

## Figure map

| Figure | Script | Main computation |
|---|---|---|
| 1 | `Fig01_representative_geometries.py` | Representative geometries: contraction, three fixed points, flip threshold, explicit CE case |
| 2 | `Fig02_quadratic_critical_point_geometry.py` | Exact map vs. quadratic Taylor approximation and quartic error |
| 3 | `Fig03_parameter_space_geometry.py` | Analytical SN/PD loci, cusp, contraction threshold, CE curve, invariant-interval wedge |
| 4 | `Fig04_critical_orbit_evolution.py` | Critical-orbit time series for four regimes along `A=-1` |
| 5 | `Fig05_cobweb_evolution.py` | Cobwebs for the same four regimes |
| 6 | `Fig06_bifurcation_diagrams.py` | Bifurcation diagrams for `A=-3,-2,-1,0` with analytical thresholds |
| 7 | `Fig07_orbit_averaged_expansion_map.py` | Parameter-space finite-orbit average of `log|T'|` with analytical curves |
| 8 | `Fig08_inverse_branch_geometry.py` | Explicit inverse branches and coalescence at the critical value |
| 9 | `Fig09_empirical_density_postcritical_peaks.py` | Two-million-iterate histogram, peak detection, post-critical matching |
| 10 | `Fig10_perron_frobenius_evolution.py` | Perron–Frobenius iterations and comparison with a long-orbit histogram |
| 11 | `Fig11_singularity_propagation.py` | Empirical density, post-critical orbit, derivative product and theoretical amplitudes |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

## Reproducing one figure

From the root of this archive:

```bash
python figures/Fig06_bifurcation_diagrams.py
```

The outputs are written to `output/Fig06.png` and `output/Fig06.pdf`.

## Reproducing all figures

```bash
python run_all_figures.py
```

To run only selected figures:

```bash
python run_all_figures.py --figures 1 3 6 8
```

Figures 7, 9, 10, and 11 are computationally heavier because they use large parameter grids, long critical orbits, or repeated transfer-operator evaluations.  The scripts intentionally retain the manuscript-resolution numerical controls rather than replacing them with quick-preview settings.

## Reproducibility notes

* No random sampling is used in the final manuscript figure scripts collected here, except where explicitly stated inside a script.  The orbit-based computations are therefore deterministic for a fixed numerical environment.
* Figure 9 uses `scipy.signal.find_peaks` for deterministic peak identification.
* The analytical curves in Figures 3, 6, and 7 use the formulas derived in the manuscript.
* Long-orbit density plots are empirical histograms; they are not presented as pointwise numerical proofs of the conditional invariant-density theorems.
* Figure 7 reproduces the final numerical atlas used in the manuscript, including its burn-in/transient stage before the finite orbit-average is accumulated.

## Citation

If these codes are used in scientific work, please cite the associated manuscript:

**A. Cunha Jr, _The Generalized Gaussian Iterated Map_, manuscript prepared for submission to the SIAM Journal on Applied Dynamical Systems (SIADS), 2026.**

Repository: <https://americocunhajr.github.io/GaussMap>

## Author

**Americo Cunha Jr**  
Laboratório Nacional de Computação Científica (LNCC), Petrópolis, Brazil  
Universidade do Estado do Rio de Janeiro (UERJ), Rio de Janeiro, Brazil  
<http://americocunha.org>
