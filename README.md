# Code and data for "Local horizontal gates with random CNOT frames converge to ground states"

X. Liu, S. Wang, W. Yang, X. Xu.

This archive contains the scripts and data behind every figure, the table, and the numerical checks of the
paper and its Supplemental Material. All simulations are exact state-vector simulations in NumPy/SciPy;
no quantum hardware or quantum SDK is used.

## Requirements

Python 3.10 or newer, with

    pip install -r requirements.txt

(tested with Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9).
The figures use the Latin Modern font from TeX Live if it is installed at
`/usr/share/texmf/fonts/opentype/public/lm/`; otherwise Matplotlib falls back to its default serif font.

## Layout

    figs/            data generation and plotting for the main-text figures and Table I
    figs/concept/    shared algorithm code (alg.py, toy.py), plot style (pcommon.py), Figs. 1, 4 and the fiber figure
    checks/          numerical checks listed in the Supplemental Material, with their reference outputs (*.out)

Each script starts with a docstring that states the Hamiltonian, step size, frame type, seeds and output file.
Some docstrings and printed labels are in Chinese.

## Figures and table

Run the scripts from the directory they are in. The data files produced by the `data_*` scripts are included,
so every figure can be redrawn without rerunning the simulations.

| Item in the paper | Data script -> data file | Plot script -> output |
|---|---|---|
| Fig. 1 (circuit) | drawn in TikZ in the LaTeX source | none |
| Fig. 2 (what one step can see) | computed inside the plot script -> `concept/geometry_c.npz` | `concept/fig_geometry.py` -> `fig_geometry.pdf` |
| Fig. 3 (trap concept) | computed inside the plot script | `concept/fig_trap.py` -> `fig_trap.pdf` |
| Fig. 4 (one fiber of the proof) | `concept/data_fiber.py` -> `concept/fiber.npz` | `concept/fig_fiber.py` -> `fig_fiber.pdf` |
| Fig. 5 (convergence) | `data_convergence.py` -> `convergence.csv`; `data_extra.py` -> `noframe.csv` | `plot_figs.py` -> `fig2_convergence.pdf` |
| Fig. 6 (scaling with n) | `data_scaling.py` -> `scaling.csv` (n = 3-9), `scaling_n10.csv` (n = 10) | `plot_scaling.py` -> `fig_scaling.pdf`; summary: `analyze_scaling.py` |
| Fig. 7 (local Pauli projection) | `data_localpauli.py` -> `localpauli.csv` | `plot_localpauli.py` -> `fig_localpauli.pdf` |
| Table I (saddle escape) | `data_escape.py` -> `escape.json` (log: `data_escape.out`) | numbers are read from `escape.json` |
| Fig. 8 (finite shot noise) | `data_shots.py` -> `shots.csv` | `plot_shots.py` -> `fig_shots.pdf` |
| Fig. 9 (saddles: Lyapunov exponents, trap) | `data_lyapunov.py` -> `lyapunov.csv`; `data_extra.py` -> `trap.csv` | `plot_figs.py` -> `fig3_saddles.pdf` |

The file names `fig2_convergence.pdf` and `fig3_saddles.pdf` are historical and do not match the final figure numbers.

`data_scaling.py` and `data_shots.py` use `multiprocessing`; set the number of worker processes with the
environment variable `NPROC`. The n = 10 runs of `data_scaling.py` take up to about 1.5 x 10^6 steps each and
can be run separately with

    python data_scaling.py 1500000 10 scaling_n10.csv

## Numerical checks

The scripts in `checks/` correspond to the rows of the table of numerical checks in the Supplemental Material;
the last column of that table names the script. Run them from `checks/`, for example

    cd checks
    python check_rgf_theorems.py > check_rgf_theorems.new
    diff check_rgf_theorems.new check_rgf_theorems.out

All seeds are fixed, so the outputs should agree with the reference `*.out` files up to floating-point
rounding. Several scripts reuse the setup of `check_rgf_hyperbolic.py` and must stay in the same directory.
In addition:

- `figs/checks_paper.py` (output `figs/checks_paper.out`) checks the two-term parameter shift rule, the
  gradient bound on good blocks, and the linearization in tubular coordinates.
- `checks/check_malvetti_assumption.py` checks that the input-side direction sets do not satisfy
  Assumption 3.1 of Malvetti et al. (Introduction).
- `checks/cnot_frame_cost.py` computes the average CNOT cost of a random Pauli rotation and of a uniformly
  random CNOT frame for n = 3, 4 (Sec. III).

## License

See `LICENSE`.
