# Latent reduced-order models for hysteretic SMA wire–spring actuators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24421134.svg)](https://doi.org/10.5281/zenodo.24421134)

Code accompanying the paper

> A. S. Jacob, S. Nagulan, S. R. Pandey and R. Datta, *What makes latent reduced-order models
> work for hysteretic shape memory alloy actuators? Dynamics, re-projection and state closure*,
> manuscript submitted for publication.

The repository contains the one-dimensional simulator of a thermally driven SMA wire–spring
actuator with partial phase transformations (the full-order model, FOM), the latent
reduced-order models (ROMs) compared in the paper, and the scripts that generate every
table and figure. All datasets are generated deterministically by these scripts; no
pre-computed data are needed except the open actuator test of Gorges et al. (2025), which
`get_data.sh` downloads.

## Installation

Python 3.10 or later. The results of the paper were obtained with Python 3.11.16,
NumPy 2.4.6 and PyTorch 2.14.0 on an Apple MacBook Air (M1, 2020, 8 GB), CPU only.

```
conda create -n sma-rom python=3.11
conda activate sma-rom
pip install -r requirements.txt
```

## Quick check (a few minutes)

```
python run.py --quick --out runs/quick
```

This runs every stage of the main study with a tiny budget; the numbers are meaningless,
but it confirms that the installation works.

## Reproducing the paper

`reproduce_all.sh` runs everything in order. On the machine above this takes roughly one to
two days in total, so it is best run in stages. Every script is resumable: a finished stage
is skipped when the command is repeated.

| Step | Command | Paper |
|---|---|---|
| 1. Data of the actuator test | `bash get_data.sh` | Sec. 4.8 |
| 2. Main study, setting B | `python run.py --preset paper` | Sec. 4.1–4.7 |
| 3. Main study, setting C | `python run.py --preset revised` | Sec. 4.1–4.7 |
| 4. Main tables and figures | `python report.py --B runs/paper --C runs/revised --out report` | Main-study tables, factorial figure |
| 5. Calibration on the actuator test | `python experimental.py --out runs/experiment` | Sec. 4.8 |
| 6. Experiment figure | `python fig_experiment_paper.py --run runs/experiment --out report` | Sec. 4.8 figure |
| 7. Supplementary set 1 | `python revision.py` | Sec. 4.1–4.5, 4.8, 4.9 |
| 8. Supplementary set 2 | `python revision2.py` | Sec. 2.5, 4.2–4.5, 4.8 |
| 9. Supplementary set 3 | `python revision3.py` | Sec. 4.3, 4.4, 4.7, 4.9, 4.10 |

Steps 7–9 reuse the data, checkpoints and calibrations of steps 2–5 and must be run after
them. Their outputs are written to `report_rev/`, `report_rev2/` and `report_rev3/`
(tables as CSV and LaTeX, values as Markdown, figures as PDF and PNG). Each script lists
its stages and the corresponding sections of the paper at the top of the file; a single
stage is run with, for example, `python revision2.py --stages matched`.

### Where each result comes from

| Result in the paper | Script and stage |
|---|---|
| Simulator verification (Sec. 2.5) | `revision2.py` `verify`; restart and mesh checks in `run.py` `audit` |
| Time-step convergence (Sec. 4.1) | `revision.py` `dtconv` |
| Factorial representation × dynamics (Sec. 4.2) | `run.py` `factorial`, `report.py` |
| Multistep polynomial dynamics, weighted POD (Sec. 4.2) | `revision2.py` `lindyn` |
| Error percentiles and error against time (Sec. 4.2) | `run.py` `summary`, `revision.py` `horizon` |
| Bootstrap intervals (Sec. 4.2–4.5) | `revision2.py` `boot` |
| Output-bound admissibility, clipping (Sec. 4.3) | `run.py` `ablation`, `revision.py` `podclip` |
| Equilibrium projection and its cost (Sec. 4.3, 4.7) | `revision2.py` `eqproj`, `revision3.py` `eqcost` |
| Re-projection, long training (Sec. 4.4) | `run.py` `ablation`, `closure`, `longtrain` |
| Re-projection diagnostics (Sec. 4.4) | `revision2.py` `reproj` |
| 600 s rollouts (Sec. 4.4) | `revision3.py` `longroll` |
| State closure, r = 32 (Sec. 4.5) | `run.py` `closure`, `revision.py` `closure32` |
| Natural-pair test (Sec. 4.5) | `revision.py` `paired300` |
| Matched twin test (Sec. 4.5) | `revision2.py` `matched` |
| Extrapolation (Sec. 4.6) | `run.py` (splits `ood_*`) |
| Cost (Sec. 4.7) | `run.py`, `revision3.py` `eqcost`, `ioseq` |
| Calibration, identifiability (Sec. 4.8) | `experimental.py`, `revision.py` `calib` |
| Parameter sets on partial cycles (Sec. 4.8) | `revision2.py` `minor` |
| ROMs of the calibrated actuator (Sec. 4.9) | `revision.py` `calrom`, `revision3.py` `calsens` |
| Direct input–output model (Sec. 4.10) | `revision3.py` `ioseq` |

Table numbers above refer to the submitted version and may change during review; the
section numbers are the stable reference.

### Reproducibility notes

- Data generation, model initialization and window sampling are seeded. Neural-network
  training on another processor or library version can still give slightly different
  numbers, typically within the seed-to-seed spread reported in the paper.
- Timings and speedups depend on the machine; they are measured single-threaded, with the
  simulator timed on the same machine.
- The calibration of Section 4.8 uses a seeded differential evolution followed by a local
  least-squares refinement; parallel evaluation may change the search path slightly.

## Code overview

| File | Content |
|---|---|
| `fom.py` | Simulator: phase-diagram transformation kinetics with first-order reversal-point memory, axial heat conduction, heater-relaxation / Joule / power heating, inclined-pair and straight-wire actuator geometries. The constitutive update is given step by step in Appendix A of the paper. |
| `data.py` | Thermal command generator, dataset splits, scaling |
| `rom.py` | POD, weighted POD and autoencoder representations; linear, polynomial, neural and recurrent latent dynamics; re-projection; training |
| `metrics.py` | Error metrics, equilibrium and compatibility residuals |
| `experiments.py` | Model family, training and evaluation of the main study |
| `paired.py` | Natural-pair test of the hidden memory |
| `run.py` | Entry point of the main study |
| `report.py`, `figures.py` | Tables and figures of the main study |
| `experimental.py`, `fig_experiment_paper.py` | Calibration on the actuator test and its figure |
| `revision.py`, `revision2.py`, `revision3.py` | Supplementary experiments, sets 1–3 |
| `report_rev.py`, `report_rev2.py` | Tables and figures of sets 1 and 2 (set 3 reports itself) |

The simulator interface used throughout is `fom.Actuator(params).run(u)`, which returns the
observations `Y` [M, 3N+2] (T, ξ and transformation strain per cell, then x and σ) and the
complete states `S` [M, 10N+2] (adding the reverse anchors, last direction, memory stage,
loop origin and two reversal points per cell); `set_full(s)` restarts exactly from a
complete state.

## Data

The actuator and tensile tests of Section 4.8 are from

> T. Gorges, T. Weber, J. Mayer and P. Motzki (2025), dataset accompanying "Experimental
> investigation of phase-dependent resistivity and Poisson's ratio in shape memory alloy
> wires", Smart Materials and Structures 34, 065033. Zenodo, doi:10.5281/zenodo.15657324,
> CC BY 4.0.

`get_data.sh` downloads the three files into `experimental/`. They are not redistributed here.

The material, transformation and thermal parameters of the benchmark NiTi alloy are those of
Table 1 of A. A. Karakalas, T. T. Machairas, A. G. Solomou and D. A. Saravanos (2019),
Smart Materials and Structures 28, 035014.

## Citation

Please cite the paper (see `CITATION.cff`) if you use this code.

## License

MIT; see `LICENSE`.
