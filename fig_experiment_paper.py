"""Manuscript figure for Section 4.8 from runs/experiment/experiment_traces.npz.

    python fig_experiment_paper.py --run runs/experiment --out report
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/experiment")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})
    d = np.load(os.path.join(a.run, "experiment_traces.npz"))
    t, I = d["t"], d["I"]
    xe, xs = d["x_exp"] * 1e3, d["x_sim"] * 1e3
    se, ss = d["sig_exp"] / 1e6, d["sig_sim"] / 1e6
    tcal = d["bounds"][-2]                      # start of the held-out (last) cycle
    fig = plt.figure(figsize=(7.0, 3.7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], hspace=0.15, wspace=0.3)
    ax = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for x_ in ax:
        x_.axvspan(tcal, t[-1], color="0.92", lw=0)
    ax[0].plot(t, I, "k")
    ax[0].set_ylabel("Current (A)")
    ax[0].text(tcal + 1, 0.82 * I.max(), "held out", fontsize=7)
    ax[1].plot(t, xe, "k", label="measured")
    ax[1].plot(t, xs, color="#3b5bdb", label="simulator, setting C, calibrated on the other cycles")
    ax[1].set_ylabel("Stroke (mm)")
    ax[2].plot(t, se, "k")
    ax[2].plot(t, ss, color="#3b5bdb")
    ax[2].set_ylabel("Stress (MPa)")
    ax[2].set_xlabel("Time (s)")
    for x_ in ax[:2]:
        x_.tick_params(labelbottom=False)
    b = fig.add_subplot(gs[:, 1])
    b.plot(I, xe, "k", lw=0.8)
    b.plot(I, xs, color="#3b5bdb", lw=0.8)
    b.set_xlabel("Current (A)")
    b.set_ylabel("Stroke (mm)")
    b.set_title("Stroke--current loops", fontsize=8)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    f = os.path.join(a.out, "fig_experiment.pdf")
    fig.savefig(f, bbox_inches="tight")
    fig.savefig(f.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
