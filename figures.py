"""Per-setting figures: representative trajectory (trajectory figures of the paper) and the
paired-history scatter. Written to <out>/figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "legend.frameon": False, "savefig.bbox": "tight", "font.family": "serif"})
COL = {"FOM": "black", "POD-L": "#d95f02", "POD-N": "#1b9e77", "CR-ROM": "#3b5bdb"}
STY = {"FOM": "-", "POD-L": ":", "POD-N": "--", "CR-ROM": "-"}


def _predict(ctx, stage, name, r, seed, split="test"):
    import torch
    from experiments import load_rom
    from rom import predict
    rom, kind = load_rom(ctx, stage, name, r, seed)
    t = ctx.t[kind][split]
    with torch.no_grad():
        Xh = predict(rom, t["X"], t["C"]).cpu().numpy()
    return ctx.sc[kind].obs(Xh.astype(np.float64))


def trajectory(ctx, r=16, seed=0):
    from metrics import traj_metrics
    d, N, p = ctx.data["test"], ctx.p.N, ctx.p
    preds = {"POD-L": _predict(ctx, "factorial", "POD-linear", r, 0),
             "POD-N": _predict(ctx, "factorial", "POD-neural", r, seed),
             "CR-ROM": _predict(ctx, "closure", "obs+reproject (V3)", r, seed)}
    errs = [traj_metrics(preds["CR-ROM"][i], d["Y"][i], d["U"][i], d["P"][i], ctx.k)["x_nrmse"]
            for i in range(len(d["Y"]))]
    i = int(np.argsort(errs)[len(errs) // 2])          # median CR-ROM error, not best case
    Y, u = d["Y"][i], d["U"][i]
    t = np.arange(len(u)) * p.dt
    ulab = "Current (A)" if p.heating == "joule" else r"$T_{\mathrm{cmd}}$ (K)"
    series = {"FOM": Y, **{k: v[i] for k, v in preds.items()}}

    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.35)
    gr = gs[:, 1].subgridspec(2, 1, hspace=0.55)
    ax = [fig.add_subplot(gs[j, 0]) for j in range(4)]
    ax[0].plot(t, u, "k")
    ax[0].set_ylabel(ulab)
    for lab, Z in series.items():
        ax[1].plot(t, Z[:, -2] * 1e3, STY[lab], color=COL[lab], label=lab)
        ax[2].plot(t, Z[:, -1] / 1e6, STY[lab], color=COL[lab])
        ax[3].plot(t, Z[:, N:2 * N].mean(1), STY[lab], color=COL[lab])
    ax[1].set_ylabel("$x$ (mm)")
    ax[2].set_ylabel(r"$\sigma$ (MPa)")
    ax[3].set_ylabel(r"mean $\xi$")
    ax[3].set_xlabel("Time (s)")
    for a in ax[:3]:
        a.tick_params(labelbottom=False)
    a1 = fig.add_subplot(gr[0])
    for lab, Z in series.items():
        a1.plot(u, Z[:, -2] * 1e3, STY[lab], color=COL[lab])
    a1.set_xlabel(ulab)
    a1.set_ylabel("$x$ (mm)")
    a1.set_title("Hysteresis loops", fontsize=8)
    a2 = fig.add_subplot(gr[1])
    E = (preds["CR-ROM"][i][:, N:2 * N] - Y[:, N:2 * N]).T
    v = max(np.abs(E).max(), 1e-6)
    im = a2.imshow(E, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                   extent=[t[0], t[-1], 0.5, N + 0.5])
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Cell")
    a2.set_title(r"CR-ROM error in $\xi$", fontsize=8)
    fig.colorbar(im, ax=a2, fraction=0.05)
    n, K = d["P"][i]
    fig.suptitle(f"Test trajectory {i}: n = {int(n)}, K = {K / 1e3:.2f} kN/m "
                 f"(median CR-ROM error, x NRMSE = {errs[i]:.3f})", fontsize=8, y=1.0)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=7)
    os.makedirs(os.path.join(ctx.out, "figures"), exist_ok=True)
    f = os.path.join(ctx.out, "figures", "fig_trajectory.pdf")
    fig.savefig(f)
    fig.savefig(f.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return f


def paired(out):
    f = os.path.join(out, "paired.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    models = list(dict.fromkeys(df["model"]))
    fig, axs = plt.subplots(1, len(models), figsize=(1.9 * len(models), 2.0), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    hi = max(df["d_fom"].max(), df["sep"].max()) * 1.05
    for a, m in zip(axs, models):
        for cls, c in (("control", "0.6"), ("hidden", "#c92a2a")):
            s = df[(df.model == m) & (df.cls == cls)]
            a.scatter(s["d_fom"], s["sep"], s=5, c=c, label=cls, alpha=0.7, linewidths=0)
        a.plot([0, hi], [0, hi], "k:", lw=0.6)
        a.set_title(m, fontsize=8)
        a.set_xlabel("FOM divergence")
    axs[0].set_ylabel("Error in predicted divergence")
    axs[0].legend(fontsize=7, markerscale=2)
    g = os.path.join(out, "figures", "fig_paired.pdf")
    os.makedirs(os.path.dirname(g), exist_ok=True)
    fig.savefig(g)
    fig.savefig(g.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return g
