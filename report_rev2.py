"""Values and figures for supplementary experiments set 2 (revision2.py --stages report2)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})


def _r(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def build(a, log):
    out, V = a.out, ["# Second revision: values\n"]
    roll = lambda d: d[(d.split == "test") & (d["mode"] == "rollout")]

    d = _r(os.path.join(out, "rev2_lindyn.csv"))
    if d is not None:
        g = roll(d).groupby(["setting", "model"], sort=False)[["x_nrmse", "sig_nrmse", "xi_rmse", "xi_viol", "speedup"]]
        V += ["## lindyn: test rollouts (mean, s.d. over seeds)", g.mean().round(4).to_string(),
              g.std().round(4).add_suffix("_sd").to_string()]
        e = d[(d.split != "test") & (d["mode"] == "rollout")].groupby(["setting", "model", "split"], sort=False).x_nrmse.mean()
        V += ["### extrapolation x NRMSE", e.unstack("split").round(4).to_string()]
        if "val_init" in d:
            V += ["### validation score before / after unrolled training",
                  roll(d).dropna(subset=["val_init"]).groupby(["setting", "model"])[["val_init", "val_best"]].mean().to_string()]

    d = _r(os.path.join(out, "rev2_reproj.csv"))
    if d is not None:
        g = roll(d).groupby(["setting", "model"])[["x_nrmse", "sig_nrmse", "xi_rmse", "xi_viol", "eq_rms_N"]]
        V += ["\n## reproj: test rollouts", g.mean().round(4).to_string(), g.std().round(4).add_suffix("_sd").to_string()]
    d = _r(os.path.join(out, "rev2_reproj_diag.csv"))
    if d is not None:
        V += ["\n## reproj: latent diagnostics (mean over seeds)",
              d.groupby(["setting", "model"], sort=False).mean(numeric_only=True).round(4).to_string()]

    m = _r(os.path.join(out, "rev2_matched_pairs.csv"))
    d = _r(os.path.join(out, "rev2_matched.csv"))
    if m is not None:
        V += ["\n## matched twins: simulator divergence (fraction of training stroke)",
              m.groupby("setting")[["d_fom", "d_fom_max", "mem_diff", "jump_orig", "jump_twin"]]
              .describe(percentiles=[0.5, 0.95]).T.round(5).to_string(),
              "### by continuation (0 heat, 1 cool, 2 heat-cool)",
              m.groupby(["setting", "cont"]).d_fom.agg(["mean", "median", "max"]).round(5).to_string()]
    if d is not None:
        V += ["### ROMs on the twins", d.groupby(["setting", "model"])[["d_fom", "d_rom", "sep", "err"]].mean().round(4).to_string()]

    d = _r(os.path.join(out, "rev2_minor.csv"))
    if d is not None:
        V += ["\n## minor: parameter sets compared with the retained set",
              d.groupby(["set", "kind"], sort=False)[["x_nrmse_vs_ref", "xi_rms_vs_ref", "stroke_mm", "peak_T", "fails"]]
              .agg(["mean", "max"]).round(4).to_string()]

    d = _r(os.path.join(out, "rev2_eqproj.csv"))
    if d is not None:
        cols = ["x_nrmse", "sig_nrmse", "xi_rmse", "eq_rms_N", "cp_rms", "failed_solves", "rev_x_err", "loop_area_err"]
        V += ["\n## eqproj (mean over seeds)", d.groupby(["setting", "model"])[cols].mean().round(5).to_string()]

    d = _r(os.path.join(out, "rev2_boot.csv"))
    if d is not None:
        V += ["\n## boot: 95% bootstrap intervals over test trajectories (seed-averaged)", d.round(4).to_string(index=False)]

    f = os.path.join(out, "rev2_verify.json")
    if os.path.exists(f):
        V += ["\n## verify", json.dumps(json.load(open(f)), indent=1)]

    open(os.path.join(out, "rev2_values.md"), "w").write("\n".join(V) + "\n")

    # ---- figure: matched twins vs natural pairs vs ROM error
    d = _r(os.path.join(out, "rev2_matched.csv"))
    if m is not None and d is not None:
        tags = list(m.setting.unique())
        fig, axs = plt.subplots(1, len(tags), figsize=(3.5 * len(tags), 2.4), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, tag in zip(axs, tags):
            v = np.sort(m[m.setting == tag].d_fom.values)
            ax.plot(np.maximum(v, 1e-7), np.arange(1, len(v) + 1) / len(v), color="#c92a2a", label="matched twins, $d_{\\mathrm{FOM}}$")
            path = os.path.join(a.B if tag == "B" else a.C, "paired300.csv")
            if os.path.exists(path):
                P = pd.read_csv(path)
                h = np.sort(P[P.cls == "hidden"].groupby("pair").d_fom.first().values)
                ax.plot(np.maximum(h, 1e-7), np.arange(1, len(h) + 1) / len(h), color="0.5", ls="--",
                        label="natural hidden pairs, $d_{\\mathrm{FOM}}$")
            for mdl, col in (("POD-N", "#1b9e77"), ("CR-ROM", "#3b5bdb")):
                q = d[(d.setting == tag) & (d.model == mdl)]
                if len(q):
                    ax.axvline(q.err.mean(), color=col, lw=0.8, ls=":", label=f"{mdl} rollout error")
            ax.set_xscale("log")
            ax.set_xlabel("fraction of stroke")
            ax.set_title(f"Setting {tag}", fontsize=8)
        axs[0].set_ylabel("cumulative fraction of pairs")
        axs[0].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"fig_matched.{ext}"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ---- figure: parameter sets on the nested partial-cycle history
    f = os.path.join(out, "rev2_minor_traces.npz")
    if os.path.exists(f):
        z = np.load(f)
        hs = [h for h in ("measured window", "nested") if f"I|{h}" in z]
        fig, axs = plt.subplots(2, len(hs), figsize=(3.5 * len(hs), 3.2), sharex="col",
                                gridspec_kw=dict(height_ratios=[1, 2]))
        axs = np.array(axs).reshape(2, -1)
        sets = sorted({k.split("|")[0] for k in z.files if not k.startswith("I|")},
                      key=lambda s: (not s.startswith("current"), s))
        for j, h in enumerate(hs):
            I = z[f"I|{h}"]
            t = np.arange(len(I)) * 0.05
            axs[0, j].plot(t, I, "k", lw=0.8)
            axs[0, j].set_title("complete cycles (measured current)" if h == "measured window" else "nested partial cycles",
                                fontsize=8)
            for s_, col, ls in zip(sets, ("k", "#e8590c", "#3b5bdb"), ("-", "--", ":")):
                axs[1, j].plot(t, z[f"{s_}|{h}"] * 1e3, color=col, ls=ls, lw=1.0, label=s_)
            axs[1, j].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Current (A)")
        axs[1, 0].set_ylabel("Stroke (mm)")
        axs[1, -1].legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"fig_minor.{ext}"), dpi=180, bbox_inches="tight")
        plt.close(fig)
    log(f"[report2] wrote {out}/rev2_values.md and figures")
