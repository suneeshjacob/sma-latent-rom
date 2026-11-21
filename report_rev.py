"""Tables, figures and values for supplementary experiments set 1 (called by revision.py --stages report)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator

plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})
R = 16


def f(v, d=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def pmf(m, s, d=3):
    if not np.isfinite(m):
        return "--"
    return f(m, d) if not np.isfinite(s) or s == 0 else f"{f(m, d)} $\\pm$ {f(s, d)}"


def table(label, caption, colspec, header, rows, wide=True):
    env = "table*" if wide else "table"
    body = "\n".join(" & ".join(r) + r" \\" if isinstance(r, (list, tuple)) else r for r in rows)
    return (f"\\begin{{{env}}}[t]\n\\small\\sf\\centering\n\\caption{{{caption}\\label{{{label}}}}}\n"
            + (r"\resizebox{\textwidth}{!}{" if wide else "") +
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}" + ("}" if wide else "") + f"\n\\end{{{env}}}\n")


def logaxes(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=6))
        axis.set_minor_formatter(NullFormatter())


def summ(path):
    f_ = os.path.join(path, "summary.csv")
    return pd.read_csv(f_) if os.path.exists(f_) else None


def srow(S, stage, model, r=R, split="test", mode="rollout"):
    if S is None:
        return None
    q = S[(S.stage == stage) & (S.model == model) & (S.r == r) & (S.split == split) & (S["mode"] == mode)]
    return None if q.empty else q.iloc[0]


def g(row, col, sd=False):
    if row is None:
        return np.nan
    v = row.get(col + "_sd" if sd else col, np.nan)
    return float(v) if pd.notna(v) else np.nan


def agg(df, cols):
    """mean and s.d. over seeds."""
    return {c: (float(df[c].mean()), float(df[c].std()) if len(df) > 1 else np.nan) for c in cols if c in df}


def build(a, log):
    out = a.out
    T, V = [], ["# Revision values\n"]
    S = {"B": summ(a.B), "C": summ(a.C)}

    # ---- POD-N + clipping
    fpc = os.path.join(out, "rev_podclip.csv")
    if os.path.exists(fpc):
        pc = pd.read_csv(fpc)
        cols = ["x_nrmse", "sig_nrmse", "xi_rmse", "xi_viol", "neg_x", "neg_sig", "eq_rms_N", "rom_ms", "speedup"]
        rows = []
        for tag in ("B", "C"):
            q = pc[(pc.setting == tag) & (pc.r == R) & (pc.split == "test") & (pc["mode"] == "rollout")]
            A = agg(q, cols)
            for lab, stage, name in (("POD-N", "factorial", "POD-neural"), ("POD-N+clip", None, None),
                                     ("AE-N+clip", "ablation", "plain+clip"), ("CR-ROM", "closure", "obs+reproject (V3)")):
                if stage is None:
                    vals = {c: A.get(c, (np.nan, np.nan)) for c in cols}
                else:
                    rw = srow(S[tag], stage, name)
                    vals = {c: (g(rw, c), g(rw, c, True)) for c in cols}
                rows.append([tag, lab, pmf(*vals["x_nrmse"]), pmf(*vals["sig_nrmse"]), pmf(*vals["xi_rmse"]),
                             f(vals["xi_viol"][0], 2), f(max(vals["neg_x"][0], vals["neg_sig"][0]), 2),
                             f(vals["eq_rms_N"][0], 2), f(vals["rom_ms"][0], 1), f(vals["speedup"][0], 0)])
            if tag == "B":
                rows.append(r"\midrule")
            V.append(f"- POD-N+clip ({tag}, r={R}): " + ", ".join(f"{c} {A[c][0]:.4f}" for c in cols if c in A))
        T.append(table("tab:podclip", f"POD-N with output clipping at $r={R}$ (test set, mean $\\pm$ s.d. over three "
                       "seeds), compared with POD-N, AE-N with clipping and CR-ROM. Viol.: fraction of samples with "
                       r"$\xi\notin[0,1]$; neg.: largest fraction with $x<0$ or $\sigma<0$.", "llllllllll",
                       r"Setting & Model & $x$ NRMSE & $\sigma$ NRMSE & $\xi$ RMSE & $\xi$ viol. & neg. & "
                       r"$R_{\mathrm{eq}}$ (N) & Time (ms) & Speedup", rows))

    # ---- quadratic operator inference
    fo = os.path.join(out, "rev_opinf.csv")
    if os.path.exists(fo):
        oq = pd.read_csv(fo)
        rows = []
        for tag in ("B", "C"):
            for r in sorted(oq.r.unique()):
                for lab in ("POD-L", "POD-Q", "POD-Q+clip", "POD-N"):
                    if lab.startswith("POD-Q"):
                        q = oq[(oq.setting == tag) & (oq.r == r) & (oq.model == lab) & (oq.split == "test") & (oq["mode"] == "rollout")]
                        if q.empty:
                            continue
                        x, s_, xi, rid = (float(q[c].iloc[0]) for c in ("x_nrmse", "sig_nrmse", "xi_rmse", "ridge"))
                        extra = f"{rid:.0e}"
                    else:
                        rw = srow(S[tag], "factorial", "POD-linear" if lab == "POD-L" else "POD-neural", r=r)
                        x, s_, xi, extra = g(rw, "x_nrmse"), g(rw, "sig_nrmse"), g(rw, "xi_rmse"), "--"
                    rows.append([tag, str(r), lab, f(x), f(s_), f(xi), extra])
            if tag == "B":
                rows.append(r"\midrule")
        T.append(table("tab:opinf", "Quadratic POD operator inference (POD-Q) compared with linear (POD-L) and neural "
                       "(POD-N, mean over seeds) latent dynamics on the test set. Ridge: regularization selected on the "
                       "validation rollouts.", "lllllll",
                       r"Setting & $r$ & Model & $x$ NRMSE & $\sigma$ NRMSE & $\xi$ RMSE & Ridge", rows))

    # ---- time-step convergence
    fd = os.path.join(out, "rev_dtconv.csv")
    if os.path.exists(fd):
        dc = pd.read_csv(fd)
        rows = []
        for (tag, dt), q in dc.groupby(["setting", "dt"], sort=False):
            rows.append([tag, f"{dt:g}", f"{q.ref_dt.iloc[0]:g}", f(q.x_nrmse.mean(), 4), f(q.rev_x_err.mean(), 4),
                         f(q.loop_area_err.mean(), 4), str(int(q.fails.sum()))])
        T.append(table("tab:dtconv", "Time-step convergence of the simulator: errors relative to the finest time step "
                       "(mean over trajectories; settings B and C with six test-like commands, straight wire with the "
                       "measured current of cycles 8--10).", "lllllll",
                       r"Setting & $\Delta t$ (s) & reference (s) & $x$ NRMSE & reversal $x$ error & loop area error & failed solves",
                       rows))

    # ---- calibration robustness
    base_dir = os.path.dirname(a.exp.rstrip("/")) or "."
    runs = [("baseline (current input)", a.exp), ("identifiability check", os.path.join(base_dir, "experiment_ident")),
            ("measured power input", os.path.join(base_dir, "experiment_power")),
            (r"fixed $C=8.34$ MPa K$^{-1}$", os.path.join(base_dir, "experiment_fixC"))]
    rows, irows = [], []
    for lab, d in runs:
        fj = os.path.join(d, "experiment.json")
        if not os.path.exists(fj):
            continue
        e = json.load(open(fj))
        th = e["theta"]
        rows.append([lab, f(e["stroke_nrmse_cal"]), f(e["stroke_nrmse_heldout"]), f"{th['Ms']:.1f}",
                     f"{th['Ms'] - th['dM']:.1f}", f"{th['Ms'] + th['dAs']:.1f}", f"{th['Ms'] + th['dAs'] + th['dA']:.1f}",
                     f"{th['C'] / 1e6:.1f}", f"{th['Hmax']:.3f}", f"{th['hc']:.0f}", f"{e['peak_T_K']:.0f}"])
        idf = e.get("identifiability")
        if idf:
            C = np.array(idf["correlation"], dtype=float)
            n = idf["free"]
            pairs = sorted(((abs(C[i, j]), n[i], n[j], C[i, j]) for i in range(len(n)) for j in range(i + 1, len(n))
                            if np.isfinite(C[i, j])), reverse=True)[:3]
            irows.append([lab, f"{idf['condition_number']:.1e}", ", ".join(f"{p1}--{p2} ({c:+.2f})" for _, p1, p2, c in pairs),
                          ", ".join(idf.get("at_bound", [])) or "none"])
        V.append(f"- calibration '{lab}': {json.dumps({k: e[k] for k in ('stroke_nrmse_cal', 'stroke_nrmse_heldout', 'theta', 'peak_T_K')})}")
    if rows:
        T.append(table("tab:calib", "Calibration of the straight-wire simulator on cycles 8--9 and prediction of cycle 10 "
                       "for different inputs and constraints (temperatures in K, $C$ in MPa K$^{-1}$, $h_c$ in "
                       "W m$^{-2}$ K$^{-1}$).", "lllllllllll",
                       r"Calibration & cal. & held-out & $M_s$ & $M_f$ & $A_s$ & $A_f$ & $C$ & $H_{\max}$ & $h_c$ & peak $T$",
                       rows))
    if irows:
        T.append(table("tab:ident", "Identifiability at the calibrated parameters: condition number of the scaled Gauss--Newton "
                       "matrix, the three strongest parameter correlations and parameters at a bound of the search range.",
                       "llll", "Calibration & condition number & strongest correlations & at bound", irows))

    # ---- closure at r = 32
    rows = []
    for tag in ("B", "C"):
        for lab, name in (("CR-ROM", "obs+reproject (V3)"), ("MC-ROM", "complete latent-only"), ("MC-ROM-R", "complete+reproject")):
            r16 = srow(S[tag], "closure", name, r=16)
            r32 = srow(S[tag], "closure32", name, r=32)
            if r32 is None:
                continue
            rows.append([tag, lab, f(g(r16, "x_nrmse")), pmf(g(r32, "x_nrmse"), g(r32, "x_nrmse", True)),
                         f(g(srow(S[tag], "closure32", name, r=32, mode="restart_active"), "x_nrmse")),
                         f(g(srow(S[tag], "closure32", name, r=32, mode="restart_inactive"), "x_nrmse")),
                         f(g(srow(S[tag], "closure", name, r=16, mode="recon"), "x_nrmse"), 4),
                         f(g(srow(S[tag], "closure32", name, r=32, mode="recon"), "x_nrmse"), 4)])
        if tag == "B" and rows:
            rows.append(r"\midrule")
    if rows:
        T.append(table("tab:closure32", "Memory-complete models with a larger latent space: displacement NRMSE from $t=0$ "
                       "at $r=16$ and $r=32$, from restarts with active and inactive memory at $r=32$, and projection "
                       "error at both dimensions.", "llllllll",
                       r"Setting & Model & $r=16$ & $r=32$ & active ($r=32$) & inactive ($r=32$) & proj. $r=16$ & proj. $r=32$",
                       rows))

    # ---- paired-history, 300 pairs per class: matched-distance analysis
    prow, figdata = [], {}
    for tag, path in (("B", a.B), ("C", a.C)):
        fp = os.path.join(path, "paired300.csv")
        if not os.path.exists(fp):
            continue
        P = pd.read_csv(fp)
        U = P.groupby("pair").agg(cls=("cls", "first"), d_obs=("d_obs", "first"), d_fom=("d_fom", "first")).reset_index()
        h, c = U[U.cls == "hidden"], U[U.cls == "control"]
        lo, hi = max(h.d_obs.min(), c.d_obs.min()), min(h.d_obs.max(), c.d_obs.max())
        m = U[(U.d_obs >= lo) & (U.d_obs <= hi)]
        X = np.column_stack([np.ones(len(U)), U.d_obs, (U.cls == "hidden").astype(float)])
        beta = np.linalg.lstsq(X, U.d_fom.values, rcond=None)[0]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(1000):
            ix = rng.integers(0, len(U), len(U))
            bs.append(np.linalg.lstsq(X[ix], U.d_fom.values[ix], rcond=None)[0][2])
        ci = np.percentile(bs, [2.5, 97.5])
        mh, mc = m[m.cls == "hidden"], m[m.cls == "control"]
        sep = P[P.cls == "hidden"].groupby("model").sep.mean()
        prow.append([tag, str(len(h)), str(len(c)), f"{lo:.4f}--{hi:.4f}" if lo < hi else "none",
                     f"{len(mh)} / {len(mc)}", f(mh.d_fom.mean(), 4), f(mc.d_fom.mean(), 4),
                     f"{beta[2]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"])
        V.append(f"- paired300 {tag}: memory effect at equal d_obs = {beta[2]:.5f} (95% CI {ci[0]:.5f}, {ci[1]:.5f}); "
                 f"slope on d_obs {beta[1]:.3f}; hidden e_sep by model: {sep.round(4).to_dict()}")
        figdata[tag] = (P, U)
    if prow:
        T.append(table("tab:paired300", "Paired-history test with 300 pairs per class. Overlap: range of observation "
                       "distance shared by hidden and control pairs; matched: number of pairs and mean simulator divergence "
                       "within that range; memory effect: coefficient of the hidden-memory indicator in a linear regression of "
                       r"$d_{\mathrm{FOM}}$ on $d_{\mathrm{obs}}$ (bootstrap 95\% interval), as a fraction of the stroke.",
                       "llllllll", r"Setting & hidden & control & overlap $d_{\mathrm{obs}}$ & matched (h / c) & "
                       r"$d_{\mathrm{FOM}}$ hidden & $d_{\mathrm{FOM}}$ control & memory effect", prow))
        fig, axs = plt.subplots(1, 2 * len(figdata), figsize=(3.6 * len(figdata), 2.6))
        axs = np.atleast_1d(axs)
        for k, (tag, (P, U)) in enumerate(figdata.items()):
            ax = axs[2 * k]
            for cls, col in (("control", "0.55"), ("hidden", "#c92a2a")):
                q = U[U.cls == cls]
                ax.scatter(q.d_obs, q.d_fom, s=4, c=col, label=cls, linewidths=0)
            logaxes(ax)
            ax.set_xlabel(r"$d_{\mathrm{obs}}$"); ax.set_ylabel(r"$d_{\mathrm{FOM}}$")
            ax.set_title(f"Setting {tag}: simulator divergence", fontsize=8)
            ax.legend(fontsize=6, markerscale=2)
            ax = axs[2 * k + 1]
            q = P[P.cls == "hidden"].groupby(["model", "pair"]).agg(d_fom=("d_fom", "first"), sep=("sep", "mean")).reset_index()
            for mdl, col, mk in zip(q.model.unique(), ["#1b9e77", "#3b5bdb", "#e8590c", "#7048e8"], "o^sx"):
                z = q[q.model == mdl]
                ax.scatter(z.d_fom, z.sep, s=6, c=col, marker=mk, label=mdl, linewidths=0.6 if mk == "x" else 0, alpha=0.8)
            lim = [max(1e-6, min(q.d_fom.min(), q.sep.min())), max(q.d_fom.max(), q.sep.max())]
            ax.plot(lim, lim, "k:", lw=0.6)
            logaxes(ax)
            ax.set_xlabel(r"$d_{\mathrm{FOM}}$"); ax.set_ylabel(r"$e_{\mathrm{sep}}$")
            ax.set_title(f"Setting {tag}: hidden pairs", fontsize=8)
            ax.legend(fontsize=6, markerscale=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_paired300.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out, "fig_paired300.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    # ---- error against rollout horizon
    fh = os.path.join(out, "rev_horizon.csv")
    if os.path.exists(fh):
        H = pd.read_csv(fh)
        tags = list(H.setting.unique())
        fig, axs = plt.subplots(1, len(tags), figsize=(3.5 * len(tags), 2.4), sharey=True)
        axs = np.atleast_1d(axs)
        cols = {"POD-L": "#d95f02", "POD-N": "#1b9e77", "AE-N": "#7570b3", "CR-ROM": "#3b5bdb", "MC-ROM": "#e7298a"}
        for ax, tag in zip(axs, tags):
            for mdl, q in H[H.setting == tag].groupby("model", sort=False):
                ax.plot(q.t, q["mean"], color=cols.get(mdl), label=mdl, lw=0.9)
                ax.plot(q.t, q.p95, color=cols.get(mdl), lw=0.5, ls=":")
            ax.set_yscale("log")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Setting {tag}", fontsize=8)
        axs[0].set_ylabel(r"$|x-\tilde x|$ / stroke")
        axs[0].legend(fontsize=6, ncol=2)
        fig.text(0.5, -0.04, "solid: mean over test trajectories and seeds; dotted: 95th percentile", ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_horizon.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out, "fig_horizon.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
        for tag in tags:
            q = H[H.setting == tag]
            last = q[q.step == q.step.max()].set_index("model")["mean"].round(4).to_dict()
            first = q[q.step == max(1, q.step.max() // 10)].set_index("model")["mean"].round(4).to_dict()
            V.append(f"- horizon {tag}: mean error at 10% of horizon {first}; at end {last}")

    # ---- percentiles of trajectory error
    rows = []
    for tag in ("B", "C"):
        for lab, stage, name in (("POD-L", "factorial", "POD-linear"), ("POD-N", "factorial", "POD-neural"),
                                 ("AE-N", "factorial", "AE-neural"), ("CR-ROM", "closure", "obs+reproject (V3)"),
                                 ("MC-ROM", "closure", "complete latent-only")):
            rw = srow(S[tag], stage, name)
            if rw is None:
                continue
            rows.append([tag, lab] + [f(g(rw, f"x_nrmse_p{q:02d}")) for q in (5, 50, 95)] + [f(g(rw, "x_nrmse"))])
        if tag == "B" and rows:
            rows.append(r"\midrule")
    if rows:
        T.append(table("tab:pct", f"Distribution of the trajectory displacement NRMSE over the test set at $r={R}$ "
                       "(percentiles over trajectories, mean over seeds).", "llllll",
                       r"Setting & Model & 5th & 50th & 95th & mean", rows, wide=False))

    # ---- ROMs of the calibrated straight-wire actuator
    fc = os.path.join(out, "rev_calrom.csv")
    if os.path.exists(fc):
        cr = pd.read_csv(fc)
        rows = []
        fom = cr[cr.model == "simulator"]
        for mdl, q in cr[cr.model != "simulator"].groupby("model", sort=False):
            A = agg(q, ["test_x_nrmse", "test_sig_nrmse", "test_xi_rmse", "test_xi_viol", "rom_fom_heldout",
                        "rom_exp_heldout", "rom_exp_all"])
            rows.append([mdl] + [pmf(*A[c]) for c in ("test_x_nrmse", "test_sig_nrmse", "test_xi_rmse")] +
                        [f(A["test_xi_viol"][0], 2)] + [pmf(*A[c]) for c in ("rom_fom_heldout", "rom_exp_heldout", "rom_exp_all")])
        rows.append(r"\midrule")
        rows.append(["Simulator", "--", "--", "--", "--", "--", f(float(fom.fom_exp_heldout.iloc[0])),
                     f(float(fom.fom_exp_all.iloc[0]))])
        T.append(table("tab:calrom", "ROMs of the experimentally calibrated straight-wire actuator ($r=16$, mean $\\pm$ s.d. "
                       "over seeds). Test: simulated current histories with complete and partial cycles. Measured window: "
                       "rollout with the measured current of cycles 8--10 from the initial state; errors in stroke as "
                       "fractions of the measured stroke.", "llllllll",
                       r" & \multicolumn{4}{c}{Simulated test set} & \multicolumn{3}{c}{Measured window} \\" "\n"
                       r"Model & $x$ NRMSE & $\sigma$ NRMSE & $\xi$ RMSE & $\xi$ viol. & ROM--FOM, cycle 10 & ROM--exp., cycle 10 & ROM--exp., all", rows))
        V.append("- calrom:\n" + cr.groupby("model", sort=False).mean(numeric_only=True).round(4).to_string())
        ft = os.path.join(out, "rev_calrom_traces.npz")
        if os.path.exists(ft):
            d = np.load(ft)
            fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
            for x_ in ax:
                x_.axvspan(float(d["hold_from"]), d["t"][-1], color="0.92", lw=0)
            ax[0].plot(d["t"], d["I"], "k")
            ax[0].set_ylabel("Current (A)")
            ax[1].plot(d["t"], d["x_exp"] * 1e3, "k", lw=1.2, label="measured")
            ax[1].plot(d["t"], d["x_fom"] * 1e3, color="0.5", lw=1.0, ls="--", label="simulator")
            for k, col in (("x_POD-N+clip", "#1b9e77"), ("x_CR-ROM", "#3b5bdb"), ("x_POD-L", "#d95f02")):
                if k in d:
                    ax[1].plot(d["t"], d[k] * 1e3, color=col, lw=0.9, label=k[2:])
            ax[1].set_ylabel("Stroke (mm)")
            ax[1].set_xlabel("Time (s)")
            ax[1].legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
            fig.savefig(os.path.join(out, "fig_calrom.pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(out, "fig_calrom.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    open(os.path.join(out, "rev_tables.tex"), "w").write("\n".join(T))
    open(os.path.join(out, "rev_values.md"), "w").write("\n".join(V) + "\n")
    log(f"[report] wrote {out}/rev_tables.tex ({len(T)} tables), rev_values.md and figures")
