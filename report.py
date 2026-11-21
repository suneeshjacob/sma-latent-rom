"""Build the manuscript tables, the factorial figure and the inline values from two runs.

    python report.py --B runs/paper --C runs/revised --out report

Writes report/tables.tex (one LaTeX table per manuscript table, same labels),
report/values.md (every inline number the text needs) and report/fig_factorial.pdf.
"""
import argparse, json, os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 16
NAMES = {"POD-L": ("factorial", "POD-linear"), "POD-N": ("factorial", "POD-neural"),
         "AE-L": ("factorial", "AE-linear"), "AE-N": ("factorial", "AE-neural"),
         "AE-N+clip": ("ablation", "plain+clip"), "AE-N-B": ("ablation", "bounded"),
         "AE-N-P": ("ablation", "penalties"), "AE-N-BP": ("ablation", "bounded+penalties"),
         "CR-ROM": ("closure", "obs+reproject (V3)"), "MC-ROM": ("closure", "complete latent-only"),
         "MC-ROM-R": ("closure", "complete+reproject"), "GRU-ROM": ("closure", "obs recurrent")}
LONG = {"AE-N-B": "AE-N-B (long)", "AE-N-BP": "AE-N-BP (long)",
        "CR-ROM": "CR-ROM (long)", "GRU-ROM": "GRU-ROM (long)"}


class Run:
    def __init__(self, path):
        self.path = path
        self.s = pd.read_csv(os.path.join(path, "summary.csv"))
        self.audit = json.load(open(os.path.join(path, "audit.json")))
        self.stats = json.load(open(os.path.join(path, "data_stats.json")))
        f = os.path.join(path, "paired_summary.csv")
        self.paired = pd.read_csv(f) if os.path.exists(f) else None
        f = os.path.join(path, "hardware.json")
        self.hw = json.load(open(f)) if os.path.exists(f) else None

    def row(self, model, split="test", mode="rollout", r=None, stage=None, name=None):
        r = R if r is None else r
        if stage is None:
            stage, name = NAMES[model]
        q = self.s[(self.s.stage == stage) & (self.s.model == name) & (self.s.r == r)
                   & (self.s.split == split) & (self.s["mode"] == mode)]
        return None if q.empty else q.iloc[0]

    def get(self, model, col, sd=False, **kw):
        q = self.row(model, **kw)
        if q is None:
            return np.nan
        v = q.get(col + "_sd" if sd else col, np.nan)
        return float(v) if pd.notna(v) else np.nan


def f(v, d=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def pm(run, model, col, d=3, **kw):
    m, s = run.get(model, col, **kw), run.get(model, col, sd=True, **kw)
    if not np.isfinite(m):
        return "--"
    return f(m, d) if not np.isfinite(s) or s == 0 else f"{f(m, d)} $\\pm$ {f(s, d)}"


def table(label, caption, colspec, header, rows, wide=True):
    env = "table*" if wide else "table"
    body = "\n".join(" & ".join(r) + r" \\" if isinstance(r, (list, tuple)) else r for r in rows)
    return (f"\\begin{{{env}}}[t]\n\\small\\sf\\centering\n\\caption{{{caption}\\label{{{label}}}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def tables(B, C):
    out = []
    # Table 3: audit
    def aud(run):
        a, st = run.audit, run.stats
        ms = [v["fom_ms_mean"] for v in st.values()]
        return dict(
            restart=f(a["restart_max_abs_diff"], 0),
            dx=f"{a['counterfactual_max_dx_mm']:.2f} mm ({a['counterfactual_dx_over_stroke']:.2f})",
            ds=f"{a['counterfactual_max_dsigma_MPa']:.1f} MPa",
            lh=f(a["latent_heat_x_nrmse_on_vs_off"]),
            mesh=f(a["mesh_x_nrmse_vs_N128"]["64"], 4),
            t=f"{min(ms) / 1e3:.2f}--{max(ms) / 1e3:.2f} s",
            fail=str(sum(v["failed_root_solves"] for v in st.values())),
            mem=f"{st['train']['memory_active_fraction']:.2f} / {st['test']['memory_active_fraction']:.2f}")
    a, b = aud(B), aud(C)
    rows = [("Maximum difference on complete-state restart", a["restart"], b["restart"]),
            (r"Counterfactual $|\Delta x|_{\max}$ (fraction of stroke)", a["dx"], b["dx"]),
            (r"Counterfactual $|\Delta\sigma|_{\max}$", a["ds"], b["ds"]),
            ("Latent heat on vs off, displacement NRMSE", a["lh"], b["lh"]),
            ("Mesh $N=64$ vs $N=128$, displacement NRMSE", a["mesh"], b["mesh"]),
            ("Simulator time per trajectory", a["t"], b["t"]),
            ("Failed root solves, all sets", a["fail"], b["fail"]),
            ("Cell-samples with active memory, train / test", a["mem"], b["mem"])]
    out.append(table("tab:audit", "Full-order model audit for settings B and C.", "lll",
                     "Quantity & Setting B & Setting C", rows))

    # Table 4: factorial
    rows = []
    for tag, run in (("B", B), ("C", C)):
        for j, m in enumerate(["POD-L", "POD-N", "AE-L", "AE-N"]):
            rows.append([f"\\multirow{{4}}{{*}}{{{tag}}}" if j == 0 else "", m,
                         pm(run, m, "x_nrmse"), pm(run, m, "sig_nrmse"), pm(run, m, "xi_rmse"),
                         f(run.get(m, "x_nrmse", mode="recon"), 5)])
        if tag == "B":
            rows.append(r"\midrule")
    out.append(table("tab:factorial", f"Factorial comparison at $r={R}$ (test set, mean $\\pm$ s.d. over "
                     "three seeds). Projection: encode--decode error without dynamics.", "llllll",
                     r"Setting & Model & $x$ NRMSE & $\sigma$ NRMSE & $\xi$ RMSE & Projection $x$ NRMSE", rows))

    # Table 5: admissibility
    rows = []
    for m in ["AE-N", "AE-N+clip", "AE-N-B", "AE-N-P", "AE-N-BP", "CR-ROM"]:
        r = [m]
        for run in (B, C):
            r += [f(run.get(m, "x_nrmse")), f(run.get(m, "xi_rmse")), f(run.get(m, "xi_viol"), 2),
                  f(run.get(m, "eq_rms_N"), 2)]
        rows.append(r)
    out.append(table("tab:admiss", f"Admissibility mechanisms at $r={R}$ (test set, mean over three seeds). "
                     r"$\xi$ viol.: fraction of samples with $\xi\notin[0,1]$.", "lllll|llll",
                     r" & \multicolumn{4}{c|}{Setting B} & \multicolumn{4}{c}{Setting C} \\" "\n"
                     r"Model & $x$ NRMSE & $\xi$ RMSE & $\xi$ viol. & $R_{\mathrm{eq}}$ (N) & "
                     r"$x$ NRMSE & $\xi$ RMSE & $\xi$ viol. & $R_{\mathrm{eq}}$ (N)", rows))

    # Table 6: closure
    rows = []
    for m in ["CR-ROM", "AE-N-BP", "MC-ROM-R", "MC-ROM", "GRU-ROM"]:
        r = [m]
        for run in (B, C):
            st, nm = NAMES[m] if m != "AE-N-BP" else ("closure", "obs latent-only")
            r += [f(run.get(m, "x_nrmse", stage=st, name=nm)),
                  f(run.get(m, "x_nrmse", mode="restart_active", stage=st, name=nm)),
                  f(run.get(m, "x_nrmse", mode="restart_inactive", stage=st, name=nm))]
        rows.append(r)
    cnt = []
    for run in (B, C):
        cnt += ["--", f(run.get("CR-ROM", "n_traj", mode="restart_active"), 0),
                f(run.get("CR-ROM", "n_traj", mode="restart_inactive"), 0)]
    rows += [r"\midrule", ["Restart count (active / inactive)"] + cnt]
    out.append(table("tab:closure", f"State closure at $r={R}$: displacement NRMSE from $t=0$ and from "
                     "mid-trajectory restarts with and without active reversal memory (mean over three seeds).",
                     "llll|lll", r" & \multicolumn{3}{c|}{Setting B} & \multicolumn{3}{c}{Setting C} \\" "\n"
                     r"Model & from $t=0$ & active & inactive & from $t=0$ & active & inactive", rows))

    # Table 7: extrapolation
    rows = []
    for tag, run in (("B", B), ("C", C)):
        for m in ["POD-L", "POD-N", "CR-ROM"]:
            rows.append([f"{tag} / {m}"] + [f(run.get(m, "x_nrmse", split=s))
                                           for s in ("ood_K", "ood_n", "ood_fast", "ood_rev")])
        if tag == "B":
            rows.append(r"\midrule")
    out.append(table("tab:ood", f"Displacement NRMSE on the extrapolation sets at $r={R}$ (mean over three seeds).",
                     "lllll", "Setting / model & Stiffness & Pair count & Fast ramps & Shallow reversals", rows))

    # Table 8: cost
    rows = []
    fom = [B.get("POD-L", "rom_ms") * B.get("POD-L", "speedup"), C.get("POD-L", "rom_ms") * C.get("POD-L", "speedup")]
    rows.append(["Simulator", f"{fom[0]:.0f} / {fom[1]:.0f}", "1"])
    for m in ["POD-L", "POD-N", "AE-N", "CR-ROM"]:
        rows.append([m, f"{B.get(m, 'rom_ms'):.1f} / {C.get(m, 'rom_ms'):.1f}",
                     f"{B.get(m, 'speedup'):.0f} / {C.get(m, 'speedup'):.0f}"])
    out.append(table("tab:cost", f"Single-thread cost of a 300-step trajectory at $r={R}$ (median; settings B / C).",
                     "lll", "Model & Time (ms) & Speedup", rows, wide=False))

    # New: long training
    if any(run.row(m, stage="longtrain", name=LONG[m]) is not None for run in (B, C) for m in LONG):
        rows = []
        for m in LONG:
            r = [m]
            for run in (B, C):
                st, nm = NAMES[m]
                r += [pm(run, m, "x_nrmse", stage=st, name=nm),
                      pm(run, m, "x_nrmse", stage="longtrain", name=LONG[m])]
            rows.append(r)
        out.append(table("tab:longtrain", f"Effect of a longer training budget at $r={R}$ (displacement NRMSE, "
                         "mean $\\pm$ s.d.). Standard: 180 epochs, learning rate $8\\times10^{-4}$; long: 540 epochs, "
                         "$3\\times10^{-4}$ with cosine decay.", "lll|ll",
                         r" & \multicolumn{2}{c|}{Setting B} & \multicolumn{2}{c}{Setting C} \\" "\n"
                         r"Model & standard & long & standard & long", rows))

    # New: paired-history test
    if B.paired is not None or C.paired is not None:
        rows = []
        models = list(dict.fromkeys(pd.concat([x.paired for x in (B, C) if x.paired is not None])["model"]))
        for m in models:
            r = [m]
            for run in (B, C):
                if run.paired is None:
                    r += ["--"] * 4
                    continue
                for cls in ("control", "hidden"):
                    q = run.paired[(run.paired.model == m) & (run.paired.cls == cls)]
                    r += ["--", "--"] if q.empty else [f(q["d_fom_mean"].iloc[0]), f(q["sep_mean"].iloc[0])]
            rows.append(r)
        out.append(table("tab:paired", "Paired-history test: simulator divergence $d_{\\mathrm{FOM}}$ and error in "
                         "the ROM-predicted divergence for pairs of near-identical observed states with similar "
                         "(control) or different (hidden) reversal memory, as fractions of the training stroke.",
                         "lllll|llll",
                         r" & \multicolumn{4}{c|}{Setting B} & \multicolumn{4}{c}{Setting C} \\" "\n"
                         r" & \multicolumn{2}{c}{control} & \multicolumn{2}{c|}{hidden} & "
                         r"\multicolumn{2}{c}{control} & \multicolumn{2}{c}{hidden} \\" "\n"
                         r"Model & $d_{\mathrm{FOM}}$ & error & $d_{\mathrm{FOM}}$ & error & "
                         r"$d_{\mathrm{FOM}}$ & error & $d_{\mathrm{FOM}}$ & error", rows))
    return "\n".join(out)


def values(B, C):
    """Every inline number the manuscript text needs, per setting."""
    L = ["# Inline values for the manuscript\n"]
    for tag, run in (("B", B), ("C", C)):
        L.append(f"## Setting {tag} ({run.path})\n")
        lin = [run.get(m, "x_nrmse", r=r) for m in ("POD-L", "AE-L") for r in (4, 8, 12, 16)]
        L.append(f"- linear-dynamics x NRMSE range over r: {np.nanmin(lin):.3f}--{np.nanmax(lin):.3f}")
        fac = [run.get(l, "x_nrmse", r=r) / run.get(n, "x_nrmse", r=r)
               for l, n in (("POD-L", "POD-N"), ("AE-L", "AE-N")) for r in (4, 8, 12, 16)]
        L.append(f"- linear/neural error ratio range: {np.nanmin(fac):.1f}--{np.nanmax(fac):.1f}")
        for r in (4, 8, 12, 16):
            L.append(f"- r={r}: POD-N {run.get('POD-N', 'x_nrmse', r=r):.4f} vs AE-N {run.get('AE-N', 'x_nrmse', r=r):.4f}"
                     f"  (projection POD {run.get('POD-N', 'x_nrmse', r=r, mode='recon'):.5f},"
                     f" AE {run.get('AE-N', 'x_nrmse', r=r, mode='recon'):.5f})")
        L.append(f"- AE-N violations: xi {run.get('AE-N', 'xi_viol'):.3f}, neg x {run.get('AE-N', 'neg_x'):.3f},"
                 f" neg sigma {run.get('AE-N', 'neg_sig'):.3f}")
        L.append(f"- re-projection: AE-N-BP {pm(run, 'AE-N-BP', 'x_nrmse')} -> CR-ROM {pm(run, 'CR-ROM', 'x_nrmse')}"
                 f" (factor {run.get('AE-N-BP', 'x_nrmse') / run.get('CR-ROM', 'x_nrmse'):.1f})")
        L.append(f"- MC-ROM projection x NRMSE {run.get('MC-ROM', 'x_nrmse', mode='recon'):.4f}")
        for m in ("CR-ROM", "POD-L"):
            L.append(f"- {m}: rev_xi_err {run.get(m, 'rev_xi_err'):.3f}, loop_area_err {run.get(m, 'loop_area_err'):.3f}")
        ood = [run.get(m, "x_nrmse", split=s) for m in ("CR-ROM", "POD-N") for s in ("ood_K", "ood_n", "ood_fast", "ood_rev")]
        L.append(f"- max OOD x NRMSE for CR-ROM and POD-N: {np.nanmax(ood):.3f}")
        L.append(f"- AE-N ood_n sigma NRMSE: {run.get('AE-N', 'sig_nrmse', split='ood_n'):.3f}")
        if run.hw:
            L.append(f"- hardware: {run.hw}")
        L.append("")
    return "\n".join(L)


def factorial_figure(B, C, path):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})
    cols = [("x_nrmse", "$x$ NRMSE"), ("sig_nrmse", r"$\sigma$ NRMSE"), ("xi_rmse", r"$\xi$ RMSE")]
    sty = {"POD-L": ("#d95f02", "o", ":"), "POD-N": ("#1b9e77", "s", "-"),
           "AE-L": ("#7570b3", "^", ":"), "AE-N": ("#3b5bdb", "D", "-")}
    rs = [4, 8, 12, 16]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True)
    for i, (tag, run) in enumerate((("B", B), ("C", C))):
        for j, (col, lab) in enumerate(cols):
            a = axs[i, j]
            for m, (c, mk, ls) in sty.items():
                mu = np.array([run.get(m, col, r=r) for r in rs])
                sd = np.nan_to_num([run.get(m, col, sd=True, r=r) for r in rs])
                a.errorbar(rs, mu, yerr=sd, color=c, marker=mk, ls=ls, ms=3, capsize=2, label=m)
            for m, c in (("POD-N", "#1b9e77"), ("AE-N", "#3b5bdb")):
                a.plot(rs, [run.get(m, col, r=r, mode="recon") for r in rs], color=c, ls="--", lw=0.6)
            a.set_yscale("log")
            a.set_title(f"Setting {tag}: {lab}", fontsize=8)
            if i == 1:
                a.set_xlabel("Reduced dimension $r$")
            a.set_xticks(rs)
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.text(0.5, -0.02, "Dashed thin lines: projection error (encode-decode, no dynamics).", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", default="runs/paper", help="run folder for setting B (paper preset)")
    ap.add_argument("--C", default="runs/revised", help="run folder for setting C (revised preset)")
    ap.add_argument("--out", default="report")
    ap.add_argument("--r", type=int, default=16, help="reduced dimension for the tables")
    a = ap.parse_args()
    global R
    R = a.r
    os.makedirs(a.out, exist_ok=True)
    B, C = Run(a.B), Run(a.C)
    open(os.path.join(a.out, "tables.tex"), "w").write(tables(B, C))
    open(os.path.join(a.out, "values.md"), "w").write(values(B, C))
    factorial_figure(B, C, os.path.join(a.out, "fig_factorial.pdf"))
    for tag, run in (("B", B), ("C", C)):          # collect per-setting figures
        d = os.path.join(run.path, "figures")
        if os.path.isdir(d):
            for fn in os.listdir(d):
                shutil.copy(os.path.join(d, fn), os.path.join(a.out, fn.replace("fig_", f"fig_{tag}_")))
    print(f"[report] wrote {a.out}/tables.tex, values.md, fig_factorial.pdf and per-setting figures")


if __name__ == "__main__":
    main()
