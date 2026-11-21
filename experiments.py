"""Experiment stages: simulator audit, factorial baselines, constraint ablation,
state-closure comparison, extrapolation tests, timings and summary tables."""
import copy, json, os, time
from dataclasses import replace
import numpy as np
import pandas as pd

from fom import Actuator, simulate
from data import command, constants, Scaler, OOD
from metrics import traj_metrics, aggregate

KEYS = ["stage", "model", "r", "seed", "split", "mode"]

# ---- model specifications --------------------------------------------------
# state: 'obs' (3N+2 observed) | 'full' (10N+2 restart-complete)
# rep:   'pod' | 'ae'      dyn: 'linear' | 'mlp' | 'gru'
# bounded: sigmoid decoder   phys: soft residual penalties
# reproject: decode -> encode after each step   also_clip: also score with output clipping
V3 = dict(rep="ae", bounded=True, phys=True, dyn="mlp")
STAGES = {
    "factorial": dict(all_r=True, restart=False, specs={
        "POD-linear": dict(rep="pod", dyn="linear"),
        "POD-neural": dict(rep="pod", dyn="mlp"),
        "AE-linear":  dict(rep="ae", dyn="linear"),
        "AE-neural":  dict(rep="ae", dyn="mlp"),
    }),
    "ablation": dict(all_r=False, restart=False, specs={
        "plain":             dict(rep="ae", dyn="mlp", also_clip=True),
        "bounded":           dict(rep="ae", dyn="mlp", bounded=True),
        "penalties":         dict(rep="ae", dyn="mlp", phys=True),
        "bounded+penalties": dict(V3),
    }),
    "closure": dict(all_r=False, restart=True, specs={
        "obs+reproject (V3)":   dict(V3, reproject=True),
        "obs latent-only":      dict(V3),
        "complete+reproject":   dict(V3, state="full", reproject=True),
        "complete latent-only": dict(V3, state="full"),
        "obs recurrent":        dict(V3, dyn="gru"),
    }),
    # same models, three times the epochs, lower learning rate with cosine decay:
    # settles whether the bounded and recurrent models were only undertrained
    "longtrain": dict(all_r=False, restart=True,
                      cfg=dict(rate_epochs=540, lr=3e-4, cosine=True), specs={
        "AE-N-B (long)":  dict(rep="ae", dyn="mlp", bounded=True),
        "AE-N-BP (long)": dict(V3),
        "CR-ROM (long)":  dict(V3, reproject=True),
        "GRU-ROM (long)": dict(V3, dyn="gru"),
    }),
}

# memory-complete models given a larger latent space (642 inputs), with CR-ROM at the same r
STAGES["closure32"] = dict(all_r=False, restart=True, r=32, specs={
    "obs+reproject (V3)":   dict(V3, reproject=True),
    "complete latent-only": dict(V3, state="full"),
    "complete+reproject":   dict(V3, state="full", reproject=True),
})

# model names used in the paper
PAPER_NAME = {
    "POD-linear": "POD-L", "POD-neural": "POD-N", "AE-linear": "AE-L", "AE-neural": "AE-N",
    "plain": "AE-N", "plain+clip": "AE-N+clip", "bounded": "AE-N-B", "penalties": "AE-N-P",
    "bounded+penalties": "AE-N-BP", "obs+reproject (V3)": "CR-ROM", "obs latent-only": "AE-N-BP",
    "complete+reproject": "MC-ROM-R", "complete latent-only": "MC-ROM", "obs recurrent": "GRU-ROM",
}


def find_spec(stage, name):
    return STAGES[stage]["specs"][name]


def ckpt_path(out, stage, name, r, seed):
    return os.path.join(out, "ckpt", f"{stage}_{name.replace(' ', '_')}_r{r}_s{seed}.pt")


def load_rom(ctx, stage, name, r, seed):
    """Rebuild a trained ROM from its checkpoint (no training). Returns (rom, state kind)."""
    import torch
    from rom import AE, POD, RateMLP, RateLinear, RateGRU, ROM
    spec = find_spec(stage, name)
    kind = spec.get("state", "obs")
    sc, dev = ctx.sc[kind], ctx.dev
    rep = POD(ctx.xn[kind], r, dev) if spec["rep"] == "pod" else \
        AE(sc.lay.D, r, bool(spec.get("bounded"))).to(dev)
    gru = spec["dyn"] == "gru"
    dyn = (RateLinear(r) if spec["dyn"] == "linear" else
           RateGRU(r, ctx.cfg.hid) if gru else RateMLP(r)).to(dev)
    rom = ROM(rep, dyn, spec.get("reproject", False), warm=ctx.cfg.warm if gru else 0).to(dev)
    sd = torch.load(ckpt_path(ctx.out, stage, name, r, seed), map_location=dev)
    rom.load_state_dict(sd)
    rom.eval()
    for prm in rom.parameters():
        prm.requires_grad_(False)
    return rom, kind


# ------------------------------------------------------------ simulator audit
def audit(out, p, log=print):
    """Restart exactness, hidden-memory counterfactual, mesh and latent-heat
    sensitivity, and simulator run time. NumPy only."""
    N, res = p.N, {}
    rng = np.random.default_rng(123)
    u = command(rng, p)
    pa = replace(p, n=1.0, K=6000.0)                 # high stress -> many partial cycles
    act = Actuator(pa)
    Y, S = act.run(u)
    active = (S[:, 6 * N:7 * N] > 0).sum(1)
    k = int(np.argmax(active[:len(u) - 60]))
    res["restart_step"], res["active_cells_at_restart"] = k, int(active[k])

    def replay(s):
        b = Actuator(pa)
        b.set_full(s)
        Yb, Sb = b.run(u[k:])
        return Yb, Sb
    Yb, Sb = replay(S[k])
    res["restart_max_abs_diff"] = float(max(np.abs(Yb - Y[k:]).max(), np.abs(Sb - S[k:]).max()))
    s = S[k].copy()
    s[6 * N:10 * N] = 0.0                            # wipe m, d0, r1, r2 only
    Yc, _ = replay(s)
    res["counterfactual_initial_obs_diff"] = float(np.abs(Yc[0] - Y[k]).max())
    res["counterfactual_max_dx_mm"] = float(np.abs(Yc[:, -2] - Y[k:, -2]).max() * 1e3)
    res["counterfactual_max_dsigma_MPa"] = float(np.abs(Yc[:, -1] - Y[k:, -1]).max() / 1e6)
    res["counterfactual_dx_over_stroke"] = float(np.abs(Yc[:, -2] - Y[k:, -2]).max() / np.ptp(Y[:, -2]))
    res["counterfactual_dsigma_over_range"] = float(np.abs(Yc[:, -1] - Y[k:, -1]).max() / np.ptp(Y[:, -1]))

    nrm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / max(np.ptp(b), 1e-12))
    xs = {n_: simulate(replace(p, N=n_), u, 4, 5000)[0][:, -2] for n_ in (8, 16, 32, 64, 128)}
    res["mesh_x_nrmse_vs_N128"] = {str(n_): nrm(xs[n_], xs[128]) for n_ in (8, 16, 32, 64)}
    x_on = simulate(replace(p, latent_heat=True), u, 4, 5000)[0][:, -2]
    x_off = simulate(replace(p, latent_heat=False), u, 4, 5000)[0][:, -2]
    res["latent_heat_x_nrmse_on_vs_off"] = nrm(x_off, x_on)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        simulate(p, u, 4, 5000)
        ts.append(time.perf_counter() - t0)
    res["fom_ms_per_trajectory"] = 1e3 * float(np.median(ts))
    with open(os.path.join(out, "audit.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("[audit] " + json.dumps(res, indent=2))
    return res


def data_stats(out, data, p, log=print):
    N = p.N
    st = {s: dict(n_traj=int(len(d["Y"])), failed_root_solves=int(d["fails"].sum()),
                  memory_active_fraction=float((d["S"][:, :, 6 * N:7 * N] > 0).mean()),
                  stage2_fraction=float((d["S"][:, :, 6 * N:7 * N] > 1.5).mean()),
                  fom_ms_mean=1e3 * float(d["tfom"].mean())) for s, d in data.items()}
    with open(os.path.join(out, "data_stats.json"), "w") as f:
        json.dump(st, f, indent=2)
    log("[data] " + json.dumps(st, indent=2))


# ------------------------------------------------------------- neural stages
class Ctx:
    def __init__(self, out, p, cfg, dev, data, log=print):
        import torch
        from rom import Physics, tt
        self.out, self.p, self.cfg, self.dev, self.data, self.log = out, p, cfg, dev, data, log
        self.k = constants(p)
        self.sc, self.t, self.w, self.phys, self.xn = {}, {}, {}, {}, {}
        for kind in ("obs", "full"):
            sc = self.sc[kind] = Scaler(data["train"], p.N, kind)
            self.t[kind] = {}
            for s, d in data.items():
                X, C = sc.tensors(d)
                self.t[kind][s] = dict(X=tt(X, dev), C=tt(C, dev), nK=tt(d["P"], dev))
                if s == "train":
                    self.xn[kind] = X
            self.w[kind] = tt(sc.w, dev)
            self.phys[kind] = Physics(sc, self.k, dev)
        self.eval_splits = ["test"] + [s for s in OOD if s in data]
        M = data["test"]["Y"].shape[1]
        self.k0s = [int(v) for v in np.linspace(0.15 * M, 0.70 * M, 5)]
        self.ae_cache = {}
        self.fom_ms = 1e3 * float(np.median(data["test"]["tfom"]))


def build(ctx, spec, r, seed, cfg=None):
    import torch
    from rom import AE, POD, RateMLP, RateLinear, RateGRU, ROM, train_ae, train_rate
    cfg, dev = cfg or ctx.cfg, ctx.dev
    kind = spec.get("state", "obs")
    sc, tr, va = ctx.sc[kind], ctx.t[kind]["train"], ctx.t[kind]["val"]
    phys = ctx.phys[kind] if spec.get("phys") else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    if spec["rep"] == "pod":
        rep = POD(ctx.xn[kind], r, dev)
    else:                                   # autoencoders are shared between specs
        key = (kind, bool(spec.get("bounded")), phys is not None, r, seed)
        if key not in ctx.ae_cache:
            ae = AE(sc.lay.D, r, bool(spec.get("bounded"))).to(dev)
            train_ae(ae, tr, va, ctx.w[kind], cfg, phys, ctx.log)
            ctx.ae_cache[key] = ae
        rep = ctx.ae_cache[key]
    torch.manual_seed(seed + 1000)
    if spec["dyn"] == "linear":
        dyn = RateLinear(r).to(dev)
        dyn.fit(rep, tr["X"], tr["C"])
        return ROM(rep, dyn, spec.get("reproject", False))
    gru = spec["dyn"] == "gru"
    dyn = (RateGRU(r, cfg.hid) if gru else RateMLP(r)).to(dev)
    with torch.no_grad():
        dyn.set_stats(rep.encode(tr["X"]))
    rom = ROM(rep, dyn, spec.get("reproject", False), warm=cfg.warm if gru else 0)
    train_rate(rom, tr, va, ctx.w[kind], cfg, sc.lay, phys,
               w_dir=cfg.w_dir if phys is not None else 0.0, log=ctx.log)
    return rom


def _score(ctx, kind, Xh, split, k0=0):
    """Per-trajectory metrics for normalised predictions Xh [ntraj, M-k0, D]."""
    d = ctx.data[split]
    Yh = ctx.sc[kind].obs(np.asarray(Xh, dtype=np.float64))
    return [traj_metrics(Yh[i], d["Y"][i, k0:], d["U"][i, k0:], d["P"][i], ctx.k,
                         ref=(np.ptp(d["Y"][i, :, -2]), np.ptp(d["Y"][i, :, -1])))
            for i in range(len(Yh))]


def time_rom(rom, t, reps=7):
    """Median single-trajectory rollout time on one CPU thread [ms]."""
    import torch
    from rom import predict
    nth = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu = copy.deepcopy(rom).to("cpu")
    cpu.clip = None
    X, C = t["X"][:1].cpu(), t["C"][:1].cpu()
    ts = []
    for i in range(reps + 2):
        t0 = time.perf_counter()
        predict(cpu, X, C)
        if i >= 2:
            ts.append(time.perf_counter() - t0)
    torch.set_num_threads(nth)
    return 1e3 * float(np.median(ts))


def evaluate(ctx, rom, kind, base, restart=False):
    import torch
    from rom import predict
    rows, N = [], ctx.p.N
    for split in ctx.eval_splits:
        t = ctx.t[kind][split]
        Xh = predict(rom, t["X"], t["C"]).cpu().numpy()
        rows.append({**base, "split": split, "mode": "rollout", **aggregate(_score(ctx, kind, Xh, split))})
    rows[0]["rom_ms"] = time_rom(rom, ctx.t[kind]["test"])
    rows[0]["fom_ms"] = ctx.fom_ms
    rows[0]["speedup"] = ctx.fom_ms / rows[0]["rom_ms"]
    t = ctx.t[kind]["test"]
    with torch.no_grad():                                   # projection error, no dynamics
        Xr = rom.rep.decode(rom.rep.encode(t["X"])).cpu().numpy()
    rows.append({**base, "split": "test", "mode": "recon", **aggregate(_score(ctx, kind, Xr, "test"))})
    if restart:        # restart from mid-trajectory states, split by hidden-memory activity
        act, inact = [], []
        mem = ctx.data["test"]["S"][:, :, 6 * N:7 * N] > 0
        for k0 in ctx.k0s:
            Xh = predict(rom, t["X"], t["C"], k0).cpu().numpy()
            for i, m in enumerate(_score(ctx, kind, Xh, "test", k0)):
                (act if mem[i, k0].mean() >= 0.05 else inact).append(m)
        for mode, rr in (("restart_active", act), ("restart_inactive", inact)):
            if rr:
                rows.append({**base, "split": "test", "mode": mode, **aggregate(rr)})
    return rows


def run_stage(ctx, stage, rs, seeds):
    import torch
    from rom import tt
    st = STAGES[stage]
    f = os.path.join(ctx.out, "results.csv")
    df = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=KEYS)
    done = set(zip(df["stage"], df["model"], df["r"], df["seed"]))
    os.makedirs(os.path.join(ctx.out, "ckpt"), exist_ok=True)
    for r in (rs if st["all_r"] else [st.get("r", max(rs))]):
        for seed in seeds:
            for name, spec in st["specs"].items():
                if (stage, name, r, seed) in done:
                    continue
                if spec["rep"] == "pod" and spec["dyn"] == "linear" and seed != seeds[0]:
                    continue                                   # deterministic model
                ctx.log(f"[{stage}] {name}  r={r}  seed={seed}")
                t0 = time.perf_counter()
                cfg = ctx.cfg
                if "cfg" in st and not getattr(ctx, "quick", False):
                    cfg = replace(ctx.cfg, **st["cfg"])
                rom = build(ctx, spec, r, seed, cfg)
                kind = spec.get("state", "obs")
                base = dict(stage=stage, model=name, r=r, seed=seed)
                rows = evaluate(ctx, rom, kind, base, st["restart"])
                if spec.get("also_clip"):
                    sc = ctx.sc[kind]
                    rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev))
                    rows += evaluate(ctx, rom, kind, {**base, "model": name + "+clip"})
                    rom.clip = None
                torch.save(rom.state_dict(), ckpt_path(ctx.out, stage, name, r, seed))
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                df.to_csv(f, index=False)
                ctx.log(f"    test x NRMSE {rows[0]['x_nrmse']:.4f}  sigma NRMSE {rows[0]['sig_nrmse']:.4f}"
                        f"  xi RMSE {rows[0]['xi_rmse']:.4f}  ({time.perf_counter() - t0:.0f} s)")
        ctx.ae_cache.clear()


# ------------------------------------------------------------------- summary
def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summarize(out, log=print):
    """Mean and standard deviation over seeds -> summary.csv and summary.md."""
    df = pd.read_csv(os.path.join(out, "results.csv"))
    num = [c for c in df.columns if c not in KEYS]
    g = df.groupby(KEYS[:3] + KEYS[4:], sort=False)[num]
    mean, std = g.mean().reset_index(), g.std().reset_index()
    full = mean.merge(std, on=KEYS[:3] + KEYS[4:], suffixes=("", "_sd"))
    full.to_csv(os.path.join(out, "summary.csv"), index=False)

    def table(sel, cols, extra=()):
        m, s = mean[sel], std[sel]
        t = m[["model", "r", *extra]].copy()
        for c in cols:
            if c in m:
                sd = s[c].fillna(0.0)
                t[c] = [f"{a:.4g} ± {b:.2g}" for a, b in zip(m[c], sd)]
        return _md(t)

    main = ["x_nrmse", "sig_nrmse", "xi_rmse", "x_nrmse_p95", "xi_viol", "neg_x", "neg_sig",
            "eq_rms_N", "cp_rms", "rev_xi_err", "rev_x_err", "loop_area_err", "rom_ms", "speedup"]
    err = ["x_nrmse", "sig_nrmse", "xi_rmse", "rev_xi_err", "loop_area_err"]
    parts = ["# Summary (mean ± s.d. over seeds)"]
    for stage in df["stage"].unique():
        base = mean["stage"] == stage
        parts += [f"\n## {stage}: test rollouts from t = 0\n",
                  table(base & (mean["split"] == "test") & (mean["mode"] == "rollout"), main),
                  f"\n## {stage}: projection only (encode-decode, no dynamics)\n",
                  table(base & (mean["mode"] == "recon"), err[:3]),
                  f"\n## {stage}: extrapolation sets\n",
                  table(base & (mean["mode"] == "rollout") & (mean["split"] != "test"), err, ("split",))]
        if mean.loc[base, "mode"].str.startswith("restart").any():
            parts += [f"\n## {stage}: mid-trajectory restarts (hidden memory active / inactive)\n",
                      table(base & mean["mode"].str.startswith("restart"), err + ["n_traj"], ("mode",))]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(parts) + "\n")
    log(f"[summary] wrote {out}/summary.csv and {out}/summary.md")
