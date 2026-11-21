"""Supplementary experiments, set 3 (results in the sections given in brackets).

    python revision3.py                    # all stages, resumable (.done markers in --out)
    python revision3.py --stages ioseq     # one stage
    python revision3.py --quick            # tiny budget, for testing

Stages (section of the paper in brackets)
  eqcost   end-to-end single-thread cost of POD-N+clip with exact equilibrium projection,
           with root-solver function evaluations per step                           [Sec. 4.3, 4.7]
  calsens  POD-N(+clip), multistep bilinear POD and CR-ROM trained on each of the three
           calibrated straight-wire parameter sets: does the ranking persist?        [Sec. 4.9]
  longroll 600 s rollouts (four chained 150 s commands) in settings B and C          [Sec. 4.4]
  ioseq    direct input-output recurrent baseline: a GRU trained on full sequences that
           maps the input history to displacement and stress                         [Sec. 4.10]
  report3  rev3_values.md and fig_long in --out
"""
import argparse, copy, json, os, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fom import Actuator
from data import to_input, command_T
from revision import make_ctx, ckpt_rs, ckpt_seeds, device, settings, straight_params, current_profile
from revision2 import rdim, safe_val

ALL = ["eqcost", "calsens", "longroll", "ioseq", "report3"]


# ------------------------------------------------------------------ eqcost
def eq_project_count(act, Yh):
    """Exact equilibrium projection (as revision2.eq_project) that also counts residual evaluations."""
    p, N = act.p, act.p.N
    Yp, nfev = Yh.copy(), []
    for k in range(1, len(Yh)):
        T, xi, et = Yh[k, :N], np.clip(Yh[k, N:2 * N], 0, 1), Yh[k, 2 * N:3 * N]
        cnt = [0]

        def res(x):
            cnt[0] += 1
            sig = act.sig_of_x(x)
            strain = np.mean(act.S(xi) * sig + act.alpha(xi) * (T - p.T0) + et - act.eref)
            geo = -x / act.L0 if act.straight else (np.hypot(p.h, p.a - x) - act.L0) / act.L0
            return geo - strain
        if not act.straight and res(0.0) <= 0.0:
            x = 0.0
        elif res(act.xmin) > 0.0 > res(act.xmax):
            x = brentq(res, act.xmin, act.xmax, xtol=1e-14, rtol=1e-12)
        else:
            nfev.append(cnt[0])
            continue
        nfev.append(cnt[0])
        Yp[k, -2], Yp[k, -1] = x, act.sig_of_x(x)
    return Yp, nfev


def stage_eqcost(a, log):
    import torch
    import experiments as ex
    from rom import predict, tt
    rows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc, d, r = ctx.sc["obs"], ctx.data["test"], rdim(out)
        t = ctx.t["obs"]["test"]
        tfom = float(np.median(d["tfom"])) if "tfom" in d else np.nan
        for seed in ckpt_seeds(out, "factorial_POD-neural"):
            rom, _ = ex.load_rom(ctx, "factorial", "POD-neural", r, seed)
            cpu = copy.deepcopy(rom).to("cpu")
            cpu.clip = (tt(sc.clip_lo, "cpu"), tt(sc.clip_hi, "cpu"))
            nth = torch.get_num_threads()
            torch.set_num_threads(1)
            X, C = t["X"][:1].cpu(), t["C"][:1].cpu()
            act = Actuator(replace(ctx.p, n=float(d["P"][0, 0]), K=float(d["P"][0, 1])))
            tc, te, nf = [], [], []
            for i in range(9):
                t0 = time.perf_counter()
                with torch.no_grad():
                    Yh = sc.obs(predict(cpu, X, C).numpy().astype(np.float64))[0]
                t1 = time.perf_counter()
                _, nfev = eq_project_count(act, Yh)
                t2 = time.perf_counter()
                if i >= 2:
                    tc.append(t1 - t0)
                    te.append(t2 - t0)
                    nf = nfev
            torch.set_num_threads(nth)
            tc_ms, te_ms = 1e3 * float(np.median(tc)), 1e3 * float(np.median(te))
            rows.append(dict(setting=tag, seed=seed, fom_ms=1e3 * tfom, clip_ms=tc_ms, clip_eq_ms=te_ms,
                             eq_ms=te_ms - tc_ms, speedup_clip=1e3 * tfom / tc_ms, speedup_clip_eq=1e3 * tfom / te_ms,
                             nfev_mean=float(np.mean(nf)), nfev_max=int(np.max(nf)), steps=len(nf)))
            log(f"[eqcost] {tag} seed {seed}: clip {tc_ms:.1f} ms, clip+eq {te_ms:.1f} ms "
                f"(speedup {rows[-1]['speedup_clip']:.0f} -> {rows[-1]['speedup_clip_eq']:.0f}), "
                f"residual evaluations per step {rows[-1]['nfev_mean']:.1f} (max {rows[-1]['nfev_max']})")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev3_eqcost.csv"), index=False)


# ------------------------------------------------------------------ calsens
def _sim_job(args):
    p, u_sim, sub = args
    t0 = time.perf_counter()
    act = Actuator(p)
    Y, S = act.run(u_sim)
    return Y[::sub], S[::sub], act.fails, time.perf_counter() - t0


def poly_bilinear(ctx, r, seed, log, verbose=False):
    """Multistep bilinear POD ROM: one-step ridge fit (regularisation selected on validation
    rollouts), then 24-step unrolled training as for the neural dynamics."""
    import torch
    from rom import POD, RatePoly, ROM, train_rate
    tr, va, lay = ctx.t["obs"]["train"], ctx.t["obs"]["val"], ctx.sc["obs"].lay
    rep = POD(ctx.xn["obs"], r, ctx.dev)
    dyn = RatePoly(r, bilinear=True).to(ctx.dev)
    dyn.fit(rep, tr["X"], tr["C"])
    rom = ROM(rep, dyn)
    best = (np.inf, None)
    for lam in 10.0 ** np.arange(-8, 2):
        dyn.set_ridge(lam)
        v = safe_val(rom, va, lay)
        if v < best[0]:
            best = (v, lam)
    if best[1] is None:
        lin = RatePoly(r).to(ctx.dev)
        lin.fit(rep, tr["X"], tr["C"], 1e-8)
        with torch.no_grad():
            dyn.W.zero_()
            dyn.W[:r + 4] = lin.W
    else:
        dyn.set_ridge(best[1])
    init, v0 = copy.deepcopy(dyn.state_dict()), safe_val(rom, va, lay)
    torch.manual_seed(seed + 1000)
    vb = train_rate(rom, tr, va, ctx.w["obs"], ctx.cfg, lay, log=log if verbose else None)
    if not vb <= v0:
        dyn.load_state_dict(init)
    return rom


def stage_calsens(a, log):
    import torch
    import experiments as ex
    import experimental as xp
    from rom import Cfg, tt, predict
    base = os.path.dirname(a.exp.rstrip("/")) or "."
    sets = [("current input (retained)", a.exp), ("measured power input", os.path.join(base, "experiment_power")),
            ("fixed C", os.path.join(base, "experiment_fixC"))]
    sets = [(l, d) for l, d in sets if os.path.exists(os.path.join(d, "experiment.json"))]
    N, sub = (16 if a.quick else 32), 2
    counts = dict(train=8, val=3, test=3) if a.quick else dict(train=64, val=16, test=20)
    T = 30.0 if a.quick else 80.0
    r = 8 if a.quick else 16
    seeds = list(range(1 if a.quick else a.seeds))
    df = xp.load(a.data_exp)
    rows = []
    for si, (label, d) in enumerate(sets):
        e = json.load(open(os.path.join(d, "experiment.json")))
        dt_sim = e["dt"]
        p = straight_params(e, N=N, dt=dt_sim)
        w = xp.window(df, e["cycles"], dt_sim)
        R_eff = float(np.sum(w["P"] * w["I"] ** 2) / np.sum(w["I"] ** 4))
        to_sim = (lambda I: R_eff * I ** 2) if p.heating == "power" else (lambda I: I)
        out = os.path.join(a.out, "calsens", f"set{si}")
        os.makedirs(os.path.join(out, "data"), exist_ok=True)
        data = {}
        for split, n in counts.items():
            f = os.path.join(out, "data", split + ".npz")
            if not os.path.exists(f):
                rng = np.random.default_rng([31, list(counts).index(split)])      # same currents as calrom
                Is = [current_profile(rng, T, dt_sim) for _ in range(n)]
                with Pool() as pool:
                    res = pool.map(_sim_job, [(p, to_sim(I), sub) for I in Is])
                dd = dict(Y=np.array([q[0] for q in res]), S=np.array([q[1] for q in res]),
                          U=np.array([I[::sub] for I in Is]), P=np.tile([1.0, p.K], (n, 1)),
                          fails=np.array([q[2] for q in res]), tfom=np.array([q[3] for q in res]))
                np.savez_compressed(f, **dd)
            data[split] = dict(np.load(f))
        log(f"[calsens] {label}: data ready, failed root solves {int(sum(data[s]['fails'].sum() for s in data))}")
        cfg = replace(Cfg(), w_dir=0.0)
        if a.quick:
            cfg = replace(cfg, ae_epochs=4, rate_epochs=4, windows=96, val_every=2)
        ctx = ex.Ctx(out, replace(p, dt=dt_sim * sub), cfg, device(), data)
        ctx.quick = a.quick
        sc = ctx.sc["obs"]
        # simulator on the measured current of cycles 8-10 (reference for ROM-FOM on the window)
        act = Actuator(p)
        Yf = act.run(to_sim(w["I"]))[0][::sub]
        I_w = w["I"][::sub]
        Yt = data["test"]["Y"]
        xr = np.ptp(Yt[:, :, -2], axis=1)
        for seed in seeds:
            models = {}
            models["POD-N"] = ex.build(ctx, dict(rep="pod", dyn="mlp"), r, seed)
            models["POD-BL-U"] = poly_bilinear(ctx, r, seed, log, a.verbose)
            models["CR-ROM"] = ex.build(ctx, dict(rep="ae", bounded=True, phys=True, dyn="mlp", reproject=True), r, seed)
            for name, rom in models.items():
                for clip in ((False, True) if name == "POD-N" else (False,)):
                    rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev)) if clip else None
                    lab = name + ("+clip" if clip else "")
                    t_ = ctx.t["obs"]["test"]
                    with torch.no_grad():
                        Yh = sc.obs(predict(rom, t_["X"], t_["C"]).cpu().numpy().astype(np.float64))
                    xn = np.sqrt(np.mean((Yh[:, :, -2] - Yt[:, :, -2]) ** 2, 1)) / xr
                    x0 = torch.as_tensor(sc.fwd(Yf[0]), dtype=torch.float32, device=ctx.dev)[None]
                    c = np.column_stack([I_w[1:], np.tile([1.0, p.K], (len(I_w) - 1, 1))])
                    C = torch.as_tensor((c - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None]
                    with torch.no_grad():
                        xe = sc.obs(rom.rollout(x0, C)[0].cpu().numpy().astype(np.float64))[:, -2]
                    xe = np.concatenate([[Yf[0, -2]], xe])
                    rows.append(dict(set=label, model=lab, seed=seed, test_x_nrmse=float(xn.mean()),
                                     test_x_nrmse_p95=float(np.percentile(xn, 95)),
                                     window_rom_fom=float(np.sqrt(np.mean((xe - Yf[:, -2]) ** 2)) / np.ptp(Yf[:, -2])),
                                     **{f"traj{i}": float(v) for i, v in enumerate(xn)}))
                    log(f"[calsens] {label} {lab} seed {seed}: test x NRMSE {rows[-1]['test_x_nrmse']:.4f}, "
                        f"measured-current window ROM-FOM {rows[-1]['window_rom_fom']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "rev3_calsens.csv"), index=False)
    log("[calsens]\n" + out.groupby(["set", "model"], sort=False)[["test_x_nrmse", "window_rom_fom"]]
        .agg(["mean", "std"]).round(4).to_string())


# ------------------------------------------------------------------ longroll
def _long_job(args):
    p, u = args
    act = Actuator(p)
    Y, S = act.run(u)
    return Y, S, act.fails


LONG_MODELS = [("factorial", "POD-linear", "POD-L", False), ("factorial", "POD-neural", "POD-N", False),
               ("factorial", "POD-neural", "POD-N+clip", True), ("factorial", "AE-neural", "AE-N", False),
               ("ablation", "bounded+penalties", "AE-N-BP", False), ("closure", "obs+reproject (V3)", "CR-ROM", False),
               ("closure", "complete latent-only", "MC-ROM", False)]


def stage_longroll(a, log):
    import torch
    import experiments as ex
    from rom import tt
    rows, curves = [], []
    nseg, ntraj = 4, (3 if a.quick else 10)
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        p, r = ctx.p, rdim(out)
        M = ctx.data["test"]["Y"].shape[1]
        rng = np.random.default_rng(11)
        jobs, PK = [], []
        for j in range(ntraj):
            Tc = np.concatenate([command_T(rng, p, M) for _ in range(nseg)])
            n, K = float(rng.choice([2, 4, 6])), float(rng.uniform(3000, 8000))
            jobs.append((replace(p, n=n, K=K), to_input(p, Tc)))
            PK.append((n, K))
        with Pool() as pl:
            res = pl.map(_long_job, jobs)
        Y = np.array([q[0] for q in res])
        S = np.array([q[1] for q in res])
        log(f"[longroll] {tag}: {ntraj} trajectories of {Y.shape[1]} steps ({Y.shape[1] * p.dt:.0f} s), "
            f"failed root solves {sum(q[2] for q in res)}")
        xr = np.ptp(Y[:, :, -2], axis=1)
        for stage, name, label, clip in LONG_MODELS:
            for seed in ckpt_seeds(out, f"{stage}_{name.replace(' ', '_')}"):
                rom, kind = ex.load_rom(ctx, stage, name, r, seed)
                sc = ctx.sc[kind]
                if clip:
                    rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev))
                for j in range(ntraj):
                    src = Y[j, 0] if kind == "obs" else S[j, 0]
                    x0 = torch.as_tensor(sc.fwd(src), dtype=torch.float32, device=ctx.dev)[None]
                    u = jobs[j][1]
                    c = np.column_stack([u[1:], np.repeat(np.array(PK[j])[None], len(u) - 1, 0)])
                    C = torch.as_tensor((c - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None]
                    with torch.no_grad():
                        Xh = rom.rollout(x0, C)[0].cpu().numpy().astype(np.float64)
                    xh = np.concatenate([[Y[j, 0, -2]], sc.obs(Xh)[:, -2]])
                    err = np.abs(xh - Y[j, :, -2]) / xr[j]
                    seg = [float(np.sqrt(np.mean(((xh - Y[j, :, -2]) / xr[j])[s * M:(s + 1) * M] ** 2))) for s in range(nseg)]
                    rows.append(dict(setting=tag, model=label, seed=seed, traj=j, x_nrmse=float(np.sqrt(np.mean(err ** 2))),
                                     max_err=float(err.max()), finite=bool(np.isfinite(xh).all()),
                                     **{f"seg{s + 1}": v for s, v in enumerate(seg)}))
                    curves.append(err)
                    rows[-1]["curve_index"] = len(curves) - 1
            q = pd.DataFrame([r_ for r_ in rows if r_["setting"] == tag and r_["model"] == label])
            log(f"[longroll] {tag} {label}: " + ", ".join(f"seg{s + 1} {q[f'seg{s + 1}'].mean():.4f}" for s in range(nseg))
                + f", max {q.max_err.max():.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev3_longroll.csv"), index=False)
    np.save(os.path.join(a.out, "rev3_longroll_curves.npy"), np.array(curves, dtype=np.float32))


# ------------------------------------------------------------------ ioseq
def make_io(nin=3, hid=64, nout=2, layers=2):
    import torch
    import torch.nn as nn

    class IOGRU(nn.Module):
        """Direct input-output recurrent model: input history -> (x, sigma)."""
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(hid, nout)
            if hasattr(nn, "GRU"):            # fused multi-layer GRU (PyTorch)
                self.gru = nn.GRU(nin, hid, num_layers=layers, batch_first=True)
            else:                             # cell-by-cell fallback (test stand-in only)
                self.gru = None
                self.cells = [nn.GRUCell(nin if i == 0 else hid, hid) for i in range(layers)]

        def forward(self, C, y0):
            """C [B, T, nin] standardized inputs for steps 1..T; y0 [B, nout] initial normalized output."""
            if self.gru is not None:
                h, _ = self.gru(C)
                return y0[:, None] + self.head(h)
            hs = [torch.zeros(C.shape[0], hid) for _ in range(layers)]
            outs = []
            for k in range(C.shape[1]):
                z = C[:, k]
                for i, c in enumerate(self.cells):
                    hs[i] = c(z, hs[i])
                    z = hs[i]
                outs.append(self.head(z))
            return y0[:, None] + torch.stack(outs, 1)
    return IOGRU()


def stage_ioseq(a, log):
    import torch
    rows = []
    epochs = 4 if a.quick else 300
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc = ctx.sc["obs"]
        T = {s: ctx.t["obs"][s] for s in ctx.eval_splits + ["train", "val"] if s in ctx.t["obs"]}
        get = lambda s: (T[s]["C"][:, 1:], T[s]["X"][:, :, -2:])          # inputs, normalized (x, sigma)
        Ctr, Ytr = get("train")
        Cva, Yva = get("val")
        w = torch.as_tensor([2.5, 1.5], dtype=torch.float32, device=ctx.dev)
        for seed in range(1 if a.quick else a.seeds):
            torch.manual_seed(seed)
            m = make_io().to(ctx.dev)
            opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-6)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            best, best_sd = np.inf, copy.deepcopy(m.state_dict())
            t0 = time.perf_counter()
            for ep in range(epochs):
                m.train()
                perm = torch.randperm(len(Ctr))
                for i in range(0, len(perm), 16):
                    b = perm[i:i + 16]
                    pred = m(Ctr[b], Ytr[b][:, 0])
                    loss = ((pred - Ytr[b][:, 1:]) ** 2 * w).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()
                if (ep + 1) % 10 == 0 or ep == epochs - 1:
                    m.eval()
                    with torch.no_grad():
                        pv = m(Cva, Yva[:, 0])
                        v = float(((pv - Yva[:, 1:]) ** 2 * w).mean())
                    if v < best:
                        best, best_sd = v, copy.deepcopy(m.state_dict())
            m.load_state_dict(best_sd)
            m.eval()
            log(f"[ioseq] {tag} seed {seed}: trained {epochs} epochs ({time.perf_counter() - t0:.0f} s), best val {best:.3e}")
            for split in ctx.eval_splits:
                C_, Y_ = get(split)
                with torch.no_grad():
                    ph = m(C_, Y_[:, 0]).cpu().numpy()
                Xh = T[split]["X"].cpu().numpy().copy()
                Xh[:, 1:, -2:] = ph
                Yh = sc.obs(Xh.astype(np.float64))
                Yt = ctx.data[split]["Y"]
                xn = np.sqrt(np.mean((Yh[:, :, -2] - Yt[:, :, -2]) ** 2, 1)) / np.ptp(Yt[:, :, -2], axis=1)
                sn = np.sqrt(np.mean((Yh[:, :, -1] - Yt[:, :, -1]) ** 2, 1)) / np.ptp(Yt[:, :, -1], axis=1)
                rows.append(dict(setting=tag, model="IO-GRU", seed=seed, split=split, x_nrmse=float(xn.mean()),
                                 sig_nrmse=float(sn.mean()), neg_x=float(np.mean(Yh[:, :, -2] < 0)),
                                 **({f"traj{i}": float(v) for i, v in enumerate(xn)} if split == "test" else {})))
            # single-thread cost of one trajectory
            nth = torch.get_num_threads()
            torch.set_num_threads(1)
            mc = copy.deepcopy(m).to("cpu")
            C1, Y1 = Ctr[:1].cpu(), Ytr[:1].cpu()
            ts = []
            for i in range(9):
                t1 = time.perf_counter()
                with torch.no_grad():
                    mc(C1, Y1[:, 0])
                if i >= 2:
                    ts.append(time.perf_counter() - t1)
            torch.set_num_threads(nth)
            tfom = float(np.median(ctx.data["test"]["tfom"])) if "tfom" in ctx.data["test"] else np.nan
            for row in rows[-len(ctx.eval_splits):]:
                row["rom_ms"] = 1e3 * float(np.median(ts))
                row["speedup"] = tfom / float(np.median(ts))
            q = rows[-len(ctx.eval_splits)]
            log(f"[ioseq] {tag} seed {seed}: test x NRMSE {q['x_nrmse']:.4f}, sigma NRMSE {q['sig_nrmse']:.4f}, "
                f"speedup {q['speedup']:.0f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "rev3_ioseq.csv"), index=False)
    log("[ioseq]\n" + out.groupby(["setting", "split"])[["x_nrmse", "sig_nrmse"]].agg(["mean", "std"]).round(4).to_string())


# ------------------------------------------------------------------ report3
def stage_report3(a, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})
    V = ["# Third revision: values\n"]
    f = os.path.join(a.out, "rev3_eqcost.csv")
    if os.path.exists(f):
        V += ["## eqcost (mean over seeds)", pd.read_csv(f).groupby("setting").mean(numeric_only=True).round(3).to_string()]
    f = os.path.join(a.out, "rev3_calsens.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        V += ["\n## calsens", d.groupby(["set", "model"], sort=False)[["test_x_nrmse", "test_x_nrmse_p95", "window_rom_fom"]]
              .agg(["mean", "std"]).round(4).to_string()]
        tc = [c for c in d.columns if c.startswith("traj")]
        rng = np.random.default_rng(0)
        lines = []
        for s_, q in d.groupby("set", sort=False):
            M_ = {m: g[tc].mean(axis=0).values for m, g in q.groupby("model")}
            B = rng.integers(0, len(tc), (2000, len(tc)))
            for m1, m2 in (("POD-N+clip", "CR-ROM"), ("POD-BL-U", "POD-N"), ("POD-N", "CR-ROM")):
                if m1 in M_ and m2 in M_:
                    rat = M_[m1][B].mean(1) / M_[m2][B].mean(1)
                    lines.append(f"{s_}: {m1}/{m2} = {M_[m1].mean() / M_[m2].mean():.3f} "
                                 f"[{np.percentile(rat, 2.5):.3f}, {np.percentile(rat, 97.5):.3f}]")
        V += ["### bootstrap ratios over test trajectories (seed-averaged)"] + lines
    f = os.path.join(a.out, "rev3_longroll.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        V += ["\n## longroll (mean over trajectories and seeds)",
              d.groupby(["setting", "model"], sort=False)[["seg1", "seg2", "seg3", "seg4", "x_nrmse", "max_err"]]
              .mean().round(4).to_string(),
              "### largest pointwise error / any non-finite",
              d.groupby(["setting", "model"], sort=False).agg(max_err=("max_err", "max"), finite=("finite", "all")).to_string()]
        cv = os.path.join(a.out, "rev3_longroll_curves.npy")
        if os.path.exists(cv):
            Cv = np.load(cv)
            tags = list(d.setting.unique())
            fig, axs = plt.subplots(1, len(tags), figsize=(3.5 * len(tags), 2.4), sharey=True)
            axs = np.atleast_1d(axs)
            cols = {"POD-L": "#d95f02", "POD-N+clip": "#1b9e77", "AE-N": "#7570b3", "AE-N-BP": "#999999",
                    "CR-ROM": "#3b5bdb", "MC-ROM": "#e7298a"}
            dt = 0.5
            for ax, tag in zip(axs, tags):
                for mdl, col in cols.items():
                    q = d[(d.setting == tag) & (d.model == mdl)]
                    if len(q):
                        mean = Cv[q.curve_index.values].mean(0)
                        ax.plot(np.arange(len(mean)) * dt, np.maximum(mean, 1e-5), color=col, lw=0.8, label=mdl)
                for s in range(1, 4):
                    ax.axvline(s * len(mean) * dt / 4, color="0.8", lw=0.5)
                ax.set_yscale("log")
                ax.set_xlabel("Time (s)")
                ax.set_title(f"Setting {tag}", fontsize=8)
            axs[0].set_ylabel(r"$|x-\tilde x|$ / stroke")
            axs[0].legend(fontsize=6, ncol=2)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(a.out, f"fig_long.{ext}"), dpi=180, bbox_inches="tight")
            plt.close(fig)
    f = os.path.join(a.out, "rev3_ioseq.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        V += ["\n## ioseq", d.groupby(["setting", "split"])[["x_nrmse", "sig_nrmse", "neg_x", "speedup"]]
              .agg(["mean", "std"]).round(4).to_string()]
    open(os.path.join(a.out, "rev3_values.md"), "w").write("\n".join(V) + "\n")
    log(f"[report3] wrote {a.out}/rev3_values.md and figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--B", default="runs/paper")
    ap.add_argument("--C", default="runs/revised")
    ap.add_argument("--exp", default="runs/experiment")
    ap.add_argument("--data-exp", default="experimental/ActuatorTest_SMA300um.txt")
    ap.add_argument("--out", default="report_rev3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print training curves")
    ap.add_argument("--force", action="store_true", help="rerun stages already done")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.stages.split(","):
        marker = os.path.join(a.out, f".{s}.done")
        if s != "report3" and os.path.exists(marker) and not a.force:
            print(f"[revision3] {s}: already done, skipped (use --force to rerun)")
            continue
        t0 = time.time()
        print(f"[revision3] === {s} ===", flush=True)
        globals()["stage_" + s](a, lambda m: print(m, flush=True))
        open(marker, "w").write(time.ctime())
        print(f"[revision3] {s} finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
