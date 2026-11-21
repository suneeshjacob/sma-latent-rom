"""Supplementary experiments, set 1 (results in the sections given in brackets).

    python revision.py                      # all stages, resumable
    python revision.py --stages calrom      # one stage
    python revision.py --quick              # tiny budget, for testing

Stages (section of the paper in brackets)
  podclip    POD-N with output clipping on the existing runs                       [Sec. 4.3]
  opinf      quadratic POD operator inference (non-neural nonlinear baseline)      [not in final paper; superseded by revision2 lindyn]
  horizon    error growth against rollout horizon                                  [Sec. 4.2]
  dtconv     time-step convergence of the simulator (settings B, C, straight wire) [Sec. 4.1]
  closure32  memory-complete models with a larger latent space (r = 32)            [Sec. 4.5]
  paired300  paired-history test with 300 pairs per class, matched-distance analysis [Sec. 4.5]
  calib      recalibration with measured power input, with a fixed slope C, and
             identifiability at the baseline optimum                                [Sec. 4.8]
  calrom     ROMs of the experimentally calibrated straight-wire actuator, evaluated
             against the simulator and against the measured held-out cycle         [Sec. 4.9]
  report     tables (LaTeX), figures and a summary in --out
"""
import argparse, json, os, subprocess, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fom import PRESETS, Actuator, simulate
from data import SPLITS, load_or_make, command_T, to_input, Scaler

ALL = ["podclip", "opinf", "horizon", "dtconv", "closure32", "paired300", "calib", "calrom", "report"]


# ------------------------------------------------------------------ helpers
def make_ctx(out, preset, quick, dev):
    import experiments as ex
    from rom import Cfg
    p = PRESETS[preset]
    data = load_or_make(out, p, list(SPLITS))
    cfg = Cfg()
    if preset == "revised":
        cfg = replace(cfg, w_dir=0.0)
    if quick:
        cfg = replace(cfg, ae_epochs=4, rate_epochs=4, windows=96, val_every=2)
    ctx = ex.Ctx(out, p, cfg, dev, data)
    ctx.quick = quick
    return ctx


def ckpt_rs(out, name="factorial_POD-neural"):
    fs = os.listdir(os.path.join(out, "ckpt"))
    return sorted({int(f.split("_r")[-1].split("_")[0]) for f in fs if f.startswith(name + "_r")})


def ckpt_seeds(out, name):
    fs = os.listdir(os.path.join(out, "ckpt"))
    return sorted({int(f.split("_s")[-1][:-3]) for f in fs if f.startswith(name + "_r")})


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def settings(a):
    return [("B", a.B, "paper"), ("C", a.C, "revised")]


# ------------------------------------------------------------------ podclip
def stage_podclip(a, log):
    import experiments as ex
    from rom import tt
    rows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc = ctx.sc["obs"]
        for r in ckpt_rs(out):
            for seed in ckpt_seeds(out, "factorial_POD-neural"):
                rom, kind = ex.load_rom(ctx, "factorial", "POD-neural", r, seed)
                rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev))
                for row in ex.evaluate(ctx, rom, kind, dict(stage="podclip", model="POD-N+clip", r=r, seed=seed)):
                    rows.append(dict(setting=tag, **row))
                log(f"[podclip] {tag} r={r} seed={seed}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev_podclip.csv"), index=False)


# ------------------------------------------------------------------ opinf
def stage_opinf(a, log):
    import torch
    import experiments as ex
    from rom import POD, RateQuad, ROM, val_score, tt
    rows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc, tr, va = ctx.sc["obs"], ctx.t["obs"]["train"], ctx.t["obs"]["val"]
        for r in [x for x in (8, 16) if x in ckpt_rs(out)] or ckpt_rs(out)[:1]:
            rep = POD(ctx.xn["obs"], r, ctx.dev)
            dyn = RateQuad(r).to(ctx.dev)
            dyn.fit(rep, tr["X"], tr["C"])
            rom = ROM(rep, dyn)
            best = (np.inf, None)
            for lam in 10.0 ** np.arange(-8, 2):
                dyn.set_ridge(lam)
                with np.errstate(all="ignore"):
                    try:
                        v = val_score(rom, va, sc.lay)
                    except Exception:
                        v = np.inf
                if np.isfinite(v) and v < best[0]:
                    best = (v, lam)
                log(f"[opinf] {tag} r={r} ridge={lam:.0e} val={v:.3e}")
            if best[1] is None:
                log(f"[opinf] {tag} r={r}: every ridge value diverged")
                continue
            dyn.set_ridge(best[1])
            for clip in (False, True):
                rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev)) if clip else None
                with np.errstate(all="ignore"):
                    for row in ex.evaluate(ctx, rom, "obs", dict(stage="opinf", model="POD-Q" + ("+clip" if clip else ""),
                                                                 r=r, seed=0)):
                        rows.append(dict(setting=tag, ridge=best[1], **row))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev_opinf.csv"), index=False)


# ------------------------------------------------------------------ horizon
HZ_MODELS = [("factorial", "POD-linear", "POD-L"), ("factorial", "POD-neural", "POD-N"),
             ("factorial", "AE-neural", "AE-N"), ("closure", "obs+reproject (V3)", "CR-ROM"),
             ("closure", "complete latent-only", "MC-ROM")]


def stage_horizon(a, log):
    import torch
    import experiments as ex
    from rom import predict
    rows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        r = 16 if 16 in ckpt_rs(out) else max(ckpt_rs(out))
        d = ctx.data["test"]
        xr = np.ptp(d["Y"][:, :, -2], axis=1)
        for stage, name, label in HZ_MODELS:
            errs = []
            for seed in ckpt_seeds(out, f"{stage}_{name.replace(' ', '_')}"):
                rom, kind = ex.load_rom(ctx, stage, name, r, seed)
                t = ctx.t[kind]["test"]
                with torch.no_grad():
                    Xh = predict(rom, t["X"], t["C"]).cpu().numpy()
                xh = ctx.sc[kind].obs(Xh.astype(np.float64))[:, :, -2]
                errs.append(np.abs(xh - d["Y"][:, :, -2]) / xr[:, None])
            if not errs:
                continue
            E = np.concatenate(errs, 0)                      # [traj*seeds, M]
            for k in range(E.shape[1]):
                rows.append(dict(setting=tag, model=label, step=k, t=k * ctx.p.dt,
                                 mean=float(E[:, k].mean()), p50=float(np.median(E[:, k])),
                                 p95=float(np.percentile(E[:, k], 95))))
            log(f"[horizon] {tag} {label}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev_horizon.csv"), index=False)


# ------------------------------------------------------------------ dtconv
def _sim_job(args):
    p, u, n, K = args
    Y, S, f = simulate(p, u, n, K)
    return Y[:, -2], f


def stage_dtconv(a, log):
    rows = []
    dts = [0.5, 0.25, 0.125, 0.0625] if not a.quick else [0.5, 0.25]
    ntraj = 2 if a.quick else 6
    for tag, preset in (("B", "paper"), ("C", "revised")):
        p0 = PRESETS[preset]
        rng = np.random.default_rng(11)
        Ts = [command_T(rng, p0) for _ in range(ntraj)]       # targets on the 0.5 s grid
        t0 = np.arange(len(Ts[0])) * p0.dt
        jobs, keys = [], []
        for dt in dts:
            p = replace(p0, dt=dt)
            t = np.arange(0, t0[-1] + 1e-9, dt)
            for j, T in enumerate(Ts):
                jobs.append((p, to_input(p, np.interp(t, t0, T)), 4.0, 5000.0))
                keys.append((dt, j))
        with Pool() as pool:
            res = pool.map(_sim_job, jobs)
        X = {k: v for k, v in zip(keys, res)}
        ref = dts[-1]
        for dt in dts[:-1]:
            for j, T in enumerate(Ts):
                xr_, fr = X[(ref, j)]
                x_, f_ = X[(dt, j)]
                step = int(round(p0.dt / dt)), int(round(p0.dt / ref))
                xa, xb = x_[::step[0]], xr_[::step[1]]
                m = min(len(xa), len(xb))
                xa, xb = xa[:m], xb[:m]
                du = np.diff(T[:m])
                rev = np.where(du[:-1] * du[1:] < 0)[0] + 1
                trap = getattr(np, "trapezoid", None) or np.trapz
                rows.append(dict(setting=tag, dt=dt, ref_dt=ref, traj=j,
                                 x_nrmse=float(np.sqrt(np.mean((xa - xb) ** 2)) / max(np.ptp(xb), 1e-12)),
                                 rev_x_err=float(np.sqrt(np.mean((xa[rev] - xb[rev]) ** 2)) / max(np.ptp(xb), 1e-12)) if len(rev) else np.nan,
                                 loop_area_err=float(abs(trap(xa, T[:m]) - trap(xb, T[:m])) / max(abs(trap(xb, T[:m])), 1e-12)),
                                 fails=int(f_)))
        log(f"[dtconv] {tag} done")
    # straight wire with the measured current
    ej = os.path.join(a.exp, "experiment.json")
    if os.path.exists(ej) and os.path.exists(a.data_exp):
        import experimental as xp
        e = json.load(open(ej))
        df = xp.load(a.data_exp)
        sdts = [0.1, 0.05, 0.025] if not a.quick else [0.1, 0.05]
        xs = {}
        for dt in sdts:
            w = xp.window(df, e["cycles"], dt)
            p = straight_params(e, N=32, dt=dt)
            act = Actuator(p)
            Y, _ = act.run(w["I"])
            xs[dt] = (w["t"], Y[:, -2], act.fails)
        tr, xr_, _ = xs[sdts[-1]]
        for dt in sdts[:-1]:
            t_, x_, f_ = xs[dt]
            xb = np.interp(t_, tr, xr_)
            rows.append(dict(setting="straight", dt=dt, ref_dt=sdts[-1], traj=0,
                             x_nrmse=float(np.sqrt(np.mean((x_ - xb) ** 2)) / np.ptp(xb)),
                             rev_x_err=np.nan, loop_area_err=np.nan, fails=int(f_)))
        log("[dtconv] straight wire done")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev_dtconv.csv"), index=False)


def straight_params(e, N=32, dt=None):
    """Simulator parameters of the calibrated straight-wire actuator from experiment.json."""
    th, rp = e["theta"], e["rho"]
    s0 = (e["preload_N"] / (np.pi * 0.3e-3 ** 2 / 4)) / 1e6
    smean = s0 + 60.0
    return replace(PRESETS["revised"], geometry="straight", dw=300e-6, L0w=e["L0_mm"] * 1e-3,
                   K=e["spring_N_per_mm"] * 1e3, F0=e["preload_N"],
                   heating="power" if e.get("input") == "power" else "joule",
                   Tamb=e["Tamb"], Tinit=e["Tamb"], dt=dt or e["dt"], N=N,
                   rhoeA=e.get("rhoeA", (rp["rhoA0"] + rp["kA"] * smean) * 1e-8),
                   rhoeM=e.get("rhoeM", (rp["rhoM0"] + rp["kM"] * smean) * 1e-8),
                   Ms=th["Ms"], Mf=th["Ms"] - th["dM"], As=th["Ms"] + th["dAs"],
                   Af=th["Ms"] + th["dAs"] + th["dA"], CM=th["C"], CA=th["C"], Hmax=th["Hmax"], hconv=th["hc"])


# ------------------------------------------------------------------ closure32
def stage_closure32(a, log):
    import experiments as ex
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        seeds = list(range(1 if a.quick else a.seeds))
        ex.run_stage(ctx, "closure32", [32], seeds)
        ex.summarize(out)
        log(f"[closure32] {tag} done")


# ------------------------------------------------------------------ paired300
def stage_paired300(a, log):
    import paired
    models = [("factorial", "POD-neural", "POD-N"), ("closure", "obs+reproject (V3)", "CR-ROM"),
              ("closure", "complete latent-only", "MC-ROM"), ("closure", "complete+reproject", "MC-ROM-R")]
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        r = 16 if 16 in ckpt_rs(out) else max(ckpt_rs(out))
        paired.run(ctx, n_pairs=15 if a.quick else 300, tag="300", models=models, r=r)
        log(f"[paired300] {tag} done")


# ------------------------------------------------------------------ calib
def stage_calib(a, log):
    base = [sys.executable, os.path.join(HERE, "experimental.py"), "--data", a.data_exp,
            "--tensile-dir", a.tensile_dir] + (["--quick", "--workers", "2"] if a.quick else [])
    runs = {"experiment_ident": ["--init", os.path.join(a.exp, "experiment.json")],
            "experiment_power": ["--input", "power"],
            "experiment_fixC": ["--fix", "C=8.34e6"]}
    for name, extra in runs.items():
        out = os.path.join(os.path.dirname(a.exp.rstrip("/")) or ".", name)
        if os.path.exists(os.path.join(out, "experiment.json")):
            log(f"[calib] {name} exists, skipped")
            continue
        log(f"[calib] {name}: " + " ".join(extra))
        subprocess.run(base + ["--out", out] + extra, check=True)


# ------------------------------------------------------------------ calrom
def current_profile(rng, T=80.0, dt=0.05):
    """Random piecewise-linear current history with complete and partial cycles."""
    knots_t, knots_i, t = [0.0, 1.0], [0.0, 0.0], 1.0
    while t < T:
        r = rng.random()
        ipk = rng.uniform(1.0, 1.3) if r < 0.3 else rng.uniform(1.3, 1.6)    # partial / complete heating
        tu = rng.uniform(2.0, 8.0)
        th = rng.uniform(0.0, 8.0)
        if rng.random() < 0.4:                                              # partial cooling
            iv, tv = rng.uniform(0.25, 0.6), 0.0
        else:
            iv, tv = 0.0, rng.uniform(1.0, 4.0)
        td = rng.uniform(5.0, 25.0)
        for dtk, ik in ((tu, ipk), (th, ipk), (td, iv), (tv, iv)):
            t += dtk
            knots_t.append(t)
            knots_i.append(ik)
    tt_ = np.arange(0.0, T, dt)
    return np.interp(tt_, knots_t, knots_i)


def _cal_job(args):
    p, I, sub = args
    t0 = time.perf_counter()
    act = Actuator(p)
    Y, S = act.run(I)
    return Y[::sub], S[::sub], act.fails, time.perf_counter() - t0


CAL_SPECS = {"POD-L": dict(rep="pod", dyn="linear"),
             "POD-N": dict(rep="pod", dyn="mlp"),
             "CR-ROM": dict(rep="ae", bounded=True, phys=True, dyn="mlp", reproject=True)}


def stage_calrom(a, log):
    import torch
    import experiments as ex
    import experimental as xp
    from rom import Cfg, tt, predict
    e = json.load(open(os.path.join(a.exp, "experiment.json")))
    N, dt_sim, sub = (16 if a.quick else 32), e["dt"], 2
    p = straight_params(e, N=N, dt=dt_sim)
    out = os.path.join(a.out, "calrom")
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    counts = dict(train=8, val=3, test=3) if a.quick else dict(train=64, val=16, test=20)
    T = 30.0 if a.quick else 80.0
    data = {}
    for split, n in counts.items():
        f = os.path.join(out, "data", split + ".npz")
        if not os.path.exists(f):
            rng = np.random.default_rng([31, list(counts).index(split)])
            Is = [current_profile(rng, T, dt_sim) for _ in range(n)]
            with Pool() as pool:
                res = pool.map(_cal_job, [(p, I, sub) for I in Is])
            d = dict(Y=np.array([r[0] for r in res]), S=np.array([r[1] for r in res]),
                     U=np.array([I[::sub] for I in Is]), P=np.tile([1.0, p.K], (n, 1)),
                     fails=np.array([r[2] for r in res]), tfom=np.array([r[3] for r in res]))
            np.savez_compressed(f, **d)
            log(f"[calrom] {split}: {n} trajectories, failed root solves {int(d['fails'].sum())}")
        data[split] = dict(np.load(f))
    pp = replace(p, dt=dt_sim * sub)                    # ROM step
    cfg = replace(Cfg(), w_dir=0.0)
    if a.quick:
        cfg = replace(cfg, ae_epochs=4, rate_epochs=4, windows=96, val_every=2)
    ctx = ex.Ctx(out, pp, cfg, device(), data)
    ctx.quick = a.quick

    # measured window (cycles 8-10) and the simulator on the measured current
    df = xp.load(a.data_exp)
    w = xp.window(df, e["cycles"], dt_sim)
    act = Actuator(p)
    Yf, Sf = act.run(w["I"])
    Yf, Sf = Yf[::sub], Sf[::sub]
    I_exp, x_exp, tw = w["I"][::sub], w["x"][::sub], w["t"][::sub]
    hold = tw >= w["bounds"][-2]
    stroke = np.ptp(x_exp)
    nr = lambda u, v, m: float(np.sqrt(np.mean((u[m] - v[m]) ** 2)) / stroke)
    allm = np.ones_like(hold)

    rows, traces = [], dict(t=tw, I=I_exp, x_exp=x_exp, x_fom=Yf[:, -2], hold_from=w["bounds"][-2])
    rows.append(dict(model="simulator", seed=0, fom_exp_all=nr(Yf[:, -2], x_exp, allm),
                     fom_exp_heldout=nr(Yf[:, -2], x_exp, hold)))
    r = 8 if a.quick else 16
    seeds = list(range(1 if a.quick else a.seeds))
    for name, spec in CAL_SPECS.items():
        for seed in seeds:
            if spec["dyn"] == "linear" and seed > 0:
                continue
            f = os.path.join(out, "ckpt", f"{name}_r{r}_s{seed}.pt")
            rom = ex.build(ctx, spec, r, seed)
            torch.save(rom.state_dict(), f)
            for clip in ((False, True) if name == "POD-N" else (False,)):
                sc = ctx.sc["obs"]
                rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev)) if clip else None
                label = name + ("+clip" if clip else "")
                # simulated test set (random protocols with partial cycles)
                tt_ = ctx.t["obs"]["test"]
                with torch.no_grad():
                    Xh = predict(rom, tt_["X"], tt_["C"]).cpu().numpy()
                Yh = sc.obs(Xh.astype(np.float64))
                Yt = data["test"]["Y"]
                xr = np.ptp(Yt[:, :, -2], axis=1)
                sr = np.ptp(Yt[:, :, -1], axis=1)
                xi_h, xi_t = Yh[:, :, N:2 * N], Yt[:, :, N:2 * N]
                # measured window: rollout from the simulator's initial state with the measured current
                x0 = torch.as_tensor(sc.fwd(Yf[0]), dtype=torch.float32, device=ctx.dev)[None]
                c = np.column_stack([I_exp[1:], np.tile([1.0, p.K], (len(I_exp) - 1, 1))])
                C = torch.as_tensor((c - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None]
                with torch.no_grad():
                    xe = sc.obs(rom.rollout(x0, C)[0].cpu().numpy().astype(np.float64))[:, -2]
                xe = np.concatenate([[Yf[0, -2]], xe])
                rows.append(dict(model=label, seed=seed, r=r,
                                 test_x_nrmse=float(np.mean(np.sqrt(np.mean((Yh[:, :, -2] - Yt[:, :, -2]) ** 2, 1)) / xr)),
                                 test_sig_nrmse=float(np.mean(np.sqrt(np.mean((Yh[:, :, -1] - Yt[:, :, -1]) ** 2, 1)) / sr)),
                                 test_xi_rmse=float(np.sqrt(np.mean((xi_h - xi_t) ** 2))),
                                 test_xi_viol=float(np.mean((xi_h < -1e-6) | (xi_h > 1 + 1e-6))),
                                 rom_fom_all=nr(xe, Yf[:, -2], allm), rom_fom_heldout=nr(xe, Yf[:, -2], hold),
                                 rom_exp_all=nr(xe, x_exp, allm), rom_exp_heldout=nr(xe, x_exp, hold),
                                 fom_exp_heldout=nr(Yf[:, -2], x_exp, hold)))
                if seed == 0:
                    traces["x_" + label] = xe
                log(f"[calrom] {label} seed {seed}: ROM-exp held-out {rows[-1]['rom_exp_heldout']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev_calrom.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, "rev_calrom_traces.npz"), **traces)


# ------------------------------------------------------------------ report
def stage_report(a, log):
    import report_rev
    report_rev.build(a, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--B", default="runs/paper")
    ap.add_argument("--C", default="runs/revised")
    ap.add_argument("--exp", default="runs/experiment")
    ap.add_argument("--data-exp", default="experimental/ActuatorTest_SMA300um.txt")
    ap.add_argument("--tensile-dir", default="experimental")
    ap.add_argument("--out", default="report_rev")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done = {"podclip": "rev_podclip.csv", "opinf": "rev_opinf.csv", "horizon": "rev_horizon.csv",
            "dtconv": "rev_dtconv.csv", "calrom": "rev_calrom.csv"}
    for s in a.stages.split(","):
        f = done.get(s)
        if f and os.path.exists(os.path.join(a.out, f)) and not a.force:
            print(f"[revision] {s}: output exists, skipped")
            continue
        t0 = time.time()
        print(f"[revision] === {s} ===", flush=True)
        globals()["stage_" + s](a, lambda m: print(m, flush=True))
        print(f"[revision] {s} finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
