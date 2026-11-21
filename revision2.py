"""Supplementary experiments, set 2 (results in the sections given in brackets).

    python revision2.py                     # all stages, resumable
    python revision2.py --stages matched    # one stage
    python revision2.py --quick             # tiny budget, for testing

Stages (section of the paper in brackets)
  lindyn   linear, bilinear and quadratic+bilinear POD dynamics trained with the SAME
           24-step unrolled loss as the neural model (one-step fits for reference), and
           POD in the weighted norm of the autoencoder loss with neural dynamics      [Sec. 4.2]
  reproj   re-projection on the plain and on the bounded autoencoder (no penalties);
           latent distance to the training encodings, fixed-point defect and Jacobian
           spectra of the one-step maps with and without re-projection               [Sec. 4.4]
  matched  matched paired-history test: twin states with IDENTICAL observed fields and a
           different, locally consistent reversal memory (cell-wise transplant)       [Sec. 4.5]
  minor    the three calibrated parameter sets on simulated partial-cycle current
           histories (does non-identifiability matter for partial transformations?)   [Sec. 4.8]
  eqproj   exact spring-equilibrium projection of the POD-N+clip output               [Sec. 4.3]
  boot     trajectory-level bootstrap intervals for the key comparisons               [Sec. 4.2-4.5]
  verify   simulator checks of Section 2.5 repeated with the current code             [Sec. 2.5]
  report2  rev2_values.md and figures in --out
"""
import argparse, copy, json, os, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fom import PRESETS, Actuator, FIELDS
from data import to_input
from revision import make_ctx, ckpt_rs, ckpt_seeds, device, settings, straight_params, current_profile

ALL = ["lindyn", "reproj", "matched", "minor", "eqproj", "boot", "verify", "report2"]


def rdim(out):
    rs = ckpt_rs(out)
    return 16 if 16 in rs else max(rs)


def append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def traj_rows(ctx, rom, kind, tag, label, seed):
    """Per-trajectory test metrics (for the bootstrap stage)."""
    import torch
    import experiments as ex
    from rom import predict
    t = ctx.t[kind]["test"]
    with torch.no_grad():
        Xh = predict(rom, t["X"], t["C"]).cpu().numpy()
    with np.errstate(all="ignore"):
        sc = ex._score(ctx, kind, Xh, "test")
    return [dict(setting=tag, model=label, seed=seed, traj=i, x_nrmse=m["x_nrmse"], sig_nrmse=m["sig_nrmse"],
                 xi_rmse=m["xi_rmse"]) for i, m in enumerate(sc)]


def safe_val(rom, va, lay):
    from rom import val_score
    with np.errstate(all="ignore"):
        try:
            v = val_score(rom, va, lay)
        except Exception:
            v = np.inf
    return v if np.isfinite(v) else np.inf


# ------------------------------------------------------------------ lindyn
POLY = {"L": dict(), "BL": dict(bilinear=True), "QB": dict(bilinear=True, quadratic=True)}


def stage_lindyn(a, log):
    import torch
    import experiments as ex
    from rom import POD, WPOD, RatePoly, RateMLP, ROM, train_rate
    res_f, trj_f = os.path.join(a.out, "rev2_lindyn.csv"), os.path.join(a.out, "rev2_traj.csv")
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc, tr, va, lay = ctx.sc["obs"], ctx.t["obs"]["train"], ctx.t["obs"]["val"], ctx.sc["obs"].lay
        r = rdim(out)
        seeds = list(range(1 if a.quick else a.seeds))
        rep = POD(ctx.xn["obs"], r, ctx.dev)
        rows, trows = [], []
        for key, kw in POLY.items():
            # one-step ridge fit, regularisation selected on validation rollouts
            dyn = RatePoly(r, **kw).to(ctx.dev)
            dyn.fit(rep, tr["X"], tr["C"])
            rom = ROM(rep, dyn)
            best = (np.inf, None)
            for lam in 10.0 ** np.arange(-8, 2):
                dyn.set_ridge(lam)
                v = safe_val(rom, va, lay)
                if v < best[0]:
                    best = (v, lam)
            if best[1] is None:                        # every one-step fit diverged: start from linear
                lin = RatePoly(r).to(ctx.dev)
                lin.fit(rep, tr["X"], tr["C"], 1e-8)
                with torch.no_grad():
                    dyn.W.zero_()
                    dyn.W[:r + 4] = lin.W
                log(f"[lindyn] {tag} POD-{key}-1: every ridge diverged; unrolled training starts from the linear fit")
            else:
                dyn.set_ridge(best[1])
                for row in ex.evaluate(ctx, rom, "obs", dict(stage="lindyn", model=f"POD-{key}-1", r=r, seed=0)):
                    rows.append(dict(setting=tag, ridge=best[1], **row))
                trows += traj_rows(ctx, rom, "obs", tag, f"POD-{key}-1", 0)
                log(f"[lindyn] {tag} POD-{key}-1 ridge {best[1]:.0e}: val {best[0]:.3e}, "
                    f"test x NRMSE {rows[-(len(ctx.eval_splits) + 1)]['x_nrmse']:.4f}")
            init = copy.deepcopy(dyn.state_dict())
            v0 = safe_val(rom, va, lay)
            for seed in seeds:                         # same unrolled objective as the neural model
                torch.manual_seed(seed + 1000)
                dyn.load_state_dict(init)
                t0 = time.perf_counter()
                vb = train_rate(rom, tr, va, ctx.w["obs"], ctx.cfg, lay, log=log if a.verbose else None)
                if not vb <= v0:                       # training never beat the one-step start
                    dyn.load_state_dict(init)
                for row in ex.evaluate(ctx, rom, "obs", dict(stage="lindyn", model=f"POD-{key}-U", r=r, seed=seed)):
                    rows.append(dict(setting=tag, ridge=best[1], val_init=v0, val_best=vb, **row))
                trows += traj_rows(ctx, rom, "obs", tag, f"POD-{key}-U", seed)
                log(f"[lindyn] {tag} POD-{key}-U seed {seed}: val {v0:.3e} -> {min(vb, v0):.3e} "
                    f"({time.perf_counter() - t0:.0f} s)")
        # weighted POD + neural dynamics (representation objective matched to the autoencoder)
        wrep = WPOD(ctx.xn["obs"], r, ctx.dev, sc.w)
        for seed in seeds:
            torch.manual_seed(seed + 1000)
            dyn = RateMLP(r).to(ctx.dev)
            with torch.no_grad():
                dyn.set_stats(wrep.encode(tr["X"]))
            rom = ROM(wrep, dyn)
            t0 = time.perf_counter()
            train_rate(rom, tr, va, ctx.w["obs"], ctx.cfg, lay, log=log if a.verbose else None)
            for row in ex.evaluate(ctx, rom, "obs", dict(stage="lindyn", model="POD-N-W", r=r, seed=seed)):
                rows.append(dict(setting=tag, **row))
            trows += traj_rows(ctx, rom, "obs", tag, "POD-N-W", seed)
            log(f"[lindyn] {tag} POD-N-W seed {seed} ({time.perf_counter() - t0:.0f} s)")
        append_csv(res_f, rows)
        append_csv(trj_f, trows)
        q = pd.DataFrame(rows)
        q = q[(q.split == "test") & (q["mode"] == "rollout")].groupby("model", sort=False).x_nrmse.agg(["mean", "std"])
        log(f"[lindyn] {tag} test x NRMSE:\n" + q.round(4).to_string())


# ------------------------------------------------------------------ reproj
def latent_rollout(rom, x0, C):
    """Rollout that also returns the latent states (after re-projection, if any)."""
    import torch
    q = rom.rep.encode(x0)
    Q = []
    with torch.no_grad():
        for k in range(C.shape[1]):
            q, _ = rom.dyn.step(q, C[:, k])
            if rom.reproject:
                q = rom.rep.encode(rom.rep.decode(q))
            Q.append(q)
    return torch.stack(Q, 1)


def latent_diag(rom, tr, te, nref=20000, nj=60, seed=0):
    """Distance of rollout latents to the training encodings (relative to the typical
    nearest-neighbour spacing of the training encodings), fixed-point defect of E(D(.)),
    and spectral radii of the one-step map and of E(D(.)) at visited states."""
    import torch
    from torch.autograd.functional import jacobian
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        Qt = rom.rep.encode(tr["X"]).reshape(-1, rom.dyn.q_mu.numel())
    sd = Qt.std(0) + 1e-8
    idx = rng.permutation(len(Qt))
    ref, hold = Qt[idx[:nref]] / sd, Qt[idx[nref:nref + 2000]] / sd

    def nn_dist(Z):
        out = []
        for i in range(0, len(Z), 1024):
            out.append(torch.cdist(Z[i:i + 1024], ref).min(1).values)
        return torch.cat(out)
    d0 = float(nn_dist(hold).median())
    Q = latent_rollout(rom, te["X"][:, 0], te["C"][:, 1:])          # [ntraj, M-1, r]
    Z = Q.reshape(-1, Q.shape[-1])
    dist = (nn_dist(Z / sd) / d0).cpu().numpy()
    with torch.no_grad():
        defect = ((rom.rep.encode(rom.rep.decode(Z)) - Z) / sd).pow(2).mean(1).sqrt().cpu().numpy()
    # Jacobians at visited states
    pick = rng.choice(len(Z), size=min(nj, len(Z)), replace=False)
    Cf = te["C"][:, 1:].reshape(-1, te["C"].shape[-1])
    rho_step, rho_ed, ed_mean = [], [], []
    for j in pick:
        qj, cj = Z[j].clone(), Cf[j]

        def f(q):
            qn, _ = rom.dyn.step(q[None], cj[None])
            if rom.reproject:
                qn = rom.rep.encode(rom.rep.decode(qn))
            return qn[0]

        def ed(q):
            return rom.rep.encode(rom.rep.decode(q[None]))[0]
        J = jacobian(f, qj).detach().cpu().numpy()
        E = jacobian(ed, qj).detach().cpu().numpy()
        rho_step.append(float(np.max(np.abs(np.linalg.eigvals(J)))))
        ev = np.abs(np.linalg.eigvals(E))
        rho_ed.append(float(ev.max()))
        ed_mean.append(float(ev.mean()))
    return dict(dist_mean=float(dist.mean()), dist_p95=float(np.percentile(dist, 95)),
                defect_mean=float(defect.mean()), rho_step_med=float(np.median(rho_step)),
                rho_step_p90=float(np.percentile(rho_step, 90)), frac_rho_gt1=float(np.mean(np.array(rho_step) > 1)),
                ed_rho_med=float(np.median(rho_ed)), ed_eig_mean=float(np.median(ed_mean)))


def stage_reproj(a, log):
    import torch
    import experiments as ex
    from rom import RateMLP, ROM, train_rate
    res_f, trj_f = os.path.join(a.out, "rev2_reproj.csv"), os.path.join(a.out, "rev2_traj.csv")
    diag = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc, tr, va, te, lay = (ctx.sc["obs"], ctx.t["obs"]["train"], ctx.t["obs"]["val"],
                               ctx.t["obs"]["test"], ctx.sc["obs"].lay)
        r = rdim(out)
        rows, trows = [], []
        for seed in ckpt_seeds(out, "ablation_plain")[:1 if a.quick else a.seeds]:
            models = {}
            for base, label in (("plain", "AE-N"), ("bounded", "AE-N-B")):
                rom0, _ = ex.load_rom(ctx, "ablation", base, r, seed)
                models[label] = rom0
                torch.manual_seed(seed + 1000)
                dyn = RateMLP(r).to(ctx.dev)
                with torch.no_grad():
                    dyn.set_stats(rom0.rep.encode(tr["X"]))
                rom = ROM(rom0.rep, dyn, reproject=True)
                t0 = time.perf_counter()
                train_rate(rom, tr, va, ctx.w["obs"], ctx.cfg, lay, log=log if a.verbose else None)
                rom.eval()
                for prm in rom.parameters():
                    prm.requires_grad_(False)
                models[label + "-R"] = rom
                for row in ex.evaluate(ctx, rom, "obs", dict(stage="reproj", model=label + "-R", r=r, seed=seed)):
                    rows.append(dict(setting=tag, **row))
                trows += traj_rows(ctx, rom, "obs", tag, label + "-R", seed)
                log(f"[reproj] {tag} {label}-R seed {seed}: test x NRMSE {rows[-(len(ctx.eval_splits) + 1)]['x_nrmse']:.4f} "
                    f"({time.perf_counter() - t0:.0f} s)")
            models["AE-N-BP"] = ex.load_rom(ctx, "ablation", "bounded+penalties", r, seed)[0]
            models["CR-ROM"] = ex.load_rom(ctx, "closure", "obs+reproject (V3)", r, seed)[0]
            for label, rom in models.items():
                d = latent_diag(rom, tr, te, nref=2000 if a.quick else 20000, nj=10 if a.quick else 60, seed=seed)
                diag.append(dict(setting=tag, model=label, seed=seed, reproject=bool(rom.reproject), **d))
                log(f"[reproj] {tag} {label} seed {seed}: " + ", ".join(f"{k} {v:.3g}" for k, v in d.items()))
        append_csv(res_f, rows)
        append_csv(trj_f, trows)
    pd.DataFrame(diag).to_csv(os.path.join(a.out, "rev2_reproj_diag.csv"), index=False)


# ------------------------------------------------------------------ matched
MEM = ["xir", "etr", "dlast", "m", "d0", "r1", "r2"]


def fs(N, f):
    i = FIELDS.index(f)
    return slice(i * N, (i + 1) * N)


def _consistent(xi, m, d0, r1, r2):
    """Reversal memory admissible for the current fraction xi (vectorised)."""
    ok = np.ones_like(xi, dtype=bool)
    ok &= ~((m == 1) & (d0 > 0) & (xi > r1 + 1e-9))
    ok &= ~((m == 2) & (d0 > 0) & ((xi > r1 + 1e-9) | (xi < r2 - 1e-9)))
    ok &= ~((m == 1) & (d0 < 0) & (xi < r1 - 1e-9))
    ok &= ~((m == 2) & (d0 < 0) & ((xi < r1 - 1e-9) | (xi > r2 + 1e-9)))
    return ok


def make_twin(s, p_rec, tree, D, N, rng, tx, te):
    """Copy of state s whose memory fields are transplanted cell by cell from donor cells
    with (nearly) the same xi and transformation strain and a different memory. Cells whose
    transplanted memory would change xi at the current temperature and stress are reverted."""
    xi, et = s[fs(N, "xi")], s[fs(N, "et")]
    twin = s.copy()
    cand = tree.query_ball_point(np.c_[xi / tx, et / te], r=1.0)
    for j in range(N):
        if not (0.01 < xi[j] < 0.99) or not cand[j]:
            continue
        c = np.array(cand[j])
        md, d0d, r1d, r2d = D["m"][c], D["d0"][c], D["r1"][c], D["r2"][c]
        m0, r10 = s[fs(N, "m")][j], s[fs(N, "r1")][j]
        ok = ((md != m0) | (np.abs(r1d - r10) > 0.05)) & _consistent(np.full(len(c), xi[j]), md, d0d, r1d, r2d)
        if not ok.any():
            continue
        dn = c[rng.choice(np.where(ok)[0])]
        for f in MEM:
            twin[fs(N, f).start + j] = D[f][dn]
        if D["dlast"][dn] < 0:            # reverse branch through the current state
            twin[fs(N, "etr").start + j] = et[j] * D["xir"][dn] / max(xi[j], 1e-12)
    act = Actuator(p_rec)
    for _ in range(3):                    # stationarity at unchanged temperature and stress
        act.set_full(twin)
        trl = act.trial(act.c["T"].copy(), act.sig)
        bad = np.abs(trl["xi"] - xi) > 1e-4
        if not bad.any():
            break
        for f in MEM:
            sl = fs(N, f)
            twin[sl][bad] = s[sl][bad]
    act.set_full(s)
    jump0 = float(np.abs(act.trial(act.c["T"].copy(), act.sig)["xi"] - xi).max())
    act.set_full(twin)
    jump1 = float(np.abs(act.trial(act.c["T"].copy(), act.sig)["xi"] - xi).max())
    return twin, jump0, jump1


def _matched_job(args):
    p, s0, s1, u = args
    xs = []
    for s in (s0, s1):
        act = Actuator(p)
        act.set_full(s)
        xs.append(act.run(u)[0][:, -2])
    return xs


def stage_matched(a, log):
    import torch
    from scipy.spatial import cKDTree
    import paired
    from experiments import load_rom, ckpt_path
    rows, meta = [], []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        p, N = ctx.p, ctx.p.N
        pool = paired.make_pool(out, p)
        S, Y = pool["S"], pool["Y"]
        ks_d = np.arange(paired.K_MIN, paired.K_MAX + 1, 2)
        Sd = S[:, ks_d].reshape(-1, S.shape[-1])
        D = {f: Sd[:, fs(N, f)].ravel() for f in MEM + ["xi", "et"]}
        tx = 0.004
        te = 0.004 * max(np.ptp(D["et"]), 1e-12)
        tree = cKDTree(np.c_[D["xi"] / tx, D["et"] / te])
        rng = np.random.default_rng(7)
        ks = np.arange(paired.K_MIN, paired.K_MAX + 1, paired.STRIDE)
        cand = [(i, k) for i in range(len(S)) for k in ks
                if Y[i, k, -2] > 1e-5 and np.mean((S[i, k, fs(N, "xi")] > 0.02) & (S[i, k, fs(N, "xi")] < 0.98)) >= 0.2]
        rng.shuffle(cand)
        n_want = 15 if a.quick else 300
        xref = ctx.sc["obs"].hi[ctx.sc["obs"].lay.ix] / 1.08
        twins, jobs = [], []
        for (i, k) in cand:
            if len(twins) >= n_want:
                break
            n, K = pool["P"][i]
            pr = replace(p, n=float(n), K=float(K))
            tw, j0, j1 = make_twin(S[i, k], pr, tree, D, N, rng, tx, te)
            md = paired._memdiff(np.stack([S[i, k], tw]), 0, 1, N)
            if md < paired.HIDDEN_MIN or j1 > 1e-4:
                continue
            q = len(twins)
            u = to_input(p, paired.continuation(pool["T"][i, k], q % 3))
            twins.append((i, k, tw, md, j0, j1, u))
            jobs.append((pr, S[i, k], tw, u))
        log(f"[matched] {tag}: {len(twins)} twins from {len(cand)} candidate states")
        with Pool() as pl:
            fom = pl.map(_matched_job, jobs)
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)) / xref)
        for q, ((i, k, tw, md, j0, j1, u), xs) in enumerate(zip(twins, fom)):
            meta.append(dict(setting=tag, pair=q, traj=i, k=k, mem_diff=md, jump_orig=j0, jump_twin=j1, cont=q % 3,
                             d_fom=rms(xs[0] - xs[1]), d_fom_max=float(np.abs(xs[0] - xs[1]).max() / xref)))
        r = rdim(out)
        models = [("factorial", "POD-neural", "POD-N"), ("closure", "obs+reproject (V3)", "CR-ROM"),
                  ("closure", "complete latent-only", "MC-ROM"), ("closure", "complete+reproject", "MC-ROM-R")]
        seeds = sorted({int(f.split("_s")[-1][:-3]) for f in os.listdir(os.path.join(out, "ckpt"))})
        for stage, name, label in models:
            for seed in seeds:
                if not os.path.exists(ckpt_path(out, stage, name, r, seed)):
                    continue
                rom, kind = load_rom(ctx, stage, name, r, seed)
                sc = ctx.sc[kind]
                for q, ((i, k, tw, md, j0, j1, u), xs) in enumerate(zip(twins, fom)):
                    xh = []
                    for st in (S[i, k], tw):
                        src = st[np.r_[0:3 * N, -2, -1]] if kind == "obs" else st
                        x0 = torch.as_tensor(sc.fwd(src), dtype=torch.float32, device=ctx.dev)[None]
                        c = np.column_stack([u[1:], np.repeat(pool["P"][i][None], len(u) - 1, 0)])
                        C = torch.as_tensor((c - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None]
                        with torch.no_grad():
                            Xh = rom.rollout(x0, C)[0].cpu().numpy()
                        xh.append(np.concatenate([[st[-2]], sc.obs(Xh.astype(np.float64))[:, -2]]))
                    dF, dR = xs[0] - xs[1], xh[0] - xh[1]
                    rows.append(dict(setting=tag, model=label, seed=seed, pair=q, cont=q % 3, d_fom=rms(dF),
                                     d_rom=rms(dR), sep=rms(dR - dF),
                                     err=0.5 * (rms(xh[0] - xs[0]) + rms(xh[1] - xs[1]))))
                log(f"[matched] {tag} {label} seed {seed} done")
    pd.DataFrame(meta).to_csv(os.path.join(a.out, "rev2_matched_pairs.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "rev2_matched.csv"), index=False)
    m = pd.DataFrame(meta)
    log("[matched] simulator divergence of twins (fraction of stroke):\n" +
        m.groupby("setting").d_fom.describe(percentiles=[0.5, 0.95]).round(5).to_string())
    log("[matched] ROMs:\n" + df.groupby(["setting", "model"])[["d_fom", "d_rom", "sep", "err"]].mean().round(4).to_string())


# ------------------------------------------------------------------ minor
def nested_profile(dt=0.05):
    """Deterministic current history with nested partial cycles."""
    kt = [0, 1, 5, 11, 26, 29, 33, 36, 51, 54, 58, 61, 81, 90]
    ki = [0, 0, 1.5, 1.5, 0.45, 0.45, 1.2, 1.2, 0.6, 0.6, 1.35, 1.35, 0.0, 0.0]
    return np.interp(np.arange(0, kt[-1], dt), kt, ki)


def _minor_job(args):
    p, I = args
    act = Actuator(p)
    Y, S = act.run(I)
    N = p.N
    return Y[:, -2], Y[:, :N].max(), Y[:, N:2 * N].mean(1), act.fails


def stage_minor(a, log):
    import experimental as xp
    base = os.path.dirname(a.exp.rstrip("/")) or "."
    sets = [("current input (retained)", a.exp), ("measured power input", os.path.join(base, "experiment_power")),
            ("fixed C", os.path.join(base, "experiment_fixC"))]
    sets = [(l, d) for l, d in sets if os.path.exists(os.path.join(d, "experiment.json"))]
    N, dt = (16 if a.quick else 32), 0.05
    params = []
    for label, d in sets:
        e = json.load(open(os.path.join(d, "experiment.json")))
        params.append((label, straight_params(e, N=N, dt=dt), e))
    rng = np.random.default_rng([31, 2])          # the current histories of the calrom test set
    nr = 3 if a.quick else 20
    hist = {f"random {j}": current_profile(rng, 30.0 if a.quick else 80.0, dt) for j in range(nr)}
    hist["nested"] = nested_profile(dt)
    df = xp.load(a.data_exp)
    w = xp.window(df, params[0][2]["cycles"], dt)
    hist["measured window"] = w["I"]
    # the power-calibrated set is driven by P = R_eff I^2, with R_eff fitted to the measured power
    R_eff = float(np.sum(w["P"] * w["I"] ** 2) / np.sum(w["I"] ** 4))
    log(f"[minor] effective resistance for the power-input set: {R_eff:.3f} Ohm "
        f"(power NRMSE of the fit on the window {np.sqrt(np.mean((R_eff * w['I'] ** 2 - w['P']) ** 2)) / np.ptp(w['P']):.3f})")
    jobs = [(p, (w["P"] if h == "measured window" else R_eff * I ** 2) if p.heating == "power" else I)
            for _, p, _ in params for h, I in hist.items()]
    with Pool() as pl:
        res = pl.map(_minor_job, jobs)
    R, k = {}, 0
    for label, _, _ in params:
        for h in hist:
            R[(label, h)] = res[k]
            k += 1
    rows = []
    ref = params[0][0]
    for label, _, _ in params:
        for h in hist:
            x, Tm, xim, f = R[(label, h)]
            x0, _, xi0, _ = R[(ref, h)]
            rows.append(dict(set=label, history=h, kind=h.split()[0] if h.startswith("random") else h,
                             x_nrmse_vs_ref=float(np.sqrt(np.mean((x - x0) ** 2)) / max(np.ptp(x0), 1e-12)),
                             xi_rms_vs_ref=float(np.sqrt(np.mean((xim - xi0) ** 2))), stroke_mm=float(np.ptp(x) * 1e3),
                             peak_T=float(Tm), fails=int(f)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "rev2_minor.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, "rev2_minor_traces.npz"),
                        **{f"{l}|{h}": R[(l, h)][0] for l, _, _ in params for h in ("nested", "random 0", "measured window")},
                        **{f"I|{h}": hist[h] for h in ("nested", "random 0", "measured window")})
    log("[minor]\n" + out.groupby(["set", "kind"])[["x_nrmse_vs_ref", "xi_rms_vs_ref", "peak_T"]]
        .agg(["mean", "max"]).round(4).to_string())


# ------------------------------------------------------------------ eqproj
def eq_project(act, Yh):
    """Replace x and sigma of predicted observations by the exact solution of the spring
    equilibrium and the compatibility condition for the predicted T, xi and et fields."""
    p, N = act.p, act.p.N
    Yp, fails = Yh.copy(), 0
    for k in range(1, len(Yh)):
        T, xi, et = Yh[k, :N], np.clip(Yh[k, N:2 * N], 0, 1), Yh[k, 2 * N:3 * N]

        def res(x):
            sig = act.sig_of_x(x)
            strain = np.mean(act.S(xi) * sig + act.alpha(xi) * (T - p.T0) + et - act.eref)
            geo = -x / act.L0 if act.straight else (np.hypot(p.h, p.a - x) - act.L0) / act.L0
            return geo - strain
        if not act.straight and res(0.0) <= 0.0:
            x = 0.0
        elif res(act.xmin) > 0.0 > res(act.xmax):
            x = brentq(res, act.xmin, act.xmax, xtol=1e-14, rtol=1e-12)
        else:
            fails += 1
            continue
        Yp[k, -2], Yp[k, -1] = x, act.sig_of_x(x)
    return Yp, fails


def stage_eqproj(a, log):
    import torch
    import experiments as ex
    from metrics import traj_metrics, aggregate
    from rom import predict, tt
    rows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        sc, d, r = ctx.sc["obs"], ctx.data["test"], rdim(out)
        for seed in ckpt_seeds(out, "factorial_POD-neural"):
            rom, _ = ex.load_rom(ctx, "factorial", "POD-neural", r, seed)
            rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev))
            t = ctx.t["obs"]["test"]
            with torch.no_grad():
                Yh = sc.obs(predict(rom, t["X"], t["C"]).cpu().numpy().astype(np.float64))
            m0, m1, nf = [], [], 0
            for i in range(len(Yh)):
                act = Actuator(replace(ctx.p, n=float(d["P"][i, 0]), K=float(d["P"][i, 1])))
                Yp, f = eq_project(act, Yh[i])
                nf += f
                ref = (np.ptp(d["Y"][i, :, -2]), np.ptp(d["Y"][i, :, -1]))
                m0.append(traj_metrics(Yh[i], d["Y"][i], d["U"][i], d["P"][i], ctx.k, ref))
                m1.append(traj_metrics(Yp, d["Y"][i], d["U"][i], d["P"][i], ctx.k, ref))
            for variant, m in (("POD-N+clip", m0), ("POD-N+clip+eq", m1)):
                rows.append(dict(setting=tag, model=variant, seed=seed, r=r, failed_solves=nf, **aggregate(m)))
            log(f"[eqproj] {tag} seed {seed}: x NRMSE {rows[-2]['x_nrmse']:.4f} -> {rows[-1]['x_nrmse']:.4f}, "
                f"R_eq {rows[-2]['eq_rms_N']:.3f} -> {rows[-1]['eq_rms_N']:.2e} N, failed solves {nf}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "rev2_eqproj.csv"), index=False)


# ------------------------------------------------------------------ boot
BOOT_MODELS = [("factorial", "POD-linear", "POD-L", False), ("factorial", "POD-neural", "POD-N", False),
               ("factorial", "POD-neural", "POD-N+clip", True), ("factorial", "AE-neural", "AE-N", False),
               ("ablation", "bounded+penalties", "AE-N-BP", False), ("closure", "obs+reproject (V3)", "CR-ROM", False),
               ("closure", "complete latent-only", "MC-ROM", False)]
BOOT_DIFFS = [("POD-N", "AE-N"), ("POD-N+clip", "CR-ROM"), ("POD-L", "POD-N"), ("POD-L-U", "POD-N"),
              ("POD-BL-U", "POD-N"), ("POD-QB-U", "POD-N"), ("POD-L-U", "POD-L-1"), ("POD-N-W", "POD-N"),
              ("AE-N-R", "AE-N"), ("AE-N-B-R", "AE-N-B"), ("CR-ROM", "AE-N-BP"), ("MC-ROM", "CR-ROM")]


def stage_boot(a, log):
    import experiments as ex
    from rom import tt
    trows = []
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        r, sc = rdim(out), ctx.sc["obs"]
        for stage, name, label, clip in BOOT_MODELS:
            for seed in ckpt_seeds(out, f"{stage}_{name.replace(' ', '_')}"):
                rom, kind = ex.load_rom(ctx, stage, name, r, seed)
                if clip:
                    rom.clip = (tt(sc.clip_lo, ctx.dev), tt(sc.clip_hi, ctx.dev))
                trows += traj_rows(ctx, rom, kind, tag, label, seed)
        # AE-N-B (bounded, no penalties) for the re-projection comparison
        for seed in ckpt_seeds(out, "ablation_bounded"):
            rom, kind = ex.load_rom(ctx, "ablation", "bounded", r, seed)
            trows += traj_rows(ctx, rom, kind, tag, "AE-N-B", seed)
    T = pd.DataFrame(trows)
    f2 = os.path.join(a.out, "rev2_traj.csv")
    if os.path.exists(f2):
        T = pd.concat([T, pd.read_csv(f2)], ignore_index=True)
    rng = np.random.default_rng(0)
    rows = []
    for tag in T.setting.unique():
        V = {m: q.groupby("traj").x_nrmse.mean().sort_index().values for m, q in T[T.setting == tag].groupby("model")}
        n = len(next(iter(V.values())))
        B = rng.integers(0, n, (2000, n))
        for m, v in V.items():
            bs = v[B].mean(1)
            rows.append(dict(setting=tag, comparison=m, kind="mean", value=float(v.mean()),
                             lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n_traj=n))
        for m1, m2 in BOOT_DIFFS:
            if m1 in V and m2 in V and len(V[m1]) == len(V[m2]):
                dv = V[m1] - V[m2]
                bs = dv[B].mean(1)
                rat = V[m1][B].mean(1) / V[m2][B].mean(1)
                rows.append(dict(setting=tag, comparison=f"{m1} - {m2}", kind="difference", value=float(dv.mean()),
                                 lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                 ratio=float(V[m1].mean() / V[m2].mean()), ratio_lo=float(np.percentile(rat, 2.5)),
                                 ratio_hi=float(np.percentile(rat, 97.5)), frac_traj_m1_better=float(np.mean(dv < 0)),
                                 n_traj=n))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "rev2_boot.csv"), index=False)
    log("[boot]\n" + out.round(4).to_string(index=False))


# ------------------------------------------------------------------ verify
def material_point(p, sig, Ts):
    """Single material point at constant stress following the temperature path Ts."""
    act = Actuator(replace(p, N=1, latent_heat=False))
    xs = []
    for T in Ts:
        trl = act.trial(np.array([T]), sig)
        act.commit(trl)
        xs.append(float(trl["xi"][0]))
    return np.array(xs)


def stage_verify(a, log):
    from metrics import traj_metrics
    res = {}
    p = PRESETS["paper"]
    step = 0.01 if not a.quick else 0.05
    lines = []
    for smpa in (0.0, 50.0, 100.0, 150.0, 200.0):
        sig = smpa * 1e6
        up = np.arange(240.0, 430.0, step)
        dn = np.arange(430.0, 220.0, -step)
        xu = material_point(p, sig, np.concatenate([up, dn]))
        xh, xc = xu[:len(up)], xu[len(up):]
        shift = max(sig / p.CA, sig / p.CM + (p.Ms + 1.0 - p.As))
        th = dict(As=p.As + shift, Af=p.Af + shift, Ms=p.Ms + sig / p.CM, Mf=p.Mf + sig / p.CM)
        sim = dict(As=up[np.argmax(xh < 1 - 1e-9)], Af=up[np.argmax(xh <= 1e-9)],
                   Ms=dn[np.argmax(xc > 1e-9)], Mf=dn[np.argmax(xc >= 1 - 1e-9)])
        lines.append(dict(sigma_MPa=smpa, **{f"{k}_sim": float(v) for k, v in sim.items()},
                          **{f"{k}_th": float(v) for k, v in th.items()},
                          max_dev_K=float(max(abs(sim[k] - th[k]) for k in th)),
                          xi_min=float(xh.min()), xi_end=float(xc[-1])))
    res["phase_lines"] = lines
    res["phase_line_max_dev_K"] = max(l["max_dev_K"] for l in lines)
    res["major_cycle_xi_end_error"] = max(abs(1 - l["xi_end"]) for l in lines)
    res["major_cycle_xi_min"] = max(l["xi_min"] for l in lines)
    # return-point memory: first-order partial loops close on the major branch
    sig = 100e6
    shift = max(sig / p.CA, sig / p.CM + (p.Ms + 1.0 - p.As))
    As_, Af_, Ms_, Mf_ = p.As + shift, p.Af + shift, p.Ms + sig / p.CM, p.Mf + sig / p.CM
    T1, T2 = As_ + 0.5 * (Af_ - As_), Ms_ - 0.3 * (Ms_ - Mf_)
    path = np.concatenate([np.arange(240, T1, step), np.arange(T1, T2, -step), np.arange(T2, Af_ + 10, step)])
    x = material_point(p, sig, path)
    gr = np.clip((Af_ - path) / (Af_ - As_), 0, 1)
    last = len(path) - len(np.arange(T2, Af_ + 10, step))
    mask = np.zeros(len(path), bool)
    mask[last:] = path[last:] > T1 + 2 * step
    res["return_point_closure_error_reverse"] = float(np.abs(x[mask] - gr[mask]).max())
    # forward-side loop from austenite: cool partially, reheat partially, recool past the first reversal
    T3, T4 = Ms_ - 0.5 * (Ms_ - Mf_), As_ + 0.3 * (Af_ - As_)
    path = np.concatenate([np.arange(240, Af_ + 10, step), np.arange(Af_ + 10, T3, -step), np.arange(T3, T4, step),
                           np.arange(T4, Mf_ - 10, -step)])
    x = material_point(p, sig, path)
    gf = np.clip((Ms_ - path) / (Ms_ - Mf_), 0, 1)
    last = len(path) - len(np.arange(T4, Mf_ - 10, -step))
    mask = np.zeros(len(path), bool)
    mask[last:] = path[last:] < T3 - 2 * step
    res["return_point_closure_error_forward"] = float(np.abs(x[mask] - gf[mask]).max())
    # residuals of the simulator on the test trajectories, restart and mesh audit
    for tag, out, preset in settings(a):
        ctx = make_ctx(out, preset, a.quick, device())
        d = ctx.data["test"]
        eq, cp = [], []
        slack = 0
        for i in range(len(d["Y"])):
            taut = d["Y"][i][:, -2] > 1e-9               # compatibility holds only for a taut wire
            slack += int((~taut).sum())
            Yt = d["Y"][i][taut]
            m = traj_metrics(Yt, Yt, d["U"][i][taut], d["P"][i], ctx.k)
            eq.append(m["eq_rms_N"])
            cp.append(m["cp_rms"])
        res[f"{tag}_slack_samples_test"] = slack
        res[f"{tag}_fom_eq_rms_N_max"] = float(max(eq))
        res[f"{tag}_fom_cp_rms_max"] = float(max(cp))
        res[f"{tag}_failed_root_solves_test"] = int(d["fails"].sum())
        fa = os.path.join(out, "audit.json")
        if os.path.exists(fa):
            au = json.load(open(fa))
            res[f"{tag}_audit"] = {k: au[k] for k in au if k.startswith(("restart", "mesh", "latent", "counterfactual"))}
    with open(os.path.join(a.out, "rev2_verify.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("[verify] " + json.dumps({k: v for k, v in res.items() if k != "phase_lines"}, indent=1))


# ------------------------------------------------------------------ report2
def stage_report2(a, log):
    import report_rev2
    report_rev2.build(a, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--B", default="runs/paper")
    ap.add_argument("--C", default="runs/revised")
    ap.add_argument("--exp", default="runs/experiment")
    ap.add_argument("--data-exp", default="experimental/ActuatorTest_SMA300um.txt")
    ap.add_argument("--out", default="report_rev2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print training curves")
    ap.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.stages.split(","):
        marker = os.path.join(a.out, f".{s}.done")
        if s != "report2" and os.path.exists(marker) and not a.force:
            print(f"[revision2] {s}: already done, skipped (use --force to rerun)")
            continue
        if s in ("lindyn", "reproj"):           # remove partial output of an interrupted run
            tf = os.path.join(a.out, "rev2_traj.csv")
            if os.path.exists(tf):
                T = pd.read_csv(tf)
                drop = T.model.str.startswith("POD-") if s == "lindyn" else T.model.str.endswith("-R")
                T[~drop].to_csv(tf, index=False)
            rf = os.path.join(a.out, f"rev2_{s}.csv")
            if os.path.exists(rf):
                os.remove(rf)
        t0 = time.time()
        print(f"[revision2] === {s} ===", flush=True)
        globals()["stage_" + s](a, lambda m: print(m, flush=True))
        open(os.path.join(a.out, f".{s}.done"), "w").write(time.ctime())
        print(f"[revision2] {s} finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
