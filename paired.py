"""Paired-history test for state closure.

Question: when two NATURAL trajectories reach almost the same observed state through
different reversal histories, does the simulator diverge afterwards, and does each
ROM reproduce that divergence?

Procedure
1. A pool of trajectories is simulated for a few fixed (n, K) groups, with ordinary
   and shallow-reversal commands. Complete states S and observations Y are kept.
2. Within each group, pairs of states from different trajectories are matched by
   weighted observation distance (the training-loss weights). Pairs are split into
     'hidden'  : memory differs in >= 20% of cells (stage or reversal point),
     'control' : memory differs in <= 2% of cells.
   The closest pairs of each class are kept.
3. Both states are continued with the SAME future input (heat, cool, or heat-then-cool).
   The simulator is restarted exactly from each complete state; every ROM is started
   from its own encoding of each state (the recurrent model also gets its own history).
4. Per pair (all quantities normalised by the training displacement stroke):
     d_fom  = RMS_t |x_A - x_B|            simulator divergence
     d_rom  = RMS_t |x^_A - x^_B|          ROM divergence
     sep    = RMS_t |(x^_A - x^_B) - (x_A - x_B)|   error in the predicted divergence
     err    = mean of RMS_t |x^ - x| over A and B
   A ROM that resolves the hidden history has sep << d_fom on 'hidden' pairs.
"""
import os
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import pandas as pd

from fom import Actuator
from data import command_T, to_input

GROUPS = [(2, 4000.0), (4, 6000.0), (6, 7500.0)]
N_TRAJ = 40          # pool trajectories per group
STRIDE = 4           # sampled instants
K_MIN, K_MAX = 40, 220
H = 80               # continuation length [steps]
N_PAIRS = 60         # pairs per class
HIDDEN_MIN, CONTROL_MAX = 0.20, 0.02

# models evaluated: (stage, checkpoint name, label)
MODELS = [
    ("factorial", "POD-neural", "POD-N"),
    ("closure", "obs+reproject (V3)", "CR-ROM"),
    ("closure", "complete latent-only", "MC-ROM"),
    ("closure", "complete+reproject", "MC-ROM-R"),
    ("closure", "obs recurrent", "GRU-ROM"),
    ("longtrain", "GRU-ROM (long)", "GRU-ROM (long)"),
    ("longtrain", "CR-ROM (long)", "CR-ROM (long)"),
]


def _sim(args):
    p, u = args
    act = Actuator(p)
    Y, S = act.run(u)
    return Y, S


def make_pool(out, p, seed=29):
    f = os.path.join(out, "data", "paired_pool.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    rng = np.random.default_rng(seed)
    jobs, meta = [], []
    for g, (n, K) in enumerate(GROUPS):
        pg = replace(p, n=float(n), K=float(K))
        for j in range(N_TRAJ):
            T = command_T(rng, p, regime="reversal" if j % 2 else "id")
            jobs.append((pg, to_input(p, T)))
            meta.append((g, n, K, T))
    with Pool() as pool:
        res = pool.map(_sim, jobs)
    d = dict(Y=np.array([r[0] for r in res]), S=np.array([r[1] for r in res]),
             U=np.array([j[1] for j in jobs]), T=np.array([m[3] for m in meta]),
             G=np.array([m[0] for m in meta]), P=np.array([[m[1], m[2]] for m in meta]))
    np.savez_compressed(f, **d)
    return d


def _memdiff(S, a, b, N):
    m, r1 = S[a][6 * N:7 * N], S[a][8 * N:9 * N]
    m2, r12 = S[b][6 * N:7 * N], S[b][8 * N:9 * N]
    return float(np.mean((m != m2) | (np.abs(r1 - r12) > 0.05)))


def find_pairs(pool, sc, N, n_pairs=None):
    """Closest observation pairs in each class. Returns list of (i, k, j, l, class, d_obs)."""
    n_pairs = n_pairs or N_PAIRS
    w = sc.w
    ks = np.arange(K_MIN, K_MAX + 1, STRIDE)
    out = {"hidden": [], "control": []}
    for g in np.unique(pool["G"]):
        tr = np.where(pool["G"] == g)[0]
        idx = np.array([(i, k) for i in tr for k in ks])
        Z = sc.fwd(pool["Y"][idx[:, 0], idx[:, 1]]) * np.sqrt(w)
        Sflat = pool["S"][idx[:, 0], idx[:, 1]]
        sq = (Z ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
        same = idx[:, 0][:, None] == idx[:, 0][None, :]
        D[same | np.tri(len(idx), dtype=bool)] = np.inf        # different trajectories, a<b
        taut = pool["Y"][idx[:, 0], idx[:, 1], -2] > 1e-5
        D[~taut, :] = np.inf
        D[:, ~taut] = np.inf
        order = np.argsort(D, axis=None)
        counts = {"hidden": 0, "control": 0}
        for flat in order:
            a, b = np.unravel_index(flat, D.shape)
            if not np.isfinite(D[a, b]) or min(counts.values()) >= n_pairs:
                break
            md = _memdiff(Sflat, a, b, N)
            cls = "hidden" if md >= HIDDEN_MIN else "control" if md <= CONTROL_MAX else None
            if cls and counts[cls] < n_pairs:
                out[cls].append((idx[a, 0], idx[a, 1], idx[b, 0], idx[b, 1], cls, float(D[a, b]), md))
                counts[cls] += 1
    # keep the N_PAIRS closest per class over all groups
    res = []
    for cls in out:
        res += sorted(out[cls], key=lambda t: t[5])[:n_pairs]
    return res


def continuation(T0, kind):
    """Common future temperature target, length H+1, starting at T0."""
    t = np.arange(H + 1)
    if kind == 0:   # heat
        return np.interp(t, [0, 40, H], [T0, 372.0, 372.0])
    if kind == 1:   # cool
        return np.interp(t, [0, 40, H], [T0, 290.0, 290.0])
    return np.interp(t, [0, 30, 55, H], [T0, 340.0, 305.0, 305.0])  # heat then cool


def run(ctx, log=print, n_pairs=None, tag="", models=None, r=None):
    import torch
    from experiments import load_rom, ckpt_path
    p, N, out = ctx.p, ctx.p.N, ctx.out
    pool = make_pool(out, p)
    pairs = find_pairs(pool, ctx.sc["obs"], N, n_pairs)
    log(f"[paired] {sum(c[4] == 'hidden' for c in pairs)} hidden-memory pairs, "
        f"{sum(c[4] == 'control' for c in pairs)} control pairs")
    xref = ctx.sc["obs"].hi[ctx.sc["obs"].lay.ix] / 1.08     # training displacement stroke

    # simulator continuations
    fom = []
    for q, (i, k, j, l, cls, dobs, md) in enumerate(pairs):
        T0 = 0.5 * (pool["T"][i, k] + pool["T"][j, l])
        u = to_input(p, continuation(T0, q % 3))
        xs = []
        for (tj, tk) in ((i, k), (j, l)):
            n, K = pool["P"][tj]
            act = Actuator(replace(p, n=float(n), K=float(K)))
            act.set_full(pool["S"][tj, tk])
            xs.append(act.run(u)[0][:, -2])
        fom.append((u, xs))

    rows = []
    seeds = sorted({int(f.split("_s")[-1][:-3]) for f in os.listdir(os.path.join(out, "ckpt"))})
    if r is None:
        r = 16 if any("_r16_" in f for f in os.listdir(os.path.join(out, "ckpt"))) else \
            max(int(f.split("_r")[-1].split("_")[0]) for f in os.listdir(os.path.join(out, "ckpt")))
    for stage, name, label in (models or MODELS):
        for seed in seeds:
            if not os.path.exists(ckpt_path(out, stage, name, r, seed)):
                continue
            rom, kind = load_rom(ctx, stage, name, r, seed)
            sc = ctx.sc[kind]
            for q, (i, k, j, l, cls, dobs, md) in enumerate(pairs):
                u, xs = fom[q]
                xh = []
                for side, (tj, tk) in enumerate(((i, k), (j, l))):
                    src = pool["Y"] if kind == "obs" else pool["S"]
                    x0 = torch.as_tensor(sc.fwd(src[tj, tk]), dtype=torch.float32, device=ctx.dev)[None]
                    c = np.column_stack([u[1:], np.repeat(pool["P"][tj][None], H, 0)])
                    C = torch.as_tensor((c - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None]
                    hist = None
                    if rom.warm:
                        hi = np.clip(tk - rom.warm + np.arange(rom.warm), 0, None)
                        hc = np.column_stack([pool["U"][tj][hi + 1], np.repeat(pool["P"][tj][None], rom.warm, 0)])
                        hist = (torch.as_tensor(sc.fwd(src[tj, hi]), dtype=torch.float32, device=ctx.dev)[None],
                                torch.as_tensor((hc - sc.c_mu) / sc.c_sd, dtype=torch.float32, device=ctx.dev)[None])
                    with torch.no_grad():
                        Xh = rom.rollout(x0, C, hist)[0].cpu().numpy()
                    xh.append(np.concatenate([[src[tj, tk][-2 if kind == "obs" else sc.lay.ix]],
                                              sc.obs(Xh.astype(np.float64))[:, -2]]))
                dF, dR = xs[0] - xs[1], xh[0] - xh[1]
                rms = lambda v: float(np.sqrt(np.mean(v ** 2)) / xref)
                rows.append(dict(model=label, seed=seed, pair=q, cls=cls, d_obs=dobs, mem_diff=md,
                                 cont=q % 3, d_fom=rms(dF), d_rom=rms(dR), sep=rms(dR - dF),
                                 err=0.5 * (rms(xh[0] - xs[0]) + rms(xh[1] - xs[1]))))
            log(f"[paired] {label} seed {seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"paired{tag}.csv"), index=False)
    s = (df.groupby(["model", "cls", "seed"])[["d_obs", "d_fom", "d_rom", "sep", "err"]].mean()
           .groupby(["model", "cls"]).agg(["mean", "std"]))
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    s.to_csv(os.path.join(out, f"paired{tag}_summary.csv"), index=False)
    log("[paired]\n" + s[["model", "cls", "d_obs_mean", "d_fom_mean", "d_rom_mean", "sep_mean", "err_mean"]]
        .round(4).to_string(index=False))
    return s
