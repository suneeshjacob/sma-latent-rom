"""Command sampling, dataset generation and scaling."""
import os, time
from multiprocessing import Pool
import numpy as np
from fom import Params, Actuator, simulate, FIELDS

M_DEFAULT = 300

# name: (number of trajectories, pair counts, stiffness intervals [N/m], command regime)
SPLITS = {
    "train":    (96, [1, 3, 5, 7], [(3000, 8000)], "id"),
    "val":      (24, [1, 3, 5, 7], [(3000, 8000)], "id"),
    "test":     (30, [2, 4, 6],    [(3000, 8000)], "id"),
    # extrapolation sets
    "ood_K":    (20, [2, 4, 6],    [(2000, 3000), (8000, 10000)], "id"),
    "ood_n":    (20, [8, 9],       [(3000, 8000)], "id"),
    "ood_fast": (20, [2, 4, 6],    [(3000, 8000)], "fast"),
    "ood_rev":  (20, [2, 4, 6],    [(3000, 8000)], "reversal"),
}
OOD = [s for s in SPLITS if s.startswith("ood")]


def command(rng, p: Params, M=M_DEFAULT, regime="id"):
    """Piecewise-linear thermal command. Returns the input sequence u [M]
    (command temperature in K, or current in A for Joule heating)."""
    return to_input(p, command_T(rng, p, M, regime))


def to_input(p: Params, Tcmd):
    """Temperature target -> simulator input (unchanged, or Joule current)."""
    Tcmd = np.asarray(Tcmd, dtype=float)
    if p.heating == "joule":      # current whose steady-state wire temperature is Tcmd
        A = np.pi * p.dw ** 2 / 4
        rhoe = 0.5 * (p.rhoeA + p.rhoeM)
        return np.sqrt(np.clip(Tcmd - p.Tamb, 0, None) * 4 * p.hconv * A ** 2 / (rhoe * p.dw))
    return Tcmd


def command_T(rng, p: Params, M=M_DEFAULT, regime="id"):
    """Piecewise-linear temperature target [K]."""
    U = rng.uniform
    start, end = p.Tinit, U(p.Tamb - 2, p.Tamb + 10)
    if regime == "reversal":      # shallow valleys / low peaks outside the training envelope
        pk1 = U(350, 380); v1 = U(325, 340); pk2 = U(v1 + 8, 352)
        v2 = U(320, 338); pk3 = U(v2 + 8, 350)
        knots = [start, pk1, v1, pk2, v2, pk3, end, end]
    else:
        def cycle(prev_valley=None, lo=350.0, hi=380.0, dmin=0.0, vlo=295.0, vhi=325.0):
            pk = U(lo, hi) if prev_valley is None else U(max(prev_valley + dmin, lo), hi)
            return pk, U(vlo, vhi)
        pk1, v1 = cycle()
        pk2, v2 = cycle(v1, 345, 378, 20, 285, 320)
        pk3, _ = cycle(v2, 340, 372, 15)
        knots = [start, pk1, v1, pk2, v2, pk3, end, end]
        if regime == "fast":      # two extra cycles in the same time span -> faster ramps
            pk4, v4 = cycle(v2, 340, 372, 15, 285, 320)
            pk5, v5 = cycle(v4, 340, 372, 15, 285, 320)
            knots = [start, pk1, v1, pk2, v2, pk4, v4, pk5, v5, pk3, end, end]
    t = np.arange(M) * p.dt
    return np.interp(t, np.linspace(0, t[-1], len(knots)), knots)


def _one(args):
    p, u, n, K = args
    t0 = time.perf_counter()
    Y, S, fails = simulate(p, u, n, K)
    return Y, S, fails, time.perf_counter() - t0


def generate(split, p: Params, seed=17, M=M_DEFAULT, count=None, workers=None):
    ntraj, ns, Kints, regime = SPLITS[split]
    ntraj = count or ntraj
    rng = np.random.default_rng([seed, list(SPLITS).index(split)])
    jobs = []
    for _ in range(ntraj):
        lo, hi = Kints[rng.integers(len(Kints))]
        jobs.append((p, command(rng, p, M, regime), float(rng.choice(ns)), float(rng.uniform(lo, hi))))
    with Pool(workers or os.cpu_count()) as pool:
        res = pool.map(_one, jobs)
    return dict(Y=np.array([r[0] for r in res]), S=np.array([r[1] for r in res]),
                U=np.array([j[1] for j in jobs]), P=np.array([[j[2], j[3]] for j in jobs]),
                fails=np.array([r[2] for r in res]), tfom=np.array([r[3] for r in res]))


def load_or_make(out, p, splits, seed=17, M=M_DEFAULT, counts=None):
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    data = {}
    for s in splits:
        f = os.path.join(out, "data", s + ".npz")
        if not os.path.exists(f):
            d = generate(s, p, seed, M, (counts or {}).get(s))
            np.savez_compressed(f, **d)
            print(f"[data] {s}: {len(d['Y'])} trajectories, failed root solves: {int(d['fails'].sum())}")
        data[s] = dict(np.load(f))
    return data


# ------------------------------------------------------------------ scaling
class Layout:
    """Index bookkeeping for the observed (3N+2) or complete (10N+2) state."""
    def __init__(self, N, kind):
        self.N, self.kind = N, kind
        names = ["T", "xi", "et"] if kind == "obs" else FIELDS
        self.sl = {f: slice(i * N, (i + 1) * N) for i, f in enumerate(names)}
        self.D = len(names) * N + 2
        self.ix, self.isig = self.D - 2, self.D - 1
        # where the 3N+2 observed variables sit inside this state
        self.obs_idx = np.r_[0:3 * N, self.ix, self.isig]


# loss weights of Eq. (15); memory fields (complete state only) get 0.3
W_FIELD = dict(T=0.20, xi=1.25, et=0.30, x=2.50, sig=1.50)
W_MEMORY = 0.30


class Scaler:
    """Maps each variable into [0, 1] using padded training envelopes."""
    def __init__(self, train, N, kind):
        self.lay = lay = Layout(N, kind)
        X = (train["Y"] if kind == "obs" else train["S"]).reshape(-1, lay.D)
        lo, hi, w = np.zeros(lay.D), np.ones(lay.D), np.zeros(lay.D)
        for f, sl in lay.sl.items():
            if f in ("T", "et", "etr"):
                a, b = X[:, sl].min(), X[:, sl].max()
                pad = 0.04 * (b - a)
                lo[sl], hi[sl] = a - pad, b + pad
            elif f in ("dlast", "d0"):
                lo[sl], hi[sl] = -1.0, 1.0
            elif f == "m":
                lo[sl], hi[sl] = 0.0, 2.0
            w[sl] = W_FIELD.get(f, W_MEMORY) / N
        hi[lay.ix] = 1.08 * X[:, lay.ix].max()
        hi[lay.isig] = 1.08 * X[:, lay.isig].max()
        w[lay.ix], w[lay.isig] = W_FIELD["x"], W_FIELD["sig"]
        self.lo, self.hi, self.w = lo, hi, w
        # clip ablation: xi in [0,1], x >= 0, sigma >= 0 (normalised coordinates)
        self.clip_lo, self.clip_hi = np.full(lay.D, -np.inf), np.full(lay.D, np.inf)
        self.clip_lo[lay.sl["xi"]], self.clip_hi[lay.sl["xi"]] = 0.0, 1.0
        self.clip_lo[lay.ix] = self.clip_lo[lay.isig] = 0.0
        # condition standardisation [u, n, K]
        C = cond(train).reshape(-1, 3)
        self.c_mu, self.c_sd = C.mean(0), C.std(0) + 1e-12

    def fwd(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def inv(self, Xn):
        return self.lo + Xn * (self.hi - self.lo)

    def obs(self, Xn):
        """Normalised state -> physical observed vector (3N+2)."""
        return self.inv(Xn)[..., self.lay.obs_idx]

    def tensors(self, d):
        """Normalised states [ntraj, M, D] and standardised conditions [ntraj, M, 3]."""
        X = d["Y"] if self.lay.kind == "obs" else d["S"]
        return (self.fwd(X).astype(np.float32),
                ((cond(d) - self.c_mu) / self.c_sd).astype(np.float32))


def cond(d):
    """Conditions per sample: [u_k, n, K]."""
    M = d["U"].shape[1]
    return np.concatenate([d["U"][..., None], np.repeat(d["P"][:, None, :], M, 1)], -1)


def constants(p: Params):
    """Physical constants needed by residuals and metrics."""
    act = Actuator(p)
    return dict(h=p.h, a=p.a, L0=act.L0, A=act.A, EA=p.EA, EM=p.EM, aA=p.aA, aM=p.aM,
                T0=p.T0, eref=act.eref, N=p.N, dt=p.dt, straight=act.straight, F0=p.F0)
