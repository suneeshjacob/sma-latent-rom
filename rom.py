"""Reduced-order models (PyTorch): representations, latent dynamics, training."""
import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class Cfg:
    ae_epochs: int = 120
    ae_bs: int = 256
    rate_epochs: int = 180
    windows: int = 768        # training windows per rate-model epoch
    bs: int = 24
    unroll: int = 24          # unrolled steps per window
    val_every: int = 10
    lr: float = 8e-4
    wd: float = 1e-6
    hid: int = 64             # GRU hidden size
    warm: int = 16            # GRU warm-up history length
    w_ae_phys: float = 0.02
    w_phys: float = 0.05
    w_dir: float = 0.05
    cosine: bool = False      # cosine learning-rate decay over the rate-model epochs


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def tt(a, dev):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)


# ------------------------------------------------------------ representations
class AE(nn.Module):
    """MLP autoencoder. bounded=True puts a sigmoid on every output, which keeps
    each variable inside its training envelope (xi in [0,1], x >= 0, sigma >= 0)."""
    def __init__(self, D, r, bounded):
        super().__init__()
        self.enc = mlp([D, 256, 128, r])
        self.dec = mlp([r, 128, 256, D])
        self.bounded = bounded

    def encode(self, x):
        return self.enc(x)

    def decode(self, q):
        y = self.dec(q)
        return torch.sigmoid(y) if self.bounded else y


class POD(nn.Module):
    def __init__(self, Xn, r, dev):
        super().__init__()
        X = Xn.reshape(-1, Xn.shape[-1]).astype(np.float64)
        mu = X.mean(0)
        V = np.linalg.svd(X - mu, full_matrices=False)[2][:r].T
        self.register_buffer("mu", tt(mu, dev))
        self.register_buffer("V", tt(V, dev))

    def encode(self, x):
        return (x - self.mu) @ self.V

    def decode(self, q):
        return self.mu + q @ self.V.T


# ------------------------------------------------------------------ dynamics
class Dyn(nn.Module):
    """q_{k+1} = q_k + f(q_k, c_{k+1}) in standardised latent coordinates
    (the time step is absorbed in f)."""
    def __init__(self, r):
        super().__init__()
        self.register_buffer("q_mu", torch.zeros(r))
        self.register_buffer("q_sd", torch.ones(r))

    def set_stats(self, q):
        q = q.reshape(-1, q.shape[-1])
        self.q_mu = q.mean(0)
        self.q_sd = q.std(0) + 1e-8

    def step(self, q, c, h=None):
        z = (q - self.q_mu) / self.q_sd
        dz, h = self.rate(z, c, h)
        return (z + dz) * self.q_sd + self.q_mu, h


class RateMLP(Dyn):
    def __init__(self, r):
        super().__init__(r)
        self.net = mlp([r + 3, 160, 160, r])
        nn.init.zeros_(self.net[-1].weight)     # start as "no change per step"
        nn.init.zeros_(self.net[-1].bias)

    def rate(self, z, c, h):
        return self.net(torch.cat([z, c], -1)), h


class RateLinear(Dyn):
    """Ridge-regressed linear rate model on [z, u, n, K, 1]."""
    def __init__(self, r):
        super().__init__(r)
        self.register_buffer("W", torch.zeros(r + 4, r))

    def rate(self, z, c, h):
        return torch.cat([z, c, torch.ones_like(z[..., :1])], -1) @ self.W, h

    @torch.no_grad()
    def fit(self, rep, X, C, ridge=1e-6):
        q = rep.encode(X)
        self.set_stats(q)
        z = ((q - self.q_mu) / self.q_sd).cpu().numpy().astype(np.float64)
        c = C.cpu().numpy().astype(np.float64)
        A = np.concatenate([z[:, :-1], c[:, 1:], np.ones_like(z[:, :-1, :1])], -1)
        A = A.reshape(-1, A.shape[-1])
        B = (z[:, 1:] - z[:, :-1]).reshape(-1, z.shape[-1])
        W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ B)
        self.W = tt(W, X.device)


class RateQuad(Dyn):
    """Quadratic rate model on [z, z (x) z (upper triangle), c, 1], ridge regression on
    one-step differences (operator inference on POD coordinates)."""
    def __init__(self, r):
        super().__init__(r)
        iu = np.triu_indices(r)
        self.register_buffer("iu0", torch.as_tensor(iu[0]))
        self.register_buffer("iu1", torch.as_tensor(iu[1]))
        self.register_buffer("W", torch.zeros(r + len(iu[0]) + 4, r))

    def feats(self, z, c):
        q2 = z[..., self.iu0] * z[..., self.iu1]
        return torch.cat([z, q2, c, torch.ones_like(z[..., :1])], -1)

    def rate(self, z, c, h):
        return self.feats(z, c) @ self.W, h

    @torch.no_grad()
    def fit(self, rep, X, C, ridge=1e-6):
        q = rep.encode(X)
        self.set_stats(q)
        z = (q - self.q_mu) / self.q_sd
        A = self.feats(z[:, :-1], C[:, 1:]).cpu().numpy().astype(np.float64)
        A = A.reshape(-1, A.shape[-1])
        B = (z[:, 1:] - z[:, :-1]).cpu().numpy().astype(np.float64).reshape(-1, z.shape[-1])
        self._AtA, self._AtB = A.T @ A, A.T @ B
        self.set_ridge(ridge)

    @torch.no_grad()
    def set_ridge(self, ridge):
        n = self._AtA.shape[0]
        scale = np.trace(self._AtA) / n
        W = np.linalg.solve(self._AtA + ridge * scale * np.eye(n), self._AtB)
        self.W = tt(W, self.q_mu.device)


class RatePoly(Dyn):
    """Trainable polynomial rate model on [z, c, 1], optionally with the state-input
    products z (x) c (bilinear) and the quadratic products z (x) z (upper triangle).
    fit() gives the one-step ridge solution; train_rate() refines it on unrolled windows."""
    def __init__(self, r, bilinear=False, quadratic=False, nc=3):
        super().__init__(r)
        self.bil, self.quad, self.nc = bilinear, quadratic, nc
        iu = np.triu_indices(r)
        self.register_buffer("iu0", torch.as_tensor(iu[0]))
        self.register_buffer("iu1", torch.as_tensor(iu[1]))
        nf = r + nc + 1 + (r * nc if bilinear else 0) + (len(iu[0]) if quadratic else 0)
        self.W = nn.Parameter(torch.zeros(nf, r))

    def feats(self, z, c):
        f = [z, c, torch.ones_like(z[..., :1])]
        if self.bil:
            f.append((z[..., :, None] * c[..., None, :]).flatten(-2))
        if self.quad:
            f.append(z[..., self.iu0] * z[..., self.iu1])
        return torch.cat(f, -1)

    def rate(self, z, c, h):
        return self.feats(z, c) @ self.W, h

    @torch.no_grad()
    def fit(self, rep, X, C, ridge=1e-6):
        q = rep.encode(X)
        self.set_stats(q)
        z = (q - self.q_mu) / self.q_sd
        A = self.feats(z[:, :-1], C[:, 1:]).cpu().numpy().astype(np.float64)
        A = A.reshape(-1, A.shape[-1])
        B = (z[:, 1:] - z[:, :-1]).cpu().numpy().astype(np.float64).reshape(-1, z.shape[-1])
        self._AtA, self._AtB = A.T @ A, A.T @ B
        self.set_ridge(ridge)

    @torch.no_grad()
    def set_ridge(self, ridge):
        n = self._AtA.shape[0]
        W = np.linalg.solve(self._AtA + ridge * np.trace(self._AtA) / n * np.eye(n), self._AtB)
        self.W.copy_(tt(W, self.W.device))


class WPOD(nn.Module):
    """POD in the weighted norm of the training loss: SVD of sqrt(w)-scaled snapshots."""
    def __init__(self, Xn, r, dev, w):
        super().__init__()
        w = np.asarray(w, dtype=np.float64)
        sw = np.sqrt(np.maximum(w, 1e-6 * w.max()))
        X = Xn.reshape(-1, Xn.shape[-1]).astype(np.float64) * sw
        mu = X.mean(0)
        V = np.linalg.svd(X - mu, full_matrices=False)[2][:r].T
        self.register_buffer("mu", tt(mu, dev))
        self.register_buffer("V", tt(V, dev))
        self.register_buffer("sw", tt(sw, dev))

    def encode(self, x):
        return (x * self.sw - self.mu) @ self.V

    def decode(self, q):
        return (self.mu + q @ self.V.T) / self.sw


class RateGRU(Dyn):
    """Recurrent rate model: the hidden state can carry reversal history."""
    def __init__(self, r, hid):
        super().__init__(r)
        self.cell = nn.GRUCell(r + 3, hid)
        self.head = nn.Linear(hid, r)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def rate(self, z, c, h):
        h = self.cell(torch.cat([z, c], -1), h)
        return self.head(h), h

    def warm(self, Q, C):
        """Build the hidden state from a history of latent states Q [B,W,r] and the
        commands C [B,W,3] that followed each of them."""
        h = None
        for j in range(Q.shape[1]):
            z = (Q[:, j] - self.q_mu) / self.q_sd
            h = self.cell(torch.cat([z, C[:, j]], -1), h)
        return h


class ROM(nn.Module):
    def __init__(self, rep, dyn, reproject=False, warm=0):
        super().__init__()
        self.rep, self.dyn = rep, dyn
        self.reproject = reproject      # decode -> re-encode after every step (V3 behaviour)
        self.warm = warm                # history length for recurrent models
        self.clip = None                # (lo, hi) tensors: output clipping ablation

    def rollout(self, x0, C, hist=None):
        """x0 [B,D] normalised start state; C [B,H,3] conditions of the H predicted
        samples; hist = (states [B,W,D], conditions [B,W,3]) for recurrent warm-up.
        Returns predictions [B,H,D]."""
        q = self.rep.encode(x0)
        h = self.dyn.warm(self.rep.encode(hist[0]), hist[1]) if hist is not None else None
        out = []
        for k in range(C.shape[1]):
            q, h = self.dyn.step(q, C[:, k], h)
            if self.reproject:
                q = self.rep.encode(self.rep.decode(q))
            out.append(self.rep.decode(q))
        X = torch.stack(out, 1)
        if self.clip is not None:
            X = torch.maximum(torch.minimum(X, self.clip[1]), self.clip[0])
        return X


@torch.no_grad()
def predict(rom, X, C, k0=0):
    """Roll every trajectory out from sample k0 to the end.
    Returns [ntraj, M-k0, D]; the first sample is the true start state."""
    hist = None
    if rom.warm:
        hi = (k0 - rom.warm + torch.arange(rom.warm, device=X.device)).clamp(min=0)
        hist = (X[:, hi], C[:, hi + 1])
    Xh = rom.rollout(X[:, k0], C[:, k0 + 1:], hist)
    return torch.cat([X[:, k0:k0 + 1], Xh], 1)


# ------------------------------------------------------------------- physics
class Physics:
    """Soft residuals of Eqs. (13)-(14), slack penalty and direction penalty."""
    def __init__(self, sc, k, dev):
        self.lay, self.k = sc.lay, k
        self.lo, self.span = tt(sc.lo, dev), tt(sc.hi - sc.lo, dev)
        self.xcap = float(sc.hi[sc.lay.ix])

    def fields(self, Xn):
        X = self.lo + Xn * self.span
        sl = self.lay.sl
        return X[..., sl["T"]], X[..., sl["xi"]], X[..., sl["et"]], X[..., self.lay.ix], X[..., self.lay.isig]

    def residuals(self, Xn, nK):
        k = self.k
        T, xi, et, x, sig = self.fields(Xn)
        n, K = nK[..., 0], nK[..., 1]
        strain = (((1 - xi) / k["EA"] + xi / k["EM"]) * sig[..., None]
                  + ((1 - xi) * k["aA"] + xi * k["aM"]) * (T - k["T0"]) + et - k["eref"]).mean(-1)
        if k.get("straight"):                    # wire in series with a preloaded spring
            req = k["F0"] + K * x - k["A"] * sig
            rcp = -x / k["L0"] - strain
            return req, rcp
        L = torch.sqrt(k["h"] ** 2 + (k["a"] - x) ** 2)
        req = K * x - 2 * n * k["A"] * sig * (k["a"] - x) / L
        rcp = (L - k["L0"]) / k["L0"] - strain
        return req, rcp

    def loss(self, Xh, Xt, nK):
        """nK holds physical [n, K], broadcastable to the batch shape of Xh."""
        req, rcp = self.residuals(Xh, nK)
        _, _, _, xt, st = self.fields(Xt)
        taut = ((xt > 1e-8) | (st > 1e4)).float()
        scaled = (req / (nK[..., 1] * self.xcap)) ** 2 + (rcp / 0.01) ** 2
        slack = Xh[..., self.lay.ix] ** 2 + Xh[..., self.lay.isig] ** 2
        return (taut * scaled).mean() + ((1 - taut) * slack).mean()

    def direction(self, Xh):
        T, xi = self.fields(Xh)[:2]
        return (torch.relu((T[:, 1:] - T[:, :-1]) / 5.0 * (xi[:, 1:] - xi[:, :-1])) ** 2).mean()


def state_loss(Xh, X, w):
    return (w * (Xh - X) ** 2).sum(-1).mean()


# ------------------------------------------------------------------ training
def train_ae(ae, tr, va, w, cfg, phys=None, log=None):
    """tr / va: dicts with X [ntraj,M,D] and nK [ntraj,2] tensors."""
    D = tr["X"].shape[-1]
    M = tr["X"].shape[1]
    X = tr["X"].reshape(-1, D)
    nK = tr["nK"].repeat_interleave(M, 0)
    Xv = va["X"].reshape(-1, D)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best, best_sd = float("inf"), None
    for ep in range(cfg.ae_epochs):
        perm = torch.randperm(X.shape[0], device=X.device)
        for i in range(0, X.shape[0], cfg.ae_bs):
            idx = perm[i:i + cfg.ae_bs]
            x = X[idx]
            xh = ae.decode(ae.encode(x))
            loss = state_loss(xh, x, w)
            if phys is not None:
                loss = loss + cfg.w_ae_phys * phys.loss(xh, x, nK[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            opt.step()
        if (ep + 1) % cfg.val_every == 0 or ep == cfg.ae_epochs - 1:
            with torch.no_grad():
                v = float(state_loss(ae.decode(ae.encode(Xv)), Xv, w))
            if v < best:
                best, best_sd = v, copy.deepcopy(ae.state_dict())
            if log:
                log(f"    AE epoch {ep + 1}: val {v:.3e}")
    ae.load_state_dict(best_sd)
    for prm in ae.parameters():
        prm.requires_grad_(False)
    return best


def val_score(rom, va, lay):
    """E_x + E_sigma + 2 E_xi + 0.25 E_T on full validation rollouts (normalised units)."""
    Xh = predict(rom, va["X"], va["C"])
    e = (Xh - va["X"]) ** 2
    sl = lay.sl
    return float(e[..., lay.ix].mean() + e[..., lay.isig].mean()
                 + 2 * e[..., sl["xi"]].mean() + 0.25 * e[..., sl["T"]].mean())


def train_rate(rom, tr, va, w, cfg, lay, phys=None, w_dir=0.0, log=None):
    """Train the latent rate model with unrolled windows; representation is frozen."""
    X, C, nK = tr["X"], tr["C"], tr["nK"]
    ntraj, M, dev = X.shape[0], X.shape[1], X.device
    H, W = cfg.unroll, rom.warm
    opt = torch.optim.AdamW(rom.dyn.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.rate_epochs) if cfg.cosine else None
    ar_h = torch.arange(1, H + 1, device=dev)[None]
    ar_w = torch.arange(W, device=dev)[None] if W else None
    best, best_sd = float("inf"), None
    for ep in range(cfg.rate_epochs):
        rom.dyn.train()
        for _ in range(max(1, cfg.windows // cfg.bs)):
            i = torch.randint(0, ntraj, (cfg.bs,), device=dev)
            k0 = torch.randint(0, M - H, (cfg.bs,), device=dev)
            steps = k0[:, None] + ar_h
            Xt = X[i[:, None], steps]
            hist = None
            if W:
                hi = (k0[:, None] - W + ar_w).clamp(min=0)
                hist = (X[i[:, None], hi], C[i[:, None], hi + 1])
            Xh = rom.rollout(X[i, k0], C[i[:, None], steps], hist)
            loss = state_loss(Xh, Xt, w)
            if phys is not None:
                loss = loss + cfg.w_phys * phys.loss(Xh, Xt, nK[i][:, None, :])
                if w_dir:
                    loss = loss + w_dir * phys.direction(Xh)
            if not torch.isfinite(loss):          # diverged window (polynomial models): skip
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rom.dyn.parameters(), 5.0)
            opt.step()
        if sched is not None:
            sched.step()
        if (ep + 1) % cfg.val_every == 0 or ep == cfg.rate_epochs - 1:
            rom.dyn.eval()
            v = val_score(rom, va, lay)
            if np.isfinite(v) and v < best:
                best, best_sd = v, copy.deepcopy(rom.dyn.state_dict())
            if log:
                log(f"    rate epoch {ep + 1}: val {v:.3e}")
    if best_sd is not None:
        rom.dyn.load_state_dict(best_sd)
    rom.dyn.eval()
    return best
