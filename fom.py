"""Full-order model: thermally driven SMA wire / bias-spring actuator (1-D).

This is the simulator used for all results of the accompanying paper.
The constitutive update is stated step by step in Appendix A of the paper.

Constitutive model
    Phase-diagram (Lagoudas-type, linear hardening) transformation surfaces with
    stress-shifted temperatures, plus first-order reversal-point memory in the
    spirit of Karakalas et al. (2019): minor branches are rescaled so that a
    return branch closes at the point where the major branch was left.
    Per-cell state: T, xi, et, xi_r, et_r, dlast, m, d0, r1, r2  (10 slots).

Actuator
    Symmetric inclined wire pairs + linear spring, quasi-static, tension only.

Thermal
    'relaxation' : paper's heater-environment command (insulated ends).
    'joule'      : electrical current input, convection, clamp heat sinks.
    Latent heat is optional and is solved semi-implicitly per cell.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq

FIELDS = ["T", "xi", "et", "xir", "etr", "dlast", "m", "d0", "r1", "r2"]
OBS_FIELDS = ["T", "xi", "et"]
TOL = 1e-12
EPS = 1e-9


@dataclass(frozen=True)
class Params:
    # material, transformation and thermal data: Table 1 of Karakalas et al. (2019),
    # Smart Mater. Struct. 28, 035014 (Table 1 of the paper)
    EA: float = 66.2e9
    EM: float = 25.6e9
    aA: float = -3.02e-5
    aM: float = -2.61e-5
    Ms: float = 311.0
    Mf: float = 283.0
    As: float = 313.0
    Af: float = 336.0
    CM: float = 8.34e6          # Pa/K
    CA: float = 9.02e6
    Hmax: float = 0.048
    Hmin: float = 0.0           # assumed for the benchmark
    kt: float = 3.0e-8          # 1/Pa, assumed
    sig_crit: float = 0.0       # assumed
    beta: float = 82.05e6
    rho: float = 6500.0
    cA: float = 470.0
    cM: float = 470.0
    kA: float = 18.0
    kM: float = 18.0
    T0: float = 293.15
    # actuator
    h: float = 0.050
    a: float = 0.080
    dw: float = 0.25e-3
    K: float = 5000.0
    n: float = 4.0
    # thermal
    heating: str = "relaxation"  # 'relaxation' | 'joule' | 'power'
    gamma_h: float = 0.090
    gamma_c: float = 0.002
    Tamb: float = 238.0
    Tinit: float = 238.0
    end_sink: bool = False       # clamps held at Tamb (used with Joule heating)
    hconv: float = 70.0          # W/m^2/K, assumed
    rhoeA: float = 8.2e-7        # ohm m, assumed
    rhoeM: float = 7.6e-7
    latent_heat: bool = False
    # discretisation
    N: int = 64
    # actuator geometry: 'inclined' (wire pairs, manuscript) or 'straight'
    # (one wire in series with a preloaded spring, as in the Gorges et al. test)
    geometry: str = "inclined"
    L0w: float = 0.107           # straight wire length at the initial state [m]
    F0: float = 0.0              # spring preload at x = 0 [N] (straight geometry)
    dt: float = 0.5


PRESETS = {
    # setting B of the paper: heater relaxation, no latent heat
    "paper": Params(),
    # setting C of the paper: Joule heating, latent heat, end heat sinks
    "revised": Params(heating="joule", latent_heat=True, end_sink=True,
                      Tamb=273.15, Tinit=273.15),
}


class Actuator:
    def __init__(self, p: Params):
        self.p = p
        N = p.N
        self.straight = p.geometry == "straight"
        self.L0 = p.L0w if self.straight else float(np.hypot(p.h, p.a))
        self.A = np.pi * p.dw ** 2 / 4.0
        self.ds = self.L0 / N
        s = (np.arange(N) + 0.5) * self.ds
        self.g = 0.88 + 0.12 * np.cos(2 * np.pi * (s / self.L0 - 0.18))
        self.rho_ds0 = -0.5 * (p.CA + p.CM) * p.Hmax  # J/m^3/K
        if self.straight:
            # initial state: detwinned martensite under the spring preload
            self.sig0 = p.F0 / self.A
            self.eps0 = float(self.Hcur(self.sig0))
            self.eref = self.sig0 / p.EM + p.aM * (p.Tinit - p.T0) + self.eps0
            self.xmin, self.xmax = -0.03 * self.L0, 0.08 * self.L0
        else:
            self.sig0 = 0.0
            self.eps0 = float(self.Hcur(0.0))        # prestrain = two-way strain at zero stress
            self.eref = p.aM * (p.Tinit - p.T0) + self.eps0
            self.xmin, self.xmax = 0.0, min(0.6 * p.a, p.a - 1e-6)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        p, N = self.p, self.p.N
        one, zero = np.ones(N), np.zeros(N)
        self.c = dict(T=np.full(N, p.Tinit), xi=one.copy(), et=np.full(N, self.eps0),
                      xir=one.copy(), etr=np.full(N, self.eps0), dlast=one.copy(),
                      m=zero.copy(), d0=zero.copy(), r1=zero.copy(), r2=zero.copy())
        self.x, self.sig, self.fails = 0.0, self.sig0, 0

    def obs(self):
        """Recorded observation y in R^(3N+2)."""
        return np.concatenate([self.c[f] for f in OBS_FIELDS] + [[self.x, self.sig]])

    def full(self):
        """Restart-complete state s in R^(10N+2)."""
        return np.concatenate([self.c[f] for f in FIELDS] + [[self.x, self.sig]])

    def set_full(self, s):
        N = self.p.N
        for i, f in enumerate(FIELDS):
            self.c[f] = np.array(s[i * N:(i + 1) * N], dtype=float)
        self.x, self.sig = float(s[-2]), float(s[-1])

    # ------------------------------------------------------------ material
    def Hcur(self, sig):
        p = self.p
        se = np.abs(sig + p.beta)
        return np.where(se > p.sig_crit,
                        p.Hmin + (p.Hmax - p.Hmin) * (1 - np.exp(-p.kt * (se - p.sig_crit))),
                        p.Hmin)

    def S(self, xi):
        return (1 - xi) / self.p.EA + xi / self.p.EM

    def alpha(self, xi):
        return (1 - xi) * self.p.aA + xi * self.p.aM

    def _branches(self, T, sig):
        """Forward / reverse candidate martensite fractions for the current memory."""
        p, c = self.p, self.c
        xi, m, d0, r1, r2, dl = c["xi"], c["m"], c["d0"], c["r1"], c["r2"], c["dlast"]
        gf = np.clip((p.Ms + sig / p.CM - T) / (p.Ms - p.Mf), 0.0, 1.0)
        # guard keeps A_s(sigma) >= M_s(sigma) + 1 K (the two lines cross above ~220 MPa otherwise)
        shift_r = max(sig / p.CA, sig / p.CM + (p.Ms + 1.0 - p.As))
        gr = np.clip((p.Af + shift_r - T) / (p.Af - p.As), 0.0, 1.0)
        r1s = np.clip(r1, EPS, 1 - EPS)
        # stage 0 (major branches; a reversal starts a rescaled branch)
        F = np.where(dl > 0, gf, xi + (1 - xi) * gf)
        R = np.where(dl > 0, xi * gr, gr)
        # stage 1 / 2, loop opened from the major FORWARD branch at r1
        a = (m == 1) & (d0 > 0)
        F = np.where(a, np.where(gf >= r1, gf, xi + (r1 - xi) * gf / r1s), F)
        R = np.where(a, r1 * gr, R)
        a = (m == 2) & (d0 > 0)
        F = np.where(a, np.where(gf >= r1, gf, r2 + (r1 - r2) * gf / r1s), F)
        R = np.where(a, xi * gr, R)                      # 3rd reversal: first-order collapse
        # stage 1 / 2, loop opened from the major REVERSE branch at r1
        a = (m == 1) & (d0 < 0)
        F = np.where(a, r1 + (1 - r1) * gf, F)
        R = np.where(a, np.where(gr <= r1, gr, r1 + (xi - r1) * (gr - r1) / (1 - r1s)), R)
        a = (m == 2) & (d0 < 0)
        R = np.where(a, np.where(gr <= r1, gr, r1 + (r2 - r1) * (gr - r1) / (1 - r1s)), R)
        F = np.where(a, xi + (1 - xi) * gf, F)
        return F, R, gf, gr

    def _xi(self, T, sig):
        F, R, gf, gr = self._branches(T, sig)
        xi = self.c["xi"]
        fwd = F > xi + TOL
        rev = (R < xi - TOL) & ~fwd
        return np.where(fwd, F, np.where(rev, R, xi)), fwd, rev, gf, gr

    def trial(self, T, sig):
        """Constitutive trial update (does not modify the state).

        With latent heat the cell temperature is corrected semi-implicitly:
        T_eff = T + lam * (xi_new - xi_old), solved with one Newton-like step.
        """
        p, c = self.p, self.c
        xi_old = c["xi"]
        Te = T
        if p.latent_heat:
            rc = p.rho * ((1 - xi_old) * p.cA + xi_old * p.cM)
            lam = -T * self.rho_ds0 / rc                       # K per unit xi (>0)
            x1 = self._xi(T, sig)[0]
            x2 = self._xi(T + 0.01, sig)[0]
            slope = np.minimum((x2 - x1) / 0.01, 0.0)
            Te = T + lam * (x1 - xi_old) / (1.0 - slope * lam)
        xin, fwd, rev, gf, gr = self._xi(Te, sig)
        if p.latent_heat:                                      # energy-consistent temperature
            Te = T + lam * (xin - xi_old)
        sw = rev & (c["dlast"] > 0)                            # new reverse anchor
        xir = np.where(sw, xi_old, c["xir"])
        etr = np.where(sw, c["et"], c["etr"])
        etn = np.where(fwd, c["et"] + self.Hcur(sig) * (xin - xi_old),
                       np.where(rev, etr * xin / np.maximum(xir, 1e-12), c["et"]))
        return dict(T=Te, xi=xin, et=etn, fwd=fwd, rev=rev, gf=gf, gr=gr, xir=xir, etr=etr)

    def commit(self, tr):
        """Accept a trial update and advance the reversal memory."""
        c = self.c
        xi, m0, d00, r10, dl0 = c["xi"], c["m"], c["d0"], c["r1"], c["dlast"]
        fwd, rev, gf, gr = tr["fwd"], tr["rev"], tr["gf"], tr["gr"]
        m, d0, r1, r2 = m0.copy(), d00.copy(), r10.copy(), c["r2"].copy()

        def put(mask, mv=None, dv=None, r1v=None, r2v=None):
            if mv is not None: m[mask] = mv
            if dv is not None: d0[mask] = dv
            if r1v is not None: r1[mask] = r1v[mask]
            if r2v is not None: r2[mask] = r2v[mask]

        put((m0 == 0) & (dl0 > 0) & rev & (xi < 1 - EPS), 1, 1, xi)      # 1st reversal
        put((m0 == 0) & (dl0 < 0) & fwd & (xi > EPS), 1, -1, xi)
        put((m0 == 1) & (d00 > 0) & fwd, 2, None, None, xi)              # 2nd reversal
        put((m0 == 1) & (d00 < 0) & rev, 2, None, None, xi)
        put((m0 == 2) & (d00 > 0) & rev, 1, 1, xi)                       # 3rd: collapse
        put((m0 == 2) & (d00 < 0) & fwd, 1, -1, xi)
        # closure on the previous major branch, or completed transformation
        m[(m0 > 0) & (d00 > 0) & fwd & (gf >= r10)] = 0
        m[(m0 > 0) & (d00 < 0) & rev & (gr <= r10)] = 0
        m[(tr["xi"] <= EPS) | (tr["xi"] >= 1 - EPS)] = 0
        d0[m == 0] = 0; r1[m == 0] = 0; r2[m < 2] = 0
        c.update(T=tr["T"], xi=tr["xi"], et=tr["et"], xir=tr["xir"], etr=tr["etr"],
                 dlast=np.where(fwd, 1.0, np.where(rev, -1.0, dl0)), m=m, d0=d0, r1=r1, r2=r2)

    # ------------------------------------------------------------- thermal
    def thermal(self, u):
        p, c = self.p, self.c
        T, xi, ds = c["T"].copy(), c["xi"], self.ds
        rc = p.rho * ((1 - xi) * p.cA + xi * p.cM)
        k = (1 - xi) * p.kA + xi * p.kM
        kf = 0.5 * (k[1:] + k[:-1])
        nsub = max(1, int(np.ceil(p.dt / (0.3 * ds ** 2 * rc.min() / k.max()))))
        hsub = p.dt / nsub
        q = np.zeros(p.N + 1)
        for _ in range(nsub):
            q[1:-1] = kf * (T[1:] - T[:-1]) / ds
            if p.end_sink:
                q[0] = k[0] * (T[0] - p.Tamb) / (0.5 * ds)
                q[-1] = k[-1] * (p.Tamb - T[-1]) / (0.5 * ds)
            if p.heating == "joule":
                rhoe = (1 - xi) * p.rhoeA + xi * p.rhoeM
                src = rhoe * u ** 2 / self.A ** 2 - 4 * p.hconv / p.dw * (T - p.Tamb)
            elif p.heating == "power":          # u = measured electrical power [W], uniform
                src = u / (self.L0 * self.A) - 4 * p.hconv / p.dw * (T - p.Tamb)
            else:
                src = rc * (p.gamma_h * self.g * (u - T) + p.gamma_c * (p.Tamb - T))
            T = T + hsub * ((q[1:] - q[:-1]) / ds + src) / rc
        c["T"] = T

    # ---------------------------------------------------------- mechanics
    def sig_of_x(self, x):
        p = self.p
        if self.straight:                       # wire contraction x stretches the spring
            return (p.F0 + p.K * x) / self.A
        return p.K * x * np.hypot(p.h, p.a - x) / (2 * p.n * self.A * (p.a - x))

    def residual(self, x, T):
        p = self.p
        sig = self.sig_of_x(x)
        tr = self.trial(T, sig)
        strain = np.mean(self.S(tr["xi"]) * sig + self.alpha(tr["xi"]) * (tr["T"] - p.T0)
                         + tr["et"] - self.eref)
        if self.straight:
            return -x / self.L0 - strain
        return (np.hypot(p.h, p.a - x) - self.L0) / self.L0 - strain

    def step(self, u):
        """Advance one time step with input u (command temperature [K] or current [A])."""
        self.thermal(u)
        T = self.c["T"]
        if not self.straight and self.residual(0.0, T) <= 0.0:   # wire slack: no tension
            x = 0.0
        elif self.residual(self.xmin, T) > 0.0 > self.residual(self.xmax, T):
            x = brentq(self.residual, self.xmin, self.xmax, args=(T,), xtol=1e-14, rtol=1e-13)
        else:                                            # no bracket: best grid point
            grid = np.linspace(self.xmin, self.xmax, 400)
            x = float(grid[np.argmin([abs(self.residual(g, T)) for g in grid])])
            self.fails += 1
        self.x, self.sig = float(x), float(self.sig_of_x(x))
        self.commit(self.trial(T, self.sig))

    def run(self, u):
        """Simulate a command sequence. Sample 0 is the initial state.
        Returns Y [M, 3N+2] and S [M, 10N+2]."""
        Y, S = [self.obs()], [self.full()]
        for uk in u[1:]:
            self.step(float(uk))
            Y.append(self.obs()); S.append(self.full())
        return np.array(Y), np.array(S)


def simulate(p: Params, u, n=None, K=None):
    p = replace(p, n=p.n if n is None else float(n), K=p.K if K is None else float(K))
    act = Actuator(p)
    Y, S = act.run(u)
    return Y, S, act.fails
