"""Comparison of the coupled simulator (setting C) with a measured actuation test.

Data: Gorges, Weber, Mayer and Motzki (2025), SMA wire (300 um) actuated against a
spring, Zenodo DOI 10.5281/zenodo.15657324 (CC BY 4.0); paper: Smart Mater. Struct.
34 (2025) 065033.

Procedure
1. The measured current is used as the simulator input (Joule heating, latent heat on).
2. Geometry from the data: wire length L0 = R A / rho at the start of the window; spring
   stiffness and preload from a linear fit of force against stroke.
3. Phase resistivities rho_A(sigma), rho_M(sigma) from the two TENSILE tests of the same
   dataset (independent of the actuator test). If those files are absent, they are taken
   from the actuator test itself and the resistivity comparison is then NOT independent.
4. Calibration on the first cycles of the window (stroke only): transformation
   temperatures, stress-temperature slope, maximum transformation strain and convection
   coefficient. Global search (differential evolution, coarse mesh), then local polish.
5. Prediction: stroke on the remaining cycle(s), and the resistivity on all cycles, which
   is not used in the stroke calibration. Two resistivity models are reported:
   (a) phase resistivities only (fully independent of the actuator test);
   (b) (a) plus a linear temperature coefficient per phase, fitted on the calibration
       cycles only and evaluated on the held-out cycle.
   The R-phase on cooling is not represented by the simulator.

    python experimental.py --data experimental/ActuatorTest_SMA300um.txt \\
        --tensile-dir experimental --out runs/experiment
"""
import argparse, json, os, sys, time
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fom import PRESETS, Actuator

COLS = ["x_mm", "sig_MPa", "d_um", "I", "V", "R", "rho_uOcm", "t"]
# calibrated quantities: (name, lower, upper)
THETA = [("Ms", 230.0, 420.0),      # martensite start at zero stress [K]
         ("dM", 3.0, 60.0),         # Ms - Mf [K]
         ("dAs", -25.0, 60.0),      # As - Ms [K]
         ("dA", 3.0, 60.0),         # Af - As [K]
         ("C", 4.0e6, 22.0e6),      # CM = CA [Pa/K]
         ("Hmax", 0.025, 0.075),    # maximum transformation strain
         ("hc", 20.0, 2000.0)]      # convection coefficient [W/m^2/K]


def load(path):
    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    df.columns = COLS[:len(df.columns)]
    return df


def cycle_starts(t, I, thr=0.1, lead=1.0):
    up = np.where((I[:-1] < thr) & (I[1:] >= thr))[0]
    return t[up] - lead


def phase_resistivity(path, smin, smax):
    """Linear fit rho = rho0 + kappa*sigma over the first loading branch in [smin, smax] MPa."""
    df = load(path)
    s, r = df.sig_MPa.values, df.rho_uOcm.values
    k = int(np.argmax(s))                               # first loading branch up to peak stress
    m = (s[:k] >= smin) & (s[:k] <= smax)
    if m.sum() < 20:                                    # window not covered: use upper 30% of branch
        lo_, hi_ = np.percentile(s[:k], [70, 99])
        print(f"[exp] WARNING: {os.path.basename(path)} has too few samples in [{smin}, {smax}] MPa;"
              f" fitting over [{lo_:.0f}, {hi_:.0f}] MPa instead")
        m = (s[:k] >= lo_) & (s[:k] <= hi_)
    kappa, rho0 = np.polyfit(s[:k][m], r[:k][m], 1)
    return float(rho0), float(kappa), (s[:k], r[:k])


def window(df, cycles, dt):
    st = cycle_starts(df.t.values, df.I.values)
    c0, c1 = cycles[0] - 1, cycles[-1]
    t0 = st[c0]
    t1 = st[c1] if c1 < len(st) else df.t.values[-1]
    w = df[(df.t >= t0) & (df.t <= t1)]
    t = np.arange(0.0, w.t.values[-1] - t0, dt)
    g = lambda c: np.interp(t, w.t.values - t0, w[c].values)
    bounds = [s - t0 for s in st[c0:c1]] + [t[-1]]
    P = np.interp(t, w.t.values - t0, (w["V"] * w["I"]).values)       # electrical power [W]
    return dict(t=t, I=g("I"), P=P, x=(g("x_mm") - g("x_mm")[0]) * 1e-3, sig=g("sig_MPa") * 1e6,
                rho=g("rho_uOcm"), R=g("R"), d=g("d_um"), bounds=np.array(bounds))


def params_from(theta, base, N):
    Ms, dM, dAs, dA, C, Hmax, hc = theta
    return replace(base, Ms=Ms, Mf=Ms - dM, As=Ms + dAs, Af=Ms + dAs + dA, CM=C, CA=C,
                   Hmax=Hmax, hconv=hc, N=N)


def simulate(p, I):
    act = Actuator(p)
    Y, _ = act.run(I)
    N = p.N
    return Y[:, -2], Y[:, -1], Y[:, N:2 * N], Y[:, :N], act.fails


class Objective:
    """Stroke misfit on the calibration window (module-level so it can be sent to workers)."""
    def __init__(self, base, I, x_target, stroke, N, step=4):
        self.base, self.I, self.x, self.stroke, self.N, self.step = base, I, x_target, stroke, N, step

    def resid(self, theta):
        try:
            x, *_ = simulate(params_from(theta, self.base, self.N), self.I)
        except Exception:
            return np.full(len(self.I), 10.0)[::self.step]
        return ((x - self.x) / self.stroke)[::self.step]

    def __call__(self, theta):
        return float(np.mean(self.resid(theta) ** 2))


class Free:
    """Objective over the free parameters only (others held at fixed values)."""
    def __init__(self, obj, free, full0):
        self.obj, self.free, self.full0 = obj, np.asarray(free), np.asarray(full0, dtype=float)

    def expand(self, v):
        th = self.full0.copy()
        th[self.free] = v
        return th

    def __call__(self, v):
        return self.obj(self.expand(v))

    def resid(self, v):
        return self.obj.resid(self.expand(v))


class Scaled:
    def __init__(self, obj, lo, scale):
        self.obj, self.lo, self.scale = obj, lo, scale

    def __call__(self, z):
        return self.obj.resid(self.lo + z * self.scale)


def rho_model(xi, sig, rp):
    s = sig[:, None] / 1e6
    return ((1 - xi) * (rp["rhoA0"] + rp["kA"] * s) + xi * (rp["rhoM0"] + rp["kM"] * s)).mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="experimental/ActuatorTest_SMA300um.txt")
    ap.add_argument("--tensile-dir", default="experimental")
    ap.add_argument("--out", default="runs/experiment")
    ap.add_argument("--cycles", default="8,9,10", help="cycles used (1-based); the last is held out")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--Tamb", type=float, default=295.15)
    ap.add_argument("--A-window", default="100,300", help="stress window [MPa] for austenite resistivity fit")
    ap.add_argument("--M-window", default="350,550", help="stress window [MPa] for martensite resistivity fit")
    ap.add_argument("--de-iter", type=int, default=25)
    ap.add_argument("--de-pop", type=int, default=10)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--input", default="current", choices=["current", "power"],
                    help="simulator input: measured current (Joule heating with resistivity) or measured power V*I")
    ap.add_argument("--fix", default="", help="fixed parameters, e.g. C=8e6,Hmax=0.04")
    ap.add_argument("--N", type=int, default=64, help="cells for the final simulation")
    ap.add_argument("--init", default="", help="experiment.json to start the local fit from (skips the global search)")
    ap.add_argument("--quick", action="store_true", help="tiny budget, for testing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cyc = [int(c) for c in a.cycles.split(",")]
    if a.quick:
        a.de_iter, a.de_pop = 2, 3

    df = load(a.data)
    w = window(df, cyc, a.dt)
    nc = len(cyc)
    tcal = w["bounds"][nc - 1]                      # calibration: all but the last cycle
    cal = w["t"] < tcal
    # geometry from the data
    A_d = np.pi * (w["d"][0] * 1e-6) ** 2 / 4
    L0 = w["R"][0] * A_d / (w["rho"][0] * 1e-8)
    dw = 300e-6
    A = np.pi * dw ** 2 / 4
    kf, f0 = np.polyfit(w["x"], w["sig"] * A, 1)
    print(f"[exp] window: cycles {cyc}, {w['t'][-1]:.1f} s, {len(w['t'])} samples at {a.dt} s")
    print(f"[exp] wire length L0 = {L0 * 1e3:.1f} mm, spring K = {kf / 1e3:.3f} N/mm, preload F0 = {f0:.2f} N")

    # phase resistivities
    fa = os.path.join(a.tensile_dir, "AusteniteTensileTest_SMA300um.txt")
    fm = os.path.join(a.tensile_dir, "MartensiteTensileTest_SMA300um.txt")
    independent = os.path.exists(fa) and os.path.exists(fm)
    curves = {}
    if independent:
        aw = [float(v) for v in a.A_window.split(",")]
        mw = [float(v) for v in a.M_window.split(",")]
        rA0, kA, curves["A"] = phase_resistivity(fa, *aw)
        rM0, kM, curves["M"] = phase_resistivity(fm, *mw)
        src = "tensile tests"
    else:
        hot = w["I"] > 0.95 * w["I"].max()
        cold = w["I"] < 0.05
        rA0, rM0 = float(np.median(w["rho"][hot])), float(np.median(w["rho"][cold]))
        kA = kM = 0.0
        src = "actuator test plateaus (NOT independent)"
        print("[exp] WARNING: tensile-test files not found; resistivity comparison is not independent")
    rp = dict(rhoA0=rA0, kA=kA, rhoM0=rM0, kM=kM)
    smean = float(w["sig"].mean() / 1e6)
    print(f"[exp] resistivity from {src}: rho_A = {rA0:.2f} + {kA:.4f} sigma, "
          f"rho_M = {rM0:.2f} + {kM:.4f} sigma  [uOhm cm, sigma in MPa]")

    base = replace(PRESETS["revised"], geometry="straight", dw=dw, L0w=L0, K=kf, F0=f0,
                   heating="joule" if a.input == "current" else "power",
                   Tamb=a.Tamb, Tinit=a.Tamb, dt=a.dt,
                   rhoeA=(rA0 + kA * smean) * 1e-8, rhoeM=(rM0 + kM * smean) * 1e-8)
    I = w["I"] if a.input == "current" else w["P"]
    stroke = np.ptp(w["x"])
    cost = Objective(base, I[cal], w["x"][cal], stroke, N=8)
    fine = Objective(base, I[cal], w["x"][cal], stroke, N=16)

    names = [b[0] for b in THETA]
    fixed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.fix.split(",") if kv}
    full0 = np.array([fixed.get(n, 0.5 * (b[1] + b[2])) for n, b in zip(names, THETA)])
    free = [i for i, n in enumerate(names) if n not in fixed]
    lo = np.array([THETA[i][1] for i in free])
    hi = np.array([THETA[i][2] for i in free])
    t0 = time.time()
    if a.init:
        e0 = json.load(open(a.init))
        x_start = np.array([e0["theta"][names[i]] for i in free])
        print(f"[exp] starting the local fit from {a.init} (no global search)")
    else:
        de = differential_evolution(Free(cost, free, full0), list(zip(lo, hi)), maxiter=a.de_iter,
                                    popsize=a.de_pop, seed=7, polish=False, tol=1e-6, workers=a.workers,
                                    updating="deferred" if a.workers != 1 else "immediate", disp=True)
        print(f"[exp] global search: cost {de.fun:.3e} ({time.time() - t0:.0f} s)")
        x_start = de.x
    scale = hi - lo
    ls = least_squares(Scaled(Free(fine, free, full0), lo, scale), np.clip((x_start - lo) / scale, 0, 1), bounds=(0, 1),
                       diff_step=1e-3, max_nfev=20 if a.quick else 200, verbose=1)
    theta = Free(fine, free, full0).expand(lo + ls.x * scale)
    print(f"[exp] local polish done ({time.time() - t0:.0f} s)")
    # identifiability at the optimum: sensitivities and correlations of the free parameters
    J = ls.jac / scale                                   # d residual / d parameter (physical units)
    Jn = ls.jac                                          # per unit of each search range
    JtJ = Jn.T @ Jn
    try:
        cov = np.linalg.pinv(JtJ)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        corr = np.full_like(JtJ, np.nan)
    ev = np.linalg.eigvalsh(JtJ)
    ident = dict(free=[names[i] for i in free],
                 sensitivity_per_range=[float(v) for v in np.linalg.norm(Jn, axis=0)],
                 correlation=np.round(corr, 3).tolist(),
                 condition_number=float(ev.max() / max(ev.min(), 1e-300)),
                 at_bound=[names[i] for i, z in zip(free, ls.x) if z < 0.01 or z > 0.99])
    print("[exp] identifiability: " + json.dumps(ident))

    p = params_from(theta, base, a.N)
    x, sig, xi, T, fails = simulate(p, I)
    rho_s = rho_model(xi, sig, rp)
    hold = ~cal
    # (b) temperature coefficients fitted on the calibration cycles only
    dT = T - a.Tamb
    Xg = np.column_stack([((1 - xi) * dT).mean(1), (xi * dT).mean(1)])
    gam, *_ = np.linalg.lstsq(Xg[cal], (w["rho"] - rho_s)[cal], rcond=None)
    rho_t = rho_s + Xg @ gam
    rmse = lambda u, v, m: float(np.sqrt(np.mean((u[m] - v[m]) ** 2)))
    res = dict(
        data="Gorges et al. (2025), Zenodo 10.5281/zenodo.15657324, ActuatorTest_SMA300um",
        cycles=cyc, calibration_cycles=cyc[:-1], held_out_cycle=cyc[-1], dt=a.dt, Tamb=a.Tamb,
        input=a.input, fixed=fixed, N=a.N, identifiability=ident, final_cost=float(np.mean(ls.fun ** 2)),
        L0_mm=L0 * 1e3, spring_N_per_mm=kf / 1e3, preload_N=f0, resistivity_source=src,
        rhoeA=base.rhoeA, rhoeM=base.rhoeM,
        rho=rp, theta={n: float(v) for (n, _, _), v in zip(THETA, theta)},
        stroke_mm=stroke * 1e3, failed_root_solves=int(fails),
        stroke_nrmse_cal=rmse(x, w["x"], cal) / stroke, stroke_nrmse_heldout=rmse(x, w["x"], hold) / stroke,
        stress_rmse_MPa=rmse(sig, w["sig"], np.ones_like(cal)) / 1e6,
        rho_rmse_uOcm=rmse(rho_s, w["rho"], np.ones_like(cal)),
        rho_nrmse=rmse(rho_s, w["rho"], np.ones_like(cal)) / np.ptp(w["rho"]),
        rho_corr=float(np.corrcoef(rho_s, w["rho"])[0, 1]),
        temp_coef_uOcm_per_K=dict(austenite=float(gam[0]), martensite=float(gam[1])),
        rho_T_nrmse_heldout=rmse(rho_t, w["rho"], hold) / np.ptp(w["rho"]),
        rho_T_corr_heldout=float(np.corrcoef(rho_t[hold], w["rho"][hold])[0, 1]),
        rho_nrmse_heldout=rmse(rho_s, w["rho"], hold) / np.ptp(w["rho"]),
        peak_T_K=float(T.max()))
    json.dump(res, open(os.path.join(a.out, "experiment.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, "experiment_traces.npz"), t=w["t"], I=w["I"], P=w["P"], u=I, x_exp=w["x"],
                        x_sim=x, sig_exp=w["sig"], sig_sim=sig, rho_exp=w["rho"], rho_sim=rho_s, rho_sim_T=rho_t,
                        T_mean=T.mean(1), xi_mean=xi.mean(1), bounds=w["bounds"])
    print("[exp] " + json.dumps(res, indent=2))
    figure(a.out, w, x, sig, rho_s, rho_t, T, xi, tcal, curves, rp, res)


def figure(out, w, x, sig, rho_s, rho_t, T, xi, tcal, curves, rp, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "legend.frameon": False})
    t = w["t"]
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], hspace=0.35, wspace=0.3)
    ax = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for a_ in ax:
        a_.axvspan(tcal, t[-1], color="0.92", lw=0)
    ax[0].plot(t, w["I"], "k")
    ax[0].set_ylabel("Current (A)")
    ax[0].text(tcal + 0.5, w["I"].max() * 0.9, "held out", fontsize=7)
    ax[1].plot(t, w["x"] * 1e3, "k", label="measured")
    ax[1].plot(t, x * 1e3, color="#3b5bdb", label="simulator (calibrated)")
    ax[1].set_ylabel("Stroke (mm)")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[2].plot(t, w["rho"], "k", label="measured")
    ax[2].plot(t, rho_s, color="#c92a2a", label="predicted, phase resistivities only")
    ax[2].plot(t, rho_t, "--", color="#e8590c", label="with temperature coefficients")
    ax[2].set_ylabel(r"Resistivity ($\mu\Omega$ cm)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(fontsize=6, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.62))
    for a_ in ax[:2]:
        a_.tick_params(labelbottom=False)
    b = fig.add_subplot(gs[0:2, 1])
    b.plot(w["I"], w["x"] * 1e3, "k", lw=0.8)
    b.plot(w["I"], x * 1e3, color="#3b5bdb", lw=0.8)
    b.set_xlabel("Current (A)")
    b.set_ylabel("Stroke (mm)")
    b.set_title("Stroke-current loops", fontsize=8)
    c = fig.add_subplot(gs[2, 1])
    if curves:
        for k, col in (("A", "#d95f02"), ("M", "#1b9e77")):
            s_, r_ = curves[k]
            c.plot(s_[::50], r_[::50], ".", ms=1, color=col, label=f"tensile, {'austenite' if k == 'A' else 'martensite'}")
        c.set_xlabel("Stress (MPa)")
        c.set_ylabel(r"$\rho$ ($\mu\Omega$ cm)")
        c.legend(fontsize=6, markerscale=5)
        c.set_title("Resistivity input (tensile tests)", fontsize=8)
    else:
        c.plot(t, T.mean(1), "k")
        c.set_xlabel("Time (s)")
        c.set_ylabel("Mean T (K)")
    fig.suptitle(f"Stroke NRMSE: calibration {res['stroke_nrmse_cal']:.3f}, held-out cycle "
                 f"{res['stroke_nrmse_heldout']:.3f};  resistivity NRMSE, held-out cycle: "
                 f"{res['rho_nrmse_heldout']:.3f} (phase only), {res['rho_T_nrmse_heldout']:.3f} (with temp. coef.)", fontsize=8)
    f = os.path.join(out, "fig_experiment.pdf")
    fig.savefig(f, bbox_inches="tight")
    fig.savefig(f.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[exp] wrote {f}")


if __name__ == "__main__":
    main()
