"""Error, admissibility, residual and partial-cycle metrics (NumPy)."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
PCT_KEYS = ("x_nrmse", "sig_nrmse", "xi_rmse")


def traj_metrics(Yh, Y, u, nK, k, ref=None):
    """Metrics for one trajectory (or a segment of one).

    Yh, Y : predicted / true observations [M, 3N+2] in physical units
    u     : input sequence [M];  nK = (n, K);  k = data.constants(...)
    ref   : (x range, sigma range) used for normalisation; defaults to the range of Y.
            Pass the full-trajectory ranges when scoring a restart segment.
    """
    N = k["N"]
    n, K = nK
    T, xi, et, x, sig = Y[:, :N], Y[:, N:2 * N], Y[:, 2 * N:3 * N], Y[:, -2], Y[:, -1]
    Th, xih, eth, xh, sigh = Yh[:, :N], Yh[:, N:2 * N], Yh[:, 2 * N:3 * N], Yh[:, -2], Yh[:, -1]
    xr, sr = ref if ref is not None else (np.ptp(x), np.ptp(sig))
    xr, sr = max(xr, 1e-12), max(sr, 1e-12)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    out = dict(x_nrmse=rmse(xh, x) / xr, sig_nrmse=rmse(sigh, sig) / sr,
               xi_rmse=rmse(xih, xi), T_rmse=rmse(Th, T), et_rmse=rmse(eth, et))
    # admissibility (fractions of predicted samples)
    out["xi_viol"] = float(np.mean((xih < -1e-6) | (xih > 1 + 1e-6)))
    out["neg_x"] = float(np.mean(xh < -1e-9))
    out["neg_sig"] = float(np.mean(sigh < -1e-3))
    # mechanical residuals of the prediction
    L = np.hypot(k["h"], k["a"] - xh)
    req = K * xh - 2 * n * k["A"] * sigh * (k["a"] - xh) / L
    strain = np.mean(((1 - xih) / k["EA"] + xih / k["EM"]) * sigh[:, None]
                     + ((1 - xih) * k["aA"] + xih * k["aM"]) * (Th - k["T0"]) + eth - k["eref"], 1)
    out["eq_rms_N"] = float(np.sqrt(np.mean(req ** 2)))
    out["cp_rms"] = float(np.sqrt(np.mean(((L - k["L0"]) / k["L0"] - strain) ** 2)))
    # partial-cycle metrics
    du = np.diff(u)
    rev = np.where(du[:-1] * du[1:] < 0)[0] + 1            # command reversal samples
    if len(rev):
        out["rev_xi_err"] = rmse(xih[rev].mean(1), xi[rev].mean(1))   # mean xi at reversals
        out["rev_x_err"] = rmse(xh[rev], x[rev]) / xr                 # displacement at reversals
    else:
        out["rev_xi_err"] = out["rev_x_err"] = np.nan
    area, area_h = _trapz(x, u), _trapz(xh, u)                        # hysteresis loop area
    out["loop_area_err"] = float(abs(area_h - area) / max(abs(area), 1e-12))
    return out


def aggregate(rows):
    """Mean over trajectories for every metric, plus percentiles for the main errors."""
    if not rows:
        return {}
    out = {key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]}
    for key in PCT_KEYS:
        v = np.array([r[key] for r in rows])
        for q in (5, 50, 95):
            out[f"{key}_p{q:02d}"] = float(np.percentile(v, q))
    out["n_traj"] = len(rows)
    return out
