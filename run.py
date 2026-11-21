"""Run the revision experiment suite.

    python run.py --quick                      # smoke test, a few minutes
    python run.py --preset revised             # full suite, revised physics
    python run.py --preset paper --out runs/paper
    python run.py --stages audit,data          # simulator only (no PyTorch needed)
    python run.py --preset paper --stages longtrain,paired,figures,summary   # pending results only
    python report.py --B runs/paper --C runs/revised   # manuscript tables and figures
"""
import argparse, os, sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fom import PRESETS
from data import load_or_make, SPLITS
import experiments as ex

ALL = ["audit", "data", "factorial", "ablation", "closure", "longtrain", "paired", "figures", "summary"]


def hardware(out):
    import json, platform
    import numpy, torch
    info = dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                python=platform.python_version(), numpy=numpy.__version__, torch=torch.__version__,
                cpu_count=os.cpu_count(), timing_threads=1)
    json.dump(info, open(os.path.join(out, "hardware.json"), "w"), indent=2)



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="revised", choices=list(PRESETS))
    ap.add_argument("--out", default=None, help="output folder (default runs/<preset>)")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--r", default="4,8,12,16", help="reduced dimensions")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data-seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="tiny data and training budget")
    a = ap.parse_args()

    p = PRESETS[a.preset]
    out = a.out or os.path.join("runs", a.preset + ("_quick" if a.quick else ""))
    os.makedirs(out, exist_ok=True)
    stages = a.stages.split(",")
    rs = [int(v) for v in a.r.split(",")]
    seeds = list(range(a.seeds))
    counts = None
    if a.quick:
        counts = {s: (12 if s == "train" else 4) for s in SPLITS}
        rs, seeds = rs[:2], seeds[:1]

    if "audit" in stages:
        ex.audit(out, p)
    neural = [s for s in stages if s in ex.STAGES]
    post = [s for s in stages if s in ("paired", "figures")]
    if "data" in stages or neural or post:
        data = load_or_make(out, p, list(SPLITS), a.data_seed, counts=counts)
        ex.data_stats(out, data, p)
    if neural or post:
        import torch
        from rom import Cfg
        cfg = Cfg()
        if a.preset == "revised":
            # heating and xi rising together is physical once latent heat is on,
            # so the direction penalty must be switched off in setting C
            cfg = replace(cfg, w_dir=0.0)
        if a.quick:
            cfg = replace(cfg, ae_epochs=4, rate_epochs=4, windows=96, val_every=2)
        dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[run] preset={a.preset} device={dev} r={rs} seeds={seeds} out={out}")
        ctx = ex.Ctx(out, p, cfg, dev, data)
        ctx.quick = a.quick
        hardware(out)
        for s in neural:
            ex.run_stage(ctx, s, rs, seeds)
        if "paired" in post:
            import paired
            if a.quick:
                paired.N_TRAJ, paired.N_PAIRS = 6, 8
            paired.run(ctx)
        if "figures" in post:
            import figures
            figures.trajectory(ctx, r=max(rs))
            figures.paired(out)
    if "summary" in stages and os.path.exists(os.path.join(out, "results.csv")):
        ex.summarize(out)


if __name__ == "__main__":
    main()
