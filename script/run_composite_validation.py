"""Per-checkpoint composite-validation driver.

For a training run, loops over EVERY saved checkpoint (params_<epoch>.pkl) and
every composite task, producing the composite-trajectory visualization for each
into a per-checkpoint subfolder so nothing is overwritten:

    <run_dir>/plots/composite_validation/ep<epoch>/composite_<name>_ep0_video_{tsne,umap}.mp4
    <run_dir>/plots/composite_validation/ep<epoch>/composite_<name>_ep0_{tsne,umap}.png

This realizes the experiment directives: visualize at every saved checkpoint,
and keep per-run / per-checkpoint folders separate.

Usage:
    python script/run_composite_validation.py --run_dir exp/<group>/<run> \
        --composites loaddishwasher_mmfeat placeveggiesindrawer_mmfeat \
                     stackbowlscabinet_mmfeat startelectrickettle_mmfeat
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_dir_epochs(run_dir: Path):
    return sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("params_*.pkl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--composites", nargs="+", required=True,
                    help="Composite feature basenames (e.g. loaddishwasher_mmfeat).")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--epochs", nargs="*", type=int, default=None,
                    help="Subset of epochs; default = all saved checkpoints.")
    ap.add_argument("--video", type=int, default=1, help="1 = render mp4 videos too.")
    ap.add_argument("--stills", type=int, default=1, help="1 = render tsne/umap still images.")
    ap.add_argument("--robocasa_dir", default="~/.robocasa/data")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    epochs = args.epochs or run_dir_epochs(run_dir)
    if not epochs:
        sys.exit(f"No params_*.pkl under {run_dir}")
    print(f"run_dir={run_dir}\nepochs={epochs}\ncomposites={args.composites}")

    n_fail = 0
    for epoch in epochs:
        for comp in args.composites:
            jobs = []
            if args.stills:
                jobs.append(("still", "script/visualize_composite_traj.py"))
            if args.video:
                jobs.append(("video", "script/render_composite_video.py"))
            for kind, script in jobs:
                cmd = [sys.executable, script,
                       "--run_dir", str(run_dir), "--epoch", str(epoch),
                       "--composite_name", comp, "--episode", str(args.episode),
                       "--robocasa_dir", args.robocasa_dir]
                print(f"\n=== ep{epoch} | {comp} | {kind} ===")
                r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
                if r.returncode != 0:
                    n_fail += 1
                    print(f"[warn] {kind} failed for ep{epoch} {comp} (rc={r.returncode})")
    print(f"\nDone. failures={n_fail}")


if __name__ == "__main__":
    main()
