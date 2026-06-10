"""Task-agnostic 3x3 scatter grids: geometric measure (rows) x embedding metric (cols).

Canonical distance config: L1 (cityblock) local cost + discounted variant.
For each (segment length, target rep, epoch): a 3x3 figure where
  rows  = geometric measure  [dtw, frechet, hausdorff]
  cols  = embedding distance  [cosine, euclidean, mahalanobis]
each tile = task-blind all-pairs scatter (embedding dist vs geometric dist) + Spearman.

Segments are sampled task-blind (uniformly at random across all episodes), since
robocasa atomic tasks are geometrically mixed -- the whole-embedding question is
metric-vs-metric over all pairs, not per task.

Output: <run>/plots/trajectory_dtw/grid3x3_L1disc/seg{L}frame_{rep}_ep{epoch}.png

Usage:
    python script/grid3x3_taskblind.py --run_dir <run> \
        --epochs 50000,200000,500000 --seg_lens 25,50,200 --n_segments 400 --seed 0
"""
import os, json, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_taskblind")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
from pathlib import Path
PROJECT_ROOT = Path("/home/iw/infom_test")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.covariance import LedoitWolf

from utils.flax_utils import restore_agent
from script.analyze_trajectory_dtw import (
    build_agent_and_pretrain_dataset, encode_z_rows, build_episodes,
    _dtw_path_stats, _frechet_coupling,
)
from script.dtw_taskblind import sample_random, build_seqs

REPS = ["state", "action", "state+action"]
MEASURES = ["dtw", "frechet", "hausdorff"]      # rows
EMB = ["cosine", "euclidean", "mahalanobis"]    # cols


def reduce_all(L, w):
    """3 reductions over the SAME (discounted, L1) local-cost matrix L."""
    raw, _plen, wsum = _dtw_path_stats(L, w, True)
    return {"dtw": raw / max(wsum, 1e-12),
            "frechet": float(_frechet_coupling(L)),
            "hausdorff": float(max(L.min(axis=1).max(), L.min(axis=0).max()))}


def geo_vectors(seqs, gamma):
    """{measure: upper-tri vector} for one rep, L1 (cityblock) + discounted."""
    n = len(seqs)
    out = {m: [] for m in MEASURES}
    for i in range(n):
        a = seqs[i]; ii = np.arange(a.shape[0])[:, None]
        for j in range(i + 1, n):
            b = seqs[j]; jj = np.arange(b.shape[0])[None, :]
            w = (gamma ** ((ii + jj) / 2.0)).astype(np.float64)
            L = (cdist(a, b, metric="cityblock") * w).astype(np.float64)
            r = reduce_all(L, w)
            for m in MEASURES:
                out[m].append(r[m])
    return {m: np.asarray(v) for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", default="50000,200000,500000")
    ap.add_argument("--seg_lens", default="25,50,200")
    ap.add_argument("--n_segments", type=int, default=400)
    ap.add_argument("--plot_max", type=int, default=0, help="0 = plot all pairs (HQ).")
    ap.add_argument("--tag", default="L1disc", help="output subfolder grid3x3_<tag>")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--point_size", type=float, default=1.6)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--reps", default="state,action,state+action",
                    help="Comma-separated target reps to plot.")
    ap.add_argument("--fit_line", action="store_true",
                    help="Overlay an OLS regression line per tile (+ Pearson r in title).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    flags = json.load(open(run_dir / "flags.json"))
    env = flags["env_name"]; cfg = flags["agent"]
    gamma = float(cfg["discount"])
    img = int(cfg.get("image_feat_dim", 512 if "multimodal" in env else 0))
    sdim = int(cfg["state_dim"]) if cfg.get("state_dim") else -1
    epochs = [int(e) for e in args.epochs.split(",")]
    seg_lens = [int(s) for s in args.seg_lens.split(",")]
    out_dir = run_dir / "plots" / "trajectory_dtw" / f"grid3x3_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"### {run_dir.name}  L1+discounted  seg_lens={seg_lens}  epochs={epochs}  -> {out_dir}")

    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    actions = np.asarray(pre_train["actions"])
    episodes = build_episodes(np.asarray(pre_train["terminals"]))
    rng_plot = np.random.default_rng(args.seed + 1)

    for L in seg_lens:
        elig = [(s, e) for (s, e) in episodes if e - s + 1 >= L]
        rng = np.random.default_rng(args.seed)
        seg_rows = sample_random(elig, L, args.n_segments, rng)
        seqs = build_seqs(raw_obs, actions, seg_rows, img, sdim)
        geo = {rep: geo_vectors(seqs[rep], gamma) for rep in reps}   # epoch-independent
        start_rows = np.array([r[0] for r in seg_rows])
        n_pairs = len(next(iter(geo["state"].values())))
        sub = (np.arange(n_pairs) if args.plot_max <= 0 or n_pairs <= args.plot_max
               else rng_plot.choice(n_pairs, size=args.plot_max, replace=False))
        print(f"  seg_len={L}f: {len(seg_rows)} segs, {n_pairs} pairs")

        for ep in epochs:
            ag = restore_agent(agent, str(run_dir), ep)
            z = encode_z_rows(ag, pre_train, start_rows, args.seed)
            prec = LedoitWolf().fit(z).precision_
            emb = {"euclidean": pdist(z, metric="euclidean"),
                   "cosine": pdist(z, metric="cosine"),
                   "mahalanobis": pdist(z, metric="mahalanobis", VI=prec)}
            for rep in reps:
                fig, axes = plt.subplots(3, 3, figsize=(19, 17), squeeze=False)
                for ri, meas in enumerate(MEASURES):
                    g = geo[rep][meas]
                    for ci, em in enumerate(EMB):
                        ax = axes[ri][ci]
                        e = emb[em]
                        ax.scatter(e[sub], g[sub], s=args.point_size, alpha=args.alpha,
                                   color="tab:blue", linewidths=0, rasterized=True)
                        rho = float(spearmanr(e, g).statistic)
                        title = f"{meas} (geo)  vs  {em} (emb)   ρ={rho:.3f}"
                        if args.fit_line:
                            slope, intercept = np.polyfit(e, g, 1)
                            xs = np.array([e.min(), e.max()])
                            ax.plot(xs, slope * xs + intercept, color="red", lw=2.2, alpha=0.9)
                            title += f"  r={pearsonr(e, g)[0]:.3f}"
                        ax.set_title(title, fontsize=15)
                        ax.tick_params(labelsize=12)
                        if ri == 2:
                            ax.set_xlabel(f"embedding distance [{em}]", fontsize=14)
                        if ci == 0:
                            ax.set_ylabel(f"{meas} distance", fontsize=14)
                        ax.grid(True, alpha=0.3)
                fig.suptitle(f"{run_dir.name}  |  {rep}  |  ep{ep}  |  seg={L} frames  "
                             f"|  L1+discounted, task-blind random (n={len(seg_rows)})",
                             fontsize=18)
                fig.tight_layout(rect=[0, 0, 1, 0.975])
                fn = f"seg{L}frame_{rep.replace('+','_')}_ep{ep}.png"
                fig.savefig(out_dir / fn, dpi=args.dpi)
                plt.close(fig)
            print(f"    ep{ep}: wrote {len(REPS)} figs (seg{L}frame_*_ep{ep})")
    print("DONE")


if __name__ == "__main__":
    main()
