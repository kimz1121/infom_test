"""Task-AGNOSTIC embedding<->trajectory evaluation.

Robocasa atomic tasks are semantically similar but geometrically MIXED, so task
identity is NOT a geometric cluster. Evaluating the embedding per-task (or
splitting same/cross-task) imposes a false structure. The honest question is
purely: does embedding distance predict geometric (DTW) trajectory distance,
over ALL pairs, with NO task grouping?

We compute a single pooled all-pairs Spearman(emb_dist, dtw_dist) over fixed-
length segments sampled two ways:
  random   : segments drawn uniformly at random across all episodes (task-blind).
  balanced : equal #segments per task (so no high-episode task dominates the pool)
             -- but the correlation is STILL one pooled all-pairs value, NOT a
             per-task average.

Embedding = z at the segment start frame (single vector, direct distance).
Geometric = DTW over the segment's proprio (+action), per-dim z-scored.

Seed is FIXED so the SAME segments are selected across runs; since proprio/action
are identical across the 1-cam/3-cam/lang datasets (only the image-feature width
differs), the DTW matrix is identical across runs and only the embedding varies
-- a clean controlled comparison.

Usage:
    python script/dtw_taskblind.py --run_dir <run> --epochs 500000,200000,50000 \
        --seg_len 25 --n_segments 400 --sampling both --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Fresh numba cache: analyze_trajectory_dtw's @njit(cache=True) was cached under
# __main__ ; importing it as a module makes numba fail to reload that stale cache
# ("No module named '<dynamic>'"). A dedicated cache dir forces a clean recompile.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache_taskblind")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.flax_utils import restore_agent  # noqa: E402
from script.analyze_trajectory_dtw import (  # noqa: E402
    build_agent_and_pretrain_dataset, encode_z_rows, geo_upper, build_episodes,
)
from script.visualize_latent_robocasa import derive_task_labels, _resolve_stats_path  # noqa: E402

# Geometric distance MEASURES (user-added in analyze_trajectory_dtw): DTW (warp-sum),
# Fréchet (bottleneck max over monotone coupling), Hausdorff (order-agnostic max-min).
MEAS = ["dtw", "frechet", "hausdorff"]
REPS_TB = ["state", "action", "state+action"]
VAR = "discounted"  # at seg_len 25 discounted≈plain; discounted matches z's discounting
COMBOS = [(m, r, VAR) for m in MEAS for r in REPS_TB]


def _seg(eps, seg_len, rng):
    s, e = eps[int(rng.integers(len(eps)))]
    t0 = int(rng.integers(s, e - seg_len + 2))
    return np.arange(t0, t0 + seg_len)


def sample_random(elig, seg_len, n, rng):
    return [_seg(elig, seg_len, rng) for _ in range(n)]


def sample_balanced(by_task, seg_len, n, rng):
    tasks = sorted(by_task)
    per = max(1, n // len(tasks))
    rows = []
    for t in tasks:
        for _ in range(per):
            rows.append(_seg(by_task[t], seg_len, rng))
    return rows


def build_seqs(raw_obs, actions_all, seg_rows, img_feat_dim, state_dim):
    all_rows = np.concatenate(seg_rows)
    end = (img_feat_dim + state_dim) if state_dim > 0 else raw_obs.shape[1]
    sr = raw_obs[all_rows][:, img_feat_dim:end].astype(np.float64)
    ar = actions_all[all_rows].astype(np.float64)
    sr = (sr - sr.mean(0, keepdims=True)) / np.clip(sr.std(0, keepdims=True), 1e-8, None)
    ar = (ar - ar.mean(0, keepdims=True)) / np.clip(ar.std(0, keepdims=True), 1e-8, None)
    st, ac, sa = [], [], []
    off = 0
    for r in seg_rows:
        sl = slice(off, off + len(r))
        s_seq = np.ascontiguousarray(sr[sl]); a_seq = np.ascontiguousarray(ar[sl])
        st.append(s_seq); ac.append(a_seq)
        sa.append(np.ascontiguousarray(np.hstack([s_seq, a_seq])))
        off += len(r)
    return {"state": st, "action": ac, "state+action": sa}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", default=None)
    ap.add_argument("--seg_len", type=int, default=25)
    ap.add_argument("--n_segments", type=int, default=400)
    ap.add_argument("--sampling", choices=["random", "balanced", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]
    cfg = flags.get("agent", {})
    gamma = float(cfg["discount"])
    horizon = int(round(1.0 / (1.0 - gamma)))
    img_feat_dim = int(cfg.get("image_feat_dim", 512 if "multimodal" in env_name else 0))
    state_dim = int(cfg["state_dim"]) if cfg.get("state_dim") else -1

    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    avail = [int(p.stem.split("_")[1]) for p in ckpts]
    epochs = [int(e) for e in args.epochs.split(",")] if args.epochs else [avail[-1]]

    print(f"run={run_dir.name} env={env_name} gamma={gamma} img_feat_dim={img_feat_dim} "
          f"state_dim={state_dim} seg_len={args.seg_len} n={args.n_segments} seed={args.seed}")

    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    actions_all = np.asarray(pre_train["actions"])
    terminals = np.asarray(pre_train["terminals"])
    episodes = build_episodes(terminals)
    dataset_len = len(raw_obs)
    end = (img_feat_dim + state_dim) if state_dim > 0 else raw_obs.shape[1]
    print(f"proprio dims = {end - img_feat_dim}, action dims = {actions_all.shape[1]}")

    # eligible episodes + per-task grouping (labels only used to BALANCE sampling).
    elig = [(s, e) for (s, e) in episodes if e - s + 1 >= args.seg_len]
    by_task = {}
    try:
        stats_path = _resolve_stats_path(env_name, "")
        starts = np.array([s for s, _ in episodes])
        labels, _ = derive_task_labels(starts, stats_path, dataset_len)
        for (s, e), t in zip(episodes, labels):
            if e - s + 1 >= args.seg_len:
                by_task.setdefault(t, []).append((s, e))
        print(f"eligible episodes={len(elig)} over {len(by_task)} tasks")
    except Exception as exc:  # noqa: BLE001
        print(f"[task labels unavailable: {exc}] balanced sampling disabled")

    modes = ["random", "balanced"] if args.sampling == "both" else [args.sampling]
    results = {}
    for mode in modes:
        rng = np.random.default_rng(args.seed)  # FIXED seed per mode -> same segs across runs
        if mode == "balanced":
            if not by_task:
                print("  (skip balanced: no task labels)")
                continue
            seg_rows = sample_balanced(by_task, args.seg_len, args.n_segments, rng)
        else:
            seg_rows = sample_random(elig, args.seg_len, args.n_segments, rng)
        n_seg = len(seg_rows)
        seqs = build_seqs(raw_obs, actions_all, seg_rows, img_feat_dim, state_dim)
        # geometric distance per (measure, rep) -- epoch-independent, computed once.
        geo = {(m, r, v): geo_upper(seqs[r], m, v, gamma, horizon) for (m, r, v) in COMBOS}
        start_rows = np.array([r[0] for r in seg_rows])

        print(f"\n=== sampling={mode}  ({n_seg} segments, {n_seg*(n_seg-1)//2} pairs) ===")
        results[mode] = {}
        emb_by_epoch = {}
        for epoch in epochs:
            agent_e = restore_agent(agent, str(run_dir), epoch)
            z = encode_z_rows(agent_e, pre_train, start_rows, args.seed)
            emb = {"euclidean": pdist(z, metric="euclidean"),
                   "cosine": pdist(z, metric="cosine")}
            emb_by_epoch[epoch] = emb["euclidean"]
            row = {}
            for em in ("euclidean", "cosine"):
                for (m, r, v) in COMBOS:
                    rho = float(spearmanr(emb[em], geo[(m, r, v)]).statistic)
                    row[f"{em}/{m}/{r}/{v}"] = rho
            results[mode][str(epoch)] = row
            print(f"  ep{epoch} [euclid, {VAR}]  ρ(state / action / state+action):")
            for m in MEAS:
                print(f"      {m:9s}  "
                      f"{row[f'euclidean/{m}/state/{VAR}']:.3f} / "
                      f"{row[f'euclidean/{m}/action/{VAR}']:.3f} / "
                      f"{row[f'euclidean/{m}/state+action/{VAR}']:.3f}")

        # scatter: emb(euclidean) dist vs DTW dist, rows=epoch x cols=[state/disc, s+a/disc]
        plot_dir = run_dir / "plots" / "trajectory_dtw" / f"taskblind_seg{args.seg_len}_n{args.n_segments}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        # scatter: emb(euclidean) dist vs geo dist, rows=epoch x cols=MEASURES.
        # One figure PER representation (state / action / state+action) so each
        # channel can be inspected on its own. New filenames -- the combined
        # scatter_{mode}.png from earlier runs is left untouched.
        for rep_plot in REPS_TB:
            fig, axes = plt.subplots(len(epochs), len(MEAS),
                                     figsize=(4.6 * len(MEAS), 3.8 * len(epochs)), squeeze=False)
            for ri, ep in enumerate(epochs):
                for ci, m in enumerate(MEAS):
                    ax = axes[ri][ci]
                    e = emb_by_epoch[ep]; g = geo[(m, rep_plot, VAR)]
                    ax.scatter(e, g, s=4, alpha=0.12, color="tab:blue", linewidths=0)
                    rho = float(spearmanr(e, g).statistic)
                    ax.set_title(f"ep{ep}  {m}  [{rep_plot}/{VAR}]   ρ={rho:.3f}", fontsize=10)
                    if ri == len(epochs) - 1:
                        ax.set_xlabel("embedding distance (euclidean, start-z)")
                    if ci == 0:
                        ax.set_ylabel("geometric distance")
                    ax.grid(True, alpha=0.3)
            fig.suptitle(f"task-agnostic ({mode}, {rep_plot}) — {run_dir.name}", fontsize=12)
            fig.tight_layout()
            tag = rep_plot.replace("+", "_")
            fig.savefig(plot_dir / f"scatter_{mode}_{tag}.png", dpi=130)
            plt.close(fig)
        print(f"  saved per-rep scatters ({mode}) -> {plot_dir}/scatter_{mode}_{{state,action,state_action}}.png")

    out = run_dir / "plots" / "trajectory_dtw" / f"taskblind_seg{args.seg_len}_n{args.n_segments}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"run_dir": str(run_dir), "env_name": env_name, "seg_len": args.seg_len,
                   "n_segments": args.n_segments, "seed": args.seed, "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
