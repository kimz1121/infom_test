"""Clustering evaluation for the DINOv3-token learnable-attention inFOM run.

Mirrors script/cluster_robocasa_embeddings.py (same internal metrics, same
run_suite / plots / metrics.csv / summary.json) but sources observations from the
streaming token dataset (utils.token_dataset.TokenDataset) instead of a flat
HDF5, because the obs are a memmap'd patch-token grid.

Two embedding sources (cluster-then-label, internal metrics only; given task
labels are reference-only for Robocasa):
  raw_dino     : mean-pooled DINOv3 tokens per camera (+ state), z-scored — the
                 "no-inFOM" visual baseline (parallels raw_resnet).
  infom_latent : posterior mean z from the trained intention encoder
                 (learnable attention pool over tokens).

Usage:
    python script/cluster_dino_tokens.py \
        --run_dir exp/debug/<run> --epoch 500000 --n_samples 4000
"""
import argparse
import json
import os.path as osp
import sys
from pathlib import Path

import jax
import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs import robocasa_utils  # noqa: E402
from script.cluster_robocasa_embeddings import (  # noqa: E402
    build_and_restore_agent, run_suite)
from script.visualize_latent_robocasa import (  # noqa: E402
    derive_task_labels, derive_task_family_labels, encode_latents)
from utils.token_dataset import load_token_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--sources", type=str, default="both",
                    choices=["both", "raw_dino", "infom_latent"])
    ap.add_argument("--hdbscan_mcs", type=int, default=25)
    ap.add_argument("--hdbscan_ms", type=int, default=5)
    ap.add_argument("--optics_ms", type=int, default=10)
    ap.add_argument("--optics_xi", type=float, default=0.05)
    ap.add_argument("--optics_mcs", type=float, default=0.03)
    ap.add_argument("--spectral_nn", type=int, default=15)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(PROJECT_ROOT) / run_dir
    flags = json.load(open(run_dir / "flags.json"))
    env_name = flags["env_name"]
    pre_name, _, _ = robocasa_utils.parse_env_name(env_name)
    data_dir = osp.expanduser(robocasa_utils.DEFAULT_DATASET_DIR)
    stats_path = osp.join(data_dir, f"{pre_name}.stats.json")
    print(f"Run dir : {run_dir}\nEpoch   : {args.epoch}\nBase    : {pre_name}")

    max_rows = flags.get("pretraining_size", np.inf) or np.inf
    ds = load_token_dataset(data_dir, pre_name, max_size=max_rows)
    n_used = ds.size
    print(f"  n_trans={n_used}  obs_dim={ds.obs_dim}  layout={ds.obs_layout}")

    rng = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(n_used, size=min(args.n_samples, n_used), replace=False))

    obs = ds._gather_obs(ds.obs_idx[idxs]).astype(np.float32)  # (N, obs_dim)
    act = ds.actions[idxs].astype(np.float32)

    task_ref, present_tasks = derive_task_labels(idxs, stats_path, n_used)
    _fam, _ = derive_task_family_labels(task_ref)
    print(f"Reference task labels: {len(present_tasks)} tasks present among samples")

    sources = {}
    want = ({"raw_dino", "infom_latent"} if args.sources == "both" else {args.sources})
    if "raw_dino" in want:
        # mean-pool the 65 tokens per camera -> (N, n_cam*feat), append state, z-score.
        nc, nt, fd = ds.n_cam, ds.n_tok, ds.feat
        tok = obs[:, :ds.token_dim].reshape(len(idxs), nc, nt, fd).mean(axis=2)  # (N,nc,fd)
        raw = np.concatenate([tok.reshape(len(idxs), nc * fd),
                              obs[:, ds.token_dim:ds.token_dim + ds.state_dim]], axis=-1)
        raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)
        sources["raw_dino"] = raw.astype(np.float32)
    if "infom_latent" in want:
        print("Building + restoring agent, encoding latents ...")
        agent = build_and_restore_agent(flags, str(run_dir), args.epoch, ds.obs_dim, act.shape[1])
        mean, std, _ = encode_latents(agent, {"observations": obs, "actions": act},
                                      jax.random.PRNGKey(args.seed))
        print(f"  latent dim={mean.shape[1]} mu[{mean.min():+.2f},{mean.max():+.2f}] "
              f"sigma[{std.min():.2f},{std.max():.2f}]")
        sources["infom_latent"] = np.asarray(mean, np.float32)

    out_dir = Path(args.outdir) if args.outdir else run_dir / "plots" / "clustering_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, summaries = [], {}
    for name, X in sources.items():
        rows, summ = run_suite(name, X, task_ref, args, out_dir)
        all_rows.extend(rows)
        summaries[name] = summ

    import csv
    csv_path = out_dir / "metrics.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(run_dir=str(run_dir), epoch=args.epoch, base=pre_name,
                       n_samples=int(len(idxs)), sources=summaries), f, indent=2, default=float)
    print(f"\nWrote {csv_path}")
    for r in all_rows:
        print(f"  {r['source']:12s} {r['method']:22s} k={r['n_clusters']:>2d} "
              f"sil={r['silhouette']:.3f} CH={r['calinski_harabasz']:.0f} DB={r['davies_bouldin']:.3f}")


if __name__ == "__main__":
    main()
