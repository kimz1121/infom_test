"""Rank a run's checkpoints by how well their latent z separates atomic tasks.

For every params_<epoch>.pkl we encode the SAME stratified sample of atomic
transitions to q(z|s,a) means and score task separability in the full latent:

  fisher_trace_ratio = trace(S_between) / trace(S_within)   over the task labels
      S_within  = sum_c sum_{i in c} ||z_i - mu_c||^2     (spread inside a task)
      S_between = sum_c n_c ||mu_c - mu||^2                (spread across tasks)
  higher  ->  tasks form tighter, more separated clusters.

Also reports a quick 1-NN leave-in task accuracy as a second, more intuitive
view. Prints the ranking and the Top-K epochs (for selective video rendering).

Usage:
    python script/rank_checkpoints.py --run_dir exp/<group>/<run> --num 13000 --top 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import jax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.visualize_latent_robocasa import (  # noqa: E402
    build_agent_and_pretrain_dataset,
    derive_task_labels,
    encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent  # noqa: E402


def fisher_trace_ratio(z: np.ndarray, labels: np.ndarray) -> float:
    mu = z.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for c in np.unique(labels):
        zc = z[labels == c]
        muc = zc.mean(axis=0)
        sw += float(((zc - muc) ** 2).sum())
        sb += float(len(zc) * ((muc - mu) ** 2).sum())
    return sb / max(sw, 1e-12)


def nn_task_accuracy(z: np.ndarray, labels: np.ndarray, k: int, seed: int) -> float:
    """1-NN task accuracy on a held-out split (cheap separability proxy)."""
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(seed)
    n = len(z)
    perm = rng.permutation(n)
    n_test = min(3000, n // 4)
    te, tr = perm[:n_test], perm[n_test:]
    clf = KNeighborsClassifier(n_neighbors=k).fit(z[tr], labels[tr].astype(str))
    return float((clf.predict(z[te]) == labels[te].astype(str)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num", type=int, default=13000, help="Atomic transitions to encode.")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robocasa_dir", default="~/.robocasa/data")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epochs = [int(p.stem.split("_")[1]) for p in ckpts]
    if not epochs:
        sys.exit(f"No params_*.pkl under {run_dir}")

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]

    print("Building agent + atomic pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    stats_path = _resolve_stats_path(env_name, args.robocasa_dir)

    rng_np = np.random.default_rng(args.seed)
    idxs = rng_np.integers(0, len(raw_obs), size=args.num)
    labels, present = derive_task_labels(idxs, stats_path, len(raw_obs))
    batch = pre_train.sample(args.num, idxs=idxs)
    print(f"Scoring {len(epochs)} checkpoints on {args.num} transitions, "
          f"{len(present)} atomic tasks present.")

    rows = []
    for ep in epochs:
        ag = restore_agent(agent, str(run_dir), ep)
        mean, _, _ = encode_latents(ag, batch, jax.random.PRNGKey(args.seed))
        z = np.asarray(mean)
        fisher = fisher_trace_ratio(z, labels)
        acc = nn_task_accuracy(z, labels, args.knn, args.seed)
        rows.append({"epoch": ep, "fisher": fisher, "knn_acc": acc})
        print(f"  ep={ep:>7}  fisher={fisher:.4f}  {args.knn}-NN task acc={acc:.3f}")

    rows.sort(key=lambda r: r["fisher"], reverse=True)
    top = [r["epoch"] for r in rows[: args.top]]
    print("\n=== ranking by Fisher trace-ratio (task separability) ===")
    for i, r in enumerate(rows):
        mark = "  <== TOP" if r["epoch"] in top else ""
        print(f"  {i+1:>2}. ep={r['epoch']:>7}  fisher={r['fisher']:.4f}  knn_acc={r['knn_acc']:.3f}{mark}")
    out = {"run_dir": str(run_dir), "n_tasks": len(present), "num": args.num,
           "ranking": rows, "top_epochs": top}
    (run_dir / "checkpoint_task_separation.json").write_text(json.dumps(out, indent=2))
    print(f"\nTop-{args.top} epochs: {top}")
    print(f"Wrote {run_dir / 'checkpoint_task_separation.json'}")
    print("TOP_EPOCHS=" + ",".join(str(e) for e in top))


if __name__ == "__main__":
    main()
