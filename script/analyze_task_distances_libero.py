"""Numerical inter-instruction distance analysis for LIBERO-Goal latents.

LIBERO analogue of script/analyze_task_distances.py. For each of the 10
LIBERO-Goal instructions we collect q(z|s,a) posterior means, take the per-task
centroid, and compute euclidean / cosine / mahalanobis pairwise distance
matrices, plus heatmaps, MDS-2D layouts and per-sample PCA/LDA scatters.

All metric/projection machinery is reused from analyze_task_distances; only the
task list and the family-aware coloring are LIBERO-specific (imported from
visualize_latent_libero).

Outputs under <run_dir>/plots/task_distances_libero/:
  {euclidean,cosine,mahalanobis}_heatmap.png / .csv / _mds2d.png
  scatter2d_<emb>_<metric>.png
  task_distances.json

Usage:
    python script/analyze_task_distances_libero.py                 # latest run
    python script/analyze_task_distances_libero.py --space obs     # baseline
    python script/analyze_task_distances_libero.py --per_task 1500 --source mean
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.flax_utils import restore_agent  # noqa: E402
# Generic, dataset-agnostic machinery.
from script.analyze_task_distances import (  # noqa: E402
    stratified_idxs,
    extract_features,
    compute_distance_matrices,
    transform_features_for_metric,
    ranked_pairs,
    save_csv,
    save_scatter_2d,
    project_2d,
    offdiag_summary,
    resolve_run_dir,
)
# LIBERO-specific constants + agent/dataset helpers.
from script.visualize_latent_libero import (  # noqa: E402
    LIBERO_GOAL_10,
    TASK_FAMILY_MAP,
    FAMILY_ORDER,
    FAMILY_COLORS,
    build_agent_and_pretrain_dataset,
    encode_latents,
    _resolve_stats_path,
)


# ---------------------------------------------------------------------------
# LIBERO-specific stratified ranges + family-aware plotting.
# ---------------------------------------------------------------------------

def build_task_ranges(stats_path: str, dataset_len: int) -> list[tuple[str, int, int]]:
    """Return [(instruction, start, end_exclusive)] for instructions in the prefix."""
    with open(stats_path) as f:
        stats = json.load(f)
    task_T = stats["tasks"]
    ranges: list[tuple[str, int, int]] = []
    cur = 0
    for t in LIBERO_GOAL_10:
        if t not in task_T:
            continue
        n = int(task_T[t]["train_T"])
        start = cur
        end = min(cur + n, dataset_len)
        if end > start:
            ranges.append((t, start, end))
        cur = end
        if cur >= dataset_len:
            break
    return ranges


def order_by_family(present: list[str]) -> list[int]:
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(
        range(len(present)),
        key=lambda i: (fam_rank.get(TASK_FAMILY_MAP.get(present[i], "Other"), 99), present[i]),
    )


def save_heatmap(mat, present, order, out_path, *, title):
    tasks = [present[i] for i in order]
    m = mat[np.ix_(order, order)]
    n = len(tasks)
    fig, ax = plt.subplots(figsize=(0.62 * n + 5, 0.62 * n + 4))
    vmax = np.max(m) if np.max(m) > 0 else 1.0
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fam_color = lambda t: FAMILY_COLORS.get(TASK_FAMILY_MAP.get(t, "Other"), "black")
    ax.set_xticklabels(tasks, rotation=90, fontsize=7)
    ax.set_yticklabels(tasks, fontsize=7)
    for lbl, t in zip(ax.get_xticklabels(), tasks):
        lbl.set_color(fam_color(t))
    for lbl, t in zip(ax.get_yticklabels(), tasks):
        lbl.set_color(fam_color(t))
    thresh = vmax * 0.6
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if m[i, j] < thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_mds_2d(mat, present, out_path, *, title, seed):
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              n_init=8, max_iter=500, normalized_stress=False)
    coords = mds.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(9, 8))
    seen_fams: set[str] = set()
    for i, t in enumerate(present):
        fam = TASK_FAMILY_MAP.get(t, "Other")
        color = FAMILY_COLORS.get(fam, "black")
        ax.scatter(coords[i, 0], coords[i, 1], s=90, color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.5,
                   label=fam if fam not in seen_fams else None)
        seen_fams.add(fam)
        ax.annotate(t, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7.0)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(f"{title}\nMDS 2D embedding (stress={mds.stress_:.2f})", fontsize=11)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    order_leg = [labels.index(f) for f in FAMILY_ORDER if f in labels]
    ax.legend([handles[i] for i in order_leg], [labels[i] for i in order_leg],
              loc="best", fontsize=9, title="instruction family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return float(mds.stress_)


# ---------------------------------------------------------------------------
# Main (mirrors analyze_task_distances.main with LIBERO constants).
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir. Defaults to latest under exp/debug/.")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--per_task", type=int, default=1500)
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "obs", "action"])
    parser.add_argument("--source", type=str, default="mean", choices=["mean", "sample"])
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--scatter_embed", type=str, default="pca",
                        choices=["pca", "lda", "both"])
    parser.add_argument("--scatter_metric", type=str, default="all",
                        choices=["euclidean", "mahalanobis", "cosine", "all"])
    parser.add_argument("--ellipse_std", type=float, default=2.0)
    parser.add_argument("--scatter_max_per_task", type=int, default=800)
    parser.add_argument("--libero_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags.get("env_name", "")
    if not env_name.startswith("libero_"):
        raise SystemExit(
            f"env_name={env_name!r} is not libero_*; task labels unavailable.")

    print("Building agent + pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    dataset_len = len(raw_obs)

    if args.space == "latent":
        ckpts = sorted(run_dir.glob("params_*.pkl"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
        epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
        print(f"Restoring agent from epoch {epoch} ...")
        agent = restore_agent(agent, str(run_dir), epoch)
    else:
        epoch = None
        print(f"space={args.space}: skipping checkpoint restore (raw feature baseline).")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    ranges = build_task_ranges(stats_path, dataset_len)
    idxs, owner, present = stratified_idxs(ranges, args.per_task, args.seed)
    counts = {present[ti]: int((owner == ti).sum()) for ti in range(len(present))}
    print(f"Instructions present: {len(present)}/10.  per-task counts: {counts}")
    print(f"Total transitions sampled: {len(idxs)}")

    print(f"Extracting features (space={args.space}, source={args.source}) ...")
    feats = extract_features(agent, pre_train, idxs, args.space, args.source, args.seed).astype(np.float64)
    print(f"Feature matrix: {feats.shape}")

    mats = compute_distance_matrices(feats, owner, present)
    order = order_by_family(present)

    out_dir = run_dir / "plots" / "task_distances_libero"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "euclidean": "Euclidean distance between instruction centroids",
        "cosine": "Cosine distance (1 - cos sim) between instruction centroids",
        "mahalanobis": "Mahalanobis distance (pooled within-task cov)",
    }
    space_tag = args.space if args.space != "latent" else f"latent.{args.source}"
    summary = {
        "run_dir": str(run_dir), "epoch": epoch, "env_name": env_name,
        "space": space_tag, "feature_dim": int(feats.shape[1]),
        "per_task": args.per_task, "n_sampled": int(len(idxs)),
        "tasks_in_family_order": [present[i] for i in order],
        "per_task_counts": counts,
        "metrics": {},
    }
    for metric, mat in mats.items():
        suffix = f"_{args.space}" if args.space != "latent" else ""
        save_heatmap(mat, present, order,
                     out_dir / f"{metric}{suffix}_heatmap.png",
                     title=f"{metric_titles[metric]}\n{env_name}  [{space_tag}]")
        save_csv(mat, present, order, out_dir / f"{metric}{suffix}.csv")
        stress = save_mds_2d(mat, present, out_dir / f"{metric}{suffix}_mds2d.png",
                             title=f"{metric_titles[metric]}\n{env_name}  [{space_tag}]",
                             seed=args.seed)
        ranks = ranked_pairs(mat, present, args.top_k)
        summary["metrics"][metric] = {
            "offdiag": offdiag_summary(mat),
            "mds_stress": stress,
            "ranked_pairs": ranks,
            "matrix_family_order": mat[np.ix_(order, order)].tolist(),
        }
        print(f"  Saved heatmap + csv + MDS-2D for {metric} → {out_dir}")

    scatter_metrics = (["euclidean", "mahalanobis", "cosine"]
                       if args.scatter_metric == "all" else [args.scatter_metric])
    embeds = ["pca", "lda"] if args.scatter_embed == "both" else [args.scatter_embed]
    suffix = f"_{args.space}" if args.space != "latent" else ""
    summary["scatter_2d"] = {}
    for metric in scatter_metrics:
        feats_t = transform_features_for_metric(feats, owner, len(present), metric)
        for emb in embeds:
            coords, info = project_2d(feats_t, owner, emb, args.seed)
            out_path = out_dir / f"scatter2d_{emb}_{metric}{suffix}.png"
            fisher = save_scatter_2d(
                coords, owner, present, out_path,
                title=(f"Per-sample distribution per instruction — {metric} geometry\n"
                       f"{env_name}  [{space_tag}]"),
                info=info, n_std=args.ellipse_std,
                max_per_task=args.scatter_max_per_task, seed=args.seed,
            )
            summary["scatter_2d"][f"{metric}/{emb}"] = {
                "fisher_ratio_2d": fisher, "evr_2d": info.get("evr_2d"),
                "ellipse_std": args.ellipse_std,
            }
            print(f"  Saved {emb.upper()} scatter+ellipse [{metric}] "
                  f"(Fisher 2D={fisher:.3f}) → {out_path.name}")

    with (out_dir / "task_distances.json").open("w") as f:
        json.dump(summary, f, indent=2)

    for metric in ("euclidean", "cosine", "mahalanobis"):
        s = summary["metrics"][metric]["offdiag"]
        ranks = summary["metrics"][metric]["ranked_pairs"]
        print(f"\n=== {metric.upper()} ({space_tag}) ===")
        print(f"  off-diagonal: mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"min={s['min']:.4f}  max={s['max']:.4f}")
        print(f"  closest {args.top_k} pairs:")
        for p in ranks["closest"]:
            print(f"    {p['dist']:.4f}  {p['a']}  <->  {p['b']}")
        print(f"  farthest {args.top_k} pairs:")
        for p in ranks["farthest"]:
            print(f"    {p['dist']:.4f}  {p['a']}  <->  {p['b']}")

    print(f"\nWrote: {out_dir}/task_distances.json")
    print(f"Heatmaps + CSVs under: {out_dir}")


if __name__ == "__main__":
    main()
