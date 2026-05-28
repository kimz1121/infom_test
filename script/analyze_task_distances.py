"""Numerical inter-task distance analysis for robocasa latents.

Where visualize_latent_robocasa.py shows *where* tasks land in a t-SNE/UMAP
projection, this script answers the quantitative question: how far apart are
the 18 atomic_seen tasks from each other in the encoder's latent space?

For each task we collect q(z|s,a) posterior means, take the per-task centroid
(the "distribution center"), and compute three pairwise distance matrices:

  euclidean   : ||mu_i - mu_j||_2  between task centroids.
  cosine      : 1 - cos(mu_i, mu_j) (cosine *distance*; 0 = same direction).
  mahalanobis : sqrt((mu_i-mu_j)^T S_W^-1 (mu_i-mu_j)), where S_W is the pooled
                within-task covariance (Ledoit-Wolf shrinkage so the 512-d
                matrix is invertible). This is the LDA-style separation that
                discounts directions in which every task is individually spread
                out — the most honest "are these clusters actually distinct".

Sampling is *stratified*: an equal number of transitions per task (so a task's
centroid/covariance is not dominated by a high-T task). Task ownership of each
row is reconstructed from the <basename>.stats.json cumulative-T table, exactly
as in visualize_latent_robocasa.derive_task_labels.

Outputs under <run_dir>/plots/task_distances/:
  {euclidean,cosine,mahalanobis}_heatmap.png   annotated 18x18 heatmaps
  {euclidean,cosine,mahalanobis}.csv           raw matrices (task-labeled)
  task_distances.json                           matrices + ranked pairs + meta

Usage:
    python script/analyze_task_distances.py                       # latest run
    python script/analyze_task_distances.py --run_dir exp/debug/<run>
    python script/analyze_task_distances.py --space obs           # baseline
    python script/analyze_task_distances.py --per_task 2000 --source sample
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402

from utils.flax_utils import restore_agent  # noqa: E402
from script.visualize_latent_robocasa import (  # noqa: E402
    ATOMIC_SEEN_18,
    TASK_FAMILY_MAP,
    FAMILY_ORDER,
    FAMILY_COLORS,
    build_agent_and_pretrain_dataset,
    encode_latents,
    _resolve_stats_path,
    _class_separation_score,
)


# ---------------------------------------------------------------------------
# Stratified per-task sampling.
# ---------------------------------------------------------------------------

def build_task_ranges(stats_path: str, dataset_len: int) -> list[tuple[str, int, int]]:
    """Return [(task, start, end_exclusive)] for tasks present in the loaded prefix.

    Mirrors derive_task_labels: the HDF5 concatenates tasks in ATOMIC_SEEN_18
    order, so cumulative train_T gives each task's contiguous row range. The
    loader reads only the first ``dataset_len`` rows, so trailing tasks may be
    absent or truncated; we clamp to dataset_len and drop empty ranges.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    task_T = stats["tasks"]
    ranges: list[tuple[str, int, int]] = []
    cur = 0
    for t in ATOMIC_SEEN_18:
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


def stratified_idxs(
    ranges: list[tuple[str, int, int]], per_task: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample up to ``per_task`` indices inside each task's row range.

    Returns (idxs, owner_task_index, present_tasks) where owner_task_index[k]
    is the position of idxs[k]'s task within present_tasks.
    """
    rng = np.random.default_rng(seed)
    idxs: list[int] = []
    owner: list[int] = []
    present: list[str] = []
    for t, start, end in ranges:
        n_avail = end - start
        take = min(per_task, n_avail)
        chosen = rng.choice(np.arange(start, end), size=take, replace=False)
        ti = len(present)
        present.append(t)
        idxs.extend(chosen.tolist())
        owner.extend([ti] * take)
    return np.asarray(idxs), np.asarray(owner), present


# ---------------------------------------------------------------------------
# Feature extraction (latent / obs / action).
# ---------------------------------------------------------------------------

def encode_latents_chunked(
    agent, pre_train, idxs: np.ndarray, source: str, seed: int, chunk: int = 4096,
) -> np.ndarray:
    """Encode q(z|s,a) for ``idxs`` in chunks; return (N, latent_dim) features."""
    rng = jax.random.PRNGKey(seed)
    out: list[np.ndarray] = []
    for s in range(0, len(idxs), chunk):
        sub = idxs[s : s + chunk]
        batch = pre_train.sample(len(sub), idxs=sub)
        rng, sub_rng = jax.random.split(rng)
        mean, _std, sample = encode_latents(agent, batch, sub_rng)
        out.append(sample if source == "sample" else mean)
    return np.concatenate(out, axis=0)


def extract_features(
    agent, pre_train, idxs: np.ndarray, space: str, source: str, seed: int,
) -> np.ndarray:
    """Return the (N, D) feature matrix for the chosen distance space."""
    if space == "latent":
        return encode_latents_chunked(agent, pre_train, idxs, source, seed)
    if space == "obs":
        # normalize_observations() already replaced the stored obs in-place.
        return np.asarray(pre_train["observations"])[idxs].astype(np.float64)
    if space == "action":
        return np.asarray(pre_train["actions"])[idxs].astype(np.float64)
    raise ValueError(f"Unknown space: {space!r}")


# ---------------------------------------------------------------------------
# Distance matrices.
# ---------------------------------------------------------------------------

def _pooled_precision(feats: np.ndarray, owner: np.ndarray, n_tasks: int) -> np.ndarray:
    """Ledoit-Wolf precision of pooled within-task (centered) features.

    Each sample is centered by its own task mean before pooling, so the
    estimate captures spread *within* tasks (the LDA denominator), not the
    between-task spread we want to measure against it.
    """
    from sklearn.covariance import LedoitWolf
    centered = np.empty_like(feats, dtype=np.float64)
    for ti in range(n_tasks):
        m = owner == ti
        centered[m] = feats[m] - feats[m].mean(axis=0, keepdims=True)
    lw = LedoitWolf().fit(centered)
    return lw.precision_


def compute_distance_matrices(
    feats: np.ndarray, owner: np.ndarray, present: list[str],
) -> dict[str, np.ndarray]:
    """Return {metric: (n_tasks, n_tasks)} for euclidean / cosine / mahalanobis."""
    n = len(present)
    centroids = np.stack([feats[owner == ti].mean(axis=0) for ti in range(n)])

    # Euclidean between centroids.
    diff = centroids[:, None, :] - centroids[None, :, :]
    euclidean = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Cosine distance between centroids.
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit = centroids / np.clip(norms, 1e-12, None)
    cos_sim = np.clip(unit @ unit.T, -1.0, 1.0)
    cosine = 1.0 - cos_sim

    # Mahalanobis with pooled within-task precision.
    precision = _pooled_precision(feats, owner, n)
    maha = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = centroids[i] - centroids[j]
            val = float(np.sqrt(max(d @ precision @ d, 0.0)))
            maha[i, j] = maha[j, i] = val

    return {"euclidean": euclidean, "cosine": cosine, "mahalanobis": maha}


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def transform_features_for_metric(
    feats: np.ndarray, owner: np.ndarray, n_tasks: int, metric: str,
) -> np.ndarray:
    """Map features so that plain Euclidean distance equals the chosen metric.

    Running PCA/LDA on the transformed space therefore yields a 2D scatter
    whose geometry reflects ``metric``:
      euclidean   : identity (raw features).
      cosine      : L2-normalize each sample; Euclidean on the unit sphere is
                    monotonic with cosine distance (||a-b||^2 = 2(1-cos)).
      mahalanobis : whiten by the pooled within-task precision P (x -> P^{1/2} x),
                    so ||x'-y'|| = sqrt((x-y)^T P (x-y)) = Mahalanobis distance.
    """
    if metric == "euclidean":
        return feats
    if metric == "cosine":
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-12, None)
    if metric == "mahalanobis":
        precision = _pooled_precision(feats, owner, n_tasks)
        vals, vecs = np.linalg.eigh(precision)
        vals = np.clip(vals, 0.0, None)
        whiten = (vecs * np.sqrt(vals)) @ vecs.T  # symmetric P^{1/2}
        return feats @ whiten
    raise ValueError(f"Unknown metric: {metric!r}")


def order_by_family(present: list[str]) -> list[int]:
    """Index order grouping present tasks by motion family for readable heatmaps."""
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(
        range(len(present)),
        key=lambda i: (fam_rank.get(TASK_FAMILY_MAP.get(present[i], "Other"), 99), present[i]),
    )


def ranked_pairs(mat: np.ndarray, present: list[str], k: int) -> dict:
    """Top-k closest and farthest off-diagonal task pairs, plus each task's NN."""
    n = len(present)
    pairs = [
        (present[i], present[j], float(mat[i, j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    pairs.sort(key=lambda x: x[2])
    nn = {}
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: mat[i, j])
        j = order[0]
        nn[present[i]] = {"task": present[j], "dist": float(mat[i, j])}
    return {
        "closest": [{"a": a, "b": b, "dist": d} for a, b, d in pairs[:k]],
        "farthest": [{"a": a, "b": b, "dist": d} for a, b, d in pairs[-k:][::-1]],
        "nearest_neighbor": nn,
    }


def save_heatmap(mat, present, order, out_path, *, title):
    tasks = [present[i] for i in order]
    m = mat[np.ix_(order, order)]
    n = len(tasks)
    fig, ax = plt.subplots(figsize=(0.62 * n + 4, 0.62 * n + 3))
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
    """Embed the task distance matrix into 2D via metric MDS, one point per task.

    MDS finds 2D coordinates whose pairwise Euclidean distances best reproduce
    ``mat`` (a precomputed dissimilarity), so the layout is the honest 2D
    picture of the full distance structure. Points are colored by motion family
    and annotated with task names. Returns the MDS stress (lower = better fit).
    """
    from sklearn.manifold import MDS
    n = len(present)
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
                    textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(f"{title}\nMDS 2D embedding (stress={mds.stress_:.2f})", fontsize=11)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    order_leg = [labels.index(f) for f in FAMILY_ORDER if f in labels]
    ax.legend([handles[i] for i in order_leg], [labels[i] for i in order_leg],
              loc="best", fontsize=9, title="motion family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return float(mds.stress_)


def project_2d(feats: np.ndarray, owner: np.ndarray, method: str, seed: int):
    """Project (N, D) features to (N, 2). Returns (coords, info_dict).

    pca : linear, unsupervised — preserves the true Euclidean geometry, so
          per-task covariance ellipses honestly reflect spread/overlap.
    lda : linear, supervised — rotates to the axes that maximize between-task
          vs within-task variance, i.e. the *best-case* separability view.
    """
    if method == "pca":
        X = feats - feats.mean(axis=0, keepdims=True)
        _u, s, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:2].T
        evr = (s ** 2) / (s ** 2).sum()
        return coords, {"method": "pca", "evr_2d": evr[:2].tolist()}
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=2)
        coords = lda.fit_transform(feats, owner)
        evr = getattr(lda, "explained_variance_ratio_", None)
        return coords, {"method": "lda",
                        "evr_2d": (evr[:2].tolist() if evr is not None else None)}
    raise ValueError(f"Unknown 2D projection: {method!r}")


def _draw_cov_ellipse(ax, points, color, n_std):
    """Outline the n_std covariance ellipse of a 2D point cloud."""
    from matplotlib.patches import Ellipse
    if len(points) < 3:
        return
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False,
                         edgecolor=color, lw=1.8, alpha=0.9))


def save_scatter_2d(coords, owner, present, out_path, *, title, info,
                    n_std, max_per_task, seed):
    """Per-sample 2D scatter colored by task, with centroid + covariance ellipse.

    Returns the Fisher ratio (between/within variance) of the 2D layout — a
    single number for "how separated are the tasks in this projection".
    """
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 9.5))

    # Background point cloud (subsampled per task for render speed).
    for ti in range(len(present)):
        m = np.flatnonzero(owner == ti)
        if len(m) > max_per_task:
            m = rng.choice(m, size=max_per_task, replace=False)
        ax.scatter(coords[m, 0], coords[m, 1], s=5, alpha=0.22,
                   color=cmap(ti % 20), linewidths=0)

    # Ellipses + centroids on top so structure stays visible over the cloud.
    for ti, t in enumerate(present):
        m = owner == ti
        color = cmap(ti % 20)
        _draw_cov_ellipse(ax, coords[m], color, n_std)
        c = coords[m].mean(axis=0)
        ax.scatter(c[0], c[1], s=140, marker="X", color=color,
                   edgecolors="black", linewidths=0.8, zorder=5, label=t)
        ax.annotate(t, (c[0], c[1]), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, zorder=6)

    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=1,
              markerscale=0.6, title="task", title_fontsize=7, borderpad=0.6,
              labelspacing=0.3)

    method = info["method"].upper()
    evr = info.get("evr_2d")
    evr_note = ""
    if evr:
        evr_note = f"  (axes capture {100*evr[0]:.1f}% / {100*evr[1]:.1f}% var)"
    fisher = _class_separation_score(coords, np.asarray([present[i] for i in owner], dtype=object))
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{title}\n{method} projection{evr_note}  |  Fisher ratio={fisher:.3f}  "
                 f"(ellipse = {n_std:g}σ)", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return float(fisher)


def save_csv(mat, present, order, out_path):
    tasks = [present[i] for i in order]
    m = mat[np.ix_(order, order)]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + tasks)
        for i, t in enumerate(tasks):
            w.writerow([t] + [f"{v:.6f}" for v in m[i]])


def offdiag_summary(mat: np.ndarray) -> dict:
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    return {
        "mean": float(vals.mean()), "std": float(vals.std()),
        "min": float(vals.min()), "max": float(vals.max()),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def resolve_run_dir(arg: str | None) -> Path:
    if arg is None:
        debug_root = PROJECT_ROOT / "exp" / "debug"
        candidates = sorted(p for p in debug_root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories under {debug_root}")
        run_dir = candidates[-1]
        print(f"(no --run_dir given; using latest: {run_dir.name})")
        return run_dir
    run_dir = Path(arg)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory. Defaults to latest under exp/debug/.")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--per_task", type=int, default=1500,
                        help="Stratified samples per task for centroids/covariance.")
    parser.add_argument("--space", type=str, default="latent",
                        choices=["latent", "obs", "action"],
                        help="latent = q(z|s,a) (default); obs/action = raw baselines.")
    parser.add_argument("--source", type=str, default="mean", choices=["mean", "sample"],
                        help="latent source: posterior mean (default) or a sample.")
    parser.add_argument("--top_k", type=int, default=8,
                        help="How many closest/farthest pairs to report.")
    parser.add_argument("--scatter_embed", type=str, default="pca",
                        choices=["pca", "lda", "both"],
                        help="2D projection for the per-sample scatter+ellipse view. "
                             "pca = honest geometry (default); lda = best-case separation.")
    parser.add_argument("--scatter_metric", type=str, default="all",
                        choices=["euclidean", "mahalanobis", "cosine", "all"],
                        help="Distance geometry the scatter should reflect. The features "
                             "are transformed so Euclidean = this metric before PCA/LDA. "
                             "all (default) emits one scatter per metric.")
    parser.add_argument("--ellipse_std", type=float, default=2.0,
                        help="Covariance-ellipse radius in std-devs (2σ≈86%% of a 2D Gaussian).")
    parser.add_argument("--scatter_max_per_task", type=int, default=800,
                        help="Cap points drawn per task in the scatter (centroid/ellipse use all).")
    parser.add_argument("--robocasa_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags.get("env_name", "")
    if not env_name.startswith("robocasa_"):
        raise SystemExit(
            f"env_name={env_name!r} is not robocasa_*; task labels unavailable.")

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

    stats_path = _resolve_stats_path(env_name, args.robocasa_dir)
    ranges = build_task_ranges(stats_path, dataset_len)
    idxs, owner, present = stratified_idxs(ranges, args.per_task, args.seed)
    counts = {present[ti]: int((owner == ti).sum()) for ti in range(len(present))}
    print(f"Tasks present: {len(present)}/18.  per-task counts: {counts}")
    print(f"Total transitions sampled: {len(idxs)}")

    print(f"Extracting features (space={args.space}, source={args.source}) ...")
    feats = extract_features(agent, pre_train, idxs, args.space, args.source, args.seed).astype(np.float64)
    print(f"Feature matrix: {feats.shape}")

    mats = compute_distance_matrices(feats, owner, present)
    order = order_by_family(present)

    out_dir = run_dir / "plots" / "task_distances"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "euclidean": "Euclidean distance between task centroids",
        "cosine": "Cosine distance (1 - cos sim) between task centroids",
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

    # Per-sample 2D scatter with covariance ellipses (shows spread + overlap).
    # Features are first transformed so plain Euclidean == the chosen metric,
    # then projected with PCA/LDA, so the scatter geometry reflects the metric.
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
                title=(f"Per-sample distribution per task — {metric} geometry\n"
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

    # Console digest.
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
