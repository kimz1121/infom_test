"""Still-image version of the t-SNE/UMAP *backdrop* shown inside the demo
videos (render_libero_video_mm.py / render_composite_video.py).

The static figures produced by visualize_latent_libero.py use a different
palette (tab10), point style (s=8, alpha=0.55) and — crucially — a different
t-SNE layout (mean-latent subsample, no PCA-frame alignment) than the backdrop
drawn in the videos. So the two never match pixel-for-pixel.

This script reproduces the video backdrop EXACTLY as a standalone PNG: same
stratified backdrop subsample, same backdrop-only t-SNE fit, same PCA-frame
Procrustes alignment, same `_draw_atomic_static` styling and `_libero_color_map`
palette. The only deliberate difference is the legend — instead of `loc="best"`
inside the axes (which overlaps the points), it sits in a single tall column
OUTSIDE the plot, so it never covers the embedding.

No demo is loaded: the backdrop t-SNE is fit on the pretrain pool ONLY, so it is
independent of any trajectory — identical to what the videos draw underneath.

Usage:
    python script/render_libero_backdrop_still.py \
        --run_dir exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from sklearn.manifold import TSNE  # noqa: E402

try:
    import umap  # umap-learn
except ImportError:
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402

# Reuse the EXACT helpers the videos use, so the layout matches pixel-for-pixel.
from script.render_composite_video import (  # noqa: E402
    _draw_atomic_static,
    _pca_ref2,
    _align_to_ref,
)
from script.render_libero_video import _libero_color_map  # noqa: E402
from script.visualize_latent_libero import (  # noqa: E402
    build_agent_and_pretrain_dataset,
    derive_task_labels,
    encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent  # noqa: E402


def _discriminant_ref2(X, labels, sel_tasks):
    """Linear 2D projection whose axes best SEPARATE the given `sel_tasks`, while
    still showing the real (linear) distribution like PCA — no t-SNE warping.

    Steps:
      1. Fit LDA on the points of `sel_tasks` only → the 2 directions that
         maximally discriminate those 3 classes (within-class spread aware).
      2. Orthonormalize those 2 directions (QR). This is the key difference from
         a raw LDA transform: we DON'T whiten/rescale per axis, so the projection
         is a faithful orthogonal projection onto the discriminant plane — real
         distances and spread are preserved, exactly like a PCA projection onto a
         chosen 2D subspace.
      3. Orthogonally project ALL points onto that plane, then fix per-axis sign
         deterministically (same cube-mean rule as _pca_ref2)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    mask = np.isin(labels, list(sel_tasks))
    if mask.sum() < 3 or len(set(labels[mask].tolist())) < 2:
        raise ValueError("Need >=2 of the requested tasks present to fit LDA.")
    lda = LinearDiscriminantAnalysis(n_components=min(2, len(sel_tasks) - 1))
    lda.fit(X[mask], labels[mask])

    W = np.asarray(lda.scalings_)[:, :2]      # discriminant directions (d, k)
    if W.shape[1] == 1:                       # only 2 classes -> pad with PCA dir
        Bc = X - X.mean(0)
        _, _, vt = np.linalg.svd(Bc, full_matrices=False)
        W = np.column_stack([W[:, 0], vt[0]])
    Q, _ = np.linalg.qr(W)                     # orthonormal basis of the plane

    emb = (X - X.mean(0)) @ Q[:, :2]
    for k in range(2):
        if np.mean(emb[:, k] ** 3) < 0:
            emb[:, k] *= -1.0
    return emb


def compute_backdrop(args):
    """Encode the pretrain pool and fit the same backdrop embedding(s) the
    videos draw. Returns (embeddings, labels, present_tasks, color_map, meta)."""
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]

    print("Building agent + libero backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    # --- identical sampling/encoding to render_libero_video_mm.py ---
    rng_np = np.random.default_rng(args.seed)
    if args.full:
        # Encode EVERY transition in the pool (no sampling at all).
        a_idxs = np.arange(len(raw_obs))
    else:
        a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(len(a_idxs), idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Backdrop {len(a_mean)} latents, dim={a_mean.shape[1]}")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    color_map = _libero_color_map()

    if args.full:
        # Plot every encoded point — no subsample.
        sel = np.arange(len(a_mean))
    elif args.uniform:
        # Uniform random subsample: point density reflects the NATURAL task
        # frequency in the pool, with zero label-driven rebalancing. This is the
        # "pure data distribution" view (a_idxs are already uniform over rows).
        take = min(args.max_atomic_plot, len(a_mean))
        sel = rng_np.choice(len(a_mean), size=take, replace=False)
    else:
        # Stratified subsample (same quota logic as the videos): equal points
        # per task, so cluster densities are artificially balanced.
        sel = []
        quota = max(1, args.max_atomic_plot // max(1, len(present_tasks)))
        for t in present_tasks:
            idx_c = np.flatnonzero(atomic_labels == t)
            sel.extend(rng_np.choice(idx_c, size=min(quota, len(idx_c)), replace=False).tolist())
        sel = np.asarray(sel)
    a_mean_plot, a_labels_plot = a_mean[sel], atomic_labels[sel]

    # --separate_tasks short-circuits the usual methods: emit ONLY the linear
    # discriminant-plane projection optimized to separate the requested tasks.
    if args.separate_tasks:
        wanted = [t.strip() for t in args.separate_tasks.split(",") if t.strip()]
        resolved = []
        for w in wanted:
            hits = [t for t in present_tasks if t == w] or \
                   [t for t in present_tasks if w in t]
            if not hits:
                raise ValueError(f"task '{w}' not found. present: {list(present_tasks)}")
            resolved.append(hits[0])
        print(f"Discriminant axes separating: {resolved}")
        emb = _discriminant_ref2(a_mean_plot, a_labels_plot, resolved)
        meta_extra = dict(run_dir=run_dir, epoch=epoch, env_name=env_name,
                          sep_tasks=resolved)
        return {"disc": emb}, a_labels_plot, present_tasks, color_map, meta_extra

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    embeddings = {}
    if "pca" in methods:
        # Pure top-2 PCA of the latents: an UNSUPERVISED LINEAR projection (no
        # task labels — the "LDA X, PCA O" view). _pca_ref2 already returns the
        # sign-fixed top-2 PCA coords, so no separate alignment is needed.
        embeddings["pca"] = _pca_ref2(a_mean_plot)
    if "umap" in methods:
        if umap is None:
            raise ImportError("umap-learn is required for --method umap.")
        m = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      random_state=args.seed, metric="euclidean")
        embeddings["umap"] = m.fit_transform(a_mean_plot)
    if "tsne" in methods:
        perp = min(30.0, max(5.0, (len(a_mean_plot) - 1) / 3.0))
        embeddings["tsne"] = TSNE(
            n_components=2, perplexity=perp, max_iter=1000, init="pca",
            learning_rate="auto", random_state=args.seed,
            metric="euclidean").fit_transform(a_mean_plot)

    # Same deterministic PCA-frame alignment the videos apply. The demo arg to
    # _align_to_ref is just transformed alongside; pass an empty (0,2) array.
    ref2 = _pca_ref2(a_mean_plot)
    empty = np.empty((0, 2), dtype=float)
    for k, ea in list(embeddings.items()):
        if k == "pca":
            continue            # already the PCA frame; aligning is a no-op
        embeddings[k], _ = _align_to_ref(ea, empty, ref2)

    return embeddings, a_labels_plot, present_tasks, color_map, dict(
        run_dir=run_dir, epoch=epoch, env_name=env_name)


def draw_still(emb, labels, present_tasks, color_map, *, embedding_name,
               out_path, meta, legend_fontsize, dpi, mono=False):
    """Backdrop scatter identical to the video panel, with the legend moved to a
    single tall column OUTSIDE the axes so it never covers the points.

    With mono=True the task coloring is dropped entirely: every point is drawn in
    one neutral color with no legend, so only the raw t-SNE cluster geometry
    (no label hint) is visible. The 2D layout is byte-identical either way."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5) if mono else (9.0, 7.5))

    # Auto-shrink points when dense so clusters don't merge into a blob.
    n = len(emb)
    if n > 20000:
        s, alpha = 2.0, 0.30
    elif n > 8000:
        s, alpha = 4.0, 0.45
    else:
        s, alpha = 9.0, 0.65        # video-identical style for the default counts
    raster = n > 8000               # rasterize dense scatters to keep the PNG small

    if mono:
        ax.scatter(emb[:, 0], emb[:, 1], s=s, alpha=alpha,
                   color="#4C72B0", edgecolors="none", zorder=1, rasterized=raster)
    elif n <= 8000:
        # Exact same backdrop draw call the videos use.
        _draw_atomic_static(ax, emb, labels, present_tasks, color_map)
    else:
        # Dense colored: same per-task split, but shrunk/rasterized.
        for c in present_tasks:
            m = labels == c
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=s, alpha=alpha,
                           color=color_map[c], edgecolors="none", zorder=1,
                           rasterized=raster)

    ax.set_title(f"{embedding_name} of intention-encoder latent"
                 + ("  (no task coloring)" if mono else ""), fontsize=16, pad=6)
    ax.set_xlabel(f"{embedding_name} 1", fontsize=12)
    ax.set_ylabel(f"{embedding_name} 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.25)
    pad_x = 0.04 * (emb[:, 0].max() - emb[:, 0].min())
    pad_y = 0.04 * (emb[:, 1].max() - emb[:, 1].min())
    ax.set_xlim(emb[:, 0].min() - pad_x, emb[:, 0].max() + pad_x)
    ax.set_ylim(emb[:, 1].min() - pad_y, emb[:, 1].max() + pad_y)

    if mono:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")
        return

    # Legend: one entry per task, single tall column, anchored to the right of
    # the axes (outside). Matches the backdrop dot style (markeredgecolor none).
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=color_map[t], markeredgecolor="none",
               label=f"{t}  (n={int((labels == t).sum())})")
        for t in present_tasks
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              ncol=1, fontsize=legend_fontsize, frameon=False,
              title=f"LIBERO-Goal tasks  (epoch={meta['epoch']})",
              title_fontsize=legend_fontsize + 1, borderaxespad=0.0)

    # bbox_inches="tight" expands the canvas to include the outside legend, so
    # nothing is clipped and nothing overlaps the embedding.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def draw_highlight(emb, labels, present_tasks, color_map, *, highlight_tasks,
                   embedding_name, out_path, meta, legend_fontsize, dpi):
    """Like the mono backdrop, but ONE OR MORE tasks are drawn in color on top of
    an all-gray background. Every other point is neutral gray so only the chosen
    task(s) pop out. The 2D layout is byte-identical to the mono/colored versions."""
    if isinstance(highlight_tasks, str):
        highlight_tasks = [highlight_tasks]
    fig, ax = plt.subplots(figsize=(9.0, 7.5))

    # Same auto-shrink schedule as draw_still so the geometry matches exactly.
    n = len(emb)
    if n > 20000:
        s, alpha = 2.0, 0.30
    elif n > 8000:
        s, alpha = 4.0, 0.45
    else:
        s, alpha = 9.0, 0.65
    raster = n > 8000

    hi = np.isin(labels, highlight_tasks)
    # Gray background: every non-highlighted point, drawn first/underneath.
    ax.scatter(emb[~hi, 0], emb[~hi, 1], s=s, alpha=min(alpha, 0.35),
               color="#cfcfcf", edgecolors="none", zorder=1, rasterized=raster)
    # Highlighted task(s) on top, each in its own color.
    for t in highlight_tasks:
        m = labels == t
        ax.scatter(emb[m, 0], emb[m, 1], s=s, alpha=alpha,
                   color=color_map[t], edgecolors="none", zorder=3,
                   rasterized=raster)

    title_tasks = highlight_tasks[0] if len(highlight_tasks) == 1 else \
        f"{len(highlight_tasks)} tasks"
    ax.set_title(f"{embedding_name} of intention-encoder latent  "
                 f"(highlight: {title_tasks})", fontsize=15, pad=6)
    ax.set_xlabel(f"{embedding_name} 1", fontsize=12)
    ax.set_ylabel(f"{embedding_name} 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.25)
    pad_x = 0.04 * (emb[:, 0].max() - emb[:, 0].min())
    pad_y = 0.04 * (emb[:, 1].max() - emb[:, 1].min())
    ax.set_xlim(emb[:, 0].min() - pad_x, emb[:, 0].max() + pad_x)
    ax.set_ylim(emb[:, 1].min() - pad_y, emb[:, 1].max() + pad_y)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=color_map[t], markeredgecolor="none",
               label=f"{t}  (n={int((labels == t).sum())})")
        for t in highlight_tasks
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              ncol=1, fontsize=legend_fontsize, frameon=False,
              title=f"LIBERO-Goal task  (epoch={meta['epoch']})",
              title_fontsize=legend_fontsize + 1, borderaxespad=0.0)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run_dir",
        default="exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "pca", "both"])
    ap.add_argument("--libero_dir", default="~/.libero/data")
    ap.add_argument("--num_atomic", type=int, default=4000)
    ap.add_argument("--max_atomic_plot", type=int, default=2000)
    ap.add_argument("--legend_fontsize", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--mono", action="store_true",
                    help="Drop task coloring: single-color points, no legend "
                         "(raw t-SNE geometry only). Layout is identical.")
    ap.add_argument("--separate_tasks", default=None,
                    help="Comma-separated task names (or substrings). Build LINEAR "
                         "discriminant axes that best separate exactly these tasks, "
                         "then orthogonally project all points (PCA-like real "
                         "distribution, no t-SNE warping). Overrides --method.")
    ap.add_argument("--highlight_each", action="store_true",
                    help="Emit one image PER task: all points gray except that "
                         "single task, which is drawn in its color. Layout is "
                         "identical to --mono.")
    ap.add_argument("--uniform", action="store_true",
                    help="Uniform random subsample instead of per-task stratified "
                         "(point density = natural task frequency, no label bias).")
    ap.add_argument("--full", action="store_true",
                    help="Encode AND plot every transition in the pool (no "
                         "sampling). Slow t-SNE + dense; points auto-shrink.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    embeddings, labels, present_tasks, color_map, meta = compute_backdrop(args)

    out_dir = meta["run_dir"] / "plots" / "latent_libero"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, emb in embeddings.items():
        base = (("_full" if args.full else "")
                + ("_uniform" if args.uniform and not args.full else ""))

        name = "Discriminant" if method == "disc" else method.upper()

        # --separate_tasks default view: the requested tasks colored together on
        # a gray backdrop, so their separation along the discriminant axes shows.
        if method == "disc" and not args.highlight_each and not args.mono:
            sep = meta["sep_tasks"]
            tag = "_".join(t.replace(" ", "_").split("_")[-1] for t in sep) or "sep"
            out_path = out_dir / f"backdrop_still_{method}{base}_sep_{tag}.png"
            draw_highlight(emb, labels, present_tasks, color_map,
                           highlight_tasks=sep, embedding_name=name,
                           out_path=out_path, meta=meta,
                           legend_fontsize=args.legend_fontsize, dpi=args.dpi)
            continue

        if args.highlight_each:
            # One image per task: gray backdrop + that task in color.
            for t in present_tasks:
                safe = "".join(c if c.isalnum() else "_" for c in str(t)).strip("_")
                out_path = out_dir / f"backdrop_still_{method}{base}_hl_{safe}.png"
                draw_highlight(emb, labels, present_tasks, color_map,
                               highlight_tasks=t, embedding_name=name,
                               out_path=out_path, meta=meta,
                               legend_fontsize=args.legend_fontsize, dpi=args.dpi)
            continue

        suffix = base + ("_mono" if args.mono else "")
        out_path = out_dir / f"backdrop_still_{method}{suffix}.png"
        draw_still(emb, labels, present_tasks, color_map,
                   embedding_name=name, out_path=out_path, meta=meta,
                   legend_fontsize=args.legend_fontsize, dpi=args.dpi,
                   mono=args.mono)


if __name__ == "__main__":
    main()
