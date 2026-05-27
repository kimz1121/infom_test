"""Reproduce InFOM paper Section 5.2 — t-SNE of z colored by behavior phase.

Difference vs. script/visualize_latent.py:
  Instead of comparing pretrain vs. finetune *splits* (an unrelated axis),
  this colors each point by the ground-truth manipulation phase of the
  underlying (state, action) pair: reach / grasp / transport / release.

Phase labels for cube-single (28-dim state) are derived directly from the
observation schema in ogbench/manipspace/envs/cube_env.py:618 — no
external label is required:

    idx 18 : gripper_contact  (>0.5 => holding the cube)
    idx 21 : block_z (scaled, on-table baseline ~0.2; >0.5 => lifted)

    reach     : not holding & block on table   (~54%)
    grasp     : holding     & block on table   (~11%)
    transport : holding     & block lifted     (~35%)
    release   : not holding & block lifted     (~0% — instantaneous)

Output: <run_dir>/plots/latent_phase/tsne_2d_phase.png  (+ summary.json)

Usage:
    python script/visualize_latent_phase.py --run_dir exp/debug/<run>
    python script/visualize_latent_phase.py --num_samples 6000 --tsne_samples 3000
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
from sklearn.manifold import TSNE

try:
    import umap  # umap-learn
except ImportError:
    umap = None

import jax
import jax.numpy as jnp
import ml_collections

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import agents  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Phase-label derivation (cube-single obs schema).
# ---------------------------------------------------------------------------

GRIPPER_CONTACT_IDX = 18
BLOCK_Z_IDX = 21
PHASE_ORDER = ["reach", "grasp", "transport", "release"]
PHASE_COLORS = {
    "reach":     "tab:blue",
    "grasp":     "tab:orange",
    "transport": "tab:green",
    "release":   "tab:red",
}


def derive_phase_labels(raw_obs: np.ndarray) -> np.ndarray:
    """Return a string phase label per row of un-normalized observations."""
    gc = raw_obs[:, GRIPPER_CONTACT_IDX] > 0.5
    hi = raw_obs[:, BLOCK_Z_IDX] > 0.5
    labels = np.empty(len(raw_obs), dtype=object)
    labels[~gc & ~hi] = "reach"
    labels[ gc & ~hi] = "grasp"
    labels[ gc &  hi] = "transport"
    labels[~gc &  hi] = "release"
    return labels


# ---------------------------------------------------------------------------
# Agent + dataset construction. Keeps a copy of the RAW (un-normalized)
# observations so labels can be read after normalize_observations() rewrites
# the dataset's obs in place for the encoder.
# ---------------------------------------------------------------------------

def build_agent_and_pretrain_dataset(flags: dict):
    env_name = flags["env_name"]
    pretraining_size = flags["pretraining_size"]
    obs_norm_type = flags["obs_norm_type"]
    config = ml_collections.ConfigDict(flags["agent"])

    _, _, pre_train_raw_dict, _ = make_env_and_datasets(
        env_name, frame_stack=flags.get("frame_stack"),
        max_size=pretraining_size, reward_free=True,
    )

    pre_train = Dataset.create(**pre_train_raw_dict)

    # Snapshot raw observations BEFORE normalization clobbers them in place.
    raw_obs = np.array(pre_train["observations"], copy=True)

    pre_train.obs_norm_type = obs_norm_type
    pre_train.p_aug = flags.get("p_aug")
    pre_train.num_aug = flags.get("num_aug", 1)
    pre_train.inplace_aug = flags.get("inplace_aug", 1)
    pre_train.frame_stack = flags.get("frame_stack")
    pre_train.return_next_actions = True
    pre_train.normalize_observations()

    example = pre_train.sample(1)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        flags["seed"], example["observations"], example["actions"], config
    )
    return agent, pre_train, raw_obs


# ---------------------------------------------------------------------------
# Encoder pass + t-SNE plot.
# ---------------------------------------------------------------------------

def encode_latents(agent, batch, rng):
    obs = jnp.asarray(batch["observations"])
    act = jnp.asarray(batch["actions"])
    latent_dist = agent.network.select("intention_encoder")(obs, act)
    mean = np.asarray(latent_dist.mean())
    std = np.asarray(latent_dist.stddev())
    sample = np.asarray(latent_dist.sample(seed=rng))
    return mean, std, sample


def _fit_embedding(
    z_in: np.ndarray,
    *,
    method: str,
    n_components: int,
    perplexity: float,
    n_iter: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
):
    """Fit a 2D or 3D embedding (t-SNE or UMAP) and return (embedded, info, tag)."""
    if method == "tsne":
        perp = min(perplexity, max(5.0, (len(z_in) - 1) / 3.0))
        model = TSNE(
            n_components=n_components,
            perplexity=perp,
            max_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            metric="euclidean",
        )
        embedded = model.fit_transform(z_in)
        info = {"perplexity": float(perp), "kl_divergence": float(model.kl_divergence_)}
        tag = f"perp={perp:.1f}, kl={model.kl_divergence_:.3f}"
    elif method == "umap":
        if umap is None:
            raise ImportError(
                "umap-learn is not installed. Run `pip install umap-learn`."
            )
        n_neigh = max(2, min(n_neighbors, len(z_in) - 1))
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neigh,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
        )
        embedded = model.fit_transform(z_in)
        info = {"n_neighbors": int(n_neigh), "min_dist": float(min_dist)}
        tag = f"n_neighbors={n_neigh}, min_dist={min_dist}"
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return embedded, info, tag


# 3D viewing angles (elev, azim) chosen to cover front-iso / side-iso /
# back-iso / top-down so cluster overlap from one angle can be checked
# against another.
VIEW_ANGLES_3D = [
    (20, -60),   # default iso
    (20, 30),    # rotated 90° around vertical
    (20, 120),   # rotated 180°
    (90, -90),   # top-down
]


def _scatter_phases_2d(ax, embedded, lab_sub, title, *, axis_label):
    for phase in PHASE_ORDER:
        m = lab_sub == phase
        if not m.any():
            continue
        ax.scatter(
            embedded[m, 0], embedded[m, 1],
            s=8, alpha=0.55, color=PHASE_COLORS[phase],
            label=f"{phase} (n={int(m.sum())})",
        )
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def _scatter_phases_3d(ax, embedded, lab_sub, title, *, axis_label, view):
    elev, azim = view
    for phase in PHASE_ORDER:
        m = lab_sub == phase
        if not m.any():
            continue
        ax.scatter(
            embedded[m, 0], embedded[m, 1], embedded[m, 2],
            s=6, alpha=0.55, color=PHASE_COLORS[phase],
            label=f"{phase} (n={int(m.sum())})",
        )
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_zlabel(f"{axis_label} 3")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title}\nelev={elev}°, azim={azim}°", fontsize=9)


def _save_3d_solo(out_path, embedded, lab_sub, *, axis_label, base_title):
    fig = plt.figure(figsize=(12, 10))
    for i, view in enumerate(VIEW_ANGLES_3D):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _scatter_phases_3d(
            ax, embedded, lab_sub, base_title,
            axis_label=axis_label, view=view,
        )
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _save_3d_combined(out_path, source_to_embedded, lab_sub, *, axis_label,
                      method_label, source_titles):
    n_sources = len(source_to_embedded)
    n_views = len(VIEW_ANGLES_3D)
    fig = plt.figure(figsize=(4.2 * n_views, 4.0 * n_sources))
    for r, (name, embedded) in enumerate(source_to_embedded.items()):
        for c, view in enumerate(VIEW_ANGLES_3D):
            ax = fig.add_subplot(
                n_sources, n_views, r * n_views + c + 1, projection="3d",
            )
            _scatter_phases_3d(
                ax, embedded, lab_sub, source_titles[name],
                axis_label=axis_label, view=view,
            )
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"{method_label} (3D) of intention latent z colored by manipulation phase",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_embedding_by_phase(
    z_sources: dict[str, np.ndarray],
    phase_labels: np.ndarray,
    out_path: Path,
    *,
    method: str,
    n_components: int,
    max_total: int,
    perplexity: float,
    n_iter: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> dict:
    """Run a 2-D embedding on each z source side-by-side, sharing the subsample.

    Sharing the stratified subsample across panels means each panel sees the
    same underlying (s, a) pairs — only the z representation differs, so any
    difference in phase separation is attributable to mean vs. sample, not to
    a different draw of points.
    """
    rng = np.random.default_rng(seed)

    sel: list[int] = []
    per_phase_counts: dict[str, int] = {}
    n_phases_present = sum(int((phase_labels == p).any()) for p in PHASE_ORDER)
    quota = max(1, max_total // max(1, n_phases_present))
    for phase in PHASE_ORDER:
        idx_p = np.flatnonzero(phase_labels == phase)
        if len(idx_p) == 0:
            per_phase_counts[phase] = 0
            continue
        take = min(quota, len(idx_p))
        chosen = rng.choice(idx_p, size=take, replace=False)
        sel.extend(chosen.tolist())
        per_phase_counts[phase] = int(take)
    sel = np.array(sel)
    lab_sub = phase_labels[sel]

    method_label = method.upper()
    axis_label = method_label
    per_source_info: dict[str, dict] = {}
    per_source_embedded: dict[str, np.ndarray] = {}
    per_source_panel_title: dict[str, str] = {}
    per_source_solo_title: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Compute embedding for each source. The figure assembly differs by
    # n_components (single 2D scatter vs. 4-view 3D grid), but the fit
    # and bookkeeping are identical.
    # ------------------------------------------------------------------
    for name, z_full in z_sources.items():
        z_sub = z_full[sel]

        if z_sub.shape[1] > 50:
            centered = z_sub - z_sub.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            z_in = centered @ vt[:50].T
            init_note = " (PCA-50)"
        else:
            z_in = z_sub
            init_note = ""

        embedded, info, tag = _fit_embedding(
            z_in,
            method=method,
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=seed,
        )
        per_source_info[name] = info
        per_source_embedded[name] = embedded
        per_source_panel_title[name] = f"z = q(z|s,a).{name}{init_note}\n{tag}"
        per_source_solo_title[name] = (
            f"{method_label} of z = q(z|s,a).{name} colored by manipulation"
            f" phase{init_note}\n{tag}"
        )

    # ------------------------------------------------------------------
    # Figure assembly.
    # ------------------------------------------------------------------
    if n_components == 2:
        n_panels = len(z_sources)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(7.5 * n_panels, 6), squeeze=False,
        )
        axes = axes[0]
        for ax, name in zip(axes, z_sources.keys()):
            _scatter_phases_2d(
                ax, per_source_embedded[name], lab_sub,
                per_source_panel_title[name], axis_label=axis_label,
            )
        fig.suptitle(
            f"{method_label} of intention latent z colored by manipulation phase",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

        for name in z_sources:
            solo_path = out_path.with_name(
                f"{out_path.stem}_{name}{out_path.suffix}"
            )
            fig_solo, ax_solo = plt.subplots(figsize=(7.5, 6))
            _scatter_phases_2d(
                ax_solo, per_source_embedded[name], lab_sub,
                per_source_solo_title[name], axis_label=axis_label,
            )
            fig_solo.tight_layout()
            fig_solo.savefig(solo_path, dpi=130)
            plt.close(fig_solo)
    elif n_components == 3:
        _save_3d_combined(
            out_path, per_source_embedded, lab_sub,
            axis_label=axis_label, method_label=method_label,
            source_titles=per_source_panel_title,
        )
        for name in z_sources:
            solo_path = out_path.with_name(
                f"{out_path.stem}_{name}{out_path.suffix}"
            )
            _save_3d_solo(
                solo_path, per_source_embedded[name], lab_sub,
                axis_label=axis_label,
                base_title=per_source_solo_title[name],
            )
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    return {"per_phase_counts": per_phase_counts, "per_source_info": per_source_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", type=str, default="exp/debug/sd000_20260526_155243",
        help="Run directory containing flags.json and params_<epoch>.pkl",
    )
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument(
        "--num_samples", type=int, default=8000,
        help="How many (s, a) pairs to encode from pretrain dataset.",
    )
    parser.add_argument(
        "--tsne_samples", type=int, default=2500,
        help="Total points fed to t-SNE (stratified across phases).",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["tsne", "umap", "both"],
        help="Which embedding to compute.",
    )
    parser.add_argument(
        "--n_components", type=int, default=2, choices=[2, 3],
        help="Embedding dimensionality. 3 produces a 4-view PNG grid.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    ckpts = sorted(
        run_dir.glob("params_*.pkl"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    print(f"Run dir : {run_dir}")
    print(f"Epoch   : {epoch}")

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)

    if "cube-single" not in flags.get("env_name", ""):
        print(
            f"[warning] env_name={flags.get('env_name')!r} is not cube-single — "
            f"phase-label indices (gripper=18, block_z=21) are calibrated for "
            f"cube-single only and may be meaningless here."
        )

    print("Building agent + pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)

    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    # Sample synchronized indices so raw_obs[idx] aligns with normalized batch.
    rng_np = np.random.default_rng(args.seed)
    idxs = rng_np.integers(0, len(raw_obs), size=args.num_samples)
    batch = pre_train.sample(args.num_samples, idxs=idxs)

    rng = jax.random.PRNGKey(args.seed)
    mean, std, z_sample = encode_latents(agent, batch, rng)
    phases = derive_phase_labels(raw_obs[idxs])
    phase_counts_total = {p: int((phases == p).sum()) for p in PHASE_ORDER}
    print(
        f"Encoded {args.num_samples} samples. "
        f"Phase distribution: {phase_counts_total}. "
        f"μ range [{mean.min():+.3f}, {mean.max():+.3f}], "
        f"σ range [{std.min():.3f}, {std.max():.3f}]"
    )

    out_dir = run_dir / "plots" / "latent_phase"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["tsne", "umap"] if args.method == "both" else [args.method]
    per_method_info: dict[str, dict] = {}
    samples_plotted: dict[str, int] | None = None
    for method in methods:
        out_path = out_dir / f"{method}_{args.n_components}d_phase.png"
        print(
            f"Running {method.upper()} n_components={args.n_components} "
            f"(max_total={args.tsne_samples}, "
            f"perplexity={args.tsne_perplexity}, n_iter={args.tsne_iter}, "
            f"n_neighbors={args.umap_n_neighbors}, "
            f"min_dist={args.umap_min_dist}) ..."
        )
        info = plot_embedding_by_phase(
            {"mean": mean, "sample": z_sample}, phases, out_path,
            method=method,
            n_components=args.n_components,
            max_total=args.tsne_samples,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )
        per_method_info[method] = info["per_source_info"]
        samples_plotted = info["per_phase_counts"]
        print(f"Saved → {out_path}")

    summary = {
        "epoch": epoch,
        "num_samples_encoded": args.num_samples,
        "samples_plotted": samples_plotted,
        "embedding_info": per_method_info,
        "phase_distribution_total": phase_counts_total,
        "latent_dim": int(mean.shape[1]),
        "posterior_stats": {
            "mu_min": float(mean.min()),
            "mu_max": float(mean.max()),
            "mu_abs_mean": float(np.abs(mean).mean()),
            "sigma_min": float(std.min()),
            "sigma_max": float(std.max()),
            "sigma_mean": float(std.mean()),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
