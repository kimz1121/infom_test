"""Robocasa analogue of script/visualize_latent_phase.py.

The cube-single phase derivation in visualize_latent_phase.py relies on
hard-coded indices (gripper=18, block_z=21) that are meaningless for the
16-d robocasa state. This script swaps the *coloring axis* but keeps the
encoder pass + t-SNE/UMAP / 3D / best-view machinery identical.

Coloring options for robocasa:
  --color_by task    : 18 atomic_seen tasks (default; the closest analogue
                       to "phase" — distinct intentions per task).
                       Task ownership is reconstructed from the
                       <basename>.stats.json sibling produced by
                       data_gen_scripts/generate_robocasa_dataset.py: tasks
                       are concatenated in ATOMIC_SEEN_18 order, so a
                       cumulative-T table maps each row index to a task.
  --color_by ep_time : 5-bin normalized time-within-episode
                       (start/early/mid/late/end). Derived from terminals;
                       no external metadata needed.
  --color_by both    : Emit plots for both axes side-by-side under the
                       same run.

Output: <run_dir>/plots/latent_robocasa/{tsne,umap}_<n>d_<axis>.png
        plus solo / 3d / best-view variants matching visualize_latent_phase.

Usage:
    python script/visualize_latent_robocasa.py --run_dir exp/debug/<run>
    python script/visualize_latent_robocasa.py --color_by ep_time
    python script/visualize_latent_robocasa.py --color_by both --n_components 3
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
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

# Keep in sync with data_gen_scripts/generate_robocasa_dataset.py.
ATOMIC_SEEN_18 = [
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
    "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
]

EP_TIME_BINS = ["start", "early", "mid", "late", "end"]
EP_TIME_COLORS = {
    "start": "tab:blue",
    "early": "tab:cyan",
    "mid":   "tab:green",
    "late":  "tab:orange",
    "end":   "tab:red",
}

# Manual mapping: 18 atomic_seen tasks → 4 motion families.
# Same motion primitives in tasks of the same family (e.g. Open/Close share
# articulated-joint manipulation), so the family axis usually separates t-SNE
# better than the 18-way task axis.
TASK_FAMILY_MAP = {
    "CloseBlenderLid": "Articulated",
    "CloseFridge": "Articulated",
    "CloseToasterOvenDoor": "Articulated",
    "OpenCabinet": "Articulated",
    "OpenDrawer": "Articulated",
    "OpenStandMixerHead": "Articulated",
    "SlideDishwasherRack": "Articulated",
    "PickPlaceCounterToCabinet": "PickPlace",
    "PickPlaceCounterToStove": "PickPlace",
    "PickPlaceDrawerToCounter": "PickPlace",
    "PickPlaceSinkToCounter": "PickPlace",
    "PickPlaceToasterToCounter": "PickPlace",
    "CoffeeSetupMug": "PickPlace",
    "TurnOffStove": "Knob",
    "TurnOnElectricKettle": "Knob",
    "TurnOnMicrowave": "Knob",
    "TurnOnSinkFaucet": "Knob",
    "NavigateKitchen": "Navigate",
}
FAMILY_ORDER = ["PickPlace", "Articulated", "Knob", "Navigate"]
FAMILY_COLORS = {
    "PickPlace": "tab:blue",
    "Articulated": "tab:orange",
    "Knob": "tab:green",
    "Navigate": "tab:red",
}


def _task_color_map() -> dict:
    """One distinct color per task using tab20."""
    cmap = plt.get_cmap("tab20")
    return {t: cmap(i % 20) for i, t in enumerate(ATOMIC_SEEN_18)}


# ---------------------------------------------------------------------------
# Label derivation.
# ---------------------------------------------------------------------------

def derive_task_labels(
    idxs: np.ndarray, stats_path: str, dataset_len: int,
) -> tuple[np.ndarray, list[str]]:
    """Map dataset row indices → task name using cumulative train_T from stats.

    The HDF5 produced by generate_robocasa_dataset.py concatenates tasks in
    ATOMIC_SEEN_18 order, so a single cumulative-T table is enough to
    recover ownership of any row.

    Robocasa loader in envs/robocasa_utils.py reads the FIRST max_size rows
    of the file, so when pretraining_size < total_T only the prefix of tasks
    is present. We clamp boundaries to dataset_len and drop tasks that
    contribute zero rows from the returned label order.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    task_T = stats["tasks"]
    boundaries = [0]
    present_tasks: list[str] = []
    for t in ATOMIC_SEEN_18:
        if t not in task_T:
            continue
        n = int(task_T[t]["train_T"])
        nxt = min(boundaries[-1] + n, dataset_len)
        if nxt > boundaries[-1]:
            present_tasks.append(t)
            boundaries.append(nxt)
        if nxt >= dataset_len:
            break
    bounds = np.asarray(boundaries[1:])  # right-exclusive task ends
    # searchsorted with 'right' gives the index of the task whose end > idx.
    owner = np.searchsorted(bounds, idxs, side="right")
    owner = np.clip(owner, 0, len(present_tasks) - 1)
    return np.array([present_tasks[i] for i in owner], dtype=object), present_tasks


def derive_task_family_labels(
    task_labels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Collapse 18 task names → 4 motion families using TASK_FAMILY_MAP."""
    fam = np.array([TASK_FAMILY_MAP.get(t, "Other") for t in task_labels], dtype=object)
    present = [f for f in FAMILY_ORDER if (fam == f).any()]
    return fam, present


def derive_action_cluster_labels(
    idxs: np.ndarray, actions_all: np.ndarray, k: int, seed: int,
) -> tuple[np.ndarray, list[str], dict]:
    """K-means on raw action vectors → unsupervised motion-primitive labels.

    Fit on up to 20k actions for speed; predict for the visualized idxs.
    Cluster ids are not semantically named — interpret post-hoc by inspecting
    each cluster's mean action.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    n_fit = min(20000, len(actions_all))
    fit_idx = rng.choice(len(actions_all), size=n_fit, replace=False)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(actions_all[fit_idx])
    cluster_ids = km.predict(actions_all[idxs])
    labels = np.array([f"a{c}" for c in cluster_ids], dtype=object)
    order = [f"a{i}" for i in range(k)]
    cmap = plt.get_cmap("tab20" if k > 10 else "tab10")
    color_map = {f"a{i}": cmap(i % cmap.N) for i in range(k)}
    # Also stash cluster centroids so caller can print interpretations.
    color_map["__centroids__"] = km.cluster_centers_  # type: ignore[assignment]
    return labels, order, color_map


def derive_gripper_labels(
    idxs: np.ndarray, actions_all: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    """Binary open/closed label from the last action dim (gripper command).

    Robocasa OSC controller convention: action[..., -1] ∈ {-1, +1}.
    """
    g = actions_all[idxs, -1]
    labels = np.where(g > 0, "open", "closed").astype(object)
    return labels, ["closed", "open"], {"closed": "tab:red", "open": "tab:blue"}


def derive_joint_labels(
    task_labels: np.ndarray, ep_time_labels: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    """Joint (task_family × ep_time) label — 4×5 = 20 cells, tab20 coloring."""
    fam = np.array([TASK_FAMILY_MAP.get(t, "Other") for t in task_labels], dtype=object)
    joint = np.array([f"{f}/{e}" for f, e in zip(fam, ep_time_labels)], dtype=object)
    order = [f"{f}/{e}" for f in FAMILY_ORDER for e in EP_TIME_BINS]
    cmap = plt.get_cmap("tab20")
    color_map = {k: cmap(i % 20) for i, k in enumerate(order)}
    return joint, order, color_map


def derive_ep_time_labels(
    idxs: np.ndarray, terminals_all: np.ndarray,
) -> np.ndarray:
    """5-bin labels for normalized time-within-episode of each idx.

    'start' = first 10% of the episode, then 25/40/65/100 cutoffs for
    'early'/'mid'/'late'/'end'. The bins are unequal-width on purpose: the
    'end' bin is short so the terminal frame doesn't get lost in 'late'.
    """
    term_locs = np.nonzero(terminals_all > 0)[0]
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    ep_of = np.searchsorted(term_locs, np.arange(len(terminals_all)))
    ep_start_of = init_locs[ep_of]
    ep_end_of = term_locs[ep_of]
    ep_len = (ep_end_of - ep_start_of + 1).astype(np.float64)
    rel = (np.arange(len(terminals_all)) - ep_start_of) / np.maximum(ep_len, 1.0)
    rel_sub = rel[idxs]
    cutoffs = np.array([0.10, 0.25, 0.40, 0.65])
    bin_idx = np.searchsorted(cutoffs, rel_sub, side="right")
    return np.array([EP_TIME_BINS[i] for i in bin_idx], dtype=object)


# ---------------------------------------------------------------------------
# Agent + dataset construction (mirrors visualize_latent_phase.py).
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


def encode_latents(agent, batch, rng):
    obs = jnp.asarray(batch["observations"])
    act = jnp.asarray(batch["actions"])
    latent_dist = agent.network.select("intention_encoder")(obs, act)
    mean = np.asarray(latent_dist.mean())
    std = np.asarray(latent_dist.stddev())
    sample = np.asarray(latent_dist.sample(seed=rng))
    return mean, std, sample


# ---------------------------------------------------------------------------
# Embedding fit + plotting (generic over label order/color).
# ---------------------------------------------------------------------------

def _fit_embedding(
    z_in: np.ndarray, *, method: str, n_components: int,
    perplexity: float, n_iter: int, n_neighbors: int, min_dist: float, seed: int,
):
    if method == "tsne":
        perp = min(perplexity, max(5.0, (len(z_in) - 1) / 3.0))
        model = TSNE(
            n_components=n_components, perplexity=perp, max_iter=n_iter,
            init="pca", learning_rate="auto", random_state=seed, metric="euclidean",
        )
        embedded = model.fit_transform(z_in)
        info = {"perplexity": float(perp), "kl_divergence": float(model.kl_divergence_)}
        tag = f"perp={perp:.1f}, kl={model.kl_divergence_:.3f}"
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. Run `pip install umap-learn`.")
        n_neigh = max(2, min(n_neighbors, len(z_in) - 1))
        model = umap.UMAP(
            n_components=n_components, n_neighbors=n_neigh, min_dist=min_dist,
            metric="euclidean", random_state=seed,
        )
        embedded = model.fit_transform(z_in)
        info = {"n_neighbors": int(n_neigh), "min_dist": float(min_dist)}
        tag = f"n_neighbors={n_neigh}, min_dist={min_dist}"
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return embedded, info, tag


VIEW_ANGLES_3D = [(20, -60), (20, 30), (20, 120), (90, -90)]


def _view_projection_2d(points_3d: np.ndarray, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    cos_a, sin_a = np.cos(azim), np.sin(azim)
    cos_e, sin_e = np.cos(elev), np.sin(elev)
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    x_screen = -sin_a * x + cos_a * y
    y_screen = -sin_e * (cos_a * x + sin_a * y) + cos_e * z
    return np.stack([x_screen, y_screen], axis=-1)


def _class_separation_score(points_2d: np.ndarray, labels: np.ndarray) -> float:
    global_mean = points_2d.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 2:
            continue
        pts_c = points_2d[m]
        mu_c = pts_c.mean(axis=0)
        between += int(m.sum()) * np.sum((mu_c - global_mean) ** 2)
        within += np.sum((pts_c - mu_c) ** 2)
    return float(between / max(within, 1e-9))


def find_best_view_3d(embedded_3d, labels, *, elev_step=10, azim_step=10):
    best_score = -np.inf
    best_view = (20, -60)
    for elev in range(-80, 81, elev_step):
        for azim in range(0, 360, azim_step):
            proj = _view_projection_2d(embedded_3d, float(elev), float(azim))
            score = _class_separation_score(proj, labels)
            if score > best_score:
                best_score = score
                best_view = (int(elev), int(azim))
    return best_view, float(best_score)


def _scatter_2d(ax, embedded, labels, label_order, color_map, title, *, axis_label):
    for c in label_order:
        m = labels == c
        if not m.any():
            continue
        ax.scatter(
            embedded[m, 0], embedded[m, 1],
            s=8, alpha=0.55, color=color_map[c],
            label=f"{c} (n={int(m.sum())})",
        )
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)


def _scatter_3d(ax, embedded, labels, label_order, color_map, title, *, axis_label, view):
    elev, azim = view
    for c in label_order:
        m = labels == c
        if not m.any():
            continue
        ax.scatter(
            embedded[m, 0], embedded[m, 1], embedded[m, 2],
            s=6, alpha=0.55, color=color_map[c],
            label=f"{c} (n={int(m.sum())})",
        )
    ax.set_xlabel(f"{axis_label} 1")
    ax.set_ylabel(f"{axis_label} 2")
    ax.set_zlabel(f"{axis_label} 3")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title}\nelev={elev}°, azim={azim}°", fontsize=9)


def _save_3d_solo(out_path, embedded, labels, label_order, color_map, *, axis_label, base_title):
    fig = plt.figure(figsize=(12, 10))
    for i, view in enumerate(VIEW_ANGLES_3D):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _scatter_3d(
            ax, embedded, labels, label_order, color_map, base_title,
            axis_label=axis_label, view=view,
        )
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _save_3d_best_solo(out_path, embedded, labels, label_order, color_map, *, axis_label, base_title):
    best, score = find_best_view_3d(embedded, labels)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_3d(
        ax, embedded, labels, label_order, color_map, base_title,
        axis_label=axis_label, view=best,
    )
    ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(
        f"auto-selected best view (max Fisher ratio): "
        f"elev={best[0]}°, azim={best[1]}°  |  score={score:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return best, score


def _save_3d_quadview_solo(out_path, embedded, labels, label_order, color_map, *, axis_label, base_title):
    best, score = find_best_view_3d(embedded, labels)
    panels = [
        (best,         f"auto-best  elev={best[0]}°, azim={best[1]}°  score={score:.3f}"),
        ((20, -60),    "elev=20°, azim=-60°"),
        ((20,  30),    "elev=20°, azim=30°"),
        ((20, 120),    "elev=20°, azim=120°"),
    ]
    fig = plt.figure(figsize=(12, 10))
    for i, (view, subtitle) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _scatter_3d(
            ax, embedded, labels, label_order, color_map, subtitle,
            axis_label=axis_label, view=view,
        )
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(base_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return best, score


def _save_pair_grid_solo(out_path, embedded, labels, label_order, color_map, *, axis_label, base_title):
    n_components = embedded.shape[1]
    pairs = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]
    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax_, (i, j) in zip(axes_flat, pairs):
        for c in label_order:
            m = labels == c
            if not m.any():
                continue
            ax_.scatter(
                embedded[m, i], embedded[m, j],
                s=8, alpha=0.55, color=color_map[c],
                label=f"{c} (n={int(m.sum())})",
            )
        ax_.set_xlabel(f"{axis_label} {i + 1}")
        ax_.set_ylabel(f"{axis_label} {j + 1}")
        ax_.grid(True, alpha=0.3)
    axes_flat[0].legend(loc="best", fontsize=7)
    for ax_ in axes_flat[len(pairs):]:
        ax_.axis("off")
    fig.suptitle(base_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_embedding(
    z_sources: dict[str, np.ndarray],
    labels: np.ndarray,
    label_order: list[str],
    color_map: dict,
    out_path: Path,
    *,
    method: str, n_components: int, max_total: int,
    perplexity: float, n_iter: int, n_neighbors: int, min_dist: float, seed: int,
) -> dict:
    """Run embedding on each z source, stratified subsample of labels."""
    rng = np.random.default_rng(seed)
    sel: list[int] = []
    per_label_counts: dict[str, int] = {}
    present = [c for c in label_order if (labels == c).any()]
    quota = max(1, max_total // max(1, len(present)))
    for c in present:
        idx_c = np.flatnonzero(labels == c)
        take = min(quota, len(idx_c))
        chosen = rng.choice(idx_c, size=take, replace=False)
        sel.extend(chosen.tolist())
        per_label_counts[c] = int(take)
    sel = np.array(sel)
    lab_sub = labels[sel]

    method_label = method.upper()
    axis_label = method_label
    per_source_info: dict[str, dict] = {}
    per_source_embedded: dict[str, np.ndarray] = {}
    per_source_solo_title: dict[str, str] = {}

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
            z_in, method=method, n_components=n_components,
            perplexity=perplexity, n_iter=n_iter,
            n_neighbors=n_neighbors, min_dist=min_dist, seed=seed,
        )
        per_source_info[name] = info
        per_source_embedded[name] = embedded
        per_source_solo_title[name] = (
            f"{method_label} of z = q(z|s,a).{name}{init_note}\n{tag}"
        )

    if n_components == 2:
        for name in z_sources:
            solo_path = out_path.with_name(f"{out_path.stem}_{name}{out_path.suffix}")
            fig, ax = plt.subplots(figsize=(8, 6.5))
            _scatter_2d(
                ax, per_source_embedded[name], lab_sub, label_order, color_map,
                per_source_solo_title[name], axis_label=axis_label,
            )
            fig.tight_layout()
            fig.savefig(solo_path, dpi=130)
            plt.close(fig)
        # Combined side-by-side.
        n_panels = len(z_sources)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6.5), squeeze=False)
        for ax, name in zip(axes[0], z_sources.keys()):
            _scatter_2d(
                ax, per_source_embedded[name], lab_sub, label_order, color_map,
                per_source_solo_title[name], axis_label=axis_label,
            )
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    elif n_components == 3:
        for name in z_sources:
            _save_3d_solo(
                out_path.with_name(f"{out_path.stem}_{name}{out_path.suffix}"),
                per_source_embedded[name], lab_sub, label_order, color_map,
                axis_label=axis_label, base_title=per_source_solo_title[name],
            )
            _save_3d_best_solo(
                out_path.with_name(f"{out_path.stem}_{name}_best{out_path.suffix}"),
                per_source_embedded[name], lab_sub, label_order, color_map,
                axis_label=axis_label, base_title=per_source_solo_title[name],
            )
            _save_3d_quadview_solo(
                out_path.with_name(f"{out_path.stem}_{name}_quad{out_path.suffix}"),
                per_source_embedded[name], lab_sub, label_order, color_map,
                axis_label=axis_label, base_title=per_source_solo_title[name],
            )
            _save_pair_grid_solo(
                out_path.with_name(
                    out_path.stem.replace(f"{n_components}d", "pairs")
                    + f"_{name}" + out_path.suffix
                ),
                per_source_embedded[name], lab_sub, label_order, color_map,
                axis_label=axis_label, base_title=per_source_solo_title[name],
            )
    elif n_components >= 4:
        for name in z_sources:
            _save_pair_grid_solo(
                out_path.with_name(
                    out_path.stem.replace(f"{n_components}d", "pairs")
                    + f"_{name}" + out_path.suffix
                ),
                per_source_embedded[name], lab_sub, label_order, color_map,
                axis_label=axis_label, base_title=per_source_solo_title[name],
            )
    else:
        raise ValueError(f"n_components must be >= 2, got {n_components}")

    return {"per_label_counts": per_label_counts, "per_source_info": per_source_info}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def _resolve_stats_path(env_name: str, robocasa_dir: str) -> str:
    """Find the .stats.json for the *pretrain* HDF5 implied by env_name."""
    from envs.robocasa_utils import parse_env_name, DEFAULT_DATASET_DIR
    pre_name, _, _ = parse_env_name(env_name)
    base = osp.expanduser(robocasa_dir or DEFAULT_DATASET_DIR)
    p = osp.join(base, f"{pre_name}.stats.json")
    if not osp.exists(p):
        raise FileNotFoundError(f"Stats file not found: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory with flags.json and params_<epoch>.pkl. "
                             "Defaults to the most recent run under exp/debug/.")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=8000)
    parser.add_argument("--tsne_samples", type=int, default=2500)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--method", type=str, default="both", choices=["tsne", "umap", "both"])
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--sources", type=str, default="mean", choices=["mean", "sample", "both"])
    parser.add_argument("--color_by", type=str, default="task",
                        choices=["task", "ep_time", "task_family", "action_cluster",
                                 "gripper", "joint", "both", "all"],
                        help="task = 18 atomic_seen tasks (default). "
                             "ep_time = 5 normalized-time bins within episode. "
                             "task_family = 4 motion families (PickPlace, Articulated, Knob, Navigate). "
                             "action_cluster = K-means clusters on actions (set k via --k_action). "
                             "gripper = open/closed binary from action last dim. "
                             "joint = task_family × ep_time (20 cells). "
                             "both = task + ep_time (backward-compatible). "
                             "all = run all six axes and report Fisher-ratio ranking.")
    parser.add_argument("--k_action", type=int, default=8,
                        help="Number of K-means clusters for --color_by action_cluster.")
    parser.add_argument("--robocasa_dir", type=str, default="",
                        help="Override ~/.robocasa/data (where the .stats.json lives).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run_dir is None:
        debug_root = PROJECT_ROOT / "exp" / "debug"
        candidates = sorted(p for p in debug_root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories under {debug_root}")
        run_dir = candidates[-1]
        print(f"(no --run_dir given; using latest: {run_dir.name})")
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    print(f"Run dir : {run_dir}")
    print(f"Epoch   : {epoch}")

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)

    env_name = flags.get("env_name", "")
    if not env_name.startswith("robocasa_"):
        print(f"[warning] env_name={env_name!r} is not robocasa_*; task labels "
              f"will be unavailable. Use script/visualize_latent_phase.py for "
              f"cube-single, or script/visualize_latent.py for env-agnostic plots.")

    print("Building agent + pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)

    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    rng_np = np.random.default_rng(args.seed)
    idxs = rng_np.integers(0, len(raw_obs), size=args.num_samples)
    batch = pre_train.sample(args.num_samples, idxs=idxs)

    rng = jax.random.PRNGKey(args.seed)
    mean, std, z_sample = encode_latents(agent, batch, rng)
    print(
        f"Encoded {args.num_samples} samples. "
        f"μ range [{mean.min():+.3f}, {mean.max():+.3f}], "
        f"σ range [{std.min():.3f}, {std.max():.3f}]"
    )

    terminals_all = np.asarray(pre_train["terminals"])

    # Resolve requested axes ("all" expands to every supported axis).
    if args.color_by == "all":
        requested = {"task", "ep_time", "task_family", "action_cluster", "gripper", "joint"}
    elif args.color_by == "both":
        requested = {"task", "ep_time"}
    else:
        requested = {args.color_by}

    # Several axes need task and/or ep_time labels as prerequisites; compute them
    # once if any consumer is requested.
    needs_task = bool(requested & {"task", "task_family", "joint"})
    needs_ep = bool(requested & {"ep_time", "joint"})
    task_labels = present_tasks = ep_labels = None
    if needs_task:
        stats_path = _resolve_stats_path(env_name, args.robocasa_dir)
        task_labels, present_tasks = derive_task_labels(idxs, stats_path, len(raw_obs))
    if needs_ep:
        ep_labels = derive_ep_time_labels(idxs, terminals_all)

    actions_all = None
    if requested & {"action_cluster", "gripper"}:
        actions_all = np.asarray(pre_train["actions"])

    # Assemble (labels, label_order, color_map) per requested coloring axis.
    color_axes: dict[str, tuple[np.ndarray, list[str], dict]] = {}
    if "task" in requested:
        color_axes["task"] = (task_labels, present_tasks, _task_color_map())
        counts = {t: int((task_labels == t).sum()) for t in present_tasks}
        print(f"Task labels — present={len(present_tasks)}/18, counts={counts}")
    if "ep_time" in requested:
        color_axes["ep_time"] = (ep_labels, EP_TIME_BINS, EP_TIME_COLORS)
        ep_counts = {b: int((ep_labels == b).sum()) for b in EP_TIME_BINS}
        print(f"Episode-time labels — counts={ep_counts}")
    if "task_family" in requested:
        fam_labels, present_families = derive_task_family_labels(task_labels)
        color_axes["task_family"] = (fam_labels, present_families, FAMILY_COLORS)
        fam_counts = {f: int((fam_labels == f).sum()) for f in present_families}
        print(f"Task-family labels — counts={fam_counts}")
    if "action_cluster" in requested:
        ac_labels, ac_order, ac_cmap = derive_action_cluster_labels(
            idxs, actions_all, args.k_action, args.seed,
        )
        centroids = ac_cmap.pop("__centroids__", None)
        color_axes["action_cluster"] = (ac_labels, ac_order, ac_cmap)
        ac_counts = {c: int((ac_labels == c).sum()) for c in ac_order}
        print(f"Action-cluster labels (k={args.k_action}) — counts={ac_counts}")
        if centroids is not None:
            print("Centroid actions (one row per cluster):")
            for i, row in enumerate(centroids):
                pretty = ", ".join(f"{v:+.2f}" for v in row)
                print(f"  a{i}: [{pretty}]")
    if "gripper" in requested:
        g_labels, g_order, g_cmap = derive_gripper_labels(idxs, actions_all)
        color_axes["gripper"] = (g_labels, g_order, g_cmap)
        g_counts = {c: int((g_labels == c).sum()) for c in g_order}
        print(f"Gripper labels — counts={g_counts}")
    if "joint" in requested:
        j_labels, j_order, j_cmap = derive_joint_labels(task_labels, ep_labels)
        color_axes["joint"] = (j_labels, j_order, j_cmap)
        j_counts = {c: int((j_labels == c).sum()) for c in j_order if (j_labels == c).any()}
        print(f"Joint (family×ep_time) labels — non-empty cells={len(j_counts)}/{len(j_order)}")

    out_dir = run_dir / "plots" / "latent_robocasa"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_z_sources = {"mean": mean, "sample": z_sample}
    if args.sources == "both":
        z_sources = all_z_sources
    else:
        z_sources = {args.sources: all_z_sources[args.sources]}
    print(f"Visualizing sources: {list(z_sources.keys())}")

    methods = ["tsne", "umap"] if args.method == "both" else [args.method]
    summary = {
        "epoch": epoch, "env_name": env_name,
        "num_samples_encoded": args.num_samples,
        "latent_dim": int(mean.shape[1]),
        "posterior_stats": {
            "mu_min": float(mean.min()), "mu_max": float(mean.max()),
            "mu_abs_mean": float(np.abs(mean).mean()),
            "sigma_min": float(std.min()), "sigma_max": float(std.max()),
            "sigma_mean": float(std.mean()),
        },
        "by_axis": {},
    }

    # Fisher ratio on the full latent space (cheap, label-only). This is the
    # structural "does the encoder separate by this label" score — independent
    # of t-SNE/UMAP. The plots are diagnostic; this number is the verdict.
    fisher_full: dict[str, dict[str, float]] = {}
    for axis_name, (labels, _, _) in color_axes.items():
        fisher_full[axis_name] = {
            src_name: float(_class_separation_score(z, labels))
            for src_name, z in z_sources.items()
        }
    summary["fisher_full_latent"] = fisher_full
    for axis_name, (labels, label_order, color_map) in color_axes.items():
        per_method_info: dict[str, dict] = {}
        samples_plotted = None
        for method in methods:
            out_path = out_dir / f"{method}_{args.n_components}d_{axis_name}.png"
            print(
                f"[{axis_name}] {method.upper()} n_components={args.n_components} "
                f"(max_total={args.tsne_samples}) ..."
            )
            info = plot_embedding(
                z_sources, labels, label_order, color_map, out_path,
                method=method, n_components=args.n_components,
                max_total=args.tsne_samples,
                perplexity=args.tsne_perplexity, n_iter=args.tsne_iter,
                n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
                seed=args.seed,
            )
            per_method_info[method] = info["per_source_info"]
            samples_plotted = info["per_label_counts"]
            print(f"  Saved → {out_path}")
        summary["by_axis"][axis_name] = {
            "samples_plotted": samples_plotted,
            "embedding_info": per_method_info,
            "label_distribution_total": {
                c: int((labels == c).sum()) for c in label_order
            },
        }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Final ranking: which label axis is most separable in the encoder's latent?
    flat = [
        (axis, src, score)
        for axis, srcs in fisher_full.items() for src, score in srcs.items()
    ]
    flat.sort(key=lambda x: -x[2])
    print("\n=== Fisher ratio ranking (between/within variance, full latent) ===")
    for axis, src, score in flat:
        print(f"  {axis:14s} ({src:6s})  {score:.4f}")
    if flat:
        top_axis, top_src, top_score = flat[0]
        print(f"\nMost separable axis: {top_axis} ({top_src})  score={top_score:.4f}")
        print(f"Open plots: {out_dir}/tsne_{args.n_components}d_{top_axis}.png "
              f"(also umap_*_{top_axis}.png)")


if __name__ == "__main__":
    main()
