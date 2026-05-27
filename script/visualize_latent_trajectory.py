"""Render a single demo with a synced t-SNE animation of its q(z|s,a) means.

What it produces
----------------
A 2-panel MP4 (one frame per env step of the chosen episode):

    [ env.render() at t ]  |  [ t-SNE of z = q(z|s,a).mean ]
                           |     background context : light gray
                           |     visited points 0..t : color by --color_by
                           |                           (time | phase)
                           |     current point t    : black ring

Previous trajectory points stay on the right panel — only the new point
gets added each frame, so the trail builds up over time.

t-SNE is fit ONCE on (background context ∪ full trajectory) so the layout
is fixed and the trajectory's motion is meaningful across frames.

Usage
-----
    python script/visualize_latent_trajectory.py --run_dir exp/debug/<run>
    python script/visualize_latent_trajectory.py --run_dir <run> --episode_idx 7 \
        --n_context 3000 --fps 20

Notes
-----
- Requires that the dataset retains qpos/qvel — this script loads OGBench
  with add_info=True directly (bypassing utils/env_utils.py which strips
  those keys).
- Reuses agent construction logic from visualize_latent_phase.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import ml_collections

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import agents  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Phase-label derivation — cube-single only (28-dim obs schema).
# Mirrors script/visualize_latent_phase.py.
# ---------------------------------------------------------------------------

GRIPPER_CONTACT_IDX = 18
BLOCK_Z_IDX = 21
# 'pick'  = closing-on-table grasp (followed by transport).
# 'place' = opening-prep grasp     (preceded by transport).
# 'grasp' = ambiguous grasp (neither side has transport within the window).
PHASE_ORDER = ["reach", "pick", "transport", "place", "release", "grasp"]
PHASE_COLORS = {
    "reach":     "tab:blue",
    "pick":      "tab:orange",   # closing gripper → about to lift
    "transport": "tab:green",
    "place":     "tab:purple",   # opening gripper → about to release
    "release":   "tab:red",
    "grasp":     "0.5",          # rarely used — only if ambiguous
}


def derive_phase_labels_raw(raw_obs: np.ndarray) -> np.ndarray:
    """4-class labels straight from the obs schema.

    Returns one of {reach, grasp, transport, release}. Used as a building
    block for the 5-class scheme below.
    """
    gc = raw_obs[:, GRIPPER_CONTACT_IDX] > 0.5
    hi = raw_obs[:, BLOCK_Z_IDX] > 0.5
    labels = np.empty(len(raw_obs), dtype=object)
    labels[~gc & ~hi] = "reach"
    labels[ gc & ~hi] = "grasp"
    labels[ gc &  hi] = "transport"
    labels[~gc &  hi] = "release"
    return labels


def derive_phase_labels(
    raw_obs_all: np.ndarray,
    terminals_all: np.ndarray,
    idxs: np.ndarray,
    *,
    window: int = 10,
) -> np.ndarray:
    """5-class labels for a subset of dataset indices.

    The base label is the 4-class (reach/grasp/transport/release) scheme.
    Each 'grasp' step is then split into 'pick' vs 'place' by looking at
    the nearest steps within the same episode:

        - 'pick'  : within the next  `window` in-episode steps a
                    'transport' step exists → about to lift.
        - 'place' : within the prev  `window` in-episode steps a
                    'transport' step exists → just put down.
        - 'grasp' : neither (ambiguous; rare except at episode edges).

    Args:
        raw_obs_all:  un-normalized observations for the WHOLE dataset.
                      Needed because 'idxs' references global indices and
                      the temporal window must stay inside an episode.
        terminals_all: terminals over the whole dataset (for episode bounds).
        idxs: which rows to label.
    """
    base = derive_phase_labels_raw(raw_obs_all)

    term_locs = np.nonzero(terminals_all > 0)[0]
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    ep_of = np.searchsorted(term_locs, np.arange(len(raw_obs_all)))
    ep_start_of = init_locs[ep_of]
    ep_end_of = term_locs[ep_of]

    out = base[idxs].copy()
    for j, i in enumerate(idxs):
        if base[i] != "grasp":
            continue
        lo = max(int(ep_start_of[i]), int(i) - window)
        hi = min(int(ep_end_of[i]) + 1, int(i) + 1 + window)
        future = base[i + 1: hi]
        past = base[lo: i]
        f_t = (future == "transport").any() if len(future) else False
        p_t = (past == "transport").any() if len(past) else False
        if f_t and not p_t:
            out[j] = "pick"
        elif p_t and not f_t:
            out[j] = "place"
        # else: leave as 'grasp'
    return out


# ---------------------------------------------------------------------------
# OGBench loader that keeps qpos/qvel (utils/env_utils.py strips them).
# ---------------------------------------------------------------------------

def load_ogbench_with_info(env_name_full: str, max_size: int | float = np.inf):
    """Replicate envs/ogbench_utils.make_env_and_datasets with add_info=True.

    Returns (env, dataset_dict). dataset_dict retains 'qpos' and 'qvel'.
    Reward-free path only — that matches the pretrain dataset used by
    visualize_latent_phase.py.
    """
    import gymnasium
    import ogbench  # noqa: F401  — registers gymnasium envs
    from ogbench.utils import DEFAULT_DATASET_DIR, download_datasets, load_dataset
    from ogbench.relabel_utils import relabel_dataset

    splits = env_name_full.split("-")
    if "singletask" in splits:
        pos = splits.index("singletask")
        env_name = "-".join(splits[: pos - 1] + splits[pos:])
        dataset_name = "-".join(splits[:pos] + splits[-1:])
    else:
        env_name = "-".join(splits[:-2] + splits[-1:])
        dataset_name = env_name_full

    env = gymnasium.make(env_name)
    dataset_dir = os.path.expanduser(DEFAULT_DATASET_DIR)
    download_datasets([dataset_name], dataset_dir)
    train_path = os.path.join(dataset_dir, f"{dataset_name}.npz")
    ob_dtype = np.uint8 if ("visual" in env_name or "powderworld" in env_name) else np.float32
    action_dtype = np.int32 if "powderworld" in env_name else np.float32
    train = load_dataset(
        train_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=False,
        add_info=True,
    )
    if train["observations"].shape[0] > max_size:
        for k in list(train.keys()):
            train[k] = train[k][: int(max_size)]
    if "singletask" in splits:
        relabel_dataset(env_name, env, train)
    return env, train


# ---------------------------------------------------------------------------
# Encoder pass.
# ---------------------------------------------------------------------------

def encode_means(agent, batch) -> np.ndarray:
    obs = jnp.asarray(batch["observations"])
    act = jnp.asarray(batch["actions"])
    return np.asarray(agent.network.select("intention_encoder")(obs, act).mean())


# ---------------------------------------------------------------------------
# Per-frame rendering.
# ---------------------------------------------------------------------------

def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory with flags.json and params_<epoch>.pkl. "
                             "Defaults to the most recent run under exp/debug/.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch (default: largest available).")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Which episode of the pretrain dataset to render.")
    parser.add_argument("--n_context", type=int, default=3000,
                        help="Background context points fed to t-SNE (0 = trajectory only).")
    parser.add_argument("--max_dataset_size", type=int, default=200_000,
                        help="Cap dataset size to keep memory + load time reasonable.")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--color_by", type=str, default="phase", choices=["time", "phase"],
        help="Trajectory point coloring: 'time' (viridis by step) or "
             "'phase' (reach/pick/transport/place/release; cube-single only).",
    )
    parser.add_argument(
        "--phase_window", type=int, default=10,
        help="Within-episode lookahead/lookback window (in steps) used to "
             "split 'grasp' into 'pick' vs 'place'. Larger → more grasp "
             "steps get classified, but with weaker temporal locality.",
    )
    parser.add_argument("--out_path", type=str, default=None,
                        help="Output video path. Default: <run_dir>/plots/latent_trajectory/ep<idx>_epoch<E>.mp4")
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

    ckpts = sorted(run_dir.glob("params_*.pkl"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)

    env_name = flags["env_name"]
    print(f"Run dir : {run_dir}")
    print(f"Epoch   : {epoch}")
    print(f"Env     : {env_name}")

    # ------------------------------------------------------------------
    # Dataset (with qpos/qvel) + env for rendering.
    # ------------------------------------------------------------------
    print(f"Loading dataset (max_size={args.max_dataset_size}, add_info=True) ...")
    env, raw_ds = load_ogbench_with_info(env_name, max_size=args.max_dataset_size)
    env.reset()

    qpos_all = raw_ds.get("qpos")
    qvel_all = raw_ds.get("qvel")
    if qpos_all is None or qvel_all is None:
        raise RuntimeError(
            "Dataset is missing qpos/qvel — cannot render. "
            "Is this a state-based OGBench env?"
        )

    terminals = raw_ds["terminals"]
    term_locs = np.nonzero(terminals > 0)[0]
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    n_episodes = len(term_locs)
    print(f"  -> {len(raw_ds['observations'])} steps, {n_episodes} episodes.")
    if args.episode_idx < 0 or args.episode_idx >= n_episodes:
        raise ValueError(f"episode_idx={args.episode_idx} out of range [0, {n_episodes - 1}]")
    ep_start = int(init_locs[args.episode_idx])
    ep_end = int(term_locs[args.episode_idx])
    ep_len = ep_end - ep_start + 1
    print(f"  -> episode {args.episode_idx}: steps [{ep_start}..{ep_end}] (len={ep_len})")

    # Build the Dataset wrapper for the encoder using only the keys it expects.
    obs_norm_type = flags["obs_norm_type"]
    config = ml_collections.ConfigDict(flags["agent"])
    dataset_keys = ("observations", "actions", "terminals", "next_observations",
                    "rewards", "masks")
    ds_fields = {k: raw_ds[k] for k in dataset_keys if k in raw_ds}
    ds = Dataset.create(**ds_fields)
    ds.obs_norm_type = obs_norm_type
    ds.p_aug = flags.get("p_aug")
    ds.num_aug = flags.get("num_aug", 1)
    ds.inplace_aug = flags.get("inplace_aug", 1)
    ds.frame_stack = flags.get("frame_stack")
    ds.return_next_actions = True
    ds.normalize_observations()

    # ------------------------------------------------------------------
    # Agent.
    # ------------------------------------------------------------------
    example = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        flags["seed"], example["observations"], example["actions"], config
    )
    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    # ------------------------------------------------------------------
    # Encode trajectory + context.
    # ------------------------------------------------------------------
    print(f"Encoding trajectory ({ep_len} steps) ...")
    ep_idxs = np.arange(ep_start, ep_end + 1)
    ep_batch = ds.sample(ep_len, idxs=ep_idxs)
    traj_mean = encode_means(agent, ep_batch)

    rng = np.random.default_rng(args.seed)
    if args.n_context > 0:
        print(f"Encoding {args.n_context} context points ...")
        # Sample uniformly; drop indices that fall inside the chosen episode.
        size_total = len(raw_ds["observations"])
        pool = rng.choice(size_total, size=min(args.n_context * 2, size_total), replace=False)
        pool = pool[(pool < ep_start) | (pool > ep_end)][: args.n_context]
        ctx_batch = ds.sample(len(pool), idxs=pool)
        ctx_mean = encode_means(agent, ctx_batch)
        ctx_idxs = pool
    else:
        ctx_mean = np.zeros((0, traj_mean.shape[1]), dtype=traj_mean.dtype)
        ctx_idxs = np.zeros((0,), dtype=np.int64)

    # ------------------------------------------------------------------
    # Phase labels (cube-single only). Computed off the RAW observations
    # that were preserved before normalization.
    # ------------------------------------------------------------------
    can_phase = "cube-single" in env_name
    if args.color_by == "phase" and not can_phase:
        print(f"[warning] --color_by=phase requested but env_name={env_name!r} is "
              f"not cube-single; falling back to --color_by=time.")
    use_phase = args.color_by == "phase" and can_phase
    if use_phase:
        raw_obs_all = raw_ds["observations"]
        terms_all = raw_ds["terminals"]
        traj_phases = derive_phase_labels(
            raw_obs_all, terms_all, ep_idxs, window=args.phase_window
        )
        ctx_phases = derive_phase_labels(
            raw_obs_all, terms_all, ctx_idxs, window=args.phase_window,
        ) if len(ctx_idxs) > 0 else np.zeros((0,), dtype=object)
        traj_phase_counts = {p: int((traj_phases == p).sum()) for p in PHASE_ORDER}
        print(f"  trajectory phase counts: {traj_phase_counts}")
    else:
        traj_phases = None
        ctx_phases = None

    # ------------------------------------------------------------------
    # t-SNE on combined (context + trajectory). Fit once.
    # ------------------------------------------------------------------
    combined = np.concatenate([ctx_mean, traj_mean], axis=0)
    if combined.shape[1] > 50:
        centered = combined - combined.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        tsne_in = centered @ vt[:50].T
        init_note = " (PCA-50)"
    else:
        tsne_in = combined
        init_note = ""
    perp = min(args.tsne_perplexity, max(5.0, (len(tsne_in) - 1) / 3.0))
    print(f"Fitting t-SNE (n={len(tsne_in)}, perplexity={perp:.1f}){init_note} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=args.tsne_iter,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        metric="euclidean",
    )
    embedded = tsne.fit_transform(tsne_in)
    ctx_emb = embedded[: len(ctx_mean)]
    traj_emb = embedded[len(ctx_mean):]
    print(f"  -> t-SNE done (kl_divergence={tsne.kl_divergence_:.3f})")

    # ------------------------------------------------------------------
    # Render the per-step video.
    # ------------------------------------------------------------------
    if args.out_path is None:
        out_dir = run_dir / "plots" / "latent_trajectory"
        out_dir.mkdir(parents=True, exist_ok=True)
        color_tag_short = "phase" if use_phase else "time"
        out_path = out_dir / f"ep{args.episode_idx:03d}_epoch{epoch}_{color_tag_short}.mp4"
    else:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed bounds so the right panel doesn't rescale per frame.
    pad_x = 0.05 * (embedded[:, 0].max() - embedded[:, 0].min() + 1e-9)
    pad_y = 0.05 * (embedded[:, 1].max() - embedded[:, 1].min() + 1e-9)
    xlim = (embedded[:, 0].min() - pad_x, embedded[:, 0].max() + pad_x)
    ylim = (embedded[:, 1].min() - pad_y, embedded[:, 1].max() + pad_y)

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(1, ep_len - 1))

    print(f"Rendering {ep_len} frames @ {args.fps} fps -> {out_path}")
    writer = imageio.get_writer(
        str(out_path), fps=args.fps, codec="libx264", quality=8,
        macro_block_size=None,  # avoid forced resizing
    )
    try:
        for t in range(ep_len):
            env.unwrapped.set_state(qpos_all[ep_start + t], qvel_all[ep_start + t])
            frame = env.render()

            fig, axes = plt.subplots(
                1, 2, figsize=(11.5, 5.4),
                gridspec_kw={"width_ratios": [1.0, 1.25]},
            )
            axes[0].imshow(frame)
            axes[0].set_title(f"Episode {args.episode_idx}   step {t + 1}/{ep_len}",
                              fontsize=11)
            axes[0].axis("off")

            ax = axes[1]
            if len(ctx_emb) > 0:
                if use_phase:
                    # Color context by phase too (very low alpha so the
                    # foreground trajectory remains the focus).
                    for phase in PHASE_ORDER:
                        m = ctx_phases == phase
                        if not m.any():
                            continue
                        ax.scatter(
                            ctx_emb[m, 0], ctx_emb[m, 1],
                            s=5, alpha=0.10, color=PHASE_COLORS[phase],
                            edgecolors="none",
                        )
                else:
                    ax.scatter(
                        ctx_emb[:, 0], ctx_emb[:, 1],
                        s=5, alpha=0.12, color="0.55", edgecolors="none",
                        label=f"context (n={len(ctx_emb)})",
                    )

            if use_phase:
                # Per-phase scatter so the legend lists each phase once,
                # with per-phase counts up to step t.
                visited_phases = traj_phases[: t + 1]
                for phase in PHASE_ORDER:
                    m = visited_phases == phase
                    if not m.any():
                        continue
                    pts = traj_emb[: t + 1][m]
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        s=24, color=PHASE_COLORS[phase], edgecolors="none",
                        label=f"{phase} ({int(m.sum())})",
                    )
            else:
                colors = cmap(norm(np.arange(t + 1)))
                ax.scatter(
                    traj_emb[: t + 1, 0], traj_emb[: t + 1, 1],
                    s=22, c=colors, edgecolors="none",
                )

            # Current-point ring — black so it doesn't clash with the
            # release-phase (tab:red) or viridis-yellow colors.
            ax.scatter(
                [traj_emb[t, 0]], [traj_emb[t, 1]],
                s=140, facecolors="none", edgecolors="black", linewidths=1.8,
                label="current",
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            color_tag = "color = phase" if use_phase else "color = time within episode"
            ax.set_title(f"z = q(z | s, a).mean   ({color_tag})", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

            fig.tight_layout()
            writer.append_data(fig_to_rgb(fig))
            plt.close(fig)

            if (t + 1) % 50 == 0 or t == ep_len - 1:
                print(f"  rendered {t + 1}/{ep_len}")
    finally:
        writer.close()

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
