"""Composite-demo trajectory overlay on the atomic-task latent map.

Encodes one composite Robocasa demo with the same intention_encoder as the
atomic pretrain scatter from `visualize_latent_robocasa.py`, then projects
both into a 2D space and overlays the composite trajectory.

For every composite timestep we also do a K-NN vote against the atomic
latents → predicted atomic-task label → a 1D "skill timeline" strip beneath
the scatter. This is the "which atomic task is the robot doing at time t"
view.

Two projections are emitted side-by-side:
  * UMAP — fit on atomic latents, .transform() applied to composite. The
    atomic backdrop matches `umap_2d_task.png` from visualize_latent_robocasa
    so trajectories are placed in the exact same coordinate frame.
  * t-SNE — refit on the concatenated (atomic ⊕ composite) set. The atomic
    cluster *structure* is preserved but absolute coordinates differ from
    `tsne_2d_task.png` (t-SNE has no .transform()).

Outputs:
  <run_dir>/plots/latent_robocasa/composite_<name>_ep<i>_{umap,tsne}.png

Usage:
    python script/visualize_composite_traj.py \
        --composite_name placeveggiesindrawer_state --episode 0
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # umap-learn
except ImportError:
    umap = None

import jax
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.visualize_latent_robocasa import (  # noqa: E402
    ATOMIC_SEEN_18,
    _task_color_map,
    build_agent_and_pretrain_dataset,
    derive_task_labels,
    encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent  # noqa: E402

# 18-color palette tuned for distinguishability *and* aesthetic restraint —
# combines seaborn 'deep' with Tol's qualitative palette. Saturation is held
# in the 50–70% range so it doesn't read as garish, while remaining clearly
# categorical when mapped onto the timeline strip.
HIGH_CONTRAST_18 = [
    "#4C72B0",  # deep blue
    "#DD8452",  # warm orange
    "#55A868",  # leaf green
    "#C44E52",  # brick red
    "#8172B3",  # dusty purple
    "#937860",  # bronze
    "#DA8BC3",  # rose pink
    "#8C8C8C",  # neutral gray
    "#CCB974",  # mustard
    "#64B5CD",  # sky cyan
    "#117733",  # forest green
    "#882255",  # wine
    "#88CCEE",  # pale blue
    "#999933",  # olive
    "#AA4499",  # plum
    "#44AA99",  # mint
    "#332288",  # indigo
    "#DDCC77",  # sand
]


def _high_contrast_color_map() -> dict:
    return {t: HIGH_CONTRAST_18[i % len(HIGH_CONTRAST_18)]
            for i, t in enumerate(ATOMIC_SEEN_18)}


def load_composite_episode(hdf5_path: str, episode_idx: int) -> dict:
    """Return raw arrays for the ep-th episode of a flat-HDF5 dataset.

    Episodes are delimited by terminals==1. Episode k spans
    [prev_term+1 .. term_k].
    """
    with h5py.File(hdf5_path, "r") as f:
        term = f["terminals"][:]
        ep_ends = np.nonzero(term > 0)[0]
        if episode_idx >= len(ep_ends):
            raise IndexError(
                f"episode {episode_idx} >= {len(ep_ends)} episodes in {hdf5_path}"
            )
        end = int(ep_ends[episode_idx])
        start = 0 if episode_idx == 0 else int(ep_ends[episode_idx - 1]) + 1
        sl = slice(start, end + 1)
        return dict(
            observations=f["observations"][sl],
            actions=f["actions"][sl],
            next_observations=f["next_observations"][sl],
            start=start, end=end, length=end - start + 1,
        )


def knn_vote_labels(
    comp_z: np.ndarray, atomic_z: np.ndarray, atomic_labels: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each comp_z row, return (voted_label, vote_confidence).

    Confidence = fraction of k neighbors that agreed with the majority.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(atomic_z)
    _, idx = nn.kneighbors(comp_z)
    neighbor_labels = atomic_labels[idx]  # (T, k) object array
    voted = np.empty(len(comp_z), dtype=object)
    conf = np.empty(len(comp_z), dtype=np.float32)
    for i in range(len(comp_z)):
        vals, counts = np.unique(neighbor_labels[i], return_counts=True)
        j = int(np.argmax(counts))
        voted[i] = vals[j]
        conf[i] = counts[j] / k
    return voted, conf


def _draw_trajectory(ax, traj_2d: np.ndarray, *, cmap_name: str = "viridis"):
    """Plot a connected-line trajectory with a time-gradient color."""
    T = len(traj_2d)
    points = traj_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=cmap_name, norm=plt.Normalize(0, max(T - 1, 1)),
        linewidth=1.8, alpha=0.9, zorder=8,
    )
    lc.set_array(np.arange(T - 1))
    line = ax.add_collection(lc)
    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker="s", s=110,
               c="lime", edgecolors="black", linewidths=1.3, zorder=10, label="start")
    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker="^", s=130,
               c="red", edgecolors="black", linewidths=1.3, zorder=10, label="end")
    return line


def _draw_atomic_scatter(ax, emb_2d: np.ndarray, labels: np.ndarray,
                         label_order: list[str], color_map: dict, *, alpha: float = 0.35):
    """Draw the atomic-task background; legend entries returned by caller."""
    for c in label_order:
        m = labels == c
        if not m.any():
            continue
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], s=6, alpha=alpha,
                   color=color_map[c], label=c, edgecolors="none")


def _contiguous_runs(voted: np.ndarray):
    """Return list of (task, start_idx, end_idx, length) for contiguous runs."""
    T = len(voted)
    if T == 0:
        return []
    runs = []
    start = 0
    for t in range(1, T):
        if voted[t] != voted[start]:
            runs.append((voted[start], start, t - 1, t - start))
            start = t
    runs.append((voted[start], start, T - 1, T - start))
    return runs


def _draw_timeline_strip(ax_strip, voted: np.ndarray, label_order: list[str],
                         color_map: dict, *, min_seg_frac: float = 0.03):
    """1D horizontal strip: each timestep colored by its KNN-voted atomic task.

    Long contiguous runs (≥ min_seg_frac * T) get an inline text label at the
    segment center; text orientation switches to vertical for narrow segments.
    """
    name_to_idx = {n: i for i, n in enumerate(label_order)}
    color_list = [mcolors.to_rgb(color_map[n]) for n in label_order]
    cmap = mcolors.ListedColormap(color_list)
    series_idx = np.array([name_to_idx.get(v, 0) for v in voted], dtype=np.int32)
    img = series_idx[None, :]
    ax_strip.imshow(img, aspect="auto", cmap=cmap,
                    vmin=-0.5, vmax=len(label_order) - 0.5,
                    interpolation="nearest")
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("composite timestep")
    ax_strip.set_xlim(-0.5, len(voted) - 0.5)

    T = len(voted)
    runs = _contiguous_runs(voted)
    # White boundary lines between contiguous runs.
    for task, r_start, r_end, _ in runs[:-1]:
        ax_strip.axvline(r_end + 0.5, color="white", lw=0.6, alpha=0.6)
    # Inline labels for long-enough segments.
    min_seg_len = max(1, int(T * min_seg_frac))
    for task, r_start, r_end, seg_len in runs:
        if seg_len < min_seg_len:
            continue
        bg = mcolors.to_rgb(color_map.get(task, (0.5, 0.5, 0.5)))
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        txt_color = "white" if lum < 0.55 else "black"
        rotation = 0 if seg_len >= T * 0.10 else 90
        mid = (r_start + r_end) / 2.0
        ax_strip.text(mid, 0, task, ha="center", va="center",
                      fontsize=6.5, color=txt_color, rotation=rotation,
                      clip_on=True)


def _save_figure(out_path: Path, emb_atomic: np.ndarray, emb_comp: np.ndarray,
                 atomic_labels: np.ndarray, voted: np.ndarray, conf: np.ndarray,
                 label_order: list[str], color_map: dict, *,
                 title: str, min_seg_frac: float = 0.03):
    """3-row layout: scatter+trajectory / timeline strip / focused strip legend.

    Strip legend lists ONLY tasks that actually appear in the voted timeline,
    sorted by descending duration — much more readable than dumping all 18
    atomic-task colors into a single big legend.
    """
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(11, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[9, 1.6, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[2, 0])

    _draw_atomic_scatter(ax, emb_atomic, atomic_labels, label_order, color_map)
    line = _draw_trajectory(ax, emb_comp)

    cb = fig.colorbar(line, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("composite timestep (start → end)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.25)
    # Scatter legend: only start/end markers (atomic task colors are explained
    # by the strip legend below).
    start_handle, end_handle = ax.get_legend_handles_labels()[0][-2:]
    ax.legend(handles=[start_handle, end_handle], loc="upper right",
              fontsize=8, framealpha=0.85)

    _draw_timeline_strip(ax_strip, voted, label_order, color_map,
                         min_seg_frac=min_seg_frac)
    ax_strip.set_title("KNN-voted atomic task per composite timestep",
                       fontsize=9, pad=4)

    # Focused strip legend: only tasks that appear, descending by duration.
    ax_leg.axis("off")
    present = sorted(
        [(t, int((voted == t).sum())) for t in label_order if (voted == t).any()],
        key=lambda x: -x[1],
    )
    if present:
        T_total = int(len(voted))
        patches = [
            Patch(facecolor=color_map[t], edgecolor="black", linewidth=0.4,
                  label=f"{t}  ({n}, {100 * n / T_total:.1f}%)")
            for t, n in present
        ]
        ax_leg.legend(handles=patches, loc="center", ncol=min(4, len(patches)),
                      fontsize=8, frameon=False, handlelength=2.0,
                      title=f"Atomic tasks in this trajectory  (sorted by duration; T={T_total})",
                      title_fontsize=9)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--composite_name", type=str,
                        default="placeveggiesindrawer_state",
                        help="HDF5 basename under ~/.robocasa/data/")
    parser.add_argument("--robocasa_dir", type=str, default="~/.robocasa/data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num_atomic", type=int, default=8000,
                        help="How many atomic transitions to encode for backdrop.")
    parser.add_argument("--max_atomic_plot", type=int, default=2500,
                        help="Subsample for plotting the atomic backdrop.")
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--palette", choices=["high_contrast", "tab20"],
                        default="high_contrast",
                        help="high_contrast (default) = 18 visually distinct colors "
                             "so the strip is legible. tab20 = match the original "
                             "visualize_latent_robocasa tab20 palette.")
    parser.add_argument("--min_seg_frac", type=float, default=0.03,
                        help="Inline-label only contiguous runs ≥ this fraction of T.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --- 1. Resolve run dir and load encoder + atomic pretrain dataset. -----
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
        raise ValueError(
            f"env_name={env_name!r} is not robocasa_*; composite-trajectory "
            f"visualization is robocasa-only.")

    print("Building agent + atomic pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    # --- 2. Sample atomic indices, encode → atomic_z. ----------------------
    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Encoded {len(a_mean)} atomic transitions; latent dim={a_mean.shape[1]}")

    stats_path = _resolve_stats_path(env_name, args.robocasa_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    color_map = (_high_contrast_color_map() if args.palette == "high_contrast"
                 else _task_color_map())
    print(f"Atomic present_tasks={present_tasks}")
    print(f"Palette: {args.palette}")

    # --- 3. Load + encode the composite episode. ---------------------------
    comp_hdf5 = osp.join(osp.expanduser(args.robocasa_dir), f"{args.composite_name}.hdf5")
    if not osp.exists(comp_hdf5):
        raise FileNotFoundError(comp_hdf5)
    ep = load_composite_episode(comp_hdf5, args.episode)
    T = ep["length"]
    print(f"Composite ep#{args.episode}: T={T} (raw rows {ep['start']}..{ep['end']})")

    # Normalize composite obs using the *atomic* normalization stats so the
    # encoder sees the same distribution it was trained on.
    comp_obs_norm = pre_train.normalize_observations(observations=ep["observations"]).astype(np.float32)
    comp_act = np.clip(ep["actions"], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)
    # Encode the whole episode in one call (T is at most ~2k; fits on CPU).
    import jax.numpy as jnp
    comp_dist = agent.network.select("intention_encoder")(
        jnp.asarray(comp_obs_norm), jnp.asarray(comp_act),
    )
    comp_mean = np.asarray(comp_dist.mean())

    # --- 4. KNN-vote: for each composite timestep, find nearest atomic task. --
    voted, conf = knn_vote_labels(comp_mean, a_mean, atomic_labels, k=args.knn)
    seg_counts = {t: int((voted == t).sum()) for t in present_tasks if (voted == t).any()}
    print(f"KNN-voted skill timeline (k={args.knn}): {seg_counts}")
    print(f"Mean vote confidence: {conf.mean():.3f}")

    # --- 5. Embeddings: UMAP transform + joint t-SNE. ----------------------
    out_dir = run_dir / "plots" / "latent_robocasa"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subsample atomic for plotting (stratified by task to keep all classes).
    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present_tasks)))
    for t in present_tasks:
        idx_c = np.flatnonzero(atomic_labels == t)
        take = min(quota, len(idx_c))
        sel.extend(rng_np.choice(idx_c, size=take, replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot = a_mean[sel]
    a_labels_plot = atomic_labels[sel]

    # 5a. UMAP
    if umap is None:
        print("[warn] umap-learn not installed — skipping UMAP plot.")
        umap_emb_atomic = umap_emb_comp = None
    else:
        print(f"Fitting UMAP on {len(a_mean_plot)} atomic latents ...")
        umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                               random_state=args.seed, metric="euclidean")
        umap_emb_atomic = umap_model.fit_transform(a_mean_plot)
        umap_emb_comp = umap_model.transform(comp_mean)
        out_path = out_dir / f"composite_{args.composite_name}_ep{args.episode}_umap.png"
        title = (f"UMAP — atomic backdrop + composite '{args.composite_name}' "
                 f"ep#{args.episode} (T={T})\n"
                 f"epoch={epoch}, knn={args.knn}, mean_conf={conf.mean():.2f}")
        _save_figure(out_path, umap_emb_atomic, umap_emb_comp,
                     a_labels_plot, voted, conf, present_tasks, color_map,
                     title=title, min_seg_frac=args.min_seg_frac)
        print(f"Saved → {out_path}")

    # 5b. Joint t-SNE: fit on stacked (atomic_plot + composite).
    print(f"Fitting joint t-SNE on {len(a_mean_plot) + T} points ...")
    z_joint = np.concatenate([a_mean_plot, comp_mean], axis=0)
    perp = min(30.0, max(5.0, (len(z_joint) - 1) / 3.0))
    tsne_model = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                      init="pca", learning_rate="auto",
                      random_state=args.seed, metric="euclidean")
    tsne_joint = tsne_model.fit_transform(z_joint)
    tsne_emb_atomic = tsne_joint[: len(a_mean_plot)]
    tsne_emb_comp = tsne_joint[len(a_mean_plot):]
    out_path = out_dir / f"composite_{args.composite_name}_ep{args.episode}_tsne.png"
    title = (f"t-SNE (joint refit) — atomic + composite '{args.composite_name}' "
             f"ep#{args.episode} (T={T})\n"
             f"epoch={epoch}, perp={perp:.1f}, kl={tsne_model.kl_divergence_:.3f}, "
             f"knn={args.knn}, mean_conf={conf.mean():.2f}")
    _save_figure(out_path, tsne_emb_atomic, tsne_emb_comp,
                 a_labels_plot, voted, conf, present_tasks, color_map,
                 title=title, min_seg_frac=args.min_seg_frac)
    print(f"Saved → {out_path}")

    # --- 6. Dump a small json summary of the skill timeline. ---------------
    runs = []
    if T > 0:
        cur = voted[0]
        run_start = 0
        for t in range(1, T):
            if voted[t] != cur:
                runs.append({"task": str(cur), "start": run_start, "end": t - 1,
                             "length": t - run_start})
                cur = voted[t]
                run_start = t
        runs.append({"task": str(cur), "start": run_start, "end": T - 1,
                     "length": T - run_start})
    summary = {
        "composite_name": args.composite_name,
        "episode": args.episode,
        "length": T,
        "knn": args.knn,
        "mean_confidence": float(conf.mean()),
        "voted_counts": seg_counts,
        "skill_runs": runs,
    }
    summary_path = out_dir / f"composite_{args.composite_name}_ep{args.episode}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {summary_path}")
    print("\nSkill timeline (contiguous runs):")
    for r in runs:
        print(f"  t={r['start']:>4}..{r['end']:<4}  ({r['length']:>4} steps)  → {r['task']}")


if __name__ == "__main__":
    main()
