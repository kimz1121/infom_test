"""Render a composite-demo trajectory as a synced mp4 video.

Each output frame shows three things at composite timestep t:
  * top-left  : robot demo frame (decoded from LeRobot mp4)
  * top-right : UMAP embedding (atomic backdrop + growing trajectory),
                with a yellow "current position" marker at traj[t]
  * bottom    : KNN-voted skill timeline strip with a vertical "current-time"
                line at t; focused legend with present atomic tasks

Embedding is UMAP fit on atomic latents and `.transform()`ed for composite,
so the atomic backdrop is in the same coordinates as the existing UMAP
still-image plot from visualize_composite_traj.py.

Usage:
    python script/render_composite_video.py \
        --composite_name startelectrickettle_state --episode 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

try:
    import umap  # umap-learn
except ImportError:
    umap = None

from sklearn.manifold import TSNE  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402


def _pca_ref2(B):
    """Deterministic 2D reference = top-2 PCA of backdrop latents, fixed per-axis
    sign. Identical across runs/demos, so aligning every video's embedding to it
    pins the axis orientation (t-SNE/UMAP have no canonical orientation)."""
    Bc = B - B.mean(0)
    _, _, vt = np.linalg.svd(Bc, full_matrices=False)
    ref = Bc @ vt[:2].T
    for k in range(2):
        if np.mean(ref[:, k] ** 3) < 0:
            ref[:, k] *= -1.0
    return ref


def _align_to_ref(emb_atomic, emb_comp, ref2):
    """Orthogonal-Procrustes align backdrop to ref2 (rotation+reflection), same
    transform applied to the demo trajectory."""
    mu = emb_atomic.mean(0)
    A = emb_atomic - mu
    R, _ = orthogonal_procrustes(A, ref2 - ref2.mean(0))
    return A @ R, (emb_comp - mu) @ R

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from script.visualize_composite_traj import (  # noqa: E402
    _contiguous_runs,
    _high_contrast_color_map,
    knn_vote_labels,
    load_composite_episode,
)
from script.visualize_latent_robocasa import (  # noqa: E402
    build_agent_and_pretrain_dataset,
    derive_task_labels,
    distinct_color_map,
    encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Video discovery + decoding.
# ---------------------------------------------------------------------------

def composite_task_name_from_basename(composite_basename: str, category: str = "composite") -> str:
    """e.g. 'startelectrickettle_state' -> 'StartElectricKettle',
    'atomic_OpenDrawer_3cam' -> 'OpenDrawer'.

    Resolves against real folder names under ~/.robocasa/raw/pretrain/<category>/
    (case-insensitive) so we don't have to hardcode camel-case.
    """
    base = composite_basename
    for suffix in ("_3cam_lang", "_3cam_state", "_3cam", "_multimodal",
                   "_lang", "_state", "_image"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith(category + "_"):   # e.g. 'atomic_OpenDrawer' -> 'OpenDrawer'
        base = base[len(category) + 1:]
    raw_dir = osp.expanduser(f"~/.robocasa/raw/pretrain/{category}")
    if osp.isdir(raw_dir):
        for entry in os.listdir(raw_dir):
            if entry.lower() == base.lower():
                return entry
    return base


def find_composite_mp4(task_name: str, episode: int, camera: str,
                       category: str = "composite") -> str:
    pattern = osp.expanduser(
        f"~/.robocasa/raw/pretrain/{category}/{task_name}/*/lerobot/"
        f"videos/chunk-*/observation.images.{camera}/episode_{episode:06d}.mp4"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def decode_all_frames(mp4_path: str) -> np.ndarray:
    """Decode the full video into (T, H, W, 3) uint8."""
    import av
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    container.close()
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Pipeline (mirrors visualize_composite_traj's main but returns intermediates).
# ---------------------------------------------------------------------------

def prepare_data(args):
    """Encode atomic + composite, fit UMAP, KNN-vote. Returns a dict of arrays
    ready for the video renderer."""
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

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags.get("env_name", "")
    if not env_name.startswith("robocasa_"):
        raise ValueError(
            f"env_name={env_name!r} is not robocasa_*; video render is robocasa-only.")

    print("Building agent + atomic pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Encoded {len(a_mean)} atomic transitions; latent dim={a_mean.shape[1]}")

    stats_path = _resolve_stats_path(env_name, args.robocasa_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    color_map = distinct_color_map(present_tasks)  # sized to all present atomic tasks (up to 65)

    # Load composite episode.
    comp_hdf5 = osp.join(osp.expanduser(args.robocasa_dir), f"{args.composite_name}.hdf5")
    if not osp.exists(comp_hdf5):
        raise FileNotFoundError(comp_hdf5)
    ep = load_composite_episode(comp_hdf5, args.episode)
    T = ep["length"]
    print(f"Composite ep#{args.episode}: T={T}")

    comp_obs_norm = pre_train.normalize_observations(observations=ep["observations"]).astype(np.float32)
    comp_act = np.clip(ep["actions"], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)
    comp_dist = agent.network.select("intention_encoder")(
        jnp.asarray(comp_obs_norm), jnp.asarray(comp_act),
    )
    comp_mean = np.asarray(comp_dist.mean())

    voted, conf = knn_vote_labels(comp_mean, a_mean, atomic_labels, k=args.knn)
    print(f"Mean vote confidence: {conf.mean():.3f}")

    # UMAP: fit on stratified subsample of atomic, transform composite.
    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present_tasks)))
    for t in present_tasks:
        idx_c = np.flatnonzero(atomic_labels == t)
        take = min(quota, len(idx_c))
        sel.extend(rng_np.choice(idx_c, size=take, replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot = a_mean[sel]
    a_labels_plot = atomic_labels[sel]

    methods = args.method.split(",") if args.method != "both" else ["umap", "tsne"]
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if "umap" in methods:
        if umap is None:
            raise ImportError("umap-learn is required for --method umap.")
        print(f"Fitting UMAP on {len(a_mean_plot)} atomic latents ...")
        umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                               random_state=args.seed, metric="euclidean")
        umap_atomic = umap_model.fit_transform(a_mean_plot)
        umap_comp = umap_model.transform(comp_mean)
        embeddings["umap"] = (umap_atomic, umap_comp)

    if "tsne" in methods:
        # Fit t-SNE on the backdrop ONLY (deterministic given fixed backdrop+seed
        # -> identical layout across videos), place the demo by kNN-interpolation
        # in that layout (t-SNE has no .transform). A per-demo joint refit instead
        # silently flips/rotates the backdrop between videos.
        print(f"Fitting t-SNE on {len(a_mean_plot)} backdrop latents ...")
        perp = min(30.0, max(5.0, (len(a_mean_plot) - 1) / 3.0))
        tsne_bd = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                       init="pca", learning_rate="auto",
                       random_state=args.seed, metric="euclidean").fit_transform(a_mean_plot)
        from sklearn.neighbors import NearestNeighbors
        _, nidx = NearestNeighbors(n_neighbors=10).fit(a_mean_plot).kneighbors(comp_mean)
        embeddings["tsne"] = (tsne_bd, tsne_bd[nidx].mean(axis=1))

    # Pin every video to a shared deterministic orientation (backdrop PCA frame)
    # so embedding axes never flip/rotate between demos/runs.
    ref2 = _pca_ref2(a_mean_plot)
    for k, (ea, ec) in list(embeddings.items()):
        embeddings[k] = _align_to_ref(ea, ec, ref2)

    return dict(
        run_dir=run_dir, epoch=epoch,
        T=T, comp_mean=comp_mean,
        embeddings=embeddings,
        atomic_labels=a_labels_plot, present_tasks=present_tasks,
        voted=voted, conf=conf, color_map=color_map,
    )


# ---------------------------------------------------------------------------
# Video rendering.
# ---------------------------------------------------------------------------

def _draw_atomic_static(ax, emb, labels, label_order, color_map):
    """Atomic backdrop — same high-contrast palette as the strip, vivid enough
    that the cluster colors are unambiguously matchable to strip segments."""
    for c in label_order:
        m = labels == c
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.65,
                   color=color_map[c], edgecolors="none", zorder=1)


def _setup_timeline_strip(ax_strip, voted, label_order, color_map):
    """Render just the colored strip — segment labels go in `_draw_diagonal_labels`.

    Strip has no xlabel and no tick labels: the diagonal labels axis sits flush
    underneath (hspace=0 in the parent gridspec) and the colored connectors at
    the top of the label axis act as a visual continuation of each strip
    segment's color down to its text label.
    """
    name_to_idx = {n: i for i, n in enumerate(label_order)}
    color_list = [mcolors.to_rgb(color_map[n]) for n in label_order]
    cmap = mcolors.ListedColormap(color_list)
    series_idx = np.array([name_to_idx.get(v, 0) for v in voted], dtype=np.int32)
    img = series_idx[None, :]
    ax_strip.imshow(img, aspect="auto", cmap=cmap,
                    vmin=-0.5, vmax=len(label_order) - 0.5,
                    interpolation="nearest", zorder=1)
    ax_strip.set_yticks([])
    ax_strip.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_strip.set_xlim(-0.5, len(voted) - 0.5)
    runs = _contiguous_runs(voted)
    for _, _, r_end, _ in runs[:-1]:
        ax_strip.axvline(r_end + 0.5, color="white", lw=0.6, alpha=0.6)


def _draw_diagonal_labels(ax_lbl, voted, color_map, *, min_seg_frac=0.03,
                          rotation=30):
    """Below-strip axis: short colored tick + 30°-rotated task label per
    contiguous run ≥ min_seg_frac * T. Mirrors the "rotated tick labels"
    pattern but placed at segment centers.
    """
    T = len(voted)
    ax_lbl.set_xlim(-0.5, T - 0.5)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.axis("off")
    runs = _contiguous_runs(voted)
    min_seg_len = max(1, int(T * min_seg_frac))
    for task, r_start, r_end, seg_len in runs:
        if seg_len < min_seg_len:
            continue
        mid = (r_start + r_end) / 2.0
        # Thin black leader line pointing from the strip's segment center
        # (top edge of label axis, hspace=0) down to the rotated text.
        ax_lbl.plot([mid, mid], [1.0, 0.82], color="black",
                    lw=0.8, solid_capstyle="butt", zorder=3)
        # Rotated text. ha='right' va='top' is the canonical "long rotated
        # x-axis label" alignment — the right edge of the unrotated text sits
        # at (mid, 0.78) and rotation tilts it down-and-left.
        ax_lbl.text(mid, 0.78, task, rotation=rotation, ha="right", va="top",
                    fontsize=11, color="black", clip_on=True)


def render_video(data: dict, video_frames: np.ndarray, out_path: Path, *,
                 emb_atomic: np.ndarray, emb_comp: np.ndarray,
                 embedding_name: str,
                 fps: int = 20, title: str = "", subtitle: str = "",
                 max_T: int = 0, min_seg_frac: float = 0.03, conf=None):
    T = data["T"]
    n_frames = T if max_T <= 0 else min(T, max_T)
    if len(video_frames) < T:
        print(f"[warn] video has {len(video_frames)} frames, HDF5 T={T}. "
              f"Truncating to {min(len(video_frames), T)}.")
        n_frames = min(n_frames, len(video_frames))

    voted = data["voted"]
    present_tasks = data["present_tasks"]
    color_map = data["color_map"]
    atomic_labels = data["atomic_labels"]

    # --- figure layout ---
    # Nested gridspecs so strip + diagonal labels can be flush (hspace=0) while
    # the top row (video+embed) and bottom row (legend) keep normal spacing.
    has_conf = conf is not None
    fig = plt.figure(figsize=(15, 12.0 if has_conf else 11.0), constrained_layout=False)
    if has_conf:
        outer = fig.add_gridspec(
            4, 1, height_ratios=[7.0, 2.6, 1.3, 1.0],
            left=0.04, right=0.985, top=0.905, bottom=0.05, hspace=0.30,
        )
        leg_row = 3
    else:
        outer = fig.add_gridspec(
            3, 1, height_ratios=[7.0, 2.6, 1.0],
            left=0.04, right=0.985, top=0.905, bottom=0.05, hspace=0.18,
        )
        leg_row = 2
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.12)
    mid = outer[1].subgridspec(2, 1, height_ratios=[0.5, 2.1], hspace=0.0)
    ax_vid = fig.add_subplot(top[0, 0])
    ax_emb = fig.add_subplot(top[0, 1])
    ax_strip = fig.add_subplot(mid[0, 0])
    ax_lbl = fig.add_subplot(mid[1, 0])
    ax_conf = fig.add_subplot(outer[2]) if has_conf else None
    ax_leg = fig.add_subplot(outer[leg_row])

    # Big main title + smaller gray subtitle. Splitting prevents the single
    # long line from overflowing the figure width at fontsize=20.
    fig.suptitle(title, fontsize=20, y=0.97)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=12, color="dimgray")

    # --- static parts ---
    ax_vid.set_xticks([]); ax_vid.set_yticks([])
    ax_vid.set_title("robot demo (composite)", fontsize=16, pad=4)
    video_im = ax_vid.imshow(video_frames[0])

    _draw_atomic_static(ax_emb, emb_atomic, atomic_labels, present_tasks, color_map)
    ax_emb.set_title(f"{embedding_name} of intention-encoder latent",
                     fontsize=16, pad=4)
    ax_emb.set_xlabel(f"{embedding_name} 1", fontsize=11)
    ax_emb.set_ylabel(f"{embedding_name} 2", fontsize=11)
    ax_emb.tick_params(axis="both", labelsize=9)
    ax_emb.grid(True, alpha=0.25)
    # Fix axis limits so trajectory animation doesn't autoshift mid-video.
    all_x = np.concatenate([emb_atomic[:, 0], emb_comp[:, 0]])
    all_y = np.concatenate([emb_atomic[:, 1], emb_comp[:, 1]])
    pad_x = 0.04 * (all_x.max() - all_x.min())
    pad_y = 0.04 * (all_y.max() - all_y.min())
    ax_emb.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_emb.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    # Start marker (static, always at traj[0]).
    ax_emb.scatter(emb_comp[0, 0], emb_comp[0, 1], marker="s", s=140,
                   c="lime", edgecolors="black", linewidths=1.4, zorder=12,
                   label="start")
    # Trajectory as a growing scatter of dots — no connecting lines (the
    # encoder may map adjacent timesteps to far-apart 2D coords; a line would
    # falsely imply traversal of the gap). Use magma cmap (black → magenta →
    # orange → white) with a black outline so the trajectory is clearly distinct
    # from the atomic backdrop palette regardless of cluster color.
    traj_scat = ax_emb.scatter(
        emb_comp[:1, 0], emb_comp[:1, 1],
        c=np.array([0.0], dtype=float),
        cmap="magma", vmin=0, vmax=max(n_frames - 1, 1),
        s=30, alpha=1.0,
        edgecolors="black", linewidths=0.5,
        zorder=8,
    )
    # Current-position marker (yellow circle, on top of everything).
    cur_pt = ax_emb.scatter([emb_comp[0, 0]], [emb_comp[0, 1]],
                            s=220, c="yellow", edgecolors="black",
                            linewidths=1.6, zorder=13, label="now")
    ax_emb.legend(loc="upper right", fontsize=12, framealpha=0.85)

    _setup_timeline_strip(ax_strip, voted, present_tasks, color_map)
    ax_strip.set_title("KNN-voted atomic task per composite timestep",
                       fontsize=16, pad=4)
    # Current-time vertical line on the strip (white over black for visibility).
    strip_vline = ax_strip.axvline(0, color="white", lw=2.4, zorder=10)
    strip_vline2 = ax_strip.axvline(0, color="black", lw=0.9, zorder=11)

    _draw_diagonal_labels(ax_lbl, voted, color_map, min_seg_frac=min_seg_frac)

    # Optional: KNN-vote-confidence curve over time (frac of k neighbours agreeing).
    conf_vline = conf_dot = None
    if has_conf:
        conf = np.asarray(conf, dtype=float)
        tt = np.arange(len(conf))
        ax_conf.plot(tt, conf, color="tab:purple", lw=1.6, zorder=5)
        ax_conf.fill_between(tt, 0, conf, color="tab:purple", alpha=0.12, zorder=4)
        ax_conf.axhline(float(conf.mean()), color="gray", lw=1.0, ls="--", zorder=3,
                        label=f"mean={conf.mean():.2f}")
        ax_conf.set_xlim(-0.5, len(conf) - 0.5)
        ax_conf.set_ylim(0.0, 1.02)
        ax_conf.set_ylabel("KNN conf", fontsize=12)
        ax_conf.set_title("KNN vote confidence per timestep  (fraction of k neighbours agreeing)",
                          fontsize=14, pad=4)
        ax_conf.set_xlabel("timestep", fontsize=11)
        ax_conf.tick_params(labelsize=9)
        ax_conf.grid(True, alpha=0.25)
        ax_conf.legend(loc="lower right", fontsize=10, framealpha=0.85)
        conf_vline = ax_conf.axvline(0, color="black", lw=1.6, zorder=10)
        conf_dot = ax_conf.scatter([0], [conf[0]], s=90, c="yellow",
                                   edgecolors="black", linewidths=1.4, zorder=11)

    # Focused legend with present atomic tasks (sorted by duration). Numbers
    # carry an explicit unit ("steps") so the count isn't ambiguous.
    ax_leg.axis("off")
    present_sorted = sorted(
        [(t, int((voted == t).sum())) for t in present_tasks if (voted == t).any()],
        key=lambda x: -x[1],
    )
    if present_sorted:
        T_total = int(T)
        patches = [Patch(facecolor=color_map[t], edgecolor="black",
                         linewidth=0.4,
                         label=f"{t}  ({n} steps, {100 * n / T_total:.1f}%)")
                   for t, n in present_sorted]
        ax_leg.legend(handles=patches, loc="center",
                      ncol=min(4, len(patches)), fontsize=10, frameon=False,
                      handlelength=2.0,
                      title=f"Atomic tasks in this trajectory  (T={T_total} steps)",
                      title_fontsize=11)

    # --- animation loop ---
    # yuv420p needs even pixel dims; matplotlib's figure-to-pipe size can be
    # odd, so add a pad filter to round each dim up to the nearest even number.
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=2500,
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
    )
    print(f"Rendering {n_frames} frames @ {fps} fps → {out_path}")
    with writer.saving(fig, str(out_path), dpi=110):
        for t in tqdm(range(n_frames)):
            video_im.set_data(video_frames[t])
            traj_scat.set_offsets(emb_comp[: t + 1])
            traj_scat.set_array(np.arange(t + 1, dtype=float))
            cur_pt.set_offsets(emb_comp[t: t + 1])
            strip_vline.set_xdata([t, t])
            strip_vline2.set_xdata([t, t])
            if has_conf:
                conf_vline.set_xdata([t, t])
                conf_dot.set_offsets([[t, conf[t]]])
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--composite_name", default="startelectrickettle_state")
    parser.add_argument("--category", default="composite", choices=["composite", "atomic"],
                        help="Raw video category for the demo frames (mp4 path + task-name "
                             "resolution). 'atomic' renders a single atomic-task episode demo.")
    parser.add_argument("--robocasa_dir", default="~/.robocasa/data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="robot0_agentview_left",
                        choices=["robot0_agentview_left", "robot0_agentview_right",
                                 "robot0_eye_in_hand"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--method", default="both",
                        choices=["umap", "tsne", "both"],
                        help="Which 2D projection(s) to render. 'both' produces "
                             "two mp4s, one per method. UMAP keeps atomic-coord "
                             "stability via .transform(); t-SNE refits jointly.")
    parser.add_argument("--num_atomic", type=int, default=4000)
    parser.add_argument("--max_atomic_plot", type=int, default=2000)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--min_seg_frac", type=float, default=0.03)
    parser.add_argument("--max_T", type=int, default=0,
                        help="If >0, render only first max_T frames (smoke test).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = prepare_data(args)

    # Locate + decode the matching mp4.
    task_name = composite_task_name_from_basename(args.composite_name, args.category)
    mp4 = find_composite_mp4(task_name, args.episode, args.camera, args.category)
    print(f"Decoding video: {mp4}")
    frames = decode_all_frames(mp4)
    print(f"  {len(frames)} frames @ {frames.shape[1]}x{frames.shape[2]}")

    # Per-checkpoint subfolder so different epochs (and runs) never overwrite.
    out_dir = data["run_dir"] / "plots" / "composite_validation" / f"ep{data['epoch']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, (emb_atomic, emb_comp) in data["embeddings"].items():
        out_path = (out_dir
                    / f"composite_{args.composite_name}_ep{args.episode}"
                      f"_video_{method}.mp4")
        main_title = f"{task_name}  ep#{args.episode}  ({method.upper()})"
        subtitle = (f"T={data['T']} steps  •  epoch={data['epoch']}  •  "
                    f"knn={args.knn}  •  "
                    f"mean_conf={data['conf'].mean():.2f}  •  "
                    f"camera={args.camera}")
        render_video(data, frames, out_path,
                     emb_atomic=emb_atomic, emb_comp=emb_comp,
                     embedding_name=method.upper(),
                     fps=args.fps, title=main_title, subtitle=subtitle,
                     max_T=args.max_T, min_seg_frac=args.min_seg_frac,
                     conf=data["conf"])


if __name__ == "__main__":
    main()
