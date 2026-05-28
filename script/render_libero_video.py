"""Render a LIBERO demo as a synced analysis mp4 (LIBERO analogue of
script/render_composite_video.py).

Each output frame shows, at demo timestep t:
  * top-left  : robot demo frame (read directly from the LIBERO HDF5 RGB;
                no separate mp4 to decode, unlike Robocasa)
  * top-right : UMAP / t-SNE embedding of the intention-encoder latent —
                libero_goal pretrain pool as backdrop (colored by the 10 goal
                instructions) + the demo's growing trajectory and a "now" marker
  * bottom    : KNN-voted instruction timeline strip with a current-time line

The "backdrop" is the libero_goal pretrain pool (the 10 instructions), and the
overlaid trajectory is one chosen demo. For a compound goal like
"open_the_top_drawer_and_put_the_bowl_inside" the KNN timeline tends to split
into a drawer-opening phase and a bowl-placing phase — the LIBERO version of
the atomic/composite skill decomposition.

The agent must be a libero_*_state run (latents are encoded from the 15-d
state); the displayed frames are RGB regardless.

Usage:
    python script/render_libero_video.py \
        --task open_the_top_drawer_and_put_the_bowl_inside --demo 0
    python script/render_libero_video.py --task turn_on_the_stove --demo 3 --max_T 40
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import numpy as np  # noqa: E402

from sklearn.manifold import TSNE  # noqa: E402

try:
    import umap  # umap-learn
except ImportError:
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from data_gen_scripts.generate_libero_dataset import (  # noqa: E402
    resolve_suite_dir,
    find_task_hdf5,
    build_state,
    DEFAULT_STATE_KEYS,
)
from script.visualize_composite_traj import (  # noqa: E402
    HIGH_CONTRAST_18,
    knn_vote_labels,
)
from script.render_composite_video import render_video  # noqa: E402
from script.visualize_latent_libero import (  # noqa: E402
    LIBERO_GOAL_10,
    build_agent_and_pretrain_dataset,
    derive_task_labels,
    encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent  # noqa: E402


def _libero_color_map() -> dict:
    return {t: HIGH_CONTRAST_18[i % len(HIGH_CONTRAST_18)]
            for i, t in enumerate(LIBERO_GOAL_10)}


def load_libero_demo(suite_dir: str, task: str, demo_idx: int,
                     state_keys: list[str], camera: str, flip: bool) -> dict:
    """Read one demo's state (for encoding), actions, and RGB frames (for display).

    All three are length T and index-aligned. LIBERO stores RGB with the OpenGL
    convention (rows bottom-to-top), so frames are flipped vertically for
    upright display when ``flip`` is set.
    """
    path = find_task_hdf5(suite_dir, task)
    with h5py.File(path, "r") as f:
        data = f["data"]
        try:
            instr = json.loads(data.attrs["problem_info"])["language_instruction"]
        except Exception:
            instr = task.replace("_", " ")
        name = f"demo_{demo_idx}"
        if name not in data:
            raise IndexError(f"{name} not in {path} (have {len(data.keys())} demos)")
        d = data[name]
        state = build_state(d["obs"], state_keys).astype(np.float32)
        actions = np.asarray(d["actions"], dtype=np.float32)
        cam_key = camera if camera in d["obs"] else f"{camera}_rgb"
        frames = np.asarray(d["obs"][cam_key])  # (T, H, W, 3) uint8
    if flip:
        frames = frames[:, ::-1, :, :]
    return dict(state=state, actions=actions, frames=frames,
                instruction=instr, length=len(actions))


def prepare_data(args):
    """Encode the libero_goal backdrop + the chosen demo, fit UMAP/t-SNE,
    KNN-vote. Returns a dict ready for render_composite_video.render_video."""
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
    if not env_name.startswith("libero_") or "state" not in env_name:
        raise ValueError(
            f"env_name={env_name!r} is not a libero_*_state run; this video "
            f"encodes latents from the 15-d state.")

    print("Building agent + libero_goal pretrain backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Encoded {len(a_mean)} backdrop transitions; latent dim={a_mean.shape[1]}")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    color_map = _libero_color_map()

    # State representation must match what the agent was trained on.
    with open(stats_path) as f:
        state_keys = json.load(f).get("state_keys", list(DEFAULT_STATE_KEYS))

    suite_dir = resolve_suite_dir(args.suite, args.raw_root)
    ep = load_libero_demo(suite_dir, args.task, args.demo, state_keys,
                          args.camera, flip=not args.no_flip)
    T = ep["length"]
    print(f"Demo: {args.task} #{args.demo}  instruction={ep['instruction']!r}  T={T}")

    comp_obs_norm = pre_train.normalize_observations(
        observations=ep["state"]).astype(np.float32)
    comp_act = np.clip(ep["actions"], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)
    comp_dist = agent.network.select("intention_encoder")(
        jnp.asarray(comp_obs_norm), jnp.asarray(comp_act),
    )
    comp_mean = np.asarray(comp_dist.mean())

    voted, conf = knn_vote_labels(comp_mean, a_mean, atomic_labels, k=args.knn)
    print(f"Mean vote confidence: {conf.mean():.3f}")

    # Stratified atomic subsample for the plotted backdrop.
    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present_tasks)))
    for t in present_tasks:
        idx_c = np.flatnonzero(atomic_labels == t)
        take = min(quota, len(idx_c))
        sel.extend(rng_np.choice(idx_c, size=take, replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot = a_mean[sel]
    a_labels_plot = atomic_labels[sel]

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "umap" in methods:
        if umap is None:
            raise ImportError("umap-learn is required for --method umap.")
        print(f"Fitting UMAP on {len(a_mean_plot)} backdrop latents ...")
        m = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      random_state=args.seed, metric="euclidean")
        embeddings["umap"] = (m.fit_transform(a_mean_plot), m.transform(comp_mean))
    if "tsne" in methods:
        print(f"Fitting joint t-SNE on {len(a_mean_plot) + T} latents ...")
        z_joint = np.concatenate([a_mean_plot, comp_mean], axis=0)
        perp = min(30.0, max(5.0, (len(z_joint) - 1) / 3.0))
        tj = TSNE(n_components=2, perplexity=perp, max_iter=1000, init="pca",
                  learning_rate="auto", random_state=args.seed,
                  metric="euclidean").fit_transform(z_joint)
        embeddings["tsne"] = (tj[: len(a_mean_plot)], tj[len(a_mean_plot):])

    return dict(
        run_dir=run_dir, epoch=epoch, T=T, comp_mean=comp_mean,
        embeddings=embeddings, frames=ep["frames"], instruction=ep["instruction"],
        atomic_labels=a_labels_plot, present_tasks=present_tasks,
        voted=voted, conf=conf, color_map=color_map,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--raw_root", default="")
    parser.add_argument("--task", default="open_the_top_drawer_and_put_the_bowl_inside",
                        help="Instruction stem (one of the libero_goal 10).")
    parser.add_argument("--demo", type=int, default=0)
    parser.add_argument("--libero_dir", default="~/.libero/data")
    parser.add_argument("--camera", default="agentview",
                        choices=["agentview", "eye_in_hand"])
    parser.add_argument("--no_flip", action="store_true",
                        help="Do not vertically flip frames (LIBERO RGB is "
                             "stored OpenGL bottom-to-top; flip is on by default).")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--method", default="both", choices=["umap", "tsne", "both"])
    parser.add_argument("--num_atomic", type=int, default=4000)
    parser.add_argument("--max_atomic_plot", type=int, default=2000)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--min_seg_frac", type=float, default=0.03)
    parser.add_argument("--max_T", type=int, default=0,
                        help="If >0, render only the first max_T frames (smoke test).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = prepare_data(args)

    out_dir = data["run_dir"] / "plots" / "latent_libero"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, (emb_atomic, emb_comp) in data["embeddings"].items():
        out_path = out_dir / f"demo_{args.task}_ep{args.demo}_video_{method}.mp4"
        main_title = f"{args.task}  demo#{args.demo}  ({method.upper()})"
        subtitle = (f"\"{data['instruction']}\"  •  T={data['T']} steps  •  "
                    f"epoch={data['epoch']}  •  knn={args.knn}  •  "
                    f"mean_conf={data['conf'].mean():.2f}  •  camera={args.camera}")
        render_video(data, data["frames"], out_path,
                     emb_atomic=emb_atomic, emb_comp=emb_comp,
                     embedding_name=method.upper(),
                     fps=args.fps, title=main_title, subtitle=subtitle,
                     max_T=args.max_T, min_seg_frac=args.min_seg_frac)


if __name__ == "__main__":
    main()
