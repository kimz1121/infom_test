"""MULTIMODAL LIBERO demo video (analogue of render_libero_video.py for
precompute multimodal runs). Same synced mp4 layout (demo RGB | latent backdrop
+ growing trajectory | KNN instruction timeline), but the demo latent is encoded
from the SAME [resnet34(image) | state] (and |language) representation the
multimodal agent was trained on -- not the raw 15-d state.

Demo image frames are processed exactly like training (generate_libero_dataset:
unflipped, resized to stats.image_size, BILINEAR) before resnet, so the demo
latent lands in the backdrop's distribution. Display frames stay flipped/native.

Works for libero_*_multimodal_state (527-d) and *_multimodal_lang_state (911-d,
appends the task's SBERT embedding). Backdrop = the run's own pretrain pool.

Usage:
    python script/render_libero_video_mm.py --run_dir <multimodal libero run> \
        --task open_the_top_drawer_and_put_the_bowl_inside --demo 0
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import jax, jax.numpy as jnp

from data_gen_scripts.generate_libero_dataset import (
    resolve_suite_dir, find_task_hdf5, build_state, load_image_obs, DEFAULT_STATE_KEYS,
)
from script.visualize_composite_traj import knn_vote_labels
from script.render_composite_video import render_video
from script.render_libero_video import _libero_color_map
from script.visualize_latent_libero import (
    build_agent_and_pretrain_dataset, derive_task_labels, encode_latents, _resolve_stats_path,
)
from utils.flax_utils import restore_agent
from utils.visual_features import FrozenResNet34Extractor
from scipy.linalg import orthogonal_procrustes


def _pca_ref2(B):
    """Deterministic 2D reference frame = top-2 PCA of the backdrop latents, with
    a fixed per-axis sign (3rd-moment). Identical across runs/tasks, so every
    video can be aligned to it -> consistent embedding-axis orientation."""
    Bc = B - B.mean(0)
    _, _, vt = np.linalg.svd(Bc, full_matrices=False)
    ref = Bc @ vt[:2].T
    for k in range(2):
        if np.mean(ref[:, k] ** 3) < 0:
            ref[:, k] *= -1.0
    return ref


def _align_to_ref(emb_atomic, emb_comp, ref2):
    """Orthogonal-Procrustes align the backdrop layout (rotation+reflection) to
    ref2, applying the SAME transform to the demo trajectory. t-SNE/UMAP have no
    canonical orientation; this pins every video to the shared PCA frame."""
    mu = emb_atomic.mean(0)
    A = emb_atomic - mu
    R, _ = orthogonal_procrustes(A, ref2 - ref2.mean(0))
    return A @ R, (emb_comp - mu) @ R


def load_demo_mm(suite_dir, task, demo_idx, state_keys, camera, image_size, flip):
    """Return display frames (flipped/native) + encode images (unflipped, resized)."""
    path = find_task_hdf5(suite_dir, task)
    with h5py.File(path, "r") as f:
        data = f["data"]
        try:
            instr = json.loads(data.attrs["problem_info"])["language_instruction"]
        except Exception:
            instr = task.replace("_", " ")
        d = data[f"demo_{demo_idx}"]
        state = build_state(d["obs"], state_keys).astype(np.float32)
        actions = np.asarray(d["actions"], dtype=np.float32)
        enc_imgs = load_image_obs(d["obs"], camera, image_size)        # unflipped, resized
        cam_key = camera if camera in d["obs"] else f"{camera}_rgb"
        disp = np.asarray(d["obs"][cam_key])                            # native
    if flip:
        disp = disp[:, ::-1, :, :]
    return dict(state=state, actions=actions, enc_imgs=enc_imgs, frames=disp,
                instruction=instr, length=len(actions))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--raw_root", default="")
    ap.add_argument("--task", default="open_the_top_drawer_and_put_the_bowl_inside")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--libero_dir", default="~/.libero/data")
    ap.add_argument("--lang_json", default="~/.libero/data/libero_goal_lang_embeddings.json")
    ap.add_argument("--camera", default="agentview", choices=["agentview", "eye_in_hand"])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "both"])
    ap.add_argument("--num_atomic", type=int, default=4000)
    ap.add_argument("--max_atomic_plot", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--min_seg_frac", type=float, default=0.03)
    ap.add_argument("--max_T", type=int, default=0)
    ap.add_argument("--no_conf", action="store_true",
                    help="Omit the KNN-confidence-over-time curve panel.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]; cfg = flags["agent"]
    img_feat_dim = int(cfg.get("image_feat_dim", 512))
    assert "multimodal" in env_name, f"{env_name} is not a multimodal run; use render_libero_video.py"

    print("Building agent + libero backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    agent = restore_agent(agent, str(run_dir), epoch)
    target_dim = raw_obs.shape[1]
    is_lang = target_dim > img_feat_dim + 15  # 527 vs 911

    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Backdrop {len(a_mean)} latents, dim={a_mean.shape[1]}; obs_dim={target_dim} (lang={is_lang})")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    with open(stats_path) as f:
        stats = json.load(f)
    state_keys = stats.get("state_keys", list(DEFAULT_STATE_KEYS))
    image_size = int(stats.get("image_size", 128))

    suite_dir = resolve_suite_dir(args.suite, args.raw_root)
    ep = load_demo_mm(suite_dir, args.task, args.demo, state_keys, args.camera,
                      image_size, flip=not args.no_flip)
    T = ep["length"]
    print(f"Demo {args.task} #{args.demo}  T={T}  instr={ep['instruction']!r}")

    # Build the multimodal demo obs: [resnet34(image) | state] (+ language).
    ext = FrozenResNet34Extractor(device="cuda", batch_size=256)
    feat = ext.extract(ep["enc_imgs"][:T])                  # (T, 512)
    demo_obs = np.concatenate([feat, ep["state"][:T]], axis=-1).astype(np.float32)
    if is_lang:
        lang = json.load(open(os.path.expanduser(args.lang_json)))
        emb = np.asarray(lang["embeddings"][args.task], dtype=np.float32)
        demo_obs = np.concatenate([demo_obs, np.broadcast_to(emb, (T, emb.shape[0]))], axis=-1)
    assert demo_obs.shape[1] == target_dim, (demo_obs.shape, target_dim)

    comp_obs_norm = pre_train.normalize_observations(observations=demo_obs).astype(np.float32)
    comp_act = np.clip(ep["actions"][:T], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
    comp_mean = np.asarray(agent.network.select("intention_encoder")(
        jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())

    voted, conf = knn_vote_labels(comp_mean, a_mean, atomic_labels, k=args.knn)
    print(f"mean vote conf={conf.mean():.3f}")

    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present_tasks)))
    for t in present_tasks:
        idx_c = np.flatnonzero(atomic_labels == t)
        sel.extend(rng_np.choice(idx_c, size=min(quota, len(idx_c)), replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot, a_labels_plot = a_mean[sel], atomic_labels[sel]
    color_map = _libero_color_map()

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    embeddings = {}
    if "umap" in methods and umap is not None:
        m = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      random_state=args.seed, metric="euclidean")
        embeddings["umap"] = (m.fit_transform(a_mean_plot), m.transform(comp_mean))
    if "tsne" in methods:
        # Fit t-SNE on the backdrop ONLY (deterministic given the fixed backdrop
        # + seed -> identical layout across every video, unlike a per-demo joint
        # refit which silently flips/rotates). Place the demo by kNN interpolation
        # in that fixed backdrop layout (t-SNE has no .transform), mirroring UMAP.
        perp = min(30.0, max(5.0, (len(a_mean_plot) - 1) / 3.0))
        bd = TSNE(n_components=2, perplexity=perp, max_iter=1000, init="pca",
                  learning_rate="auto", random_state=args.seed).fit_transform(a_mean_plot)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10).fit(a_mean_plot)
        _, nidx = nn.kneighbors(comp_mean)
        demo2d = bd[nidx].mean(axis=1)
        embeddings["tsne"] = (bd, demo2d)

    # Pin every video's embedding to a shared, deterministic orientation (the
    # backdrop-latent PCA frame) so axes never flip/rotate between videos.
    ref2 = _pca_ref2(a_mean_plot)
    for k, (ea, ec) in list(embeddings.items()):
        embeddings[k] = _align_to_ref(ea, ec, ref2)

    data = dict(run_dir=run_dir, epoch=epoch, T=T, comp_mean=comp_mean,
                embeddings=embeddings, frames=ep["frames"], instruction=ep["instruction"],
                atomic_labels=a_labels_plot, present_tasks=present_tasks,
                voted=voted, conf=conf, color_map=color_map)

    out_dir = run_dir / "plots" / "latent_libero"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, (emb_a, emb_c) in embeddings.items():
        out_path = out_dir / f"demo_{args.task}_ep{args.demo}_video_{method}.mp4"
        sub = (f"\"{ep['instruction']}\"  •  T={T}  •  epoch={epoch}  •  knn={args.knn}  •  "
               f"mean_conf={conf.mean():.2f}  •  {'multimodal+lang' if is_lang else 'multimodal'}")
        render_video(data, ep["frames"], out_path, emb_atomic=emb_a, emb_comp=emb_c,
                     embedding_name=method.upper(), fps=args.fps,
                     title=f"{args.task}  demo#{args.demo}  ({method.upper()})",
                     subtitle=sub, max_T=args.max_T, min_seg_frac=args.min_seg_frac,
                     conf=(None if args.no_conf else conf))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
