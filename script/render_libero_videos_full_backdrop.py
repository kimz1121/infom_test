"""Batch-render the LIBERO demo videos using the FULL pretrain pool (all 60k
transitions) as the t-SNE/UMAP backdrop — the same embedding produced by
`render_libero_backdrop_still.py --full`.

Why a separate driver (not render_libero_video_mm.py per task):
  * The backdrop layout is identical across all demos, so we encode the pool
    and fit each embedding ONCE here, then place every demo into that shared
    layout. Running render_libero_video_mm.py 10× would refit a 60k-point t-SNE
    10 times (~15 min wasted) and is not guaranteed to land on the same layout.
  * Output filenames get an explicit suffix (default "_fullbg") so the existing
    2k-backdrop videos in plots/latent_libero/ are NEVER overwritten.

The dense 60k backdrop is drawn with small/rasterized points (backdrop_size,
backdrop_alpha) so it doesn't blob over the trajectory in the small video panel.

Usage:
    python script/render_libero_videos_full_backdrop.py \
        --run_dir exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import jax
import jax.numpy as jnp

from data_gen_scripts.generate_libero_dataset import (
    resolve_suite_dir, DEFAULT_STATE_KEYS,
)
from script.visualize_composite_traj import knn_vote_labels
from script.render_composite_video import render_video, _pca_ref2, _align_to_ref
from script.render_libero_video import _libero_color_map
from script.render_libero_video_mm import load_demo_mm
from script.visualize_latent_libero import (
    LIBERO_GOAL_10,
    build_agent_and_pretrain_dataset, derive_task_labels, encode_latents,
    _resolve_stats_path,
)
from utils.flax_utils import restore_agent
from utils.visual_features import FrozenResNet34Extractor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--raw_root", default="")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of instruction stems; default = all 10.")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--libero_dir", default="~/.libero/data")
    ap.add_argument("--lang_json",
                    default="~/.libero/data/libero_goal_lang_embeddings.json")
    ap.add_argument("--camera", default="agentview",
                    choices=["agentview", "eye_in_hand"])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "pca", "both"])
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--min_seg_frac", type=float, default=0.03)
    ap.add_argument("--max_T", type=int, default=0)
    ap.add_argument("--no_conf", action="store_true")
    ap.add_argument("--out_suffix", default="_fullbg",
                    help="Appended to each mp4 name so existing videos are kept.")
    ap.add_argument("--backdrop_size", type=float, default=3.0)
    ap.add_argument("--backdrop_alpha", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]
    cfg = flags["agent"]
    img_feat_dim = int(cfg.get("image_feat_dim", 512))
    assert "multimodal" in env_name, f"{env_name} is not a multimodal run."

    # ----- backdrop: encode ALL transitions, fit each embedding ONCE -----
    print("Building agent + FULL libero backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    agent = restore_agent(agent, str(run_dir), epoch)
    target_dim = raw_obs.shape[1]
    is_lang = target_dim > img_feat_dim + 15

    a_idxs = np.arange(len(raw_obs))            # every transition, no sampling
    a_batch = pre_train.sample(len(a_idxs), idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Backdrop {len(a_mean)} latents (FULL pool), dim={a_mean.shape[1]}")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    atomic_labels, present_tasks = derive_task_labels(a_idxs, stats_path, len(raw_obs))
    color_map = _libero_color_map()
    with open(stats_path) as f:
        stats = json.load(f)
    state_keys = stats.get("state_keys", list(DEFAULT_STATE_KEYS))
    image_size = int(stats.get("image_size", 128))

    # Plot the whole pool (a_mean_plot == a_mean).
    a_mean_plot, a_labels_plot = a_mean, atomic_labels
    ref2 = _pca_ref2(a_mean_plot)

    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    backdrops = {}   # method -> dict(bd_raw, ref2, placer)
    if "umap" in methods:
        if umap is None:
            raise ImportError("umap-learn required for --method umap.")
        print(f"Fitting UMAP on {len(a_mean_plot)} latents ...")
        m = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      random_state=args.seed, metric="euclidean")
        bd = m.fit_transform(a_mean_plot)
        backdrops["umap"] = dict(bd=bd, place=lambda z, _m=m: _m.transform(z))
    if "tsne" in methods:
        print(f"Fitting t-SNE on {len(a_mean_plot)} latents (one-time) ...")
        perp = min(30.0, max(5.0, (len(a_mean_plot) - 1) / 3.0))
        bd = TSNE(n_components=2, perplexity=perp, max_iter=1000, init="pca",
                  learning_rate="auto", random_state=args.seed,
                  metric="euclidean").fit_transform(a_mean_plot)
        nn = NearestNeighbors(n_neighbors=10).fit(a_mean_plot)
        def _tsne_place(z, _bd=bd, _nn=nn):
            _, nidx = _nn.kneighbors(z)
            return _bd[nidx].mean(axis=1)
        backdrops["tsne"] = dict(bd=bd, place=_tsne_place)
    if "pca" in methods:
        # Top-2 PCA of the latents: UNSUPERVISED LINEAR projection (no task
        # labels). Because it's linear we project the demo into the SAME basis
        # exactly (no kNN/transform). Mirrors _pca_ref2's mean-center + per-axis
        # 3rd-moment sign fix, so this matches backdrop_still_pca_full.png.
        print(f"Projecting top-2 PCA on {len(a_mean_plot)} latents ...")
        mu_pca = a_mean_plot.mean(0)
        _, _, vt = np.linalg.svd(a_mean_plot - mu_pca, full_matrices=False)
        basis = vt[:2]                                   # (2, D)
        bd = (a_mean_plot - mu_pca) @ basis.T
        signs = np.where(np.mean(bd ** 3, axis=0) < 0, -1.0, 1.0)
        bd = bd * signs
        def _pca_place(z, _mu=mu_pca, _b=basis, _s=signs):
            return ((z - _mu) @ _b.T) * _s
        # already in the PCA frame -> no _align_to_ref needed.
        backdrops["pca"] = dict(bd=bd, place=_pca_place, no_align=True)

    # ----- per-demo: encode, place into the shared backdrop, render -----
    lang_emb = None
    if is_lang:
        lang_emb = json.load(open(os.path.expanduser(args.lang_json)))
    ext = FrozenResNet34Extractor(device="cuda", batch_size=256)
    suite_dir = resolve_suite_dir(args.suite, args.raw_root)
    tasks = args.tasks if args.tasks else list(LIBERO_GOAL_10)

    out_dir = run_dir / "plots" / "latent_libero"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ti, task in enumerate(tasks):
        print(f"\n[{ti+1}/{len(tasks)}] {task}")
        ep = load_demo_mm(suite_dir, task, args.demo, state_keys, args.camera,
                          image_size, flip=not args.no_flip)
        T = ep["length"]
        feat = ext.extract(ep["enc_imgs"][:T])
        demo_obs = np.concatenate([feat, ep["state"][:T]], axis=-1).astype(np.float32)
        if is_lang:
            emb = np.asarray(lang_emb["embeddings"][task], dtype=np.float32)
            demo_obs = np.concatenate(
                [demo_obs, np.broadcast_to(emb, (T, emb.shape[0]))], axis=-1)
        assert demo_obs.shape[1] == target_dim, (demo_obs.shape, target_dim)

        comp_obs_norm = pre_train.normalize_observations(
            observations=demo_obs).astype(np.float32)
        comp_act = np.clip(ep["actions"][:T], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
        comp_mean = np.asarray(agent.network.select("intention_encoder")(
            jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())

        voted, conf = knn_vote_labels(comp_mean, a_mean, atomic_labels, k=args.knn)
        data = dict(run_dir=run_dir, epoch=epoch, T=T, comp_mean=comp_mean,
                    frames=ep["frames"], instruction=ep["instruction"],
                    atomic_labels=a_labels_plot, present_tasks=present_tasks,
                    voted=voted, conf=conf, color_map=color_map)

        for method, bk in backdrops.items():
            demo2d = bk["place"](comp_mean)
            if bk.get("no_align"):
                emb_a, emb_c = bk["bd"], demo2d        # PCA: already aligned
            else:
                emb_a, emb_c = _align_to_ref(bk["bd"], demo2d, ref2)
            out_path = out_dir / f"demo_{task}_ep{args.demo}_video_{method}{args.out_suffix}.mp4"
            sub = (f"\"{ep['instruction']}\"  •  T={T}  •  epoch={epoch}  •  "
                   f"knn={args.knn}  •  mean_conf={conf.mean():.2f}  •  "
                   f"FULL backdrop ({len(a_mean)})  •  "
                   f"{'multimodal+lang' if is_lang else 'multimodal'}")
            render_video(data, ep["frames"], out_path, emb_atomic=emb_a,
                         emb_comp=emb_c, embedding_name=method.upper(),
                         fps=args.fps,
                         title=f"{task}  demo#{args.demo}  ({method.upper()})",
                         subtitle=sub, max_T=args.max_T,
                         min_seg_frac=args.min_seg_frac,
                         conf=(None if args.no_conf else conf),
                         backdrop_size=args.backdrop_size,
                         backdrop_alpha=args.backdrop_alpha)
            print(f"  wrote {out_path.name}")


if __name__ == "__main__":
    main()
