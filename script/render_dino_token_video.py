"""Render a DINOv3-attnpool inFOM trajectory as a synced mp4 (action segmentation).

Self-contained for the token run (the token dataset stores no raw frames): for a
chosen atomic (task, episode) we decode the raw mp4, run the frozen DINOv3
extractor + trained intention encoder to get a per-frame latent z_t, and render:
  * left   : robot demo frame
  * right  : UMAP of intention-encoder latents (atomic backdrop, colored by KMeans
             cluster) + this episode's growing trajectory + current marker
  * bottom : per-frame KMeans-cluster timeline strip (post-hoc "skill"/action ids)

The cluster-then-label philosophy: clusters are unnamed action groups (C0..Ck-1);
the point is to SEE whether the DINOv3 embedding segments the trajectory into
coherent action phases. Reuses render_video/strip/backdrop from
render_composite_video.py. Writes to run_dir/plots/videos/epoch_<E>/ (no clobber).

Usage:
    python script/render_dino_token_video.py --run_dir exp/dinov3_attnpool/<run> \
        --epoch 40000 --task PickPlaceCounterToCabinet --episode 0 --n_clusters 8
"""
from __future__ import annotations
import argparse, json, os, os.path as osp, sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import jax
import umap
from sklearn.cluster import KMeans

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_gen_scripts.generate_robocasa_dataset import (
    find_task_lerobot_dir, list_episode_parquets, load_episode_state, load_episode_image)
from envs import robocasa_utils
from script.cluster_robocasa_embeddings import build_and_restore_agent
from script.render_composite_video import render_video
from script.visualize_latent_robocasa import encode_latents
from utils.token_dataset import load_token_dataset
from utils.visual_features import FrozenDINOv3Extractor


def build_episode_obs(task, episode, cameras, image_size, lang_vec, pool_grid):
    raw_root = osp.expanduser("~/.robocasa/raw")
    lerobot_dir = find_task_lerobot_dir(raw_root, "pretrain", task, category="atomic")
    p = list_episode_parquets(lerobot_dir)[episode]
    ep = load_episode_state(p); state = ep["state"]; T = len(state)
    cam_imgs = [load_episode_image(p, lerobot_dir, c, image_size) for c in cameras]
    T_eff = min([T] + [len(im) for im in cam_imgs])
    ex = FrozenDINOv3Extractor(device="cpu", batch_size=64, pool_grid=pool_grid, keep_cls=True)
    cam_toks = [ex.extract(im[:T_eff]) for im in cam_imgs]          # each (T_eff,65,384)
    tokens = np.stack(cam_toks, axis=1).reshape(T_eff, -1).astype(np.float32)
    state = state[:T_eff].astype(np.float32)
    parts = [tokens, state]
    if lang_vec is not None:
        parts.append(np.tile(np.asarray(lang_vec, np.float32)[None], (T_eff, 1)))
    obs = np.concatenate(parts, axis=-1)
    act = np.clip(ep["action"][:T_eff - 1], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
    disp = cam_imgs[0][:T_eff]                                       # agentview_left for display
    return obs[:T_eff - 1], act, disp[:T_eff - 1]                    # align to transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    ap.add_argument("--tasks", default="", help="comma list; overrides --task, reuses one backdrop.")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n_clusters", type=int, default=8)
    ap.add_argument("--n_backdrop", type=int, default=4000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang_json", default="~/.robocasa/data/task_lang_embeddings.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(PROJECT_ROOT) / run_dir
    flags = json.load(open(run_dir / "flags.json"))
    cfg = flags["agent"]
    pool_grid = int(round((cfg["n_tok"] - 1) ** 0.5))               # 65 -> 8
    pre, _, _ = robocasa_utils.parse_env_name(flags["env_name"])
    dd = osp.expanduser(robocasa_utils.DEFAULT_DATASET_DIR)
    stats = json.load(open(osp.join(dd, f"{pre}.stats.json")))
    cameras = stats["cameras"]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or [args.task]
    lang_emb = (json.load(open(osp.expanduser(args.lang_json)))["embeddings"]
                if stats.get("has_lang", False) else None)

    # --- backdrop computed ONCE, reused across all tasks -------------------
    ds = load_token_dataset(dd, pre, max_size=flags.get("pretraining_size", np.inf) or np.inf)
    agent = build_and_restore_agent(flags, str(run_dir), args.epoch,
                                    ds.obs_dim, ds.actions.shape[1])
    print("Encoding atomic backdrop latents (once) ...")
    rng = np.random.default_rng(args.seed)
    bidx = np.sort(rng.choice(ds.size, size=min(args.n_backdrop, ds.size), replace=False))
    b_obs = ds._gather_obs(ds.obs_idx[bidx]).astype(np.float32)
    b_act = ds.actions[bidx].astype(np.float32)
    z_atomic, _, _ = encode_latents(agent, {"observations": b_obs, "actions": b_act},
                                    jax.random.PRNGKey(args.seed))
    z_atomic = np.asarray(z_atomic, np.float32)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10).fit(z_atomic)
    lab_atomic = np.array([f"C{c}" for c in km.labels_], dtype=object)
    order = [f"C{i}" for i in range(args.n_clusters)]
    palette = matplotlib.colormaps["tab10" if args.n_clusters <= 10 else "tab20"]
    color_map = {f"C{i}": palette(i % palette.N) for i in range(args.n_clusters)}
    print(f"Fitting UMAP on {len(z_atomic)} atomic latents (once) ...")
    um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed)
    emb_atomic = um.fit_transform(z_atomic)

    out_dir = run_dir / "plots" / "videos" / f"epoch_{args.epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        try:
            print(f"\n=== {task} ep{args.episode} ===")
            lang_vec = lang_emb[task] if lang_emb is not None else None
            obs, act, frames = build_episode_obs(task, args.episode, cameras, 256, lang_vec, pool_grid)
            z_ep, _, _ = encode_latents(agent, {"observations": obs, "actions": act},
                                        jax.random.PRNGKey(args.seed))
            z_ep = np.asarray(z_ep, np.float32)
            emb_comp = um.transform(z_ep)
            voted = np.array([f"C{c}" for c in km.predict(z_ep)], dtype=object)
            data = dict(T=len(obs), voted=voted, present_tasks=order,
                        color_map=color_map, atomic_labels=lab_atomic)
            out_path = out_dir / f"{task}_ep{args.episode}_k{args.n_clusters}.mp4"
            render_video(data, frames, out_path,
                         emb_atomic=emb_atomic, emb_comp=emb_comp, embedding_name="UMAP",
                         fps=args.fps,
                         title=f"DINOv3-attnpool inFOM — {task} ep{args.episode}",
                         subtitle=f"epoch {args.epoch} · {args.n_clusters} KMeans action clusters (post-hoc)")
            print(f"[saved] {out_path}")
        except Exception as e:
            print(f"[FAIL] {task}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
