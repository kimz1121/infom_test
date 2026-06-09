"""Build the multimodal-precompute inFOM dataset: frozen resnet34(image) + state.

For every transition the observation is the 528-d vector
    concat([ frozen_resnet34(agentview image)  (512) , proprio state (16) ]).
This is stored as an ordinary flat HDF5 (single float32 `observations` /
`next_observations` array), so the existing state-style loader + `encoder=None`
inFOM pipeline trains on it unchanged.

Per episode the resnet runs ONCE over all T frames (next_obs[t] = obs[t+1]).
Visual features come from utils.visual_features.FrozenResNet34Extractor — the
same module inference/viz must use, so features are train/inference-identical.

Usage:
    python data_gen_scripts/extract_mm_features.py \
        --split pretrain --category atomic --tasks all \
        --name atomic_65_multimodal_precompute_state \
        --image_size 256 --camera robot0_agentview_left
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from typing import List

import numpy as np

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_gen_scripts.generate_robocasa_dataset import (  # noqa: E402
    find_task_lerobot_dir,
    list_episode_parquets,
    load_episode_image,
    load_episode_state,
    save_hdf5,
    split_train_val,
)
from utils.visual_features import FrozenResNet34Extractor  # noqa: E402


def resolve_tasks(raw_root, split, category, tasks_arg) -> List[str]:
    if tasks_arg == "all":
        # Use whatever tasks are actually extracted on disk for this split/category.
        base = osp.join(raw_root, split, category)
        found = sorted(
            d for d in os.listdir(base)
            if osp.isdir(osp.join(base, d)) and glob.glob(osp.join(base, d, "*", "lerobot"))
        )
        if not found:
            raise FileNotFoundError(f"No extracted tasks under {base}")
        return found
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def build(args):
    raw_root = osp.expanduser(args.raw_root)
    out_dir = osp.expanduser(args.out_dir)
    if args.flat:
        args.val_fraction = 0.0  # keep every episode in the single flat file
    tasks = resolve_tasks(raw_root, args.split, args.category, args.tasks)
    print(f"tasks ({len(tasks)}): {tasks}")

    extractor = FrozenResNet34Extractor(device=args.device, batch_size=args.batch_size)
    feat_dim = extractor.feat_dim

    train_buf = {k: [] for k in ["observations", "next_observations", "actions", "terminals", "masks"]}
    val_buf = {k: [] for k in train_buf}
    stats = dict(tasks={}, total_train_T=0, total_val_T=0,
                 feature="frozen_resnet34_imagenet", feat_dim=feat_dim,
                 camera=args.camera, image_size=args.image_size)
    t0 = time.time()

    for task in tasks:
        lerobot_dir = find_task_lerobot_dir(raw_root, args.split, task, category=args.category)
        eps = list_episode_parquets(lerobot_dir)
        if args.max_episodes_per_task and len(eps) > args.max_episodes_per_task:
            eps = eps[: args.max_episodes_per_task]
        tr_eps, val_eps = split_train_val(eps, args.val_fraction)
        task_T_tr = task_T_val = 0

        for which, ep_list, buf in [("train", tr_eps, train_buf), ("val", val_eps, val_buf)]:
            for p in ep_list:
                ep = load_episode_state(p)
                state = ep["state"]
                T = len(state)
                if T < 2:
                    continue
                imgs = load_episode_image(p, lerobot_dir, args.camera, args.image_size)  # (T,H,W,3)
                T_eff = min(T, len(imgs))
                if T_eff < 2:
                    continue
                feats = extractor.extract(imgs[:T_eff])  # (T_eff, 512) — once per episode
                state = state[:T_eff].astype(np.float32)
                fused = np.concatenate([feats, state], axis=-1)  # (T_eff, 528)

                obs = fused[: T_eff - 1]
                next_obs = fused[1:T_eff]
                act = np.clip(ep["action"][: T_eff - 1], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)
                term = np.zeros(T_eff - 1, dtype=np.float32)
                term[-1] = 1.0
                mask = 1.0 - term

                buf["observations"].append(obs)
                buf["next_observations"].append(next_obs)
                buf["actions"].append(act)
                buf["terminals"].append(term)
                buf["masks"].append(mask)
                if which == "train":
                    task_T_tr += len(act)
                else:
                    task_T_val += len(act)

        stats["tasks"][task] = dict(train_eps=len(tr_eps), val_eps=len(val_eps),
                                    train_T=task_T_tr, val_T=task_T_val)
        stats["total_train_T"] += task_T_tr
        stats["total_val_T"] += task_T_val
        print(f"[{task}] tr_eps={len(tr_eps)} val_eps={len(val_eps)} "
              f"tr_T={task_T_tr} val_T={task_T_val} ({time.time()-t0:.1f}s)")

    def concat(buf):
        return {k: np.concatenate(v, axis=0) for k, v in buf.items() if v}

    tr, val = concat(train_buf), concat(val_buf)

    def _enforce_terminal(d):
        if d and len(d.get("terminals", [])) and d["terminals"][-1] < 0.5:
            d["terminals"][-1] = 1.0
            d["masks"][-1] = 0.0

    _enforce_terminal(tr)
    _enforce_terminal(val)

    if args.flat:
        # Single flat file (no train/val split, no _pretrain suffix) — used for
        # composite episodes that the visualization reads via load_composite_episode.
        save_hdf5(osp.join(out_dir, f"{args.name}.hdf5"), tr)
        with open(osp.join(out_dir, f"{args.name}.stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        save_hdf5(osp.join(out_dir, f"{args.name}_pretrain.hdf5"), tr)
        if val:
            save_hdf5(osp.join(out_dir, f"{args.name}_pretrain_val.hdf5"), val)
        with open(osp.join(out_dir, f"{args.name}_pretrain.stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    print(f"done. total_train_T={stats['total_train_T']} feat_dim={feat_dim} (+state)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="~/.robocasa/raw")
    ap.add_argument("--out_dir", default="~/.robocasa/data")
    ap.add_argument("--split", default="pretrain", choices=["target", "pretrain"])
    ap.add_argument("--category", default="atomic", choices=["atomic", "composite"])
    ap.add_argument("--tasks", default="all", help="'all' (extracted dirs) or comma-separated names.")
    ap.add_argument("--name", required=True, help="Output basename (without _pretrain.hdf5).")
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--flat", action="store_true",
                    help="Write a single <name>.hdf5 (no split/_pretrain) — for composite episodes.")
    ap.add_argument("--max_episodes_per_task", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
