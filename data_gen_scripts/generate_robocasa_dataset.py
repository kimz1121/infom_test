"""Convert Robocasa LeRobot v2.1 datasets to inFOM flat HDF5 format.

Produces files compatible with envs/dmc_utils.py:get_dataset conventions:
  observations, actions, next_observations, terminals, masks
  + rewards when --relabel_reward (i.e. include rewards) is set.

Two modalities:
  --modality state  : observation.state (16-d float32)
  --modality image  : decodes one camera mp4, resizes to --image_size, uint8

Two splits per task-set:
  --relabel_reward 0 → reward_free (drop rewards) for pretraining
  --relabel_reward 1 → reward_labeled (keep next.reward) for finetuning

A held-out _val.hdf5 sibling is written using --val_fraction of episodes (by index).
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd

# 18 atomic-seen tasks (mirrors download_robocasa.py).
ATOMIC_SEEN_18 = [
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
    "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
]


def find_task_lerobot_dir(raw_root: str, split: str, task: str, category: str = "atomic") -> str:
    """Resolve the latest-dated lerobot/ directory for (split, category, task)."""
    pattern = osp.join(raw_root, split, category, task, "*", "lerobot")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No extracted lerobot dir under {pattern}")
    return matches[-1]


def list_episode_parquets(lerobot_dir: str) -> List[str]:
    return sorted(glob.glob(osp.join(lerobot_dir, "data", "chunk-*", "episode_*.parquet")))


def split_train_val(eps: List[str], val_fraction: float, seed: int = 0) -> Tuple[List[str], List[str]]:
    if val_fraction <= 0:
        return eps, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(eps))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(eps) * val_fraction)))
    val_idx = sorted(idx[:n_val].tolist())
    tr_idx = sorted(idx[n_val:].tolist())
    return [eps[i] for i in tr_idx], [eps[i] for i in val_idx]


def stack_obj_col(series: pd.Series) -> np.ndarray:
    """Stack a parquet 'object' column of fixed-length 1-D arrays into (T, D)."""
    arr = np.asarray(series.tolist())  # shape (T, D)
    return arr


def load_episode_state(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    state = stack_obj_col(df["observation.state"]).astype(np.float32)
    action = stack_obj_col(df["action"]).astype(np.float32)
    # next.reward/done may be scalar columns.
    reward = df["next.reward"].to_numpy().astype(np.float32).reshape(-1)
    done = df["next.done"].to_numpy().astype(bool).reshape(-1)
    return dict(state=state, action=action, reward=reward, done=done)


def _decode_video(mp4_path: str, image_size: int) -> np.ndarray:
    """Decode an mp4 into (T, H, W, 3) uint8 array, resized to image_size."""
    import av  # PyAV — installed via lerobot dependencies in this env.
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        if img.shape[0] != image_size or img.shape[1] != image_size:
            # use PIL for resize (no scipy/cv2 dep).
            from PIL import Image
            img = np.array(Image.fromarray(img).resize((image_size, image_size), Image.BILINEAR))
        frames.append(img)
    container.close()
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def load_episode_image(parquet_path: str, lerobot_dir: str, camera: str, image_size: int) -> np.ndarray:
    """Resolve the mp4 sibling for an episode parquet and decode frames."""
    base = osp.basename(parquet_path)  # episode_000123.parquet
    ep_id = base.replace(".parquet", "").split("_")[-1]
    chunk = osp.basename(osp.dirname(parquet_path))  # chunk-000
    # LeRobot v2.1 video subdir is prefixed with "observation.images.".
    cam_dir = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
    mp4 = osp.join(lerobot_dir, "videos", chunk, cam_dir, f"episode_{ep_id}.mp4")
    if not osp.exists(mp4):
        raise FileNotFoundError(mp4)
    return _decode_video(mp4, image_size)


def build_dataset(
    raw_root: str,
    split: str,
    tasks: List[str],
    val_fraction: float,
    modality: str,
    camera: str,
    image_size: int,
    relabel_reward: bool,
    max_episodes_per_task: int,
    category: str = "atomic",
    log_every: int = 50,
) -> Tuple[dict, dict, dict]:
    """Return (train_dict, val_dict, stats_dict) holding flat arrays.

    The dicts contain numpy arrays with the inFOM convention:
      observations, actions, next_observations, terminals, masks, [rewards]
    """
    train_buf = {k: [] for k in ["observations", "next_observations", "actions",
                                  "terminals", "masks"] + (["rewards"] if relabel_reward else [])}
    val_buf = {k: [] for k in train_buf}

    stats = dict(tasks={}, total_train_T=0, total_val_T=0)
    t_start = time.time()

    for task in tasks:
        lerobot_dir = find_task_lerobot_dir(raw_root, split, task, category=category)
        eps = list_episode_parquets(lerobot_dir)
        if max_episodes_per_task and len(eps) > max_episodes_per_task:
            eps = eps[:max_episodes_per_task]
        tr_eps, val_eps = split_train_val(eps, val_fraction)
        task_T_tr = task_T_val = 0

        for which, ep_list, buf in [("train", tr_eps, train_buf), ("val", val_eps, val_buf)]:
            for i, p in enumerate(ep_list):
                ep = load_episode_state(p)
                T = len(ep["state"])
                if T < 2:
                    continue

                obs = ep["state"][:T - 1]
                next_obs = ep["state"][1:]
                act = np.clip(ep["action"][:T - 1], -1.0 + 1e-5, 1.0 - 1e-5)
                # terminals: only last transition of episode is 1.
                term = np.zeros(T - 1, dtype=np.float32)
                term[-1] = 1.0
                mask = 1.0 - term

                if modality == "image":
                    img_obs = load_episode_image(p, lerobot_dir, camera, image_size)
                    # img_obs has T frames; align to T-1 transitions.
                    if len(img_obs) != T:
                        # rare: if video has off-by-one, truncate
                        T_eff = min(T, len(img_obs))
                        if T_eff < 2:
                            continue
                        img_obs = img_obs[:T_eff]
                        obs = obs[: T_eff - 1]
                        next_obs = ep["state"][1:T_eff]
                        act = np.clip(ep["action"][: T_eff - 1], -1.0 + 1e-5, 1.0 - 1e-5)
                        term = np.zeros(T_eff - 1, dtype=np.float32); term[-1] = 1.0
                        mask = 1.0 - term
                    buf["observations"].append(img_obs[:-1])
                    buf["next_observations"].append(img_obs[1:])
                else:
                    buf["observations"].append(obs)
                    buf["next_observations"].append(next_obs)

                buf["actions"].append(act)
                buf["terminals"].append(term)
                buf["masks"].append(mask)
                if relabel_reward:
                    rew = ep["reward"][1:1 + len(act)].astype(np.float32)
                    if len(rew) < len(act):  # pad if mismatch
                        rew = np.concatenate([rew, np.zeros(len(act) - len(rew), dtype=np.float32)])
                    buf["rewards"].append(rew)

                if which == "train":
                    task_T_tr += len(act)
                else:
                    task_T_val += len(act)
                if (i + 1) % log_every == 0:
                    print(f"  [{task} {which}] {i + 1}/{len(ep_list)} eps, T_so_far={task_T_tr if which=='train' else task_T_val}")

        stats["tasks"][task] = dict(train_eps=len(tr_eps), val_eps=len(val_eps),
                                    train_T=task_T_tr, val_T=task_T_val)
        stats["total_train_T"] += task_T_tr
        stats["total_val_T"] += task_T_val
        elapsed = time.time() - t_start
        print(f"[{task}] tr_eps={len(tr_eps)} val_eps={len(val_eps)} tr_T={task_T_tr} val_T={task_T_val} ({elapsed:.1f}s elapsed)")

    def concat(buf):
        return {k: np.concatenate(v, axis=0) for k, v in buf.items() if v}

    return concat(train_buf), concat(val_buf), stats


def save_hdf5(path: str, data: dict):
    os.makedirs(osp.dirname(path), exist_ok=True)
    print(f"Writing {path} ({len(next(iter(data.values()))):,} transitions)")
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def parse_tasks(arg: str) -> List[str]:
    if arg == "atomic_seen_18":
        return list(ATOMIC_SEEN_18)
    return [t.strip() for t in arg.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="~/.robocasa/raw")
    ap.add_argument("--out_dir", default="~/.robocasa/data")
    ap.add_argument("--split", default="target", choices=["target", "pretrain"])
    ap.add_argument("--category", default="atomic", choices=["atomic", "composite"],
                    help="Subdirectory under raw_root/{split}/. Composite tasks are "
                         "multi-skill chained demos.")
    ap.add_argument("--tasks", default="atomic_seen_18",
                    help="'atomic_seen_18' or comma-separated task names.")
    ap.add_argument("--name", required=True,
                    help="Basename for output hdf5, e.g. 'atomic_seen_state_pretrain'.")
    ap.add_argument("--modality", choices=["state", "image"], default="state")
    ap.add_argument("--camera", default="robot0_agentview_left",
                    help="Which camera to use for image modality.")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--relabel_reward", type=int, default=0,
                    help="0 = drop rewards (pretrain), 1 = include real next.reward (finetune).")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--max_episodes_per_task", type=int, default=0,
                    help="Cap demos per task for fast smoke tests (0 = use all).")
    args = ap.parse_args()

    raw_root = osp.expanduser(args.raw_root)
    out_dir = osp.expanduser(args.out_dir)
    tasks = parse_tasks(args.tasks)

    print(f"raw_root={raw_root}")
    print(f"split={args.split}  tasks={len(tasks)} modality={args.modality} reward={args.relabel_reward}")
    for t in tasks:
        print(f"  - {t}")

    tr, val, stats = build_dataset(
        raw_root=raw_root,
        split=args.split,
        tasks=tasks,
        val_fraction=args.val_fraction,
        modality=args.modality,
        camera=args.camera,
        image_size=args.image_size,
        relabel_reward=bool(args.relabel_reward),
        max_episodes_per_task=args.max_episodes_per_task,
        category=args.category,
    )

    # Ensure terminals invariant: last transition must be a terminal.
    def _enforce_terminal(d):
        if d and "terminals" in d and len(d["terminals"]) and d["terminals"][-1] < 0.5:
            d["terminals"][-1] = 1.0
            d["masks"][-1] = 0.0

    _enforce_terminal(tr)
    _enforce_terminal(val)

    train_path = osp.join(out_dir, f"{args.name}.hdf5")
    val_path = osp.join(out_dir, f"{args.name}_val.hdf5")
    save_hdf5(train_path, tr)
    if val:
        save_hdf5(val_path, val)

    with open(osp.join(out_dir, f"{args.name}.stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
