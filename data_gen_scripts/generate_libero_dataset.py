"""Convert LIBERO (robomimic-style) HDF5 demos to inFOM flat HDF5 format.

This is the LIBERO analogue of generate_robocasa_dataset.py. The LIBERO-Goal
suite ships one HDF5 per language instruction (e.g.
``put_the_bowl_on_the_plate_demo.hdf5``), each containing ``data/demo_<i>``
groups with ``actions``, ``dones``, ``rewards`` and an ``obs/`` subgroup of
proprio + RGB streams. Unlike Robocasa, RGB frames live directly in the HDF5
(no mp4 to decode).

Produces files compatible with envs/libero_utils.py:get_dataset conventions:
  observations, actions, next_observations, terminals, masks
  + rewards when --relabel_reward (i.e. include rewards) is set.

Two modalities:
  --modality state  : concat of obs proprio keys (default 15-d).
  --modality image  : one RGB camera, resized to --image_size, uint8.

Two splits per task-set:
  --relabel_reward 0 -> reward_free (drop rewards) for pretraining.
  --relabel_reward 1 -> reward_labeled (keep sparse reward) for finetuning.

A held-out _val.hdf5 sibling is written using --val_fraction of demos (by index).

Tasks are concatenated in the order of LIBERO_GOAL_10 (sorted instruction
stems), and per-task transition counts are recorded in <name>.stats.json so
that script/visualize_latent_libero.py can recover each row's owning
instruction exactly as the Robocasa pipeline does.

Example:
    python data_gen_scripts/generate_libero_dataset.py \
        --name libero_goal_state_pretrain --modality state --relabel_reward 0
    python data_gen_scripts/generate_libero_dataset.py \
        --name libero_goal_state_ft_put_the_bowl_on_the_plate \
        --tasks put_the_bowl_on_the_plate --modality state --relabel_reward 1
"""
import argparse
import glob
import json
import os
import os.path as osp
import time
from typing import List, Tuple

import h5py
import numpy as np

# LIBERO-Goal: 10 instructions, identified by their HDF5 filename stem
# (without the trailing "_demo"). Kept in sorted order so the concat layout is
# deterministic and matches a plain `ls` of the suite directory.
LIBERO_GOAL_10 = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]

# Proprio keys concatenated (in this order) to form the state vector.
# ee_pos(3) + ee_ori(3) + gripper_states(2) + joint_states(7) = 15-d.
DEFAULT_STATE_KEYS = ["ee_pos", "ee_ori", "gripper_states", "joint_states"]

# LIBERO HF cache (yifengzhu-hf/LIBERO-datasets). The snapshot hash is resolved
# at runtime so this keeps working if the cache is re-pulled.
DEFAULT_HF_GLOB = (
    "~/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/"
    "snapshots/*/{suite}"
)


def resolve_suite_dir(suite: str, raw_root: str = "") -> str:
    """Return the directory holding the suite's per-instruction HDF5s."""
    if raw_root:
        cand = osp.expanduser(raw_root)
        if osp.isdir(cand):
            return cand
    pattern = osp.expanduser(DEFAULT_HF_GLOB.format(suite=suite))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No LIBERO suite dir found at {pattern}. Pass --raw_root or download "
            f"yifengzhu-hf/LIBERO-datasets first.")
    return matches[-1]


def find_task_hdf5(suite_dir: str, task: str) -> str:
    """Resolve the HDF5 file for an instruction stem within the suite dir."""
    for cand in (f"{task}_demo.hdf5", f"{task}.hdf5"):
        p = osp.join(suite_dir, cand)
        if osp.exists(p):
            return p
    raise FileNotFoundError(f"No HDF5 for task {task!r} under {suite_dir}")


def list_demos(data_group: h5py.Group) -> List[str]:
    return sorted(data_group.keys(), key=lambda k: int(k.split("_")[1]))


def split_train_val(demos: List[str], val_fraction: float, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
    if val_fraction <= 0:
        return demos, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(demos))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(demos) * val_fraction)))
    val_idx = sorted(idx[:n_val].tolist())
    tr_idx = sorted(idx[n_val:].tolist())
    return [demos[i] for i in tr_idx], [demos[i] for i in val_idx]


def build_state(obs_group: h5py.Group, state_keys: List[str]) -> np.ndarray:
    """Concatenate the requested proprio keys into a (T, D) float32 array."""
    parts = [np.asarray(obs_group[k], dtype=np.float32) for k in state_keys]
    return np.concatenate(parts, axis=-1)


def load_image_obs(obs_group: h5py.Group, camera: str, image_size: int) -> np.ndarray:
    """Read an RGB stream as (T, H, W, 3) uint8, resized to image_size."""
    key = camera if camera in obs_group else f"{camera}_rgb"
    if key not in obs_group:
        raise KeyError(
            f"Camera {camera!r} not in obs (have: {list(obs_group.keys())})")
    img = np.asarray(obs_group[key])  # (T, H, W, 3) uint8
    if img.shape[1] != image_size or img.shape[2] != image_size:
        from PIL import Image
        img = np.stack([
            np.asarray(Image.fromarray(f).resize(
                (image_size, image_size), Image.BILINEAR))
            for f in img
        ], axis=0)
    return img.astype(np.uint8)


def build_dataset(
    suite_dir: str,
    tasks: List[str],
    val_fraction: float,
    modality: str,
    camera: str,
    image_size: int,
    state_keys: List[str],
    relabel_reward: bool,
    max_episodes_per_task: int,
) -> Tuple[dict, dict, dict]:
    """Return (train_dict, val_dict, stats_dict) of flat inFOM-convention arrays."""
    fields = ["observations", "next_observations", "actions", "terminals", "masks"]
    if relabel_reward:
        fields.append("rewards")
    train_buf = {k: [] for k in fields}
    val_buf = {k: [] for k in fields}

    stats = dict(tasks={}, total_train_T=0, total_val_T=0,
                 modality=modality, state_keys=list(state_keys),
                 camera=camera, image_size=image_size,
                 instructions={})
    t_start = time.time()

    for task in tasks:
        path = find_task_hdf5(suite_dir, task)
        with h5py.File(path, "r") as f:
            data = f["data"]
            try:
                instr = json.loads(data.attrs["problem_info"])["language_instruction"]
            except Exception:
                instr = task.replace("_", " ")
            stats["instructions"][task] = instr
            demos = list_demos(data)
            if max_episodes_per_task and len(demos) > max_episodes_per_task:
                demos = demos[:max_episodes_per_task]
            tr_demos, val_demos = split_train_val(demos, val_fraction)
            task_T_tr = task_T_val = 0

            for which, demo_list, buf in [("train", tr_demos, train_buf),
                                          ("val", val_demos, val_buf)]:
                for name in demo_list:
                    d = data[name]
                    act_full = np.asarray(d["actions"], dtype=np.float32)
                    T = act_full.shape[0]
                    if T < 2:
                        continue
                    if modality == "image":
                        frames = load_image_obs(d["obs"], camera, image_size)
                        obs = frames[:-1]
                        next_obs = frames[1:]
                    else:
                        state = build_state(d["obs"], state_keys)
                        obs = state[:T - 1]
                        next_obs = state[1:T]
                    act = np.clip(act_full[:T - 1], -1.0 + 1e-5, 1.0 - 1e-5)
                    term = np.zeros(T - 1, dtype=np.float32)
                    term[-1] = 1.0  # last transition enters the terminal state
                    mask = 1.0 - term

                    buf["observations"].append(obs)
                    buf["next_observations"].append(next_obs)
                    buf["actions"].append(act)
                    buf["terminals"].append(term)
                    buf["masks"].append(mask)
                    if relabel_reward:
                        rew_full = np.asarray(d["rewards"], dtype=np.float32).reshape(-1)
                        # reward received on entering next_obs: rewards[1:T].
                        rew = rew_full[1:T]
                        if len(rew) < len(act):
                            rew = np.concatenate(
                                [rew, np.zeros(len(act) - len(rew), dtype=np.float32)])
                        buf["rewards"].append(rew)

                    if which == "train":
                        task_T_tr += len(act)
                    else:
                        task_T_val += len(act)

        stats["tasks"][task] = dict(train_eps=len(tr_demos), val_eps=len(val_demos),
                                    train_T=task_T_tr, val_T=task_T_val)
        stats["total_train_T"] += task_T_tr
        stats["total_val_T"] += task_T_val
        elapsed = time.time() - t_start
        print(f"[{task}] tr_eps={len(tr_demos)} val_eps={len(val_demos)} "
              f"tr_T={task_T_tr} val_T={task_T_val} ({elapsed:.1f}s)")

    def concat(buf):
        return {k: np.concatenate(v, axis=0) for k, v in buf.items() if v}

    return concat(train_buf), concat(val_buf), stats


def save_hdf5(path: str, data: dict):
    os.makedirs(osp.dirname(path), exist_ok=True)
    n = len(next(iter(data.values())))
    print(f"Writing {path} ({n:,} transitions)")
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def parse_tasks(arg: str) -> List[str]:
    if arg in ("goal_10", "libero_goal", "all"):
        return list(LIBERO_GOAL_10)
    return [t.strip() for t in arg.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal",
                    help="LIBERO suite subdir name in the HF cache.")
    ap.add_argument("--raw_root", default="",
                    help="Override the resolved suite dir (else use HF cache).")
    ap.add_argument("--out_dir", default="~/.libero/data")
    ap.add_argument("--tasks", default="goal_10",
                    help="'goal_10' or comma-separated instruction stems.")
    ap.add_argument("--name", required=True,
                    help="Output basename, e.g. 'libero_goal_state_pretrain'.")
    ap.add_argument("--modality", choices=["state", "image"], default="state")
    ap.add_argument("--state_keys", default=",".join(DEFAULT_STATE_KEYS),
                    help="Comma-separated obs keys for the state vector.")
    ap.add_argument("--camera", default="agentview",
                    help="RGB camera for image modality (agentview / eye_in_hand).")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--relabel_reward", type=int, default=0,
                    help="0 = drop rewards (pretrain), 1 = keep sparse reward (finetune).")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--max_episodes_per_task", type=int, default=0,
                    help="Cap demos per task for fast smoke tests (0 = use all).")
    args = ap.parse_args()

    out_dir = osp.expanduser(args.out_dir)
    suite_dir = resolve_suite_dir(args.suite, args.raw_root)
    tasks = parse_tasks(args.tasks)
    state_keys = [k.strip() for k in args.state_keys.split(",") if k.strip()]

    print(f"suite_dir={suite_dir}")
    print(f"tasks={len(tasks)} modality={args.modality} reward={args.relabel_reward}")
    for t in tasks:
        print(f"  - {t}")

    tr, val, stats = build_dataset(
        suite_dir=suite_dir,
        tasks=tasks,
        val_fraction=args.val_fraction,
        modality=args.modality,
        camera=args.camera,
        image_size=args.image_size,
        state_keys=state_keys,
        relabel_reward=bool(args.relabel_reward),
        max_episodes_per_task=args.max_episodes_per_task,
    )

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
