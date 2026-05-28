"""LIBERO dataset loader for inFOM.

LIBERO analogue of envs/robocasa_utils.py. Loads pre-converted HDF5 datasets
produced by data_gen_scripts/generate_libero_dataset.py. Expected on-disk
layout under ``~/.libero/data``:

    <basename>.hdf5          # train split
    <basename>_val.hdf5      # held-out val split (optional)
    <basename>.stats.json    # per-instruction transition counts

env_name convention (mirrors robocasa):
  libero_goal_state                               -> reward_free pretrain pool;
                                                     finetune = libero_goal_state_ft
  libero_goal_state_ft_<instruction>              -> reward_free pretrain = libero_goal_state_pretrain
                                                     reward_labeled finetune = libero_goal_state_ft_<instruction>
  libero_goal_image[...]                          -> image variants.

No robosuite-backed eval env is wired up; callers must use --eval_interval=0.
"""
import os.path as osp

import h5py
import numpy as np
from tqdm import tqdm

from utils.datasets import Dataset

DEFAULT_DATASET_DIR = "~/.libero/data"


def _h5_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def _load_hdf5(path: str, max_size=np.inf):
    """Read at most ``max_size`` rows per dataset, slicing inside h5py.

    Caveat: reads the FIRST ``max_size`` transitions, biasing toward early
    instructions in the multi-task concat. For full experiments either use all
    rows or regenerate with a per-task cap.
    """
    out = {}
    with h5py.File(path, "r") as f:
        keys = _h5_keys(f)
        if "observations" in f:
            n_total = f["observations"].shape[0]
            n = int(min(n_total, max_size)) if np.isfinite(max_size) else n_total
        else:
            n = None
        for k in tqdm(keys, desc=f"load {osp.basename(path)}"):
            ds = f[k]
            if n is not None and ds.shape and ds.shape[0] >= n:
                out[k] = ds[:n]
            else:
                out[k] = ds[:]
    if "terminals" in out and len(out["terminals"]) and out["terminals"][-1] < 0.5:
        out["terminals"][-1] = 1.0
        if "masks" in out:
            out["masks"][-1] = 0.0
    return out


def _build_dataset(raw: dict, has_reward: bool) -> Dataset:
    obs = raw["observations"]
    next_obs = raw["next_observations"]
    if obs.dtype == np.uint8:
        obs_field = obs
        next_obs_field = next_obs
    else:
        obs_field = obs.astype(np.float32)
        next_obs_field = next_obs.astype(np.float32)

    fields = dict(
        observations=obs_field,
        actions=raw["actions"].astype(np.float32),
        next_observations=next_obs_field,
        terminals=raw["terminals"].astype(np.float32),
        masks=raw["masks"].astype(np.float32),
    )
    if has_reward and "rewards" in raw:
        fields["rewards"] = raw["rewards"].astype(np.float32)
    return Dataset.create(**fields)


def parse_env_name(env_name: str):
    """Map a LIBERO env_name to (pretrain_basename, finetune_basename, modality).

    The "libero_" prefix is retained in the basenames so the on-disk files are
    self-descriptive (e.g. libero_goal_state_pretrain.hdf5).
    """
    s = env_name
    if not s.startswith("libero_"):
        raise ValueError(s)
    if "_ft_" in s:
        base, ft_task = s.split("_ft_", 1)
        pretrain_name = base + "_pretrain"
        finetune_name = base + "_ft_" + ft_task
    else:
        pretrain_name = s + "_pretrain"
        finetune_name = s + "_ft"
    modality = "image" if "image" in s else "state"
    return pretrain_name, finetune_name, modality


def get_dataset(env_name, reward_free=False, max_size=np.inf,
                dataset_dir=DEFAULT_DATASET_DIR):
    """Return (train_dataset, val_dataset) matching robocasa_utils.get_dataset."""
    dataset_dir = osp.expanduser(dataset_dir)
    pre_name, ft_name, _ = parse_env_name(env_name)
    base = pre_name if reward_free else ft_name

    train_path = osp.join(dataset_dir, f"{base}.hdf5")
    val_path = osp.join(dataset_dir, f"{base}_val.hdf5")
    if not osp.exists(train_path):
        raise FileNotFoundError(
            f"LIBERO dataset not found: {train_path}\n"
            f"Run data_gen_scripts/generate_libero_dataset.py with --name {base}.")

    train_raw = _load_hdf5(train_path, max_size=max_size)
    val_raw = _load_hdf5(val_path, max_size=max_size) if osp.exists(val_path) else None

    has_reward = (not reward_free)
    train_dataset = _build_dataset(train_raw, has_reward=has_reward)
    val_dataset = _build_dataset(val_raw, has_reward=has_reward) if val_raw is not None else None
    return train_dataset, val_dataset


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5,
                          reward_free=False, max_size=np.inf):
    """Returns (None, None, train_dataset, val_dataset); no eval env wired up."""
    train_dataset, val_dataset = get_dataset(
        env_name, reward_free=reward_free, max_size=max_size)

    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(add_or_replace=dict(
            actions=np.clip(train_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps)))
        if val_dataset is not None:
            val_dataset = val_dataset.copy(add_or_replace=dict(
                actions=np.clip(val_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps)))

    return None, None, train_dataset, val_dataset
