"""Robocasa dataset loader for inFOM.

Loads pre-converted HDF5 datasets produced by
`data_gen_scripts/generate_robocasa_dataset.py`. The expected on-disk layout is:

    ~/.robocasa/data/
        <basename>.hdf5
        <basename>_val.hdf5

For smoke tests, evaluation env is not constructed; callers must allow
`eval_env=None`. A real env wrapper around robosuite/Robocasa can be added later.
"""
import os
import os.path as osp

import h5py
import numpy as np
from tqdm import tqdm

from utils.datasets import Dataset

DEFAULT_DATASET_DIR = "~/.robocasa/data"


def _h5_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def _load_hdf5(path: str, max_size=np.inf):
    """Read at most ``max_size`` rows per dataset, slicing inside h5py so we
    never materialize the full file in RAM (important for image HDF5s).

    Caveat: this reads the FIRST ``max_size`` transitions, which biases toward
    early tasks in a multi-task concat. For smoke tests that's acceptable;
    for real experiments, regenerate the HDF5 with the desired episode cap
    or extend this loader with random episode sampling.
    """
    out = {}
    with h5py.File(path, "r") as f:
        keys = _h5_keys(f)
        # Resolve effective cap from observations length.
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
    # Enforce terminal invariant after slicing (last transition must be terminal).
    if "terminals" in out and len(out["terminals"]) and out["terminals"][-1] < 0.5:
        out["terminals"][-1] = 1.0
        if "masks" in out:
            out["masks"][-1] = 0.0
    return out


def _build_dataset(raw: dict, has_reward: bool) -> Dataset:
    """Wrap a dict of arrays into a frozen Dataset matching the inFOM convention."""
    obs = raw["observations"]
    next_obs = raw["next_observations"]
    if obs.dtype == np.uint8:
        # Visual obs: keep as uint8; inFOM's encoder paths normalize internally.
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
    """Map an inFOM env_name string to (basename_pretrain, basename_finetune, modality).

    Convention (smoke-friendly):
      robocasa_atomic_seen_state                              -> reward_free pretrain pool;
                                                                 finetune = same pool (smoke).
      robocasa_atomic_seen_state_ft_<TaskName>                -> reward_free pretrain = atomic_seen_state_pretrain
                                                                 reward_labeled finetune = atomic_seen_state_ft_<task>
      robocasa_atomic_seen_image[...]                          -> image variants.
    """
    s = env_name
    if not s.startswith("robocasa_"):
        raise ValueError(s)
    s = s[len("robocasa_"):]
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
    """Return (train_dataset, val_dataset) matching dmc_utils.get_dataset semantics."""
    dataset_dir = osp.expanduser(dataset_dir)
    pre_name, ft_name, _ = parse_env_name(env_name)
    base = pre_name if reward_free else ft_name

    train_path = osp.join(dataset_dir, f"{base}.hdf5")
    val_path = osp.join(dataset_dir, f"{base}_val.hdf5")
    if not osp.exists(train_path):
        raise FileNotFoundError(
            f"Robocasa dataset not found: {train_path}\n"
            f"Run data_gen_scripts/generate_robocasa_dataset.py with --name {base}.")

    train_raw = _load_hdf5(train_path, max_size=max_size)
    val_raw = _load_hdf5(val_path, max_size=max_size) if osp.exists(val_path) else None

    has_reward = (not reward_free)
    train_dataset = _build_dataset(train_raw, has_reward=has_reward)
    val_dataset = _build_dataset(val_raw, has_reward=has_reward) if val_raw is not None else None
    return train_dataset, val_dataset


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5,
                          reward_free=False, max_size=np.inf):
    """Smoke-friendly variant: returns (None, None, train_dataset, val_dataset).

    A proper robosuite-backed eval env can be plugged in later; for the
    smoke test we run pretraining/finetuning loss only (eval_interval=0).
    """
    # DINOv3 token-grid datasets stream from a memmap and use the token loader
    # instead of the flat HDF5 path (detected by a *_frame_tokens.f16 sibling).
    from utils.token_dataset import is_token_dataset, load_token_dataset
    pre_name, ft_name, _ = parse_env_name(env_name)
    base = pre_name if reward_free else ft_name
    if is_token_dataset(DEFAULT_DATASET_DIR, base):
        train_dataset = load_token_dataset(
            DEFAULT_DATASET_DIR, base, max_size=max_size, action_clip_eps=action_clip_eps)
        val_base = f"{base}_val"
        val_dataset = (load_token_dataset(DEFAULT_DATASET_DIR, val_base,
                                          max_size=max_size, action_clip_eps=action_clip_eps)
                       if is_token_dataset(DEFAULT_DATASET_DIR, val_base) else None)
        return None, None, train_dataset, val_dataset

    train_dataset, val_dataset = get_dataset(
        env_name, reward_free=reward_free, max_size=max_size)

    # Action clip parity with dmc_utils branch.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(add_or_replace=dict(
            actions=np.clip(train_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps)))
        if val_dataset is not None:
            val_dataset = val_dataset.copy(add_or_replace=dict(
                actions=np.clip(val_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps)))

    return None, None, train_dataset, val_dataset
