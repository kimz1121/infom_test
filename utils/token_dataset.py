"""Streaming Dataset for DINOv3 token-grid observations (learnable-attention inFOM).

The big per-frame token grid (~213 GB for atomic-65) lives on disk as a raw
float16 memmap; only the sampled batch's frames are read per step, so RAM stays
flat regardless of dataset size. Small per-transition arrays (actions/terminals/
masks) and per-frame state/lang live in RAM (tiny).

Batch contract matches what inFOM consumes from utils.datasets.Dataset:
    observations, next_observations, actions, next_actions, terminals, masks,
    rewards, observation_min, observation_max.

Each observation is returned FLAT as
    concat([ tokens(n_cam*n_tok*feat) , state(state_dim) , lang(lang_dim) ])
so the downstream (JAX) TokenAttnPoolEncoder can reshape the token block back to
(n_cam, n_tok, feat), run its learnable attention pool, and re-concat state+lang.
The DINOv3 tokens are already transformer-normalized, so observation
normalization is a no-op here (parity with reward-free precompute training).
"""
from __future__ import annotations

import json
import os.path as osp

import h5py
import numpy as np


class TokenDataset:
    """Disk-streaming token-grid dataset with the inFOM batch interface."""

    def __init__(self, tokens_path, meta_path, stats, max_size=np.inf,
                 action_clip_eps=1e-5):
        shape = tuple(stats["frame_tokens_shape"])
        self.tokens = np.memmap(tokens_path, dtype="<f2", mode="r", shape=shape)
        self.n_frames, self.n_cam, self.n_tok, self.feat = shape
        with h5py.File(meta_path, "r") as f:
            self.obs_idx = f["obs_idx"][:]
            self.next_idx = f["next_idx"][:]
            self.frame_state = f["frame_state"][:].astype(np.float32)
            self.frame_lang = f["frame_lang"][:].astype(np.float32) if "frame_lang" in f else None
            actions = f["actions"][:].astype(np.float32)
            self.terminals = f["terminals"][:].astype(np.float32)
            self.masks = f["masks"][:].astype(np.float32)
        if action_clip_eps is not None:
            actions = np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps)
        self.actions = actions

        n = len(self.obs_idx)
        if np.isfinite(max_size) and n > max_size:
            n = int(max_size)
            self.obs_idx = self.obs_idx[:n]
            self.next_idx = self.next_idx[:n]
            self.actions = self.actions[:n]
            self.terminals = self.terminals[:n].copy()
            self.masks = self.masks[:n].copy()
            # keep the terminal invariant after truncation
            if self.terminals[-1] < 0.5:
                self.terminals[-1] = 1.0
                self.masks[-1] = 0.0
        self.size = n

        self.state_dim = self.frame_state.shape[1]
        self.lang_dim = self.frame_lang.shape[1] if self.frame_lang is not None else 0
        self.token_dim = self.n_cam * self.n_tok * self.feat
        self.obs_dim = self.token_dim + self.state_dim + self.lang_dim

        # Layout metadata the encoder/agent config reads back.
        self.obs_layout = dict(n_cam=self.n_cam, n_tok=self.n_tok, feat_dim=self.feat,
                               token_dim=self.token_dim, state_dim=self.state_dim,
                               lang_dim=self.lang_dim, obs_dim=self.obs_dim)

        # Attributes main.py sets/reads (mirror utils.datasets.Dataset surface).
        self.obs_norm_type = "none"
        self.p_aug = None
        self.num_aug = 1
        self.inplace_aug = False
        self.frame_stack = None
        self.return_next_actions = False
        # inFOM's state-decoder flow clips goals to the proprio-state box via
        # obs_min/max sliced at [token_dim : token_dim+state_dim]. Only that slice
        # is ever read, so fill the (unused) token/lang regions with wide bounds.
        smin = self.frame_state.min(axis=0)
        smax = self.frame_state.max(axis=0)
        self.normalized_obs_min = np.full(self.obs_dim, -1e6, dtype=np.float32)
        self.normalized_obs_max = np.full(self.obs_dim, 1e6, dtype=np.float32)
        self.normalized_obs_min[self.token_dim:self.token_dim + self.state_dim] = smin
        self.normalized_obs_max[self.token_dim:self.token_dim + self.state_dim] = smax
        self.terminal_locs = np.nonzero(self.terminals > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    # -- interface used by main.py / agents ---------------------------------
    def normalize_observations(self, observations=None):
        """No-op: DINOv3 tokens are pre-normalized; keep obs untouched."""
        return None

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def _gather_obs(self, frame_ids):
        tok = np.asarray(self.tokens[frame_ids], dtype=np.float32)
        tok = tok.reshape(len(frame_ids), self.token_dim)
        parts = [tok, self.frame_state[frame_ids]]
        if self.frame_lang is not None:
            parts.append(self.frame_lang[frame_ids])
        return np.concatenate(parts, axis=-1)

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = dict(
            observations=self._gather_obs(self.obs_idx[idxs]),
            next_observations=self._gather_obs(self.next_idx[idxs]),
            actions=self.actions[idxs],
            terminals=self.terminals[idxs],
            masks=self.masks[idxs],
            # reward-free pretrain: zeros satisfy inFOM's reward_loss without a
            # rewards field on disk (matches reward_free semantics).
            rewards=np.zeros(len(idxs), dtype=np.float32),
        )
        if self.return_next_actions:
            batch["next_actions"] = self.actions[np.minimum(idxs + 1, self.size - 1)]
        batch["observation_min"] = self.normalized_obs_min
        batch["observation_max"] = self.normalized_obs_max
        return batch


def is_token_dataset(dataset_dir, base) -> bool:
    return osp.exists(osp.join(osp.expanduser(dataset_dir), f"{base}_frame_tokens.f16"))


def load_token_dataset(dataset_dir, base, max_size=np.inf, action_clip_eps=1e-5):
    dataset_dir = osp.expanduser(dataset_dir)
    stats = json.load(open(osp.join(dataset_dir, f"{base}.stats.json")))
    return TokenDataset(
        tokens_path=osp.join(dataset_dir, f"{base}_frame_tokens.f16"),
        meta_path=osp.join(dataset_dir, f"{base}.hdf5"),
        stats=stats, max_size=max_size, action_clip_eps=action_clip_eps)
