"""Memory-safe launcher for main.py.

Wraps three pieces of utils/datasets.py and envs/env_utils.py via
monkey-patching, without modifying any source file. Designed for
visual-* OGBench environments on a 32 GB-RAM host.

Patches applied (in order, before main.py is imported):

  1. envs.env_utils.make_env_and_datasets
     The original truncates with `train_dataset[k] = v[:max_size]`
     (envs/env_utils.py:156-161). `v[:max_size]` is a numpy view, so
     the full (~1M-frame) array stays alive through `v.base` even when
     --pretraining_size is small. After action-clip and Dataset.create,
     the same views end up inside a FrozenDict, holding gigabytes that
     never get freed. The wrapper materializes each view into a fresh
     contiguous copy and rebuilds the Dataset, dropping the references
     to the originals so the GC can reclaim them.

  2. utils.datasets.Dataset.normalize_observations
     The original calls np.mean / np.var / np.max / np.min on the
     full observations array. For uint8 image obs of shape
     (N, 64, 64, 3), np.mean and np.var upcast internally to float64
     and try to allocate ~91 GiB. The replacement:
       * obs_norm_type == 'none': skips mean/var entirely (their
         results are unused in the 'none' branch of normalize()), but
         still computes obs_min / obs_max so that batches expose
         valid observation_min / observation_max (consumed by
         agents/infom.py:271 etc.). np.min/np.max preserve uint8
         dtype, so they cost almost nothing.
       * 'normal' / 'bounded': computes mean / var / min / max in a
         chunked, float32 streaming pass (Welford's algorithm), so
         peak memory stays bounded.

  3. utils.datasets.Dataset._prestack_frames
     The original eagerly stacks frames for the WHOLE dataset by
     fancy-indexing `frame_stack` copies and concatenating -- peak
     ~6x raw obs size, which is what triggers the OOM killer
     (`Killed`). Replaced with a no-op so that Dataset.sample falls
     through to the per-batch on-demand stacking path that already
     exists in the original code (utils/datasets.py:187-201).

  4. utils.datasets.Dataset.sample  (background prefetch)
     Per-batch on-demand stacking + augment runs in the main Python
     thread and serialises with the GPU step, capping GPU utilisation
     near 50%. The wrapper spawns one daemon thread per dataset that
     keeps a small queue of pre-built batches ready, so subsequent
     .sample() calls return instantly while the next batch is being
     prepared on CPU concurrently with the GPU step. Bypassed when
     idxs is explicit (deterministic eval-style call) or batch_size
     is small (the one-off main.py:115 setup call).

What the user must still pass on the command line for visual envs:

  --obs_norm_type=none
      Image observations should not be mean/var normalised; the
      default 'normal' rewrites the entire obs buffer to float, which
      is unrelated to our patches.

  --agent.clip_flow_goals=False
      With config.encoder set (e.g. impala_small), observations are
      encoded to a (B, latent_dim) tensor inside compute_fwd_flow_goals
      (agents/infom.py:120-124), but batch['observation_min'/'max']
      are pixel-shape (64, 64, 3). The default clip_flow_goals=True
      then tries to broadcast (64,64,3) against (B, latent_dim) and
      raises a shape error -- this is a pre-existing limitation of
      the original code, not something this launcher can paper over.

  --pretraining_size / --finetuning_size
      Cap dataset sizes to something that fits in RAM. OGBench loads
      the full file regardless; only the in-memory truncation matters
      (and patch #1 makes that truncation actually free memory).

Recommended invocation on a 32 GB-RAM host:

    python script/run_memsafe.py \
        --env_name=visual-cube-single-play-singletask-task1-v0 \
        --obs_norm_type=none \
        --pretraining_steps=250_000 --pretraining_size=250_000 \
        --finetuning_steps=100_000  --finetuning_size=100_000 \
        --eval_interval=10_000 --save_interval=750_000 \
        --p_aug=0.5 --frame_stack=3 \
        --agent=agents/infom.py \
        --agent.expectile=0.95 --agent.kl_weight=0.025 --agent.alpha=30 \
        --agent.encoder=impala_small \
        --agent.clip_flow_goals=False
"""

import gc
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.datasets import Dataset  # noqa: E402
import envs.env_utils as _env_utils_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Patch 1: envs.env_utils.make_env_and_datasets
# ---------------------------------------------------------------------------

def _materialize_views(dataset):
    """Return a new Dataset whose view-backed arrays have been copied.

    The original Dataset (FrozenDict) is immutable, so we extract its
    contents, replace any ndarray view with a fresh contiguous copy,
    and hand the result to Dataset.create. Once the caller drops the
    old Dataset, the originals (and their .base full arrays) become
    GC-able.
    """
    if dataset is None:
        return None
    new_data = {}
    changed = False
    for k in dataset.keys():
        v = dataset[k]
        if isinstance(v, np.ndarray) and v.base is not None:
            new_data[k] = np.ascontiguousarray(v)
            changed = True
        else:
            new_data[k] = v
    if not changed:
        return dataset
    return Dataset.create(**new_data)


_orig_make_env_and_datasets = _env_utils_mod.make_env_and_datasets


def _memsafe_make_env_and_datasets(*args, **kwargs):
    env, eval_env, train_dataset, val_dataset = _orig_make_env_and_datasets(
        *args, **kwargs)
    train_dataset = _materialize_views(train_dataset)
    val_dataset = _materialize_views(val_dataset)
    gc.collect()
    return env, eval_env, train_dataset, val_dataset


_env_utils_mod.make_env_and_datasets = _memsafe_make_env_and_datasets


# ---------------------------------------------------------------------------
# Patch 2: Dataset.normalize_observations
# ---------------------------------------------------------------------------

def _chunked_stats(arr, chunk_size=4096):
    """Compute (mean, var, min, max) along axis=0 with bounded memory.

    Streams over the input in chunks, casting each chunk to float32
    (instead of letting numpy upcast the whole array to float64), and
    accumulates mean/variance with Welford's online algorithm.
    """
    n = arr.shape[0]
    feature_shape = arr.shape[1:]

    mean = np.zeros(feature_shape, dtype=np.float64)
    M2 = np.zeros(feature_shape, dtype=np.float64)
    arr_min = None
    arr_max = None
    count = 0

    for start in range(0, n, chunk_size):
        block = np.asarray(arr[start:start + chunk_size], dtype=np.float32)
        b = block.shape[0]
        if b == 0:
            continue

        block_mean = block.mean(axis=0, dtype=np.float64)
        block_var = block.var(axis=0, dtype=np.float64)

        new_count = count + b
        delta = block_mean - mean
        mean = mean + delta * (b / new_count)
        M2 = M2 + block_var * b + (delta ** 2) * (count * b / new_count)
        count = new_count

        block_min = block.min(axis=0)
        block_max = block.max(axis=0)
        if arr_min is None:
            arr_min = block_min.astype(np.float64)
            arr_max = block_max.astype(np.float64)
        else:
            np.minimum(arr_min, block_min, out=arr_min)
            np.maximum(arr_max, block_max, out=arr_max)

    var = M2 / max(count, 1)
    return mean, var, arr_min, arr_max


def _memsafe_normalize_observations(self, observations=None):
    """Drop-in replacement for Dataset.normalize_observations."""
    if observations is None:
        assert 'observations' in self
        assert 'next_observations' in self

        obs = self['observations']

        if self.obs_norm_type == 'none':
            # 'none' branch of Dataset.normalize() ignores mean/var,
            # so skip the expensive float64 reductions. We still need
            # min/max because batches expose them as observation_min/
            # observation_max (datasets.py:185-186), which downstream
            # agents (e.g. infom.py:271) read.
            if isinstance(obs, np.ndarray):
                self.obs_min = obs.min(axis=0)
                self.obs_max = obs.max(axis=0)
            else:
                self.obs_min = np.min(obs, axis=0)
                self.obs_max = np.max(obs, axis=0)
            # normalize() with 'none' returns inputs unchanged.
            self.normalized_obs_min = self.obs_min
            self.normalized_obs_max = self.obs_max
            return

        # 'normal' or 'bounded'.
        if isinstance(obs, np.ndarray):
            self.obs_mean, self.obs_var, self.obs_min, self.obs_max = \
                _chunked_stats(obs)
        else:
            # Tree/dict observations: fall back to the original path.
            self.obs_mean = np.mean(obs, axis=0)
            self.obs_var = np.var(obs, axis=0)
            self.obs_max = np.max(obs, axis=0)
            self.obs_min = np.min(obs, axis=0)

        self.normalized_obs_max = self.normalize(
            self.obs_max, self.obs_mean, self.obs_var,
            self.obs_max, self.obs_min,
            self.obs_norm_type, self.epsilon,
        )
        self.normalized_obs_min = self.normalize(
            self.obs_min, self.obs_mean, self.obs_var,
            self.obs_max, self.obs_min,
            self.obs_norm_type, self.epsilon,
        )

        observations = self['observations']
        self._dict['observations'] = self.normalize(
            self['observations'], self.obs_mean, self.obs_var,
            self.obs_max, self.obs_min,
            self.obs_norm_type, self.epsilon,
        )
        self._dict['next_observations'] = self.normalize(
            self['next_observations'], self.obs_mean, self.obs_var,
            self.obs_max, self.obs_min,
            self.obs_norm_type, self.epsilon,
        )

    observations = self.normalize(
        observations, self.obs_mean, self.obs_var,
        self.obs_max, self.obs_min,
        self.obs_norm_type, self.epsilon,
    )
    return observations


Dataset.normalize_observations = _memsafe_normalize_observations


# ---------------------------------------------------------------------------
# Patch 3: Dataset._prestack_frames -> no-op
# ---------------------------------------------------------------------------

def _noop_prestack_frames(self):
    """No-op: keep self._prestacked = False so Dataset.sample uses the
    per-batch on-demand stacking branch at datasets.py:187-201."""
    return


Dataset._prestack_frames = _noop_prestack_frames


# ---------------------------------------------------------------------------
# Patch 4: Dataset.sample -- background prefetch
# ---------------------------------------------------------------------------

import threading  # noqa: E402
import queue as _queue  # noqa: E402

_PREFETCH_MIN_BATCH = 32        # skip prefetch for tiny setup calls
_PREFETCH_QUEUE_SIZE = 4        # batches kept in flight per dataset

_orig_sample = Dataset.sample
_prefetch_state = {}            # id(dataset) -> dict


def _prefetch_worker(dataset_ref, batch_size, q, stop_flag):
    while not stop_flag[0]:
        try:
            batch = _orig_sample(dataset_ref, batch_size, idxs=None)
        except BaseException as exc:
            # Propagate to the consumer on next get().
            try:
                q.put(('__error__', exc), timeout=1.0)
            except _queue.Full:
                pass
            return
        # Blocking put with periodic stop-check.
        while not stop_flag[0]:
            try:
                q.put(batch, timeout=0.5)
                break
            except _queue.Full:
                continue


def _prefetched_sample(self, batch_size, idxs=None):
    # Deterministic / tiny calls bypass prefetch entirely.
    if idxs is not None or batch_size < _PREFETCH_MIN_BATCH:
        return _orig_sample(self, batch_size, idxs=idxs)

    key = id(self)
    state = _prefetch_state.get(key)
    if state is None:
        q = _queue.Queue(maxsize=_PREFETCH_QUEUE_SIZE)
        stop = [False]
        t = threading.Thread(
            target=_prefetch_worker,
            args=(self, batch_size, q, stop),
            name=f'prefetch-{key:x}',
            daemon=True,
        )
        t.start()
        state = dict(queue=q, batch_size=batch_size, stop=stop, thread=t)
        _prefetch_state[key] = state

    if state['batch_size'] != batch_size:
        # Batch size changed mid-run: skip the queue for this call, keep
        # the existing worker (the eventual mismatched batches stay in
        # the queue for whenever the original batch_size resumes).
        return _orig_sample(self, batch_size, idxs=idxs)

    item = state['queue'].get()
    if isinstance(item, tuple) and len(item) == 2 and item[0] == '__error__':
        raise item[1]
    return item


Dataset.sample = _prefetched_sample


print('[run_memsafe] patches active: '
      'make_env_and_datasets (view materialization), '
      'normalize_observations (chunked + none-aware), '
      '_prestack_frames (no-op -> per-batch stacking), '
      'sample (background prefetch).',
      flush=True)


# ---------------------------------------------------------------------------
# Hand off to main.py with the patches in place.
# ---------------------------------------------------------------------------

from absl import app  # noqa: E402
from main import main  # noqa: E402


if __name__ == '__main__':
    app.run(main)
