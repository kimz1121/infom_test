"""Build the DINOv3 patch-TOKEN precompute dataset for the learnable-attention inFOM.

Unlike the ResNet flat pipeline (one 512-d vector/frame in a single obs array),
DINOv3 gives a small GRID of fine-grained patch tokens per frame. A downstream
*learnable* attention pool (trained inside inFOM) selects relevant tokens, so we
must store the token grid — not a pre-pooled vector.

The full 256-patch grid would be ~0.8 TB; we adaptive-pool to POOL_GRID x POOL_GRID
(+ CLS) via utils.visual_features.FrozenDINOv3Extractor. With the default 8x8+CLS
(65 tokens, 384-d, 3 cams) that is ~213 GB for atomic-65 — so obs/next_obs are NOT
duplicated. Instead frames are stored ONCE and transitions reference them by index.

Layout (per split):
  <name>_frame_tokens.f16   raw C-order float16, shape (N_frames, n_cam, n_tok, feat)
                            written sequentially episode-by-episode (O(1) RAM).
  <name>.hdf5               small metadata:
      frame_state (N_frames, state_dim) f32
      frame_lang  (N_frames, lang_dim)  f32   [only if --lang given]
      obs_idx     (N_trans,)  int64           frame index of each transition's obs
      next_idx    (N_trans,)  int64           = obs_idx + 1
      actions     (N_trans, act_dim) f32
      terminals   (N_trans,)  f32
      masks       (N_trans,)  f32
      ep_start    (N_eps,)    int64
      ep_len      (N_eps,)    int64
  <name>.stats.json         metadata incl. frame_tokens shape/dtype/file, per-task counts.

The (encoder=None flat) contract is deliberately broken here; the token-aware
streaming loader + learnable-attention inFOM agent read this layout instead.

Usage:
    python data_gen_scripts/extract_dino_tokens.py \
        --split pretrain --category atomic --tasks all \
        --name atomic_65_dinov3_tokens_3cam_state --image_size 256 --pool_grid 8
    # small validation subset:
    python data_gen_scripts/extract_dino_tokens.py --tasks CloseCabinet,OpenDrawer \
        --name dbg_dinov3_tokens --max_episodes_per_task 4
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
    split_train_val,
)
from utils.visual_features import FrozenDINOv3Extractor  # noqa: E402


def resolve_tasks(raw_root, split, category, tasks_arg) -> List[str]:
    if tasks_arg == "all":
        base = osp.join(raw_root, split, category)
        found = sorted(
            d for d in os.listdir(base)
            if osp.isdir(osp.join(base, d)) and glob.glob(osp.join(base, d, "*", "lerobot"))
        )
        if not found:
            raise FileNotFoundError(f"No extracted tasks under {base}")
        return found
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def load_lang_table(lang_path):
    """Return (embeddings_dict, dim). Keys are raw task names as embedded."""
    d = json.load(open(osp.expanduser(lang_path)))
    emb = d["embeddings"]
    dim = int(d.get("dim", len(next(iter(emb.values())))))
    return emb, dim


def lang_for_task(emb, task):
    """Match a stats task name to its SBERT vector (tolerant to spacing/case)."""
    if task in emb:
        return emb[task]
    # normalize: CamelCase / underscores -> lowercased spaced, like embed_task_language.
    import re
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", task).replace("_", " ").lower().strip()
    for k, v in emb.items():
        if k.lower().strip() == spaced:
            return v
    raise KeyError(f"No language embedding for task '{task}' (tried '{spaced}')")


class TokenSplitWriter:
    """Sequentially writes the big frame-token binary + accumulates small metadata."""

    def __init__(self, out_dir, name, n_cam, n_tok, feat_dim, state_dim, lang_dim):
        self.tokens_path = osp.join(out_dir, f"{name}_frame_tokens.f16")
        self.meta_path = osp.join(out_dir, f"{name}.hdf5")
        self.name = name
        self.n_cam, self.n_tok, self.feat_dim = n_cam, n_tok, feat_dim
        self.state_dim, self.lang_dim = state_dim, lang_dim
        self._fh = open(self.tokens_path, "wb")
        self.n_frames = 0
        self.state, self.lang = [], []
        self.obs_idx, self.next_idx = [], []
        self.actions, self.terminals, self.masks = [], [], []
        self.ep_start, self.ep_len = [], []

    def add_episode(self, tokens, state, action, lang_vec=None):
        """tokens (T,n_cam,n_tok,feat) f16; state (T,state_dim); action (T-1,act_dim)."""
        T = tokens.shape[0]
        assert state.shape[0] == T and action.shape[0] == T - 1, (T, state.shape, action.shape)
        base = self.n_frames
        # big array: sequential write, O(1) RAM.
        np.ascontiguousarray(tokens, dtype=np.float16).tofile(self._fh)
        self.state.append(np.asarray(state, dtype=np.float32))
        if lang_vec is not None:
            self.lang.append(np.tile(np.asarray(lang_vec, np.float32)[None], (T, 1)))
        # transitions t in [0, T-1): obs=frame base+t, next=base+t+1
        idx = base + np.arange(T - 1, dtype=np.int64)
        self.obs_idx.append(idx)
        self.next_idx.append(idx + 1)
        self.actions.append(np.asarray(action, dtype=np.float32))
        term = np.zeros(T - 1, dtype=np.float32); term[-1] = 1.0
        self.terminals.append(term)
        self.masks.append(1.0 - term)
        self.ep_start.append(base)
        self.ep_len.append(T)
        self.n_frames += T

    @property
    def n_trans(self):
        return int(sum(len(a) for a in self.actions))

    def finalize(self):
        import h5py
        self._fh.close()
        if self.n_frames == 0:
            # nothing written (e.g. empty val split); drop the empty binary.
            if osp.exists(self.tokens_path):
                os.remove(self.tokens_path)
            return False
        cat = lambda xs: np.concatenate(xs, axis=0)
        with h5py.File(self.meta_path, "w") as f:
            f.create_dataset("frame_state", data=cat(self.state), compression="gzip", compression_opts=4)
            if self.lang:
                f.create_dataset("frame_lang", data=cat(self.lang), compression="gzip", compression_opts=4)
            f.create_dataset("obs_idx", data=cat(self.obs_idx))
            f.create_dataset("next_idx", data=cat(self.next_idx))
            f.create_dataset("actions", data=cat(self.actions), compression="gzip", compression_opts=4)
            f.create_dataset("terminals", data=cat(self.terminals))
            f.create_dataset("masks", data=cat(self.masks))
            f.create_dataset("ep_start", data=np.asarray(self.ep_start, np.int64))
            f.create_dataset("ep_len", data=np.asarray(self.ep_len, np.int64))
        return True


def build(args):
    raw_root = osp.expanduser(args.raw_root)
    out_dir = osp.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.flat:
        args.val_fraction = 0.0
    tasks = resolve_tasks(raw_root, args.split, args.category, args.tasks)
    print(f"tasks ({len(tasks)}): {tasks}")

    cameras = [args.camera] if args.camera else \
        [c.strip() for c in args.cameras.split(",") if c.strip()]
    print(f"cameras ({len(cameras)}, in concat order): {cameras}")

    emb = lang_dim = None
    if args.lang:
        emb, lang_dim = load_lang_table(args.lang)
        print(f"language: {len(emb)} tasks, {lang_dim}-d")

    extractor = FrozenDINOv3Extractor(device=args.device, batch_size=args.batch_size,
                                      pool_grid=args.pool_grid, keep_cls=not args.no_cls)
    n_tok, feat_dim = extractor.n_tokens, extractor.feat_dim
    print(f"tokens/frame/cam = {n_tok} (pool_grid={args.pool_grid}, cls={not args.no_cls}), feat_dim={feat_dim}")

    suffix = "" if args.flat else "_pretrain"
    train_name = f"{args.name}{suffix}"
    state_dim_probe = None
    train_w = None  # created lazily once we know state_dim

    stats = dict(tasks={}, encoder="dinov3_vits16", cameras=cameras, n_cam=len(cameras),
                 n_tokens=n_tok, pool_grid=args.pool_grid, keep_cls=(not args.no_cls),
                 feat_dim=feat_dim, image_size=args.image_size,
                 lang_dim=(lang_dim or 0), has_lang=bool(args.lang))
    writers = {}
    t0 = time.time()

    for task in tasks:
        lerobot_dir = find_task_lerobot_dir(raw_root, args.split, task, category=args.category)
        eps = list_episode_parquets(lerobot_dir)
        if args.max_episodes_per_task and len(eps) > args.max_episodes_per_task:
            eps = eps[: args.max_episodes_per_task]
        tr_eps, val_eps = split_train_val(eps, args.val_fraction)
        lang_vec = lang_for_task(emb, task) if emb is not None else None
        counts = {}

        splits = [("train", tr_eps, train_name)]
        if not args.flat:
            splits.append(("val", val_eps, f"{args.name}_pretrain_val"))

        for which, ep_list, wname in splits:
            for p in ep_list:
                ep = load_episode_state(p)
                state = ep["state"]; T = len(state)
                if T < 2:
                    continue
                cam_imgs = [load_episode_image(p, lerobot_dir, c, args.image_size) for c in cameras]
                T_eff = min([T] + [len(im) for im in cam_imgs])
                if T_eff < 2:
                    continue
                # (T_eff, n_cam, n_tok, feat)
                cam_toks = [extractor.extract(im[:T_eff]) for im in cam_imgs]
                tokens = np.stack(cam_toks, axis=1).astype(np.float16)
                state = state[:T_eff].astype(np.float32)
                act = np.clip(ep["action"][: T_eff - 1], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)

                if wname not in writers:
                    writers[wname] = TokenSplitWriter(
                        out_dir, wname, len(cameras), n_tok, feat_dim,
                        state.shape[1], lang_dim or 0)
                writers[wname].add_episode(tokens, state, act, lang_vec)
                counts[which] = counts.get(which, 0) + (T_eff - 1)

        stats["tasks"][task] = dict(train_T=counts.get("train", 0), val_T=counts.get("val", 0))
        print(f"[{task}] train_T={counts.get('train',0)} val_T={counts.get('val',0)} "
              f"({time.time()-t0:.1f}s)")

    # finalize + stats per split
    for wname, w in writers.items():
        ok = w.finalize()
        if not ok:
            continue
        st = dict(stats)
        st.update(split_name=wname, n_frames=w.n_frames, n_trans=w.n_trans,
                  state_dim=w.state_dim,
                  frame_tokens_file=osp.basename(w.tokens_path),
                  frame_tokens_shape=[w.n_frames, w.n_cam, w.n_tok, w.feat_dim],
                  frame_tokens_dtype="float16")
        with open(osp.join(out_dir, f"{wname}.stats.json"), "w") as f:
            json.dump(st, f, indent=2)
        gb = w.n_frames * w.n_cam * w.n_tok * w.feat_dim * 2 / 1e9
        print(f"  wrote {wname}: n_frames={w.n_frames:,} n_trans={w.n_trans:,} "
              f"frame_tokens={gb:.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="~/.robocasa/raw")
    ap.add_argument("--out_dir", default="~/.robocasa/data")
    ap.add_argument("--split", default="pretrain", choices=["target", "pretrain"])
    ap.add_argument("--category", default="atomic", choices=["atomic", "composite"])
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cameras",
                    default="robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--pool_grid", type=int, default=8)
    ap.add_argument("--no_cls", action="store_true", help="drop the CLS token (patch grid only).")
    ap.add_argument("--lang", default=None, help="task_lang_embeddings.json to bake per-frame lang.")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--max_episodes_per_task", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
