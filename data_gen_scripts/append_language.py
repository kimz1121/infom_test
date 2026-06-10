"""Append per-task SBERT language embedding to a precompute dataset's observations.

Turns obs = [resnet feat (512) | state (16)] (528-d) into
     obs = [resnet feat (512) | state (16) | language (384)] (928-d),
so the language-conditioned agent (infom_lang_state_decoder) sees language as part
of the observation (encoder-only; the decoder still targets only the state slice).

Per-row task identity comes from the stats.json cumulative T table (same logic as
visualize_latent_robocasa.derive_task_labels), so each transition gets its task's
language vector. Composite files are single-task -> one vector for all rows.

Usage:
    # atomic train / val
    python data_gen_scripts/append_language.py --obs_hdf5 .../atomic_..._pretrain.hdf5 \
        --stats .../atomic_..._pretrain.stats.json --which train \
        --lang ~/.robocasa/data/task_lang_embeddings.json --out .../atomic_..._lang_..._pretrain.hdf5
    # composite (single task)
    python data_gen_scripts/append_language.py --obs_hdf5 .../loaddishwasher_multimodal.hdf5 \
        --which composite --composite_task LoadDishwasher \
        --lang ... --out .../loaddishwasher_lang.hdf5
"""
import argparse
import json
import os.path as osp

import h5py
import numpy as np


def per_row_task(stats_path, n_rows, which):
    stats = json.load(open(osp.expanduser(stats_path)))
    key = "train_T" if which == "train" else "val_T"
    names, bounds, cur = [], [], 0
    for t, d in stats["tasks"].items():
        n = int(d[key])
        if n <= 0:
            continue
        cur += n
        names.append(t)
        bounds.append(cur)
    bounds = np.asarray(bounds)
    owner = np.clip(np.searchsorted(bounds, np.arange(n_rows), side="right"), 0, len(names) - 1)
    return [names[i] for i in owner]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_hdf5", required=True)
    ap.add_argument("--stats", default="")
    ap.add_argument("--which", choices=["train", "val", "composite"], required=True)
    ap.add_argument("--composite_task", default="")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lang = json.load(open(osp.expanduser(args.lang)))
    emb = {k: np.asarray(v, dtype=np.float32) for k, v in lang["embeddings"].items()}
    dim = lang["dim"]

    in_path = osp.expanduser(args.obs_hdf5)
    out = osp.expanduser(args.out)
    with h5py.File(in_path, "r") as fin:
        n = fin["observations"].shape[0]
        if args.which == "composite":
            if args.composite_task not in emb:
                raise KeyError(f"{args.composite_task} not in language embeddings")
            lang_rows = np.broadcast_to(emb[args.composite_task], (n, dim)).astype(np.float32)
        else:
            tasks = per_row_task(args.stats, n, args.which)
            lang_rows = np.stack([emb[t] for t in tasks], axis=0).astype(np.float32)

        # Stream key-by-key so the two big (obs/next_obs) arrays never coexist:
        # peak ~= one input + lang_rows + one output, instead of all of them at once.
        with h5py.File(out, "w") as fout:
            for k in fin.keys():
                arr = fin[k][:]
                if k in ("observations", "next_observations"):
                    arr = np.concatenate([arr, lang_rows], axis=-1)
                fout.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
                del arr
    print(f"{n} rows: appended {dim}-d language -> wrote {out}")


if __name__ == "__main__":
    main()
