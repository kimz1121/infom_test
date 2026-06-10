"""Extract ONE atomic-task episode from a precompute pretrain HDF5 into a flat
single-episode HDF5 (composite-style), so render_composite_video.py can render it
as a demo on the atomic backdrop. The k-th episode of a task in the concatenated
pretrain HDF5 aligns with raw mp4 episode_{k} of that task (extract_mm_features
preserves task + episode order).

Usage:
    python data_gen_scripts/extract_atomic_demo.py \
        --pretrain_hdf5 ~/.robocasa/data/atomic_65_multimodal_precompute_3cam_state_pretrain.hdf5 \
        --stats        ~/.robocasa/data/atomic_65_multimodal_precompute_3cam_state_pretrain.stats.json \
        --task OpenDrawer --episode 0 \
        --out ~/.robocasa/data/atomic_OpenDrawer_3cam.hdf5
"""
import argparse, json, os.path as osp
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_hdf5", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, default=0, help="episode index within the task")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = json.load(open(osp.expanduser(args.stats)))
    # cumulative train_T (stats task insertion order == HDF5 concat order).
    start = 0
    for t, d in stats["tasks"].items():
        n = int(d["train_T"])
        if t == args.task:
            break
        start += n
    else:
        raise SystemExit(f"task {args.task!r} not in stats")
    end_task = start + n  # task row range [start, end_task)

    with h5py.File(osp.expanduser(args.pretrain_hdf5), "r") as f:
        term = f["terminals"][start:end_task]
        ep_ends = np.nonzero(term > 0)[0]  # relative to `start`
        if args.episode >= len(ep_ends):
            raise SystemExit(f"task has {len(ep_ends)} episodes; episode {args.episode} OOR")
        e_rel = int(ep_ends[args.episode])
        s_rel = 0 if args.episode == 0 else int(ep_ends[args.episode - 1]) + 1
        sl = slice(start + s_rel, start + e_rel + 1)
        out = {k: f[k][sl] for k in ["observations", "next_observations", "actions",
                                     "terminals", "masks"]}
    out["terminals"] = out["terminals"].astype(np.float32); out["terminals"][-1] = 1.0
    out["masks"] = 1.0 - out["terminals"]
    with h5py.File(osp.expanduser(args.out), "w") as fo:
        for k, v in out.items():
            fo.create_dataset(k, data=v, compression="gzip", compression_opts=4)
    print(f"{args.task} ep{args.episode}: {len(out['actions'])} steps, obs "
          f"{out['observations'].shape[-1]}-d -> {args.out}")


if __name__ == "__main__":
    main()
