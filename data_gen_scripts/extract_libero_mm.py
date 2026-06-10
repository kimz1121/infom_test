"""Build a LIBERO multimodal-precompute dataset: frozen resnet34(image) + proprio.

LIBERO already ships row-aligned image and state HDF5s (same episodes/order from
generate_libero_dataset.py). This fuses them into the flat
    obs = [ frozen_resnet34(agentview image) (512) | proprio state (15) ] = 527-d
exactly like the robocasa precompute, so the state-style loader + state-decoder
inFOM (encoder sees image+state, decoder targets only the proprio slice) trains
on it unchanged. Image features are encoder-only (never decoded).

Usage:
    python data_gen_scripts/extract_libero_mm.py \
        --img_base ~/.libero/data/libero_goal_image \
        --state_base ~/.libero/data/libero_goal_state \
        --out_base ~/.libero/data/libero_goal_multimodal_state
"""
import argparse, json, os.path as osp
import h5py
import numpy as np
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.visual_features import FrozenResNet34Extractor


def process(img_path, state_path, out_path, extractor):
    with h5py.File(osp.expanduser(img_path), "r") as fi, \
         h5py.File(osp.expanduser(state_path), "r") as fs:
        assert fi["actions"].shape[0] == fs["actions"].shape[0], "img/state row mismatch"
        n = fi["actions"].shape[0]
        img_obs = fi["observations"][:]; img_next = fi["next_observations"][:]
        st_obs = fs["observations"][:].astype(np.float32)
        st_next = fs["next_observations"][:].astype(np.float32)
        out = {
            "observations": extractor.fuse(img_obs, st_obs),
            "next_observations": extractor.fuse(img_next, st_next),
            "actions": fs["actions"][:].astype(np.float32),
            "terminals": fs["terminals"][:].astype(np.float32),
            "masks": fs["masks"][:].astype(np.float32),
        }
    with h5py.File(osp.expanduser(out_path), "w") as fo:
        for k, v in out.items():
            fo.create_dataset(k, data=v, compression="gzip", compression_opts=4)
    print(f"  {n} rows -> {out['observations'].shape[-1]}-d  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_base", required=True)
    ap.add_argument("--state_base", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    ext = FrozenResNet34Extractor(device=args.device, batch_size=args.batch_size)
    for split in ["_pretrain", "_pretrain_val"]:
        img = f"{args.img_base}{split}.hdf5"
        st = f"{args.state_base}{split}.hdf5"
        if not osp.exists(osp.expanduser(img)) or not osp.exists(osp.expanduser(st)):
            print(f"  [skip {split}: missing img or state file]")
            continue
        print(f"[{split}]")
        process(img, st, f"{args.out_base}{split}.hdf5", ext)

    # Stats: reuse the STATE stats (tasks/instructions/state_keys), tag with feature info.
    ss = osp.expanduser(f"{args.state_base}_pretrain.stats.json")
    if osp.exists(ss):
        stats = json.load(open(ss))
        stats.update(feature="frozen_resnet34_imagenet", feat_dim=ext.feat_dim,
                     image_feat_dim=ext.feat_dim, modality="state_multimodal")
        out_stats = osp.expanduser(f"{args.out_base}_pretrain.stats.json")
        with open(out_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  wrote {out_stats}")


if __name__ == "__main__":
    main()
