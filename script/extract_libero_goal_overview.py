"""Extract representative images for LIBERO-goal tasks (for explanatory material).

Picks 5 representative tasks from the libero_goal suite and saves the
goal-achieved (final) frame from demo_0 for BOTH cameras (agentview and
eye_in_hand). LIBERO renders with the OpenGL convention, so frames are
flipped vertically before saving.

Outputs (under --out_dir):
  - per-image PNGs:  <task>__<camera>.png
  - a combined contact sheet: libero_goal_overview.png  (rows=tasks, cols=cameras)
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 5 representative tasks covering diverse goal types:
#   open-drawer+place / place-on-stove / place-bottle / push / turn-knob
REPRESENTATIVE_TASKS = [
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_the_rack",
    "push_the_plate_to_the_front_of_the_stove",
    "turn_on_the_stove",
]

CAMERAS = ["agentview", "eye_in_hand"]

DEFAULT_GLOB = (
    "~/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/"
    "snapshots/*/libero_goal"
)


def resolve_suite_dir(pattern):
    hits = glob.glob(os.path.expanduser(pattern))
    if not hits:
        raise FileNotFoundError(f"No libero_goal suite dir matched: {pattern}")
    return hits[0]


def load_frame(h5_path, demo, camera, frame="last", flip=True):
    cam_key = f"{camera}_rgb"
    with h5py.File(h5_path, "r") as h:
        data = h["data"]
        obs = data[demo]["obs"]
        arr = obs[cam_key]
        idx = arr.shape[0] - 1 if frame == "last" else int(frame)
        img = np.asarray(arr[idx])  # (H, W, 3) uint8
        lang = json.loads(data.attrs["problem_info"])["language_instruction"]
    if flip:
        img = img[::-1]  # OpenGL -> top-down
    return img, lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_glob", default=DEFAULT_GLOB)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--frame", default="last", help="'last' or an int frame index")
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "figures", "libero_goal_overview"),
    )
    args = ap.parse_args()

    suite_dir = resolve_suite_dir(args.suite_glob)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    flip = not args.no_flip

    n_tasks, n_cams = len(REPRESENTATIVE_TASKS), len(CAMERAS)
    fig, axes = plt.subplots(
        n_tasks,
        n_cams,
        figsize=(2.4 * n_cams + 2.0, 2.4 * n_tasks),
        gridspec_kw={"wspace": 0.02, "hspace": 0.08},
    )
    if n_tasks == 1:
        axes = axes[None, :]

    for r, task in enumerate(REPRESENTATIVE_TASKS):
        h5_path = os.path.join(suite_dir, f"{task}_demo.hdf5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
        for c, camera in enumerate(CAMERAS):
            img, lang = load_frame(h5_path, args.demo, camera, args.frame, flip)

            png_path = os.path.join(out_dir, f"{task}__{camera}.png")
            Image.fromarray(img).save(png_path)
            print(f"saved {png_path}  ({img.shape[1]}x{img.shape[0]})")

            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(camera, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(lang, fontsize=10, rotation=0, ha="right", va="center")
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

    fig.suptitle("LIBERO-goal: representative tasks (goal-achieved frame)", fontsize=14)
    sheet = os.path.join(out_dir, "libero_goal_overview.png")
    fig.savefig(sheet, dpi=150, bbox_inches="tight")
    print(f"\nsaved contact sheet: {sheet}")


if __name__ == "__main__":
    main()
