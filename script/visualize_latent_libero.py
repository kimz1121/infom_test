"""LIBERO analogue of script/visualize_latent_robocasa.py.

Encodes q(z|s,a) intention latents for a pretrained inFOM agent on the
LIBERO-Goal pretrain pool and projects them with t-SNE / UMAP, colored by a
chosen label axis. All the embedding / 3D / best-view / Fisher-ratio machinery
is reused verbatim from visualize_latent_robocasa; only the label axes are
LIBERO-specific.

Coloring options for LIBERO-Goal:
  --color_by task         : the 10 goal instructions (default). Each
                            instruction = one task, recovered from the
                            <basename>.stats.json cumulative-T table written by
                            data_gen_scripts/generate_libero_dataset.py.
  --color_by task_family  : verb/object instruction families (PutBowl,
                            PutWineBottle, OpenDrawer, ...). This is the
                            "instruction type" grouping.
  --color_by ep_time      : 5-bin normalized time-within-episode.
  --color_by gripper      : open/closed from the action's last dim.
  --color_by action_cluster : K-means clusters on raw actions.
  --color_by joint        : task_family x ep_time.
  --color_by both         : task + ep_time.
  --color_by all          : every axis + Fisher-ratio ranking.

Output: <run_dir>/plots/latent_libero/{tsne,umap}_<n>d_<axis>.png

Usage:
    python script/visualize_latent_libero.py --run_dir exp/debug/<run>
    python script/visualize_latent_libero.py --color_by all --n_components 3
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402

from utils.flax_utils import restore_agent  # noqa: E402
# Reuse the dataset-agnostic machinery from the robocasa script.
from script.visualize_latent_robocasa import (  # noqa: E402
    EP_TIME_BINS,
    EP_TIME_COLORS,
    derive_ep_time_labels,
    derive_action_cluster_labels,
    build_agent_and_pretrain_dataset,
    encode_latents,
    plot_embedding,
    _class_separation_score,
)

# Keep in sync with data_gen_scripts/generate_libero_dataset.py.
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

# Instruction-type families: group the 10 goals by verb + manipulated object.
# This is the LIBERO analogue of robocasa's TASK_FAMILY_MAP and is what the
# user means by "distinguish task types by instruction type".
TASK_FAMILY_MAP = {
    "put_the_bowl_on_the_plate": "PutBowl",
    "put_the_bowl_on_the_stove": "PutBowl",
    "put_the_bowl_on_top_of_the_cabinet": "PutBowl",
    "put_the_wine_bottle_on_the_rack": "PutWineBottle",
    "put_the_wine_bottle_on_top_of_the_cabinet": "PutWineBottle",
    "put_the_cream_cheese_in_the_bowl": "PutCreamCheese",
    "open_the_middle_drawer_of_the_cabinet": "OpenDrawer",
    "open_the_top_drawer_and_put_the_bowl_inside": "OpenDrawer",
    "push_the_plate_to_the_front_of_the_stove": "Push",
    "turn_on_the_stove": "TurnOnStove",
}
FAMILY_ORDER = ["PutBowl", "PutWineBottle", "PutCreamCheese",
                "OpenDrawer", "Push", "TurnOnStove"]
FAMILY_COLORS = {
    "PutBowl": "tab:blue",
    "PutWineBottle": "tab:orange",
    "PutCreamCheese": "tab:green",
    "OpenDrawer": "tab:red",
    "Push": "tab:purple",
    "TurnOnStove": "tab:brown",
}


def _task_color_map() -> dict:
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(LIBERO_GOAL_10)}


# ---------------------------------------------------------------------------
# LIBERO-specific label derivation.
# ---------------------------------------------------------------------------

def derive_task_labels(
    idxs: np.ndarray, stats_path: str, dataset_len: int,
) -> tuple[np.ndarray, list[str]]:
    """Map dataset row indices -> instruction using cumulative train_T.

    Identical mechanism to visualize_latent_robocasa.derive_task_labels: the
    HDF5 concatenates instructions in LIBERO_GOAL_10 order, so a single
    cumulative-T table recovers ownership of any row. The loader reads the
    first ``dataset_len`` rows, so trailing instructions may be absent/truncated.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    task_T = stats["tasks"]
    boundaries = [0]
    present_tasks: list[str] = []
    for t in LIBERO_GOAL_10:
        if t not in task_T:
            continue
        n = int(task_T[t]["train_T"])
        nxt = min(boundaries[-1] + n, dataset_len)
        if nxt > boundaries[-1]:
            present_tasks.append(t)
            boundaries.append(nxt)
        if nxt >= dataset_len:
            break
    bounds = np.asarray(boundaries[1:])
    owner = np.searchsorted(bounds, idxs, side="right")
    owner = np.clip(owner, 0, len(present_tasks) - 1)
    return np.array([present_tasks[i] for i in owner], dtype=object), present_tasks


def derive_task_family_labels(
    task_labels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    fam = np.array([TASK_FAMILY_MAP.get(t, "Other") for t in task_labels], dtype=object)
    present = [f for f in FAMILY_ORDER if (fam == f).any()]
    return fam, present


def derive_joint_labels(
    task_labels: np.ndarray, ep_time_labels: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    fam = np.array([TASK_FAMILY_MAP.get(t, "Other") for t in task_labels], dtype=object)
    joint = np.array([f"{f}/{e}" for f, e in zip(fam, ep_time_labels)], dtype=object)
    order = [f"{f}/{e}" for f in FAMILY_ORDER for e in EP_TIME_BINS]
    cmap = plt.get_cmap("tab20")
    color_map = {k: cmap(i % 20) for i, k in enumerate(order)}
    return joint, order, color_map


def derive_gripper_labels(
    idxs: np.ndarray, actions_all: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    """Binary open/closed from the action's last dim.

    LIBERO/robosuite OSC convention: action[..., -1] > 0 commands *close*,
    < 0 commands *open* (opposite sign meaning from some robocasa controllers).
    """
    g = actions_all[idxs, -1]
    labels = np.where(g > 0, "closed", "open").astype(object)
    return labels, ["closed", "open"], {"closed": "tab:red", "open": "tab:blue"}


def _resolve_stats_path(env_name: str, libero_dir: str) -> str:
    from envs.libero_utils import parse_env_name, DEFAULT_DATASET_DIR
    pre_name, _, _ = parse_env_name(env_name)
    base = osp.expanduser(libero_dir or DEFAULT_DATASET_DIR)
    p = osp.join(base, f"{pre_name}.stats.json")
    if not osp.exists(p):
        raise FileNotFoundError(f"Stats file not found: {p}")
    return p


# ---------------------------------------------------------------------------
# Main (mirrors visualize_latent_robocasa.main with LIBERO axes).
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir with flags.json and params_<epoch>.pkl. "
                             "Defaults to the most recent run under exp/debug/.")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=8000)
    parser.add_argument("--tsne_samples", type=int, default=2500)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--method", type=str, default="both", choices=["tsne", "umap", "both"])
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument("--sources", type=str, default="mean", choices=["mean", "sample", "both"])
    parser.add_argument("--color_by", type=str, default="task",
                        choices=["task", "ep_time", "task_family", "action_cluster",
                                 "gripper", "joint", "both", "all"],
                        help="task = 10 goal instructions (default). "
                             "task_family = verb/object instruction families. "
                             "ep_time = 5 normalized-time bins. "
                             "action_cluster = K-means on actions (--k_action). "
                             "gripper = open/closed from action last dim. "
                             "joint = task_family x ep_time. "
                             "both = task + ep_time. "
                             "all = every axis + Fisher-ratio ranking.")
    parser.add_argument("--k_action", type=int, default=8)
    parser.add_argument("--libero_dir", type=str, default="",
                        help="Override ~/.libero/data (where the .stats.json lives).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run_dir is None:
        debug_root = PROJECT_ROOT / "exp" / "debug"
        candidates = sorted(p for p in debug_root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories under {debug_root}")
        run_dir = candidates[-1]
        print(f"(no --run_dir given; using latest: {run_dir.name})")
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    print(f"Run dir : {run_dir}")
    print(f"Epoch   : {epoch}")

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)

    env_name = flags.get("env_name", "")
    if not env_name.startswith("libero_"):
        print(f"[warning] env_name={env_name!r} is not libero_*; task labels "
              f"will be unavailable. Use script/visualize_latent_robocasa.py for "
              f"robocasa runs.")

    print("Building agent + pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)

    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    rng_np = np.random.default_rng(args.seed)
    idxs = rng_np.integers(0, len(raw_obs), size=args.num_samples)
    batch = pre_train.sample(args.num_samples, idxs=idxs)

    rng = jax.random.PRNGKey(args.seed)
    mean, std, z_sample = encode_latents(agent, batch, rng)
    print(
        f"Encoded {args.num_samples} samples. "
        f"mu range [{mean.min():+.3f}, {mean.max():+.3f}], "
        f"sigma range [{std.min():.3f}, {std.max():.3f}]"
    )

    terminals_all = np.asarray(pre_train["terminals"])

    if args.color_by == "all":
        requested = {"task", "ep_time", "task_family", "action_cluster", "gripper", "joint"}
    elif args.color_by == "both":
        requested = {"task", "ep_time"}
    else:
        requested = {args.color_by}

    needs_task = bool(requested & {"task", "task_family", "joint"})
    needs_ep = bool(requested & {"ep_time", "joint"})
    task_labels = present_tasks = ep_labels = None
    if needs_task:
        stats_path = _resolve_stats_path(env_name, args.libero_dir)
        task_labels, present_tasks = derive_task_labels(idxs, stats_path, len(raw_obs))
    if needs_ep:
        ep_labels = derive_ep_time_labels(idxs, terminals_all)

    actions_all = None
    if requested & {"action_cluster", "gripper"}:
        actions_all = np.asarray(pre_train["actions"])

    color_axes: dict[str, tuple[np.ndarray, list[str], dict]] = {}
    if "task" in requested:
        color_axes["task"] = (task_labels, present_tasks, _task_color_map())
        counts = {t: int((task_labels == t).sum()) for t in present_tasks}
        print(f"Task labels — present={len(present_tasks)}/10, counts={counts}")
    if "ep_time" in requested:
        color_axes["ep_time"] = (ep_labels, EP_TIME_BINS, EP_TIME_COLORS)
        ep_counts = {b: int((ep_labels == b).sum()) for b in EP_TIME_BINS}
        print(f"Episode-time labels — counts={ep_counts}")
    if "task_family" in requested:
        fam_labels, present_families = derive_task_family_labels(task_labels)
        color_axes["task_family"] = (fam_labels, present_families, FAMILY_COLORS)
        fam_counts = {f: int((fam_labels == f).sum()) for f in present_families}
        print(f"Task-family labels — counts={fam_counts}")
    if "action_cluster" in requested:
        ac_labels, ac_order, ac_cmap = derive_action_cluster_labels(
            idxs, actions_all, args.k_action, args.seed,
        )
        centroids = ac_cmap.pop("__centroids__", None)
        color_axes["action_cluster"] = (ac_labels, ac_order, ac_cmap)
        ac_counts = {c: int((ac_labels == c).sum()) for c in ac_order}
        print(f"Action-cluster labels (k={args.k_action}) — counts={ac_counts}")
        if centroids is not None:
            print("Centroid actions (one row per cluster):")
            for i, row in enumerate(centroids):
                pretty = ", ".join(f"{v:+.2f}" for v in row)
                print(f"  a{i}: [{pretty}]")
    if "gripper" in requested:
        g_labels, g_order, g_cmap = derive_gripper_labels(idxs, actions_all)
        color_axes["gripper"] = (g_labels, g_order, g_cmap)
        g_counts = {c: int((g_labels == c).sum()) for c in g_order}
        print(f"Gripper labels — counts={g_counts}")
    if "joint" in requested:
        j_labels, j_order, j_cmap = derive_joint_labels(task_labels, ep_labels)
        color_axes["joint"] = (j_labels, j_order, j_cmap)
        j_counts = {c: int((j_labels == c).sum()) for c in j_order if (j_labels == c).any()}
        print(f"Joint (family x ep_time) labels — non-empty cells={len(j_counts)}/{len(j_order)}")

    out_dir = run_dir / "plots" / "latent_libero"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_z_sources = {"mean": mean, "sample": z_sample}
    if args.sources == "both":
        z_sources = all_z_sources
    else:
        z_sources = {args.sources: all_z_sources[args.sources]}
    print(f"Visualizing sources: {list(z_sources.keys())}")

    methods = ["tsne", "umap"] if args.method == "both" else [args.method]
    summary = {
        "epoch": epoch, "env_name": env_name,
        "num_samples_encoded": args.num_samples,
        "latent_dim": int(mean.shape[1]),
        "posterior_stats": {
            "mu_min": float(mean.min()), "mu_max": float(mean.max()),
            "mu_abs_mean": float(np.abs(mean).mean()),
            "sigma_min": float(std.min()), "sigma_max": float(std.max()),
            "sigma_mean": float(std.mean()),
        },
        "by_axis": {},
    }

    fisher_full: dict[str, dict[str, float]] = {}
    for axis_name, (labels, _, _) in color_axes.items():
        fisher_full[axis_name] = {
            src_name: float(_class_separation_score(z, labels))
            for src_name, z in z_sources.items()
        }
    summary["fisher_full_latent"] = fisher_full
    for axis_name, (labels, label_order, color_map) in color_axes.items():
        per_method_info: dict[str, dict] = {}
        samples_plotted = None
        for method in methods:
            out_path = out_dir / f"{method}_{args.n_components}d_{axis_name}.png"
            print(
                f"[{axis_name}] {method.upper()} n_components={args.n_components} "
                f"(max_total={args.tsne_samples}) ..."
            )
            info = plot_embedding(
                z_sources, labels, label_order, color_map, out_path,
                method=method, n_components=args.n_components,
                max_total=args.tsne_samples,
                perplexity=args.tsne_perplexity, n_iter=args.tsne_iter,
                n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
                seed=args.seed,
            )
            per_method_info[method] = info["per_source_info"]
            samples_plotted = info["per_label_counts"]
            print(f"  Saved → {out_path}")
        summary["by_axis"][axis_name] = {
            "samples_plotted": samples_plotted,
            "embedding_info": per_method_info,
            "label_distribution_total": {
                c: int((labels == c).sum()) for c in label_order
            },
        }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    flat = [
        (axis, src, score)
        for axis, srcs in fisher_full.items() for src, score in srcs.items()
    ]
    flat.sort(key=lambda x: -x[2])
    print("\n=== Fisher ratio ranking (between/within variance, full latent) ===")
    for axis, src, score in flat:
        print(f"  {axis:14s} ({src:6s})  {score:.4f}")
    if flat:
        top_axis, top_src, top_score = flat[0]
        print(f"\nMost separable axis: {top_axis} ({top_src})  score={top_score:.4f}")
        print(f"Open plots: {out_dir}/tsne_{args.n_components}d_{top_axis}.png "
              f"(also umap_*_{top_axis}.png)")


if __name__ == "__main__":
    main()
