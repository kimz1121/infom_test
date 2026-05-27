"""Visualize the InFoM intention-encoder latent distribution q(z | obs, action).

Loads a trained InFoM checkpoint, samples a batch of (obs, action) transitions
from the dataset, runs them through the intention encoder, and produces
several diagnostic plots under <run_dir>/plots/latent/.

What is plotted:
  1. Per-dim activity: variance across the batch of the posterior mean per dim.
     Sorted descending — flat tail = "collapsed" / inactive dims.
  2. Per-dim posterior std: average q(z|x).stddev() per dim.
     Values close to 1 mean q(z|x) ≈ prior N(0,1) (no info encoded).
  3. Global z-element histogram vs N(0,1).
  4. Per-dim KL[q(z|x) || N(0,I)] averaged over the batch.
  5. PCA 2D scatter of sampled z, optionally comparing two splits.
  6. t-SNE 2D scatter of sampled z, optionally comparing two splits.

Usage:
    python script/visualize_latent.py --run_dir exp/debug/sd000_20260526_080508
    python script/visualize_latent.py --epoch 1500000 --num_samples 4096
    python script/visualize_latent.py --tsne_samples 1500 --tsne_perplexity 40
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Reduce JAX/TF GPU memory footprint for a quick analysis run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # latent inference is cheap; avoid GPU contention
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import jax
import jax.numpy as jnp
import ml_collections

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents import agents  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


def build_agent_and_datasets(flags: dict):
    """Reproduce main.py's dataset + agent construction enough to run the encoder."""
    env_name = flags["env_name"]
    pretraining_size = flags["pretraining_size"]
    finetuning_size = flags["finetuning_size"]
    obs_norm_type = flags["obs_norm_type"]
    config = ml_collections.ConfigDict(flags["agent"])

    _, _, pre_train_raw, pre_val_raw = make_env_and_datasets(
        env_name, frame_stack=flags.get("frame_stack"), max_size=pretraining_size, reward_free=True
    )
    _, _, ft_train_raw, ft_val_raw = make_env_and_datasets(
        env_name, frame_stack=flags.get("frame_stack"), max_size=finetuning_size, reward_free=False
    )

    pre_train = Dataset.create(**pre_train_raw)
    ft_train = Dataset.create(**ft_train_raw)
    pre_val = Dataset.create(**pre_val_raw) if pre_val_raw is not None else None
    ft_val = Dataset.create(**ft_val_raw) if ft_val_raw is not None else None

    for dataset in [pre_train, pre_val, ft_train, ft_val]:
        if dataset is None:
            continue
        dataset.obs_norm_type = obs_norm_type
        dataset.p_aug = flags.get("p_aug")
        dataset.num_aug = flags.get("num_aug", 1)
        dataset.inplace_aug = flags.get("inplace_aug", 1)
        dataset.frame_stack = flags.get("frame_stack")
        dataset.return_next_actions = True
        dataset.normalize_observations()

    example = pre_train.sample(1)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        flags["seed"], example["observations"], example["actions"], config
    )
    return agent, {"pretrain": pre_train, "finetune": ft_train}


def encode_latents(agent, batch, rng):
    """Run the intention encoder on (obs, action) → returns (mean, std, sample) numpy arrays."""
    obs = jnp.asarray(batch["observations"])
    act = jnp.asarray(batch["actions"])
    latent_dist = agent.network.select("intention_encoder")(obs, act)
    mean = np.asarray(latent_dist.mean())
    std = np.asarray(latent_dist.stddev())
    sample = np.asarray(latent_dist.sample(seed=rng))
    return mean, std, sample


def kl_to_standard_normal(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Closed-form KL[N(mu, sigma^2) || N(0, 1)] per element. Shape: (B, D)."""
    var = std ** 2
    return 0.5 * (mean ** 2 + var - 1.0 - 2.0 * np.log(std + 1e-12))


def plot_per_dim_activity(mean: np.ndarray, std: np.ndarray, out_path: Path) -> None:
    """Per-dim posterior mean variance + per-dim posterior std, sorted by activity."""
    activity = mean.var(axis=0)  # (D,)
    order = np.argsort(activity)[::-1]
    mean_var_sorted = activity[order]
    std_mean_sorted = std.mean(axis=0)[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(mean_var_sorted, linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_title("Per-dim activity: Var_batch(μ_d)\n(higher = more informative)")
    axes[0].set_xlabel("latent dim (sorted)")
    axes[0].set_ylabel("variance of posterior mean across batch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(std_mean_sorted, linewidth=1.2)
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.6, label="prior std = 1")
    axes[1].set_title("Per-dim mean posterior std σ_d\n(near 1 = collapsed to prior)")
    axes[1].set_xlabel("latent dim (sorted by activity)")
    axes[1].set_ylabel("E_batch[σ_d]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    n_active = int((activity > 0.01).sum())
    fig.suptitle(f"Active dims (Var > 0.01): {n_active} / {mean.shape[1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_global_histogram(sample: np.ndarray, mean: np.ndarray, out_path: Path) -> None:
    """Histogram of sampled z and posterior means, overlaid with N(0,1)."""
    xs = np.linspace(-5, 5, 400)
    pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [sample.ravel(), mean.ravel()], ["z ~ q(z|x)", "posterior mean μ"]):
        ax.hist(data, bins=120, density=True, alpha=0.6, label=label)
        ax.plot(xs, pdf, "r--", linewidth=1.4, label="N(0,1)")
        ax.set_title(f"{label}  (mean={data.mean():+.3f}, std={data.std():.3f})")
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_per_dim_kl(mean: np.ndarray, std: np.ndarray, out_path: Path) -> None:
    kl = kl_to_standard_normal(mean, std).mean(axis=0)  # (D,)
    order = np.argsort(kl)[::-1]
    kl_sorted = kl[order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kl_sorted, linewidth=1.2)
    ax.set_title(
        f"Per-dim KL[q(z|x) || N(0,1)]   total KL = {kl.sum():.2f} nats   "
        f"(top-10 carry {100 * kl_sorted[:10].sum() / max(kl.sum(), 1e-9):.1f}%)"
    )
    ax.set_xlabel("latent dim (sorted)")
    ax.set_ylabel("E_batch[KL_d]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (projected (N,2), explained variance ratio (2,)) using numpy SVD."""
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    total_var = (s ** 2).sum()
    evr = (s[:2] ** 2) / max(total_var, 1e-12)
    return proj, evr


def plot_tsne(
    samples: dict[str, np.ndarray],
    out_path: Path,
    *,
    max_per_split: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 0,
) -> None:
    """t-SNE 2D scatter of sampled z — fit jointly on all splits, color per split.

    t-SNE has no out-of-sample projection, so we concatenate all splits, fit
    once on the union, then slice the embedding back per split for coloring.
    Each split is subsampled to ``max_per_split`` for tractable runtime.
    """
    rng = np.random.default_rng(seed)
    sub: dict[str, np.ndarray] = {}
    for name, x in samples.items():
        if len(x) > max_per_split:
            idx = rng.choice(len(x), size=max_per_split, replace=False)
            sub[name] = x[idx]
        else:
            sub[name] = x

    counts = [len(v) for v in sub.values()]
    boundaries = np.cumsum([0] + counts)
    stacked = np.concatenate(list(sub.values()), axis=0)

    # If z is very high-dim, PCA-preprocess to ~50 dims first (standard t-SNE practice).
    if stacked.shape[1] > 50:
        centered = stacked - stacked.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        stacked_in = centered @ vt[:50].T
        init_note = " (PCA-50 preprocessed)"
    else:
        stacked_in = stacked
        init_note = ""

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(stacked_in) - 1) / 3.0)),
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    embedded = tsne.fit_transform(stacked_in)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, _), color, lo, hi in zip(sub.items(), colors, boundaries[:-1], boundaries[1:]):
        e = embedded[lo:hi]
        ax.scatter(e[:, 0], e[:, 1], s=6, alpha=0.45, label=f"{name} (n={hi - lo})", color=color)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        f"t-SNE of sampled z ~ q(z | obs, action){init_note}\n"
        f"perplexity={tsne.perplexity:.1f}, max_iter={n_iter}, kl={tsne.kl_divergence_:.3f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pca(samples: dict[str, np.ndarray], out_path: Path) -> None:
    """PCA of z samples — fit on the union, project each split."""
    stacked = np.concatenate(list(samples.values()), axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    total_var = (s ** 2).sum()
    evr = (s[:2] ** 2) / max(total_var, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, x), color in zip(samples.items(), colors):
        c = (x - stacked.mean(axis=0, keepdims=True)) @ components.T
        ax.scatter(c[:, 0], c[:, 1], s=6, alpha=0.4, label=f"{name} (n={len(x)})", color=color)
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var)")
    ax.set_title("PCA of sampled z ~ q(z | obs, action)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", type=str, default="exp/debug/sd000_20260526_080508",
        help="Run directory containing flags.json and params_<epoch>.pkl",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Checkpoint epoch to load. Defaults to the largest available.",
    )
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tsne_samples", type=int, default=2000,
        help="Max points per split fed to t-SNE (O(N^2) so keep ≲ a few thousand).",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--skip_tsne", action="store_true", help="Skip the t-SNE plot.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # Find checkpoint (sort numerically, not lexicographically).
    ckpts = sorted(
        run_dir.glob("params_*.pkl"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    if args.epoch is None:
        epoch = int(ckpts[-1].stem.split("_")[1])
    else:
        epoch = args.epoch
    print(f"Run dir   : {run_dir}")
    print(f"Epoch     : {epoch}")

    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)

    print(f"Building agent + datasets for env_name={flags['env_name']} ...")
    agent, datasets = build_agent_and_datasets(flags)

    print(f"Restoring agent from epoch {epoch} ...")
    agent = restore_agent(agent, str(run_dir), epoch)

    rng = jax.random.PRNGKey(args.seed)

    print(f"Encoding {args.num_samples} samples per split ...")
    means_by_split: dict[str, np.ndarray] = {}
    stds_by_split: dict[str, np.ndarray] = {}
    samples_by_split: dict[str, np.ndarray] = {}
    for split_name, dataset in datasets.items():
        batch = dataset.sample(args.num_samples)
        rng, enc_rng = jax.random.split(rng)
        m, s, z = encode_latents(agent, batch, enc_rng)
        means_by_split[split_name] = m
        stds_by_split[split_name] = s
        samples_by_split[split_name] = z
        print(
            f"  [{split_name}]  μ range [{m.min():+.3f}, {m.max():+.3f}]   "
            f"σ range [{s.min():.3f}, {s.max():.3f}]   z var = {z.var():.3f}"
        )

    out_dir = run_dir / "plots" / "latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {out_dir} ...")

    # Use the pretrain split for the per-dim diagnostics — it matches the q(z|x) training distribution.
    diag_mean = means_by_split["pretrain"]
    diag_std = stds_by_split["pretrain"]
    diag_sample = samples_by_split["pretrain"]

    plot_per_dim_activity(diag_mean, diag_std, out_dir / "per_dim_activity.png")
    plot_global_histogram(diag_sample, diag_mean, out_dir / "global_histogram.png")
    plot_per_dim_kl(diag_mean, diag_std, out_dir / "per_dim_kl.png")
    plot_pca(samples_by_split, out_dir / "pca_2d.png")
    if not args.skip_tsne:
        print(
            f"Running t-SNE (max_per_split={args.tsne_samples}, "
            f"perplexity={args.tsne_perplexity}, n_iter={args.tsne_iter}) ..."
        )
        plot_tsne(
            samples_by_split,
            out_dir / "tsne_2d.png",
            max_per_split=args.tsne_samples,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            seed=args.seed,
        )

    # Summary statistics
    summary = {
        "epoch": epoch,
        "latent_dim": int(diag_mean.shape[1]),
        "num_samples_per_split": args.num_samples,
        "splits": {},
    }
    for name in means_by_split:
        m, s = means_by_split[name], stds_by_split[name]
        kl = kl_to_standard_normal(m, s)
        summary["splits"][name] = {
            "mean_of_means": float(m.mean()),
            "std_of_means": float(m.std()),
            "mean_posterior_std": float(s.mean()),
            "n_active_dims_var_gt_0p01": int((m.var(axis=0) > 0.01).sum()),
            "total_kl_per_sample": float(kl.sum(axis=1).mean()),
        }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
