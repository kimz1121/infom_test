"""Visualize a completed InFoM training run.

Reads pretraining_train.csv, finetuning_train.csv, and finetuning_eval.csv from a
run directory and produces PNG figures under <run_dir>/plots/.

Usage:
    python script/visualize_run.py --run_dir exp/debug/sd000_20260526_080508
    python script/visualize_run.py                       # uses latest run under exp/debug
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_run(root: Path) -> Path:
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1 or len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def plot_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    metrics: list[tuple[str, str]],
    title: str,
    smooth_window: int = 1,
    logy: bool = False,
) -> None:
    """Plot multiple (column, label) pairs on a single axes."""
    plotted = False
    for col, label in metrics:
        if col not in df.columns:
            continue
        y = df[col].astype(float)
        if y.isna().all():
            continue
        ax.plot(df[x], smooth(y, smooth_window), label=label, linewidth=1.4)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if plotted:
        ax.legend(fontsize=8, loc="best")


def plot_train_val_pair(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    base: str,
    title: str,
    smooth_window: int = 1,
    logy: bool = False,
) -> None:
    """Plot training/<base> and validation/<base> on the same axes."""
    train_col = f"training/{base}"
    val_col = f"validation/{base}"
    metrics = []
    if train_col in df.columns:
        metrics.append((train_col, "train"))
    if val_col in df.columns:
        metrics.append((val_col, "val"))
    plot_group(ax, df, x, metrics, title, smooth_window=smooth_window, logy=logy)


def make_grid(n: int, ncols: int = 3) -> tuple[plt.Figure, list[plt.Axes]]:
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    # Hide unused
    for ax in axes[n:]:
        ax.axis("off")
    return fig, list(axes[:n])


def plot_pretraining(df: pd.DataFrame, out_path: Path, smooth_window: int) -> None:
    bases = [
        ("bc/bc_loss", "BC loss", False),
        ("bc/mse", "BC MSE", False),
        ("bc/bc_log_prob", "BC log prob", False),
        ("flow_occupancy/flow_matching_loss", "Flow matching loss", False),
        ("flow_occupancy/current_flow_matching_loss", "Current flow matching", False),
        ("flow_occupancy/future_flow_matching_loss", "Future flow matching", False),
        ("flow_occupancy/kl_loss", "KL loss", False),
        ("flow_occupancy/neg_elbo_loss", "Neg ELBO", False),
    ]
    fig, axes = make_grid(len(bases) + 1, ncols=3)
    for ax, (base, title, logy) in zip(axes, bases):
        plot_train_val_pair(ax, df, "step", base, title, smooth_window=smooth_window, logy=logy)
    # Grad norm (training only)
    plot_group(
        axes[len(bases)],
        df,
        "step",
        [("training/grad/norm", "grad norm")],
        "Gradient norm",
        smooth_window=smooth_window,
    )
    fig.suptitle("Pretraining", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_finetuning(df: pd.DataFrame, out_path: Path, smooth_window: int) -> None:
    bases = [
        ("actor/actor_loss", "Actor loss", False),
        ("actor/bc_loss", "Actor BC loss", False),
        ("actor/q_loss", "Actor Q loss", False),
        ("actor/q_mean", "Actor Q mean", False),
        ("critic/critic_loss", "Critic loss", False),
        ("critic/q_mean", "Critic Q mean", False),
        ("reward/reward_loss", "Reward loss", False),
        ("flow_occupancy/flow_matching_loss", "Flow matching loss", False),
        ("flow_occupancy/kl_loss", "KL loss", False),
        ("flow_occupancy/neg_elbo_loss", "Neg ELBO", False),
    ]
    # Add a critic Q min/max range plot and grad norm
    n_extra = 2
    fig, axes = make_grid(len(bases) + n_extra, ncols=3)
    for ax, (base, title, logy) in zip(axes, bases):
        plot_train_val_pair(ax, df, "step", base, title, smooth_window=smooth_window, logy=logy)

    # Critic Q min/mean/max
    ax = axes[len(bases)]
    plot_group(
        ax,
        df,
        "step",
        [
            ("training/critic/q_min", "Q min (train)"),
            ("training/critic/q_mean", "Q mean (train)"),
            ("training/critic/q_max", "Q max (train)"),
        ],
        "Critic Q range",
        smooth_window=smooth_window,
    )
    # Grad norm
    plot_group(
        axes[len(bases) + 1],
        df,
        "step",
        [("training/grad/norm", "grad norm")],
        "Gradient norm",
        smooth_window=smooth_window,
    )

    fig.suptitle("Finetuning", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_eval(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("evaluation/episode.return", "Episode return"),
        ("evaluation/episode.success", "Success rate"),
        ("evaluation/episode.length", "Episode length"),
        ("evaluation/episode.final_reward", "Final reward"),
    ]
    available = [(c, t) for c, t in metrics if c in df.columns]
    fig, axes = make_grid(len(available), ncols=2)
    for ax, (col, title) in zip(axes, available):
        y = df[col].astype(float)
        ax.plot(df["step"], y, marker="o", linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Finetuning evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_timing(pre_df: pd.DataFrame, ft_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, df, label in zip(axes, [pre_df, ft_df], ["Pretraining", "Finetuning"]):
        if "time/epoch_time" in df.columns:
            ax.plot(df["step"], df["time/epoch_time"], linewidth=1.4)
            ax.set_title(f"{label} — epoch time (s)")
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a run directory. Defaults to the most recent under exp/debug.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window for training curves (1 = no smoothing).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if args.run_dir is None:
        run_dir = find_latest_run(project_root / "exp" / "debug")
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    print(f"Run dir : {run_dir}")
    print(f"Out dir : {out_dir}")

    # Print run config summary if present
    flags_path = run_dir / "flags.json"
    if flags_path.exists():
        with flags_path.open() as f:
            flags = json.load(f)
        summary = {
            "env_name": flags.get("env_name"),
            "agent_name": flags.get("agent", {}).get("agent_name"),
            "pretraining_steps": flags.get("pretraining_steps"),
            "finetuning_steps": flags.get("finetuning_steps"),
            "batch_size": flags.get("agent", {}).get("batch_size"),
            "lr": flags.get("agent", {}).get("lr"),
        }
        print("Config  :", json.dumps(summary, indent=2))

    pre_csv = run_dir / "pretraining_train.csv"
    ft_csv = run_dir / "finetuning_train.csv"
    eval_csv = run_dir / "finetuning_eval.csv"

    if pre_csv.exists():
        pre_df = pd.read_csv(pre_csv)
        plot_pretraining(pre_df, out_dir / "pretraining.png", args.smooth)
        print(f"  wrote {out_dir / 'pretraining.png'}  ({len(pre_df)} rows)")
    else:
        pre_df = pd.DataFrame()
        print("  skip pretraining (no CSV)")

    if ft_csv.exists():
        ft_df = pd.read_csv(ft_csv)
        plot_finetuning(ft_df, out_dir / "finetuning.png", args.smooth)
        print(f"  wrote {out_dir / 'finetuning.png'}  ({len(ft_df)} rows)")
    else:
        ft_df = pd.DataFrame()
        print("  skip finetuning (no CSV)")

    if eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)
        plot_eval(eval_df, out_dir / "finetuning_eval.png")
        print(f"  wrote {out_dir / 'finetuning_eval.png'}  ({len(eval_df)} rows)")
    else:
        print("  skip eval (no CSV)")

    if not pre_df.empty or not ft_df.empty:
        plot_timing(pre_df, ft_df, out_dir / "timing.png")
        print(f"  wrote {out_dir / 'timing.png'}")


if __name__ == "__main__":
    main()
