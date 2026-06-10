"""Embedding-distance vs geometric-trajectory-distance (DTW) analysis.

Research question
-----------------
The InFOM intention latent z = q(z|s,a) is trained against a *discounted*
state-occupancy measure (flow_occupancy_loss weights current/future by
1-gamma / gamma; see agents/infom.py:172-173). Hypothesis: if two trajectories
land close in embedding space, their raw state / action *trajectories* should
also be geometrically similar. This script tests that by correlating, over
episode PAIRS:

  embedding distance   ||mean_z_i - mean_z_j||   (euclidean and cosine)
      vs.
  geometric distance   DTW(traj_i, traj_j)       (per-dim z-scored sequences)

Geometric distance is computed over three trajectory representations
    {state, action, state+action}
crossed with three discount treatments
    plain        : standard, normalized by alignment-path length (DTW only).
    truncated    : over the first H = round(1/(1-gamma)) steps only
                   (matches the embedding's effective discount horizon).
    discounted   : local cost weighted by gamma**((i+j)/2),
                   normalized by the path's accumulated weight (DTW). The
                   analogue of the embedding's discounting -- lets you check
                   whether the embedding tracks a *discounted* geometric distance
                   better than an undiscounted one (the central worry here).
optionally crossed with extra trajectory-distance MEASURES (--measures, default
"dtw" -> identical behaviour/outputs to before; opt in with
--measures dtw,frechet,hausdorff). Every measure is a reduction over the SAME
(possibly truncated/discounted) local-cost matrix L = ||x_i - y_j|| -- they
differ ONLY in how they aggregate it, which is exactly what we want to compare:
    dtw          : sum-min along a monotone warp (many-to-many), normalized.
    frechet      : discrete Frechet -- min over monotone couplings of the MAX
                   coupled cost (bottleneck). Same alignment family as DTW but a
                   max instead of a sum: sensitive to the single worst-aligned
                   frame rather than the accumulated mismatch.
    hausdorff    : max over each point of its nearest neighbour in the other
                   sequence (symmetric). ORDER-AGNOSTIC -- ignores time entirely,
                   so contrasting it with DTW/Frechet tells you whether the
                   embedding cares about temporal ordering or only spatial cover.
=> 9 DTW matrices by default; 9 per extra measure when requested.

A correlation across all pairs can be driven purely by task identity (same-task
pairs are close in both spaces). We therefore also report Spearman restricted to
cross-task pairs, and color the scatter by same/cross task.

Key efficiency note: the 9 geometric matrices depend only on raw trajectories,
NOT on the checkpoint. They are computed ONCE; each requested epoch only re-
encodes embeddings (cheap) and recomputes correlations. So multi-epoch is nearly
free on top of a single epoch.

Outputs under <run_dir>/plots/trajectory_dtw/ (DTW names UNCHANGED; extra
measures only appear when requested via --measures):
  corr_summary.json                      all correlations + per-epoch rankings
                                         (DTW under the original keys; extra
                                         measures additive under by_epoch[ep]["measures"])
  scatter_<epoch>_<embmetric>.png        3x3 grid (rep x variant), colored by same/cross task
  corr_bars_<epoch>.png                  Spearman bar chart across the 9 geo distances
  dtw_<rep>_<variant>_heatmap.png        episode x episode geometric distance (epoch-independent)
  emb_<epoch>_<embmetric>_heatmap.png    episode x episode embedding distance
  -- extra measures (only with --measures ...) --
  measure_compare_<epoch>.png            dtw vs frechet vs hausdorff Spearman (headline)
  scatter_<epoch>_<embmetric>_<measure>.png   per-measure scatter grid
  corr_bars_<epoch>_<measure>.png        per-measure Spearman bars
  <measure>_<rep>_<variant>_heatmap.png  per-measure geometric distance heatmap

Usage:
    python script/analyze_trajectory_dtw.py --run_dir exp/atomic_65_multimodal_precompute_mlpconcat/<run>
    python script/analyze_trajectory_dtw.py --run_dir <run> --epochs 500000,200000,50000
    python script/analyze_trajectory_dtw.py --run_dir <run> --n_episodes 120 --max_len 250
    python script/analyze_trajectory_dtw.py --run_dir <run> --measures dtw,frechet,hausdorff
    python script/analyze_trajectory_dtw.py --run_dir <run> --timing_probe   # estimate runtime, then exit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr, pearsonr

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax  # noqa: E402

from utils.flax_utils import restore_agent  # noqa: E402
from script.visualize_latent_robocasa import (  # noqa: E402
    build_agent_and_pretrain_dataset,
    encode_latents,
    derive_task_labels,
    family_of,
    FAMILY_ORDER,
    FAMILY_COLORS,
    _resolve_stats_path,
)

REPS = ["state", "action", "state+action"]
VARIANTS = ["plain", "truncated", "discounted"]
EMB_METRICS = ["euclidean", "cosine", "mahalanobis"]
MEASURES = ["dtw", "frechet", "hausdorff"]  # selectable via --measures; default dtw only

# Local (point-to-point) distance for the cost matrix L -- an axis ORTHOGONAL to the
# reduction (dtw/frechet/hausdorff): it sets L[i,j], which every measure then reduces.
# L1=cityblock suits robot joint-configuration space (sum of per-joint deltas); L2 is
# the usual euclidean. The module default stays L2 so importers / prior outputs are
# byte-for-byte unchanged; the CLI (main) defaults to L1 and sets this global.
LOCAL_METRICS = {"L1": "cityblock", "L2": "euclidean"}
_LOCAL_METRIC_CDIST = "euclidean"


# ---------------------------------------------------------------------------
# DTW core (numba). Returns (raw_cost_along_path, path_len, weight_sum).
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _dtw_path_stats(L, W, use_w):
    """DTW DP over local-cost matrix L; backtrack the optimal path.

    L[i,j] is the (possibly weighted) local cost of aligning frame i to j.
    W[i,j] is the weight applied (only read when use_w); used so the caller can
    normalize a *discounted* DTW by the path's accumulated weight instead of its
    raw length. Returns (sum L over path, number of path cells, sum W over path).
    """
    Tx, Ty = L.shape
    D = np.full((Tx + 1, Ty + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Tx + 1):
        for j in range(1, Ty + 1):
            m = D[i - 1, j - 1]
            d1 = D[i - 1, j]
            d2 = D[i, j - 1]
            if d1 < m:
                m = d1
            if d2 < m:
                m = d2
            D[i, j] = L[i - 1, j - 1] + m

    i = Tx
    j = Ty
    raw = 0.0
    plen = 0.0
    wsum = 0.0
    while i > 0 and j > 0:
        raw += L[i - 1, j - 1]
        plen += 1.0
        if use_w:
            wsum += W[i - 1, j - 1]
        diag = D[i - 1, j - 1]
        up = D[i - 1, j]
        left = D[i, j - 1]
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    return raw, plen, wsum


@njit(cache=True, fastmath=True)
def _frechet_coupling(L):
    """Discrete Frechet distance: bottleneck over the optimal monotone coupling.

    ca(i,j) = max( L[i,j], min( ca(i-1,j), ca(i-1,j-1), ca(i,j-1) ) ); the result
    ca(Tx-1,Ty-1) is the smallest "leash length" over all monotone couplings --
    i.e. the single worst aligned-frame cost on the best alignment, in contrast to
    DTW which sums aligned costs. Iterative (row-major) DP so the recursion can't
    overflow the stack for long sequences.
    """
    Tx, Ty = L.shape
    ca = np.empty((Tx, Ty))
    for i in range(Tx):
        for j in range(Ty):
            d = L[i, j]
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                prev = ca[i, j - 1]
                ca[i, j] = prev if prev > d else d
            elif j == 0:
                prev = ca[i - 1, j]
                ca[i, j] = prev if prev > d else d
            else:
                m = ca[i - 1, j - 1]
                if ca[i - 1, j] < m:
                    m = ca[i - 1, j]
                if ca[i, j - 1] < m:
                    m = ca[i, j - 1]
                ca[i, j] = m if m > d else d
    return ca[Tx - 1, Ty - 1]


def _local_cost(a, b, variant, gamma, horizon):
    """Return (L, W) where L is the (possibly truncated/discounted) local-cost
    matrix shared by every measure, and W the discount weights (None unless
    discounted). All measures reduce the SAME L -- only the reduction differs."""
    if variant == "truncated":
        a = a[:horizon]
        b = b[:horizon]
    cost = cdist(a, b, metric=_LOCAL_METRIC_CDIST)
    if variant == "discounted":
        ii = np.arange(a.shape[0])[:, None]
        jj = np.arange(b.shape[0])[None, :]
        w = gamma ** ((ii + jj) / 2.0)
        return (cost * w).astype(np.float64), w.astype(np.float64)
    return cost.astype(np.float64), None


def geo_distance(a, b, measure, variant, gamma, horizon):
    """Geometric trajectory distance for one (measure, variant) over (T, D) seqs.

    All measures operate on the shared local-cost matrix L = _local_cost(...).
    They differ only in the reduction: dtw sums along a monotone warp (normalized
    by path length / discount weight), frechet takes the bottleneck (max) over the
    optimal monotone coupling, hausdorff takes the order-agnostic symmetric
    max-of-nearest-neighbour. frechet/hausdorff are maxes of (weighted) euclidean
    costs, already comparable across pairs, so they are NOT length-normalized.
    """
    L, w = _local_cost(a, b, variant, gamma, horizon)
    if measure == "dtw":
        if variant == "discounted":
            raw, _plen, wsum = _dtw_path_stats(L, w, True)
            return raw / max(wsum, 1e-12)
        raw, plen, _wsum = _dtw_path_stats(L, L, False)
        return raw / max(plen, 1e-12)
    if measure == "frechet":
        return float(_frechet_coupling(L))
    if measure == "hausdorff":
        # directed A->B: max_i min_j L[i,j]; B->A: max_j min_i L[i,j]; symmetric max.
        d_ab = L.min(axis=1).max()
        d_ba = L.min(axis=0).max()
        return float(max(d_ab, d_ba))
    raise ValueError(f"unknown measure {measure!r}")


def geo_matrix(seqs, measure, variant, gamma, horizon):
    """Symmetric (N, N) geometric distance matrix over a list of (T, D) sequences."""
    n = len(seqs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = geo_distance(seqs[i], seqs[j], measure, variant, gamma, horizon)
            mat[i, j] = mat[j, i] = d
    return mat


def geo_upper(seqs, measure, variant, gamma, horizon):
    """Upper-triangular (i<j) geometric distances, in scipy.pdist order."""
    n = len(seqs)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(geo_distance(seqs[i], seqs[j], measure, variant, gamma, horizon))
    return np.asarray(vals)


# Back-compat thin wrappers (measure="dtw") -- kept so existing call sites
# (rank mode, emb-DTW, timing probe) and any external caller keep working verbatim.
def dtw_distance(a, b, variant, gamma, horizon):
    """Normalized DTW distance between two (T, D) sequences for one variant."""
    return geo_distance(a, b, "dtw", variant, gamma, horizon)


def dtw_matrix(seqs, variant, gamma, horizon):
    """Symmetric (N, N) DTW distance matrix over a list of (T, D) sequences."""
    return geo_matrix(seqs, "dtw", variant, gamma, horizon)


def dtw_upper(seqs, variant, gamma, horizon):
    """Upper-triangular (i<j) DTW distances, in scipy.pdist order."""
    return geo_upper(seqs, "dtw", variant, gamma, horizon)


# ---------------------------------------------------------------------------
# Episode extraction + stratified sampling.
# ---------------------------------------------------------------------------

def build_episodes(terminals):
    """Return [(start, end_inclusive)] for complete episodes (terminals==1)."""
    term_locs = np.nonzero(terminals > 0)[0]
    init_locs = np.concatenate([[0], term_locs[:-1] + 1])
    return [(int(s), int(e)) for s, e in zip(init_locs, term_locs)]


def sample_segments(episodes, ep_tasks, seg_len, n_focus, seg_per_task, explicit, seed):
    """Sample fixed-length sub-trajectories (segments) of length `seg_len`.

    The comparison unit becomes a short segment whose length matches the discount
    horizon (the discounted occupancy = expected trajectory over a Geom(1-gamma)
    horizon). For each focus task we draw `seg_per_task` segments: pick a random
    episode of that task long enough to hold the segment, then a random start.
    Returns (seg_rows, seg_tasks, used_tasks) where seg_rows[k] is the length-L
    array of global row indices for segment k.
    """
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[tuple[int, int]]] = {}
    for (s, e), t in zip(episodes, ep_tasks):
        if e - s + 1 >= seg_len:
            by_task.setdefault(t, []).append((s, e))
    if explicit:
        names = [t.strip() for t in explicit.split(",") if t.strip() in by_task]
    else:
        names = sorted(by_task, key=lambda t: -len(by_task[t]))[:n_focus]
    seg_rows, seg_tasks, used = [], [], []
    for t in names:
        eps = by_task[t]
        for _ in range(seg_per_task):
            s, e = eps[int(rng.integers(len(eps)))]
            t0 = int(rng.integers(s, e - seg_len + 2))  # ensure t0+seg_len-1 <= e
            seg_rows.append(np.arange(t0, t0 + seg_len))
            seg_tasks.append(t)
        used.append(t)
    return seg_rows, seg_tasks, used


def focus_episodes(episodes, ep_tasks, per_task, n_focus, explicit, seed):
    """Focus mode: sample `per_task` episodes from each of a few tasks.

    Yields many same-task pairs so within-task correlation is measurable. Tasks
    are the explicit list if given, else the n_focus tasks with the most episodes
    (so each can supply `per_task` samples). Returns (chosen_idxs, used_tasks).
    """
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = {}
    for idx, t in enumerate(ep_tasks):
        by_task.setdefault(t, []).append(idx)
    if explicit:
        names = [t.strip() for t in explicit.split(",") if t.strip() in by_task]
    else:
        elig = [t for t in by_task if len(by_task[t]) >= per_task]
        elig.sort(key=lambda t: -len(by_task[t]))
        names = elig[:n_focus]
    chosen: list[int] = []
    used: list[str] = []
    for t in names:
        pool = by_task[t]
        take = min(per_task, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
        used.append(t)
    return sorted(chosen), used


def stratified_episodes(episodes, ep_tasks, n_target, seed):
    """Round-robin pick across tasks so the sample spans task diversity."""
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = {}
    for idx, t in enumerate(ep_tasks):
        by_task.setdefault(t, []).append(idx)
    for t in by_task:
        rng.shuffle(by_task[t])
    order = sorted(by_task)
    chosen: list[int] = []
    pos = {t: 0 for t in order}
    while len(chosen) < n_target:
        progressed = False
        for t in order:
            if pos[t] < len(by_task[t]):
                chosen.append(by_task[t][pos[t]])
                pos[t] += 1
                progressed = True
                if len(chosen) >= n_target:
                    break
        if not progressed:
            break
    return sorted(chosen)


def subsample_rows(start, end, max_len):
    """Row indices for an episode, evenly thinned to at most max_len frames."""
    rows = np.arange(start, end + 1)
    if len(rows) > max_len:
        pick = np.unique(np.linspace(0, len(rows) - 1, max_len).round().astype(int))
        rows = rows[pick]
    return rows


# ---------------------------------------------------------------------------
# Correlation + plotting.
# ---------------------------------------------------------------------------

def upper_tri(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu], iu


def corr_block(emb_mat, geo_mat, same_task_pair):
    """Spearman/Pearson over all pairs and over cross-task pairs only."""
    e, _ = upper_tri(emb_mat)
    g, _ = upper_tri(geo_mat)
    cross = ~same_task_pair
    out = {
        "spearman_all": float(spearmanr(e, g).statistic),
        "pearson_all": float(pearsonr(e, g)[0]),
        "n_pairs": int(len(e)),
    }
    if cross.sum() >= 10:
        out["spearman_cross"] = float(spearmanr(e[cross], g[cross]).statistic)
        out["n_cross_pairs"] = int(cross.sum())
    else:
        out["spearman_cross"] = None
        out["n_cross_pairs"] = int(cross.sum())
    return out


def per_task_corr(emb_mat, geo_mat, ep_tasks, min_eps=5):
    """Spearman of embedding-vs-DTW distance computed WITHIN each task separately.

    Each task's correlation uses only its own episode pairs, so per-task distance
    scale differences never leak in (unlike pooling all same-task pairs). Returns
    {task: rho} for tasks with >= min_eps episodes in the sample.
    """
    arr = np.array(ep_tasks, dtype=object)
    out = {}
    for t in sorted(set(ep_tasks)):
        E = np.flatnonzero(arr == t)
        if len(E) < min_eps:
            continue
        e, _ = upper_tri(emb_mat[np.ix_(E, E)])
        g, _ = upper_tri(geo_mat[np.ix_(E, E)])
        if np.std(e) < 1e-12 or np.std(g) < 1e-12:
            continue
        out[t] = float(spearmanr(e, g).statistic)
    return out


def within_task_summary(emb_mats, geo_mats, ep_tasks):
    """Per-(emb metric, rep, variant) mean within-task Spearman + per-task values."""
    res = {}
    for em in EMB_METRICS:
        res[em] = {}
        for rep in REPS:
            for var in VARIANTS:
                pt = per_task_corr(emb_mats[em], geo_mats[(rep, var)], ep_tasks)
                vals = list(pt.values())
                res[em][f"{rep}/{var}"] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "n_tasks": len(vals),
                    "per_task": pt,
                }
    return res


def save_per_task_bars(within, out_path, *, title, em="euclidean"):
    """Bar chart of per-task within-task Spearman (discounted variant)."""
    keys = ["state/discounted", "state+action/discounted", "action/discounted"]
    colors = {"state/discounted": "tab:blue", "state+action/discounted": "tab:green",
              "action/discounted": "tab:orange"}
    tasks = sorted(within[em]["state/discounted"]["per_task"].keys())
    if not tasks:
        return
    x = np.arange(len(tasks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(tasks) + 3), 5))
    for i, k in enumerate(keys):
        pt = within[em][k]["per_task"]
        vals = [pt.get(t, np.nan) for t in tasks]
        ax.bar(x + (i - 1) * w, vals, width=w, label=f"{k} (mean={within[em][k]['mean']:.3f})",
               color=colors[k])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"within-task Spearman ρ (emb={em})")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_scatter_grid(emb_mat, geo_mats, same_task_pair, out_path, *, title):
    """3x3 grid: rows = representation, cols = variant. x=emb dist, y=DTW dist."""
    e, _ = upper_tri(emb_mat)
    cross = ~same_task_pair
    fig, axes = plt.subplots(len(REPS), len(VARIANTS),
                             figsize=(4.2 * len(VARIANTS), 3.8 * len(REPS)),
                             squeeze=False)
    for r, rep in enumerate(REPS):
        for c, var in enumerate(VARIANTS):
            ax = axes[r][c]
            g, _ = upper_tri(geo_mats[(rep, var)])
            ax.scatter(e[cross], g[cross], s=6, alpha=0.30, color="tab:blue",
                       label="cross-task", linewidths=0)
            ax.scatter(e[~cross], g[~cross], s=6, alpha=0.30, color="tab:red",
                       label="same-task", linewidths=0)
            rho = spearmanr(e, g).statistic
            rho_c = (spearmanr(e[cross], g[cross]).statistic
                     if cross.sum() >= 10 else float("nan"))
            ax.set_title(f"{rep} / {var}\nρ_all={rho:.3f}  ρ_cross={rho_c:.3f}",
                         fontsize=9)
            if r == len(REPS) - 1:
                ax.set_xlabel("embedding distance")
            if c == 0:
                ax.set_ylabel("DTW distance")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(loc="upper left", fontsize=7, markerscale=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_corr_bars(corr, out_path, *, title):
    """Bar chart of Spearman (all & cross) for the 9 geo distances, per emb metric."""
    keys = [f"{rep}/{var}" for rep in REPS for var in VARIANTS]
    fig, axes = plt.subplots(1, len(EMB_METRICS),
                             figsize=(7.5 * len(EMB_METRICS), 5), squeeze=False)
    x = np.arange(len(keys))
    for ci, em in enumerate(EMB_METRICS):
        ax = axes[0][ci]
        all_v = [corr[em][k]["spearman_all"] for k in keys]
        cross_v = [corr[em][k]["spearman_cross"] if corr[em][k]["spearman_cross"]
                   is not None else 0.0 for k in keys]
        ax.bar(x - 0.2, all_v, width=0.4, label="all pairs", color="tab:blue")
        ax.bar(x + 0.2, cross_v, width=0.4, label="cross-task", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"emb={em}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_measure_compare_bars(corr_by_measure, measures, out_path, *, title,
                              var="discounted", which="spearman_all"):
    """Headline measure comparison: dtw vs frechet vs hausdorff, grouped by rep.

    corr_by_measure[em][measure][f"{rep}/{var}"] -> corr_block dict. One subplot per
    embedding metric; within it, bars grouped by representation, one bar per measure
    at the fixed discount `var`. `which` selects spearman_all or spearman_cross.
    """
    fig, axes = plt.subplots(1, len(EMB_METRICS),
                             figsize=(7.5 * len(EMB_METRICS), 5), squeeze=False)
    x = np.arange(len(REPS))
    width = 0.8 / max(len(measures), 1)
    for ci, em in enumerate(EMB_METRICS):
        ax = axes[0][ci]
        for mi, meas in enumerate(measures):
            vals = []
            for rep in REPS:
                d = corr_by_measure[em][meas].get(f"{rep}/{var}", {})
                v = d.get(which)
                vals.append(v if v is not None else 0.0)
            ax.bar(x + (mi - (len(measures) - 1) / 2.0) * width, vals,
                   width=width, label=meas)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(REPS, fontsize=9)
        ax.set_ylabel(f"Spearman ρ ({which.replace('spearman_', '')})")
        ax.set_title(f"emb={em}  (var={var})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_heatmap(mat, order, ep_tasks, out_path, *, title):
    """Episode x episode distance heatmap, episodes ordered by motion family."""
    m = mat[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Family band boundaries.
    fams = [family_of(ep_tasks[i]) for i in order]
    bounds = [k for k in range(1, len(fams)) if fams[k] != fams[k - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=0.5, alpha=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.5, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("episode (by family)")
    ax.set_ylabel("episode (by family)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def save_task_rank_bars(per_task, key, out_path, *, title):
    """Horizontal bar chart of per-task rho for `key`, sorted, colored by family."""
    items = sorted(((t, d[key]) for t, d in per_task.items() if d.get(key) is not None),
                   key=lambda kv: kv[1])
    if not items:
        return
    tasks = [t for t, _ in items]
    vals = [v for _, v in items]
    colors = [FAMILY_COLORS.get(family_of(t), "tab:gray") for t in tasks]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(tasks) + 2)))
    y = np.arange(len(tasks))
    ax.barh(y, vals, color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=7)
    ax.set_xlabel(f"within-task Spearman ρ  [{key}]")
    mean_v = float(np.mean(vals))
    ax.set_title(f"{title}\nmean={mean_v:.3f}  (n_tasks={len(tasks)})", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILY_ORDER
               if f in FAMILY_COLORS]
    ax.legend(handles=handles, fontsize=8, loc="lower right", title="family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_family_pair_scatter(emb_vec, geo_vec, fam_vec, out_path, *, title):
    """Pooled within-task pairs scatter, colored by motion family."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for fam in FAMILY_ORDER:
        m = fam_vec == fam
        if not m.any():
            continue
        ax.scatter(emb_vec[m], geo_vec[m], s=6, alpha=0.3,
                   color=FAMILY_COLORS.get(fam, "tab:gray"), label=fam, linewidths=0)
    ax.set_xlabel("embedding distance (euclidean, start-z)")
    ax.set_ylabel("DTW distance")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, markerscale=2, title="family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_task_ranking(run_dir, env_name, agent, pre_train, raw_obs, actions_all,
                     episodes, ep_task_labels, epochs, *, seg_len, seg_per_task,
                     min_task_eps, img_feat_dim, state_dim, gamma, horizon, seed,
                     tag_suffix):
    """Rank EVERY eligible task by within-task embedding<->trajectory correlation.

    Block-diagonal (per-task) DTW only, so this scales to all 65 tasks. For each
    task: sample seg_per_task segments, embed each by its START-frame z, and
    correlate embedding distance vs segment DTW among that task's segments. A high
    rho => the embedding faithfully tracks intra-task trajectory variation for that
    task ("well embedded"); a low rho => it does not.
    """
    rng = np.random.default_rng(seed)
    by_task = {}
    for (s, e), t in zip(episodes, ep_task_labels):
        if e - s + 1 >= seg_len:
            by_task.setdefault(t, []).append((s, e))
    tasks = sorted([t for t, v in by_task.items() if len(v) >= min_task_eps])
    if not tasks:
        raise SystemExit(f"No task has >= {min_task_eps} episodes of length >= {seg_len}.")
    print(f"RANK mode: {len(tasks)} eligible tasks, {seg_per_task} segments each, L={seg_len}")

    seg_rows, seg_task = [], []
    for t in tasks:
        eps = by_task[t]
        for _ in range(seg_per_task):
            s, e = eps[int(rng.integers(len(eps)))]
            t0 = int(rng.integers(s, e - seg_len + 2))
            seg_rows.append(np.arange(t0, t0 + seg_len))
            seg_task.append(t)
    seg_task = np.array(seg_task, dtype=object)

    # z-scored per-dim state/action sequences (global stats over all segment frames).
    all_rows = np.concatenate(seg_rows)
    state_end = (img_feat_dim + state_dim) if state_dim > 0 else raw_obs.shape[1]
    sr = raw_obs[all_rows][:, img_feat_dim:state_end].astype(np.float64)
    ar = actions_all[all_rows].astype(np.float64)
    sr = (sr - sr.mean(0, keepdims=True)) / np.clip(sr.std(0, keepdims=True), 1e-8, None)
    ar = (ar - ar.mean(0, keepdims=True)) / np.clip(ar.std(0, keepdims=True), 1e-8, None)
    state_seqs, action_seqs, sa_seqs = [], [], []
    off = 0
    for r in seg_rows:
        sl = slice(off, off + len(r))
        s_seq = np.ascontiguousarray(sr[sl]); a_seq = np.ascontiguousarray(ar[sl])
        state_seqs.append(s_seq); action_seqs.append(a_seq)
        sa_seqs.append(np.ascontiguousarray(np.hstack([s_seq, a_seq])))
        off += len(r)
    rep_seqs = {"state": state_seqs, "action": action_seqs, "state+action": sa_seqs}
    start_rows = np.array([r[0] for r in seg_rows])
    REP_VARS = [("state", "discounted"),
                ("action", "discounted"), ("state+action", "discounted")]
    HEAD = "state+action/discounted"

    tag = f"rank_seg{seg_len}_{len(tasks)}tasks"
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
    out_dir = run_dir / "plots" / "trajectory_dtw" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    summary = {"run_dir": str(run_dir), "env_name": env_name, "tag": tag,
               "local_metric": _LOCAL_METRIC_CDIST,
               "seg_len": seg_len, "seg_per_task": seg_per_task,
               "n_tasks": len(tasks), "gamma": gamma, "by_epoch": {}}

    for epoch in epochs:
        print(f"\n=== epoch {epoch} ===")
        agent_e = restore_agent(agent, str(run_dir), epoch)
        start_z = encode_z_rows(agent_e, pre_train, start_rows, seed)
        per_task = {}
        for t in tasks:
            idx = np.flatnonzero(seg_task == t)
            ez = start_z[idx]
            emb_e = pdist(ez, metric="euclidean")
            if np.std(emb_e) < 1e-12:
                continue
            d = {}
            for rep, var in REP_VARS:
                sub = [rep_seqs[rep][i] for i in idx]
                gv = dtw_upper(sub, var, gamma, horizon)
                d[f"{rep}/{var}"] = (float(spearmanr(emb_e, gv).statistic)
                                     if np.std(gv) > 1e-12 else None)
            per_task[t] = d
        summary["by_epoch"][str(epoch)] = per_task

        save_task_rank_bars(per_task, HEAD, out_dir / f"task_rank_{epoch}.png",
                            title=f"Per-task embedding↔trajectory ρ  |  {env_name}  ep={epoch}")
        save_task_rank_bars(per_task, "state/discounted",
                            out_dir / f"task_rank_state_{epoch}.png",
                            title=f"Per-task ρ [state/discounted]  |  ep={epoch}")
        # pooled within-task pairs scatter, colored by family (headline channel).
        all_emb, all_geo, all_fam = [], [], []
        for t in tasks:
            idx = np.flatnonzero(seg_task == t)
            ez = start_z[idx]
            all_emb.append(pdist(ez, metric="euclidean"))
            all_geo.append(dtw_upper([sa_seqs[i] for i in idx], "discounted", gamma, horizon))
            all_fam.append(np.array([family_of(t)] * (len(idx) * (len(idx) - 1) // 2),
                                    dtype=object))
        save_family_pair_scatter(
            np.concatenate(all_emb), np.concatenate(all_geo), np.concatenate(all_fam),
            out_dir / f"family_pairs_{epoch}.png",
            title=f"within-task pairs [{HEAD}] colored by family  |  ep={epoch}")

        ranked = sorted(((t, v.get(HEAD)) for t, v in per_task.items() if v.get(HEAD) is not None),
                        key=lambda kv: -kv[1])
        print(f"  [{HEAD}] best/worst tasks:")
        for t, v in ranked[:5]:
            print(f"    +best {v:+.3f}  {t}")
        for t, v in ranked[-5:]:
            print(f"    -worst {v:+.3f}  {t}")
        print(f"  mean ρ across tasks = {np.mean([v for _, v in ranked]):+.3f}")

    with (out_dir / "task_ranking.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/task_ranking.json")


def resolve_run_dir(arg):
    run_dir = Path(arg)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return run_dir


def encode_z_rows(agent, pre_train, rows, seed, chunk=4096):
    """Posterior-mean z for a flat array of row indices -> (len(rows), latent_dim)."""
    rng = jax.random.PRNGKey(seed)
    out = []
    for s in range(0, len(rows), chunk):
        sub = rows[s:s + chunk]
        batch = pre_train.sample(len(sub), idxs=sub)
        rng, sub_rng = jax.random.split(rng)
        mean, _std, _sample = encode_latents(agent, batch, sub_rng)
        out.append(np.asarray(mean))
    return np.concatenate(out, axis=0)


def encode_episode_zseqs(agent, pre_train, ep_rows, seed, chunk=4096):
    """Return [ (T_k, latent_dim) ] per-frame posterior-mean z for each episode.

    z_t = mean of q(z|s_t, a_t). Kept as a SEQUENCE (not averaged) so callers can
    either mean-pool it or run DTW over the time-varying embedding -- the latter
    respects that z_t encodes the discounted future occupancy *from step t*, which
    differs along the episode and must not be blurred by averaging.
    """
    flat = np.concatenate(ep_rows)
    rng = jax.random.PRNGKey(seed)
    zs = []
    for s in range(0, len(flat), chunk):
        sub = flat[s:s + chunk]
        batch = pre_train.sample(len(sub), idxs=sub)
        rng, sub_rng = jax.random.split(rng)
        mean, _std, _sample = encode_latents(agent, batch, sub_rng)
        zs.append(np.asarray(mean))
    z_all = np.concatenate(zs, axis=0)
    seqs = []
    off = 0
    for r in ep_rows:
        seqs.append(np.ascontiguousarray(z_all[off:off + len(r)].astype(np.float64)))
        off += len(r)
    return seqs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs (e.g. 500000,200000,50000). "
                             "Default: latest checkpoint only.")
    parser.add_argument("--n_episodes", type=int, default=120)
    parser.add_argument("--max_len", type=int, default=250)
    parser.add_argument("--per_task", type=int, default=0,
                        help="Focus mode: sample this many episodes from each of a few "
                             "tasks (instead of round-robin across all). >0 enables "
                             "within-task analysis (many same-task pairs).")
    parser.add_argument("--n_focus_tasks", type=int, default=6,
                        help="Focus mode: number of tasks to focus on (most-episodes first).")
    parser.add_argument("--focus_tasks", type=str, default=None,
                        help="Focus/segment mode: explicit comma-separated task names "
                             "(overrides --n_focus_tasks).")
    parser.add_argument("--seg_len", type=int, default=0,
                        help="SEGMENT mode (overrides episode mode): the comparison unit "
                             "is a fixed-length sub-trajectory of this many steps, matched "
                             "to the discount's effective horizon. Embedding = z at the "
                             "segment START frame, q(obs[img+proprio]_t0, a_t0) -- a single "
                             "vector compared by DIRECT distance (no mean-pool, no emb-DTW). "
                             "Geometric = DTW over the segment's proprio/action. 0 = off.")
    parser.add_argument("--seg_per_task", type=int, default=40,
                        help="Segments sampled per focus task in --seg_len mode.")
    parser.add_argument("--tag_suffix", type=str, default="",
                        help="Appended to the output subfolder tag, e.g. 'fixed' -> "
                             "seg25_t6x40_fixed. Use to keep variants from clobbering.")
    parser.add_argument("--rank_tasks", action="store_true",
                        help="RANK mode: for EVERY eligible task, compute the within-task "
                             "embedding<->trajectory ρ (start-z direct distance vs segment "
                             "DTW, block-diagonal) and rank tasks by how well they embed. "
                             "Uses --seg_len (default 25), --seg_per_task, --min_task_eps.")
    parser.add_argument("--min_task_eps", type=int, default=10,
                        help="RANK mode: minimum episodes (len>=seg_len) for a task to be ranked.")
    parser.add_argument("--measures", type=str, default="dtw",
                        help="Comma-separated geometric distance measures from "
                             f"{MEASURES}. Default 'dtw' reproduces the original "
                             "behaviour/outputs exactly. Add frechet/hausdorff to "
                             "also emit per-measure heatmaps/scatter/corr_bars plus "
                             "a measure_compare_<epoch>.png; extra measures are "
                             "additive under by_epoch[ep]['measures'] in the JSON "
                             "and never alter the DTW outputs. NOTE: only affects "
                             "episode/focus/segment mode, not --rank_tasks/--emb_dtw.")
    parser.add_argument("--local_metric", choices=["L1", "L2"], default="L1",
                        help="Per-frame (point-to-point) distance for the cost matrix L "
                             "that EVERY measure reduces -- an axis orthogonal to "
                             "--measures. L1=cityblock (default; suits joint-config space, "
                             "empirically higher embedding<->trajectory rho across all "
                             "measures), L2=euclidean (the previous behaviour; pass "
                             "--local_metric L2 to reproduce older outputs).")
    parser.add_argument("--discount", type=float, default=None,
                        help="Override discount gamma (default: agent's flags value).")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Truncated-DTW horizon H (default: round(1/(1-gamma))).")
    parser.add_argument("--img_feat_dim", type=int, default=-1,
                        help="Leading obs dims that are image features. -1 = auto from "
                             "agent flags (image_feat_dim), else 512 if multimodal else 0.")
    parser.add_argument("--state_dim", type=int, default=-1,
                        help="Proprio-state dims right after the image features -- the "
                             "DTW 'state' channel. -1 = auto from agent flags (state_dim); "
                             "if unavailable, use ALL remaining dims. Must be set for "
                             "lang_state envs so the trailing language embedding "
                             "(obs[img+state:]) is excluded from the state trajectory.")
    parser.add_argument("--emb_dtw", action="store_true",
                        help="Also compute embedding distance as DTW over the z-SEQUENCE "
                             "(z_t = q(z|s_t,a_t) per frame) instead of only mean-z. "
                             "Respects the time-varying, discounted nature of z.")
    parser.add_argument("--z_pca", type=int, default=50,
                        help="PCA dims to reduce the 512-d z-sequence to before emb-DTW "
                             "(speed; ~all variance kept). 0 = no reduction.")
    parser.add_argument("--no_heatmaps", action="store_true")
    parser.add_argument("--timing_probe", action="store_true",
                        help="Measure load + a small DTW batch, print an estimate, exit.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set the local-metric for this CLI run (importers keep the L2 module default).
    global _LOCAL_METRIC_CDIST
    _LOCAL_METRIC_CDIST = LOCAL_METRICS[args.local_metric]
    print(f"local_metric: {args.local_metric} ({_LOCAL_METRIC_CDIST}) -- cost-matrix point distance")

    run_dir = resolve_run_dir(args.run_dir)
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags.get("env_name", "")
    agent_cfg = flags.get("agent", {})
    gamma = args.discount if args.discount is not None else float(agent_cfg["discount"])
    horizon = args.horizon if args.horizon is not None else int(round(1.0 / (1.0 - gamma)))
    img_feat_dim = (args.img_feat_dim if args.img_feat_dim >= 0
                    else int(agent_cfg.get("image_feat_dim",
                                           512 if "multimodal" in env_name else 0)))
    if args.state_dim >= 0:
        state_dim = args.state_dim
    elif agent_cfg.get("state_dim"):
        state_dim = int(agent_cfg["state_dim"])
    else:
        state_dim = -1  # use all remaining dims after the image features

    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    available = [int(p.stem.split("_")[1]) for p in ckpts]
    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(",")]
        missing = [e for e in epochs if e not in available]
        if missing:
            raise SystemExit(f"Requested epochs not found: {missing}. Available: {available}")
    else:
        epochs = [available[-1]]

    measures = [m.strip() for m in args.measures.split(",") if m.strip()]
    bad = [m for m in measures if m not in MEASURES]
    if bad:
        raise SystemExit(f"Unknown --measures {bad}. Choose from {MEASURES}.")
    if "dtw" not in measures:  # DTW is the backbone of every downstream output.
        measures = ["dtw"] + measures
    seen = set()
    measures = [m for m in measures if not (m in seen or seen.add(m))]

    print(f"Run dir   : {run_dir}")
    print(f"env_name  : {env_name}")
    print(f"measures  : {measures}")
    state_end_note = (img_feat_dim + state_dim) if state_dim > 0 else "rest"
    print(f"gamma={gamma}  horizon(truncated)={horizon}")
    print(f"proprio state slice: obs[{img_feat_dim}:{state_end_note}]"
          f"  (img_feat_dim={img_feat_dim}, state_dim={state_dim})")
    print(f"epochs    : {epochs}  (available: {available})")

    print("Building agent + pretrain dataset ...")
    t0 = time.time()
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    dataset_len = len(raw_obs)
    load_s = time.time() - t0
    print(f"  dataset_len={dataset_len}  load={load_s:.1f}s")

    actions_all = np.asarray(pre_train["actions"])
    terminals_all = np.asarray(pre_train["terminals"])
    episodes = build_episodes(terminals_all)
    print(f"Complete episodes: {len(episodes)}")

    # Task label per episode (robocasa stats-based; degrade gracefully otherwise).
    try:
        stats_path = _resolve_stats_path(env_name, "")
        starts = np.array([s for s, _ in episodes])
        ep_task_labels, _present = derive_task_labels(starts, stats_path, dataset_len)
        ep_task_labels = list(ep_task_labels)
    except Exception as exc:  # noqa: BLE001
        print(f"  [task labels unavailable: {exc}] -> single-task fallback")
        ep_task_labels = ["task0"] * len(episodes)

    if args.rank_tasks:
        run_task_ranking(
            run_dir, env_name, agent, pre_train, raw_obs, actions_all,
            episodes, ep_task_labels, epochs,
            seg_len=(args.seg_len or 25), seg_per_task=args.seg_per_task,
            min_task_eps=args.min_task_eps, img_feat_dim=img_feat_dim,
            state_dim=state_dim, gamma=gamma, horizon=horizon, seed=args.seed,
            tag_suffix=args.tag_suffix)
        return

    seg_mode = args.seg_len > 0
    if seg_mode:
        ep_rows, ep_tasks, used_tasks = sample_segments(
            episodes, ep_task_labels, args.seg_len, args.n_focus_tasks,
            args.seg_per_task, args.focus_tasks, args.seed)
        focus_mode = True  # within-task grouping applies (segments carry a task)
        n = len(ep_rows)
        print(f"SEGMENT mode: L={args.seg_len}, {len(used_tasks)} tasks x "
              f"{args.seg_per_task} segs -> {n} segments  tasks={used_tasks}")
    else:
        focus_mode = args.per_task > 0
        if focus_mode:
            chosen, used_tasks = focus_episodes(
                episodes, ep_task_labels, args.per_task, args.n_focus_tasks,
                args.focus_tasks, args.seed)
            print(f"FOCUS mode: {len(used_tasks)} tasks x up to {args.per_task} eps "
                  f"-> {used_tasks}")
        else:
            chosen = stratified_episodes(episodes, ep_task_labels, args.n_episodes, args.seed)
        n = len(chosen)
        ep_tasks = [ep_task_labels[i] for i in chosen]
        ep_rows = [subsample_rows(*episodes[i], args.max_len) for i in chosen]
    lens = np.array([len(r) for r in ep_rows])
    print(f"Chosen episodes: {n}  (frames/ep min/med/max = "
          f"{lens.min()}/{int(np.median(lens))}/{lens.max()})")

    # Per-episode z-scored sequences (global per-dim stats over selected frames).
    all_rows = np.concatenate(ep_rows)
    state_end = (img_feat_dim + state_dim) if state_dim > 0 else raw_obs.shape[1]
    state_raw = raw_obs[all_rows][:, img_feat_dim:state_end].astype(np.float64)
    action_raw = actions_all[all_rows].astype(np.float64)
    print(f"State trajectory dim={state_raw.shape[1]}, action dim={action_raw.shape[1]}")

    def zscore(x):
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        return (x - mu) / np.clip(sd, 1e-8, None)

    state_z = zscore(state_raw)
    action_z = zscore(action_raw)

    state_seqs, action_seqs, sa_seqs = [], [], []
    off = 0
    for r in ep_rows:
        sl = slice(off, off + len(r))
        s_seq = np.ascontiguousarray(state_z[sl])
        a_seq = np.ascontiguousarray(action_z[sl])
        state_seqs.append(s_seq)
        action_seqs.append(a_seq)
        sa_seqs.append(np.ascontiguousarray(np.hstack([s_seq, a_seq])))
        off += len(r)
    rep_seqs = {"state": state_seqs, "action": action_seqs, "state+action": sa_seqs}

    # Same-task mask over upper-triangular episode pairs.
    iu = np.triu_indices(n, k=1)
    tasks_arr = np.array(ep_tasks, dtype=object)
    same_task_pair = tasks_arr[iu[0]] == tasks_arr[iu[1]]

    # --- timing probe: measure one variant on a small subset and extrapolate ---
    if args.timing_probe:
        probe_n = min(20, n)
        sub = rep_seqs["state+action"][:probe_n]
        # warm up numba
        _ = dtw_distance(sub[0], sub[1], "discounted", gamma, horizon)
        t1 = time.time()
        _ = dtw_matrix(sub, "discounted", gamma, horizon)
        probe_s = time.time() - t1
        pairs_probe = probe_n * (probe_n - 1) / 2
        per_pair = probe_s / max(pairs_probe, 1)
        total_pairs = n * (n - 1) / 2
        # 9 matrices per measure; truncated is cheaper and DTW (sum-DP) is the most
        # expensive reduction, so per-pair DTW x measure-count is a safe upper bound.
        est_dtw = per_pair * total_pairs * len(REPS) * len(VARIANTS) * len(measures)
        est_encode = 0.0  # measured below per epoch is small; estimate ~load fraction
        print("\n=== TIMING PROBE ===")
        print(f"  dataset load           : {load_s:.1f}s (one-time)")
        print(f"  DTW per pair (~upper)  : {per_pair*1000:.2f} ms")
        print(f"  total pairs            : {int(total_pairs)} x 9 x {len(measures)} measure(s)")
        print(f"  est. geo (one-time)    : {est_dtw:.0f}s")
        print(f"  est. per-epoch encode  : a few seconds (only {len(all_rows)} frames)")
        print(f"  => total for {len(epochs)} epoch(s): "
              f"~{load_s + est_dtw + 5*len(epochs):.0f}s "
              f"(DTW computed once, shared across epochs)")
        return

    # Tag the output subfolder by sampling config so different experiments
    # (cross-task vs focus, different task/episode counts) never clobber each other.
    if seg_mode:
        tag = f"seg{args.seg_len}_t{len(set(ep_tasks))}x{args.seg_per_task}"
    elif focus_mode:
        tag = f"focus{len(used_tasks)}x{args.per_task}"
    else:
        tag = f"cross_n{n}_t{len(set(ep_tasks))}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    out_dir = run_dir / "plots" / "trajectory_dtw" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    summary_tag = tag

    # --- geometric matrices: computed ONCE (checkpoint-independent) ---
    # geo_all keyed by (measure, rep, variant); 9 matrices per measure. The DTW
    # block (geo_mats below, a 2-key view) drives every original output verbatim;
    # extra measures only feed the additive per-measure outputs.
    print(f"Computing {9*len(measures)} geometric matrices "
          f"({len(measures)} measure(s) x 9; once, shared across epochs) ...")
    geo_all: dict[tuple[str, str, str], np.ndarray] = {}
    family_order_idx = sorted(range(n),
                              key=lambda i: (FAMILY_ORDER.index(family_of(ep_tasks[i]))
                                             if family_of(ep_tasks[i]) in FAMILY_ORDER
                                             else 99, ep_tasks[i]))
    for meas in measures:
        for rep in REPS:
            for var in VARIANTS:
                tg = time.time()
                geo_all[(meas, rep, var)] = geo_matrix(rep_seqs[rep], meas, var, gamma, horizon)
                print(f"  {meas:9s} {rep:13s}/{var:10s}  {time.time()-tg:.1f}s")
                if not args.no_heatmaps:
                    save_heatmap(geo_all[(meas, rep, var)], family_order_idx, ep_tasks,
                                 out_dir / f"{meas}_{rep.replace('+','_')}_{var}_heatmap.png",
                                 title=f"{meas.upper()} [{rep} / {var}]  ({env_name})")
    # DTW view with the ORIGINAL 2-tuple keys -> all existing per-epoch code,
    # plotting, and correlation calls below remain byte-for-byte unchanged.
    geo_mats = {(rep, var): geo_all[("dtw", rep, var)]
                for rep in REPS for var in VARIANTS}

    summary = {
        "run_dir": str(run_dir), "env_name": env_name, "tag": summary_tag,
        "gamma": gamma, "horizon": horizon, "local_metric": _LOCAL_METRIC_CDIST,
        "img_feat_dim": img_feat_dim, "state_dim": int(state_raw.shape[1]),
        "n_episodes": n, "max_len": args.max_len,
        "focus_mode": focus_mode,
        "seg_mode": seg_mode,
        "seg_len": (args.seg_len if seg_mode else None),
        "focus_tasks": (used_tasks if focus_mode else None),
        "per_task": (args.per_task if seg_mode else (args.per_task if focus_mode else None)),
        "n_tasks_in_sample": len(set(ep_tasks)),
        "frames_per_ep": {"min": int(lens.min()), "median": int(np.median(lens)),
                          "max": int(lens.max())},
        "by_epoch": {},
    }

    # --- per-epoch: re-encode embeddings, correlate ---
    for epoch in epochs:
        print(f"\n=== epoch {epoch} ===")
        agent_e = restore_agent(agent, str(run_dir), epoch)
        zseqs = encode_episode_zseqs(agent_e, pre_train, ep_rows, args.seed)
        if seg_mode:
            # z at the segment START frame: q(obs_t0, a_t0). Single vector, compared
            # by direct distance -- z already embeds the forward (discounted) trajectory.
            mean_z = np.stack([s[0] for s in zseqs])
        else:
            mean_z = np.stack([s.mean(axis=0) for s in zseqs])
        # Mahalanobis whitens by the covariance of the episode embeddings (Ledoit-Wolf
        # shrinkage so the 512-d precision is invertible from ~N episodes).
        from sklearn.covariance import LedoitWolf
        precision = LedoitWolf().fit(mean_z).precision_
        emb_mats = {
            "euclidean": squareform(pdist(mean_z, metric="euclidean")),
            "cosine": squareform(pdist(mean_z, metric="cosine")),
            "mahalanobis": squareform(pdist(mean_z, metric="mahalanobis", VI=precision)),
        }
        corr: dict[str, dict[str, dict]] = {em: {} for em in EMB_METRICS}
        for em in EMB_METRICS:
            for rep in REPS:
                for var in VARIANTS:
                    corr[em][f"{rep}/{var}"] = corr_block(
                        emb_mats[em], geo_mats[(rep, var)], same_task_pair)
            save_scatter_grid(
                emb_mats[em], geo_mats, same_task_pair,
                out_dir / f"scatter_{epoch}_{em}.png",
                title=f"embedding({em}) vs DTW  |  {env_name}  epoch={epoch}")
            if not args.no_heatmaps:
                save_heatmap(emb_mats[em], family_order_idx, ep_tasks,
                             out_dir / f"emb_{epoch}_{em}_heatmap.png",
                             title=f"embedding distance [{em}]  epoch={epoch}")
        save_corr_bars(corr, out_dir / f"corr_bars_{epoch}.png",
                       title=f"Spearman: embedding vs DTW  |  epoch={epoch}")

        within = None
        if focus_mode:
            within = within_task_summary(emb_mats, geo_mats, ep_tasks)
            save_per_task_bars(
                within, out_dir / f"within_task_bars_{epoch}.png",
                title=f"Within-task Spearman (emb=euclidean)  |  epoch={epoch}")

        # Best geo distance per embedding metric (by cross-task Spearman if available).
        def score(d):
            return d["spearman_cross"] if d["spearman_cross"] is not None else d["spearman_all"]
        best = {}
        for em in EMB_METRICS:
            ranked = sorted(corr[em].items(), key=lambda kv: -score(kv[1]))
            best[em] = {"key": ranked[0][0], "spearman_all": ranked[0][1]["spearman_all"],
                        "spearman_cross": ranked[0][1]["spearman_cross"]}
        ep_entry = {"correlations": corr, "best": best}
        if within is not None:
            ep_entry["within_task"] = within

        # --- embedding-DTW: distance = DTW over the z-SEQUENCE (no mean-pooling) ---
        if args.emb_dtw:
            print("  computing embedding-DTW (z-sequence) ...")
            if args.z_pca and args.z_pca < zseqs[0].shape[1]:
                from sklearn.decomposition import PCA
                allz = np.vstack(zseqs)
                pca = PCA(n_components=args.z_pca).fit(allz)
                evr = float(pca.explained_variance_ratio_.sum())
                zred = [np.ascontiguousarray(pca.transform(s)) for s in zseqs]
                print(f"    PCA z {zseqs[0].shape[1]}->{args.z_pca} (var kept={evr:.3f})")
            else:
                zred = zseqs
            t_e = time.time()
            emb_dtw_mat = dtw_matrix(zred, "discounted", gamma, horizon)
            print(f"    emb-DTW matrix in {time.time()-t_e:.1f}s")
            edtw_corr = {f"{rep}/{var}": corr_block(emb_dtw_mat, geo_mats[(rep, var)],
                                                    same_task_pair)
                         for rep in REPS for var in VARIANTS}
            edtw_block = {"correlations": edtw_corr}
            save_scatter_grid(emb_dtw_mat, geo_mats, same_task_pair,
                              out_dir / f"scatter_{epoch}_embdtw.png",
                              title=f"embedding-DTW(z-seq) vs traj-DTW  |  epoch={epoch}")
            if focus_mode:
                edtw_within = {f"{rep}/{var}":
                               per_task_corr(emb_dtw_mat, geo_mats[(rep, var)], ep_tasks)
                               for rep in REPS for var in VARIANTS}
                edtw_block["within_task_mean"] = {
                    k: (float(np.mean(list(v.values()))) if v else None)
                    for k, v in edtw_within.items()}
                edtw_block["within_task_per_task"] = edtw_within
            ep_entry["emb_dtw"] = edtw_block

        # --- extra trajectory-distance measures (additive; DTW outputs untouched) ---
        extra = [m for m in measures if m != "dtw"]
        if extra:
            # corr_by_measure[em][measure][key]; dtw reuses the block computed above.
            corr_by_measure = {em: {"dtw": corr[em]} for em in EMB_METRICS}
            meas_blocks = {}
            for meas in extra:
                geo_view = {(rep, var): geo_all[(meas, rep, var)]
                            for rep in REPS for var in VARIANTS}
                mcorr = {em: {} for em in EMB_METRICS}
                for em in EMB_METRICS:
                    for rep in REPS:
                        for var in VARIANTS:
                            mcorr[em][f"{rep}/{var}"] = corr_block(
                                emb_mats[em], geo_view[(rep, var)], same_task_pair)
                    save_scatter_grid(
                        emb_mats[em], geo_view, same_task_pair,
                        out_dir / f"scatter_{epoch}_{em}_{meas}.png",
                        title=f"embedding({em}) vs {meas.upper()}  |  {env_name}  epoch={epoch}")
                    corr_by_measure[em][meas] = mcorr[em]
                save_corr_bars(mcorr, out_dir / f"corr_bars_{epoch}_{meas}.png",
                               title=f"Spearman: embedding vs {meas.upper()}  |  epoch={epoch}")
                block = {"correlations": mcorr}
                if focus_mode:
                    block["within_task"] = within_task_summary(emb_mats, geo_view, ep_tasks)
                meas_blocks[meas] = block
            ep_entry["measures"] = meas_blocks

            # Headline: dtw vs frechet vs hausdorff at the discounted variant.
            save_measure_compare_bars(
                corr_by_measure, measures, out_dir / f"measure_compare_{epoch}.png",
                title=f"trajectory-distance measure comparison (ρ_all, discounted)  |  "
                      f"{env_name}  epoch={epoch}", var="discounted", which="spearman_all")
            print("  --- measure comparison (euclidean, ρ_all @ discounted) ---")
            for rep in REPS:
                cells = "  ".join(
                    f"{meas}={corr_by_measure['euclidean'][meas][f'{rep}/discounted']['spearman_all']:+.3f}"
                    for meas in measures)
                print(f"    [{rep:13s}] {cells}")

        summary["by_epoch"][str(epoch)] = ep_entry

        for em in EMB_METRICS:
            print(f"  best ({em:11s}): {best[em]}")
        if within is not None:
            print("  --- within-task mean ρ (per-task averaged) ---")
            for em in EMB_METRICS:
                row = {f"{rep}/{var}": within[em][f"{rep}/{var}"]["mean"]
                       for rep in REPS for var in ["discounted"]}
                best_w = max(within[em].items(),
                             key=lambda kv: (kv[1]["mean"] if kv[1]["mean"] is not None else -9))
                print(f"    [{em:11s}] state={row['state/discounted']:+.3f}  "
                      f"action={row['action/discounted']:+.3f}  "
                      f"state+action={row['state+action/discounted']:+.3f}   "
                      f"| best: {best_w[0]} ρ={best_w[1]['mean']:+.3f}")
        if args.emb_dtw and "emb_dtw" in ep_entry:
            ec = ep_entry["emb_dtw"]["correlations"]
            print("  --- embedding-DTW (z-sequence) vs traj-DTW ---")
            print(f"    all-pairs ρ: state/disc={ec['state/discounted']['spearman_all']:+.3f}  "
                  f"action/disc={ec['action/discounted']['spearman_all']:+.3f}  "
                  f"state+action/disc={ec['state+action/discounted']['spearman_all']:+.3f}")
            if "within_task_mean" in ep_entry["emb_dtw"]:
                wm = ep_entry["emb_dtw"]["within_task_mean"]
                print(f"    within-task ρ: state/disc={wm['state/discounted']:+.3f}  "
                      f"action/disc={wm['action/discounted']:+.3f}  "
                      f"state+action/disc={wm['state+action/discounted']:+.3f}  "
                      f"(compare mean-z within state={within['euclidean']['state/discounted']['mean']:+.3f})")
        # Discount comparison digest: plain vs truncated vs discounted (state+action, euclidean).
        for rep in REPS:
            row = {v: corr["euclidean"][f"{rep}/{v}"]["spearman_all"] for v in VARIANTS}
            print(f"  [{rep:13s}] ρ_all  plain={row['plain']:.3f}  "
                  f"truncated={row['truncated']:.3f}  discounted={row['discounted']:.3f}")

    with (out_dir / "corr_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/corr_summary.json")
    print(f"Plots under {out_dir}")


if __name__ == "__main__":
    main()
