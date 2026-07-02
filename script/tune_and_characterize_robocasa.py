"""Robocasa 클러스터링: (3) 파라미터 튜닝 → (1) k≈4 덩어리 사후 특성화.

cluster_robocasa_embeddings.py 로 이미 확인한 결과 위에서:
  Phase 3 (tune)       : infom_latent(PCA-50)에 대해 OPTICS(xi×min_samples),
                         HDBSCAN(min_cluster_size×min_samples) 그리드 스윕.
                         k / noise / silhouette 히트맵 → 추천 config 출력.
  Phase 1 (characterize): 안정적 분할(Agglomerative ward k=K, 기본 4)의 각 클러스터가
                         무엇인지 해석. cluster × {ep_time, gripper, task_family,
                         action_cluster, mean-state} 교차표 + medoid(글로벌 idx→렌더용).

임베딩은 한 번 빌드해 <outdir>/embed_cache.npz 에 저장하고 재사용(재-스트리밍/복원 회피).

Usage:
  python script/tune_and_characterize_robocasa.py --phase both
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import HDBSCAN, OPTICS, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.cluster_robocasa_embeddings import (  # noqa: E402
    DATASET_DEFAULTS,
    build_and_restore_agent,
    get_label_funcs,
    streaming_obs_stats,
)
from script.visualize_latent_robocasa import (  # noqa: E402  (env-agnostic)
    derive_ep_time_labels,
    encode_latents,
)

import jax  # noqa: E402

# obs 레이아웃별 state 블록 위치.
# robocasa: [image 1536 | state 16 | lang 384]=1936 → state 1536:1552
# libero  : [image  512 | state 15 | lang 384]= 911 → state  512:527
DATASET_STATE_SLICE = {
    "robocasa": slice(1536, 1552),
    "libero": slice(512, 527),
}


# --------------------------------------------------------------------------- #
def build_or_load_cache(args, run_dir, flags):
    cache = Path(args.outdir) / "embed_cache.npz"
    if cache.exists() and not args.rebuild:
        print(f"[cache] loading {cache}")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    env_name = flags["env_name"]
    if args.hdf5:
        h5_path = osp.expanduser(args.hdf5)
    else:
        import importlib
        eu = importlib.import_module(
            "envs.libero_utils" if args.dataset == "libero" else "envs.robocasa_utils")
        pre_name, _, _ = eu.parse_env_name(env_name)
        h5_path = osp.join(osp.expanduser(eu.DEFAULT_DATASET_DIR), f"{pre_name}.hdf5")
    stats_path = h5_path.replace(".hdf5", ".stats.json")
    max_rows = flags.get("pretraining_size", np.inf) or np.inf

    print("[build] streaming obs normalization stats ...")
    obs_mean, obs_var, n_total = streaming_obs_stats(h5_path, "observations", max_rows)
    n_used = int(min(n_total, max_rows))
    rng = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(n_used, size=min(args.n_samples, n_used), replace=False))

    with h5py.File(h5_path, "r") as f:
        obs_raw = f["observations"][idxs].astype(np.float32)
        act = f["actions"][idxs].astype(np.float32)
        terminals_all = f["terminals"][:].astype(np.float32)
    obs_norm = (obs_raw - obs_mean) / np.sqrt(obs_var + 1e-8)
    state_raw = obs_raw[:, DATASET_STATE_SLICE[args.dataset]].copy()
    del obs_raw

    print("[build] restoring agent + encoding latents ...")
    agent = build_and_restore_agent(flags, str(run_dir), args.epoch, obs_norm.shape[1], act.shape[1])
    mean, _, _ = encode_latents(agent, {"observations": obs_norm, "actions": act},
                                jax.random.PRNGKey(args.seed))

    derive_task_labels, derive_task_family_labels = get_label_funcs(args.dataset)
    task_ref, present = derive_task_labels(idxs, stats_path, n_used)
    fam_ref, _ = derive_task_family_labels(task_ref)
    ep_time = derive_ep_time_labels(idxs, terminals_all)

    out = {
        "raw_resnet": obs_norm.astype(np.float32),
        "infom_latent": np.asarray(mean, np.float32),
        "actions": act, "state_raw": state_raw,
        "global_idxs": idxs, "task_ref": task_ref.astype(str),
        "family_ref": fam_ref.astype(str), "ep_time": ep_time.astype(str),
    }
    np.savez(cache, **out)
    print(f"[cache] saved {cache}")
    return out


def reduce_pca(X, pca_dim, seed):
    if pca_dim and X.shape[1] > pca_dim:
        p = PCA(n_components=pca_dim, random_state=seed)
        return p.fit_transform(X).astype(np.float64), float(p.explained_variance_ratio_.sum())
    return X.astype(np.float64), 1.0


# --------------------------------------------------------------------------- #
# Phase 3: 파라미터 튜닝.
# --------------------------------------------------------------------------- #
def _grid_stats(Xr, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    k = len(set(labels[mask]))
    noise = float((~mask).mean())
    if k >= 2 and mask.sum() > k:
        sil = float(silhouette_score(Xr[mask], labels[mask]))
    else:
        sil = float("nan")
    return k, noise, sil


def _heat(ax, M, rows, cols, title, fmt="{:.2f}", cmap="viridis"):
    im = ax.imshow(M, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            txt = "-" if (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.nanmax(M) and v < np.nanmax(M) * 0.6 else "black")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)


def tune(source, X, args, out_dir):
    Xr, evr = reduce_pca(X, args.pca_dim, args.seed)
    print(f"\n===== TUNE {source}  PCA→{args.pca_dim}d (evr={evr:.3f}) =====")

    # HDBSCAN grid: min_cluster_size × min_samples.
    hdb_mcs = [15, 25, 50, 100, 150, 250]
    hdb_ms = [3, 5, 10, 20]
    Hk = np.zeros((len(hdb_mcs), len(hdb_ms)))
    Hn = np.zeros_like(Hk); Hs = np.zeros_like(Hk)
    for i, mcs in enumerate(hdb_mcs):
        for j, ms in enumerate(hdb_ms):
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(Xr)
            Hk[i, j], Hn[i, j], Hs[i, j] = _grid_stats(Xr, lab)

    # OPTICS grid: xi × min_samples.
    opt_xi = [0.01, 0.02, 0.03, 0.05, 0.10]
    opt_ms = [5, 10, 20]
    Ok = np.zeros((len(opt_xi), len(opt_ms)))
    On = np.zeros_like(Ok); Os = np.zeros_like(Ok)
    for i, xi in enumerate(opt_xi):
        for j, ms in enumerate(opt_ms):
            lab = OPTICS(min_samples=ms, xi=xi, min_cluster_size=0.03).fit_predict(Xr)
            Ok[i, j], On[i, j], Os[i, j] = _grid_stats(Xr, lab)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    _heat(axes[0, 0], Hk, [f"mcs{m}" for m in hdb_mcs], [f"ms{m}" for m in hdb_ms],
          "HDBSCAN: n_clusters", fmt="{:.0f}", cmap="cividis")
    _heat(axes[0, 1], Hn * 100, [f"mcs{m}" for m in hdb_mcs], [f"ms{m}" for m in hdb_ms],
          "HDBSCAN: noise %", fmt="{:.0f}", cmap="Reds")
    _heat(axes[0, 2], Hs, [f"mcs{m}" for m in hdb_mcs], [f"ms{m}" for m in hdb_ms],
          "HDBSCAN: silhouette", fmt="{:.2f}")
    _heat(axes[1, 0], Ok, [f"xi{x}" for x in opt_xi], [f"ms{m}" for m in opt_ms],
          "OPTICS: n_clusters", fmt="{:.0f}", cmap="cividis")
    _heat(axes[1, 1], On * 100, [f"xi{x}" for x in opt_xi], [f"ms{m}" for m in opt_ms],
          "OPTICS: noise %", fmt="{:.0f}", cmap="Reds")
    _heat(axes[1, 2], Os, [f"xi{x}" for x in opt_xi], [f"ms{m}" for m in opt_ms],
          "OPTICS: silhouette", fmt="{:.2f}")
    fig.suptitle(f"[{source}] clustering hyperparameter sweep (PCA-{args.pca_dim})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / f"{source}_param_tuning.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")

    # 추천: 2<=k<=8, noise<40%, silhouette 최대.
    recs = []
    for i, mcs in enumerate(hdb_mcs):
        for j, ms in enumerate(hdb_ms):
            if 2 <= Hk[i, j] <= 8 and Hn[i, j] < 0.4 and not np.isnan(Hs[i, j]):
                recs.append(("HDBSCAN", {"min_cluster_size": mcs, "min_samples": ms},
                             int(Hk[i, j]), float(Hn[i, j]), float(Hs[i, j])))
    recs.sort(key=lambda r: -r[4])
    print("추천 config (2≤k≤8, noise<40%, silhouette 내림차순):")
    for m, p_, k, n, s in recs[:5]:
        print(f"  {m} {p_} → k={k} noise={n*100:.1f}% sil={s:.3f}")
    return {"hdbscan": {"mcs": hdb_mcs, "ms": hdb_ms, "k": Hk.tolist(),
                        "noise": Hn.tolist(), "sil": Hs.tolist()},
            "optics": {"xi": opt_xi, "ms": opt_ms, "k": Ok.tolist(),
                       "noise": On.tolist(), "sil": Os.tolist()},
            "recommendations": [{"method": m, "params": p_, "k": k, "noise": n, "sil": s}
                                for m, p_, k, n, s in recs[:5]]}


# --------------------------------------------------------------------------- #
# Phase 1: 사후 특성화.
# --------------------------------------------------------------------------- #
def _crosstab(labels, axis_vals, axis_order):
    """행=클러스터, 열=axis 범주. 각 행을 비율(합=1)로 정규화."""
    clusters = sorted(set(labels))
    M = np.zeros((len(clusters), len(axis_order)))
    for i, c in enumerate(clusters):
        m = labels == c
        for j, a in enumerate(axis_order):
            M[i, j] = np.sum((axis_vals[m] == a))
        if M[i].sum() > 0:
            M[i] /= M[i].sum()
    return clusters, M


def characterize(source, X, cache, args, out_dir, tune_rec=None):
    Xr, evr = reduce_pca(X, args.pca_dim, args.seed)
    print(f"\n===== CHARACTERIZE {source}  (canonical: ward k={args.char_k}) =====")

    # 안정적 분할: Agglomerative ward k (noise 없음, 결정론적).
    Z = linkage(Xr, method="ward")
    labels = fcluster(Z, t=args.char_k, criterion="maxclust")

    ep_time = cache["ep_time"]
    fam = cache["family_ref"]
    task = cache["task_ref"]
    act = cache["actions"]
    state = cache["state_raw"]
    gripper = np.where(act[:, -1] > 0, "open", "closed").astype(object)
    # action_cluster: 샘플 액션에 KMeans.
    ac = KMeans(n_clusters=args.k_action, random_state=args.seed, n_init=10).fit_predict(act)
    ac_lab = np.array([f"a{c}" for c in ac], dtype=object)

    EP = ["start", "early", "mid", "late", "end"]
    FAM = sorted(set(fam.tolist()))  # dataset별 family(robocasa 4~5 / libero 6)
    GR = ["closed", "open"]
    AC = [f"a{i}" for i in range(args.k_action)]

    axes_defs = [("ep_time", ep_time, EP), ("gripper", gripper, GR),
                 ("task_family", fam, FAM), (f"action_cluster(k{args.k_action})", ac_lab, AC)]

    fig, axs = plt.subplots(1, len(axes_defs) + 1, figsize=(4.2 * (len(axes_defs) + 1), 4.6))
    clusters = None
    comp = {}
    for ax, (name, vals, order) in zip(axs[:-1], axes_defs):
        clusters, M = _crosstab(labels, vals, order)
        comp[name] = {int(c): dict(zip(order, M[i].round(3).tolist()))
                      for i, c in enumerate(clusters)}
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(clusters))); ax.set_yticklabels([f"c{c}" for c in clusters], fontsize=8)
        ax.set_title(f"cluster × {name}\n(row-normalized)", fontsize=9)
        for i in range(len(clusters)):
            for j in range(len(order)):
                if M[i, j] > 0.08:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < 0.6 else "black")

    # 마지막 패널: 클러스터별 평균 state(16-d) 히트맵(정규화된 상대 비교).
    sizes = np.array([np.sum(labels == c) for c in clusters])
    state_mean = np.stack([state[labels == c].mean(0) for c in clusters])
    sm = (state_mean - state_mean.mean(0, keepdims=True)) / (state_mean.std(0, keepdims=True) + 1e-8)
    axL = axs[-1]
    im = axL.imshow(sm, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axL.set_xticks(range(16)); axL.set_xticklabels([f"s{i}" for i in range(16)], fontsize=6, rotation=90)
    axL.set_yticks(range(len(clusters)))
    axL.set_yticklabels([f"c{c} (n={sizes[i]})" for i, c in enumerate(clusters)], fontsize=8)
    axL.set_title("cluster × mean state (z-scored)", fontsize=9)
    plt.colorbar(im, ax=axL, fraction=0.046)

    fig.suptitle(f"[{source}] post-hoc characterization of ward-k{args.char_k} clusters "
                 f"(n={len(Xr)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / f"{source}_characterize_k{args.char_k}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[saved] {p}")

    # medoids → 글로벌 idx + 해석 라벨(렌더용).
    med = {}
    for c in clusters:
        idx = np.flatnonzero(labels == c)
        sub = Xr[idx]
        d = np.linalg.norm(sub - sub.mean(0, keepdims=True), axis=1)
        loc = int(idx[int(np.argmin(d))])
        med[int(c)] = {
            "global_idx": int(cache["global_idxs"][loc]),
            "size": int(len(idx)),
            "given_task": str(task[loc]), "given_family": str(fam[loc]),
            "ep_time": str(ep_time[loc]), "gripper": str(gripper[loc]),
        }
    print("클러스터별 요약(비율 최댓 범주):")
    for c in clusters:
        top = {name: max(comp[name][int(c)], key=comp[name][int(c)].get) for name, *_ in axes_defs}
        print(f"  c{c} (n={med[int(c)]['size']:>4d}): "
              + " ".join(f"{k}={v}" for k, v in top.items()))

    return {"source": source, "char_k": args.char_k, "sizes": sizes.tolist(),
            "composition": comp, "medoids": med}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, default="robocasa", choices=["robocasa", "libero"])
    ap.add_argument("--run_dir", type=str, default="", help="비우면 --dataset 기본 run")
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--hdf5", type=str, default="")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--phase", type=str, default="both", choices=["tune", "characterize", "both"])
    ap.add_argument("--sources", type=str, default="infom_latent",
                    choices=["raw_resnet", "infom_latent", "both"])
    ap.add_argument("--char_k", type=int, default=4)
    ap.add_argument("--k_action", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--rebuild", action="store_true", help="캐시 무시하고 임베딩 재빌드")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir or DATASET_DEFAULTS[args.dataset]["run_dir"])
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    if not args.outdir:
        args.outdir = str(run_dir / "plots" / "clustering_compare")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = build_or_load_cache(args, run_dir, flags)
    want = {"raw_resnet", "infom_latent"} if args.sources == "both" else {args.sources}

    result = {"run_dir": str(run_dir), "epoch": args.epoch, "char_k": args.char_k,
              "note": "Robocasa given labels are reference-only (not ground truth)."}
    for src in want:
        X = cache[src]
        result.setdefault(src, {})
        if args.phase in ("tune", "both"):
            result[src]["tuning"] = tune(src, X, args, out_dir)
        if args.phase in ("characterize", "both"):
            rec = result[src].get("tuning", {}).get("recommendations", [None])
            result[src]["characterization"] = characterize(
                src, X, cache, args, out_dir, tune_rec=rec[0] if rec else None)

    with (out_dir / "tune_characterize.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {out_dir/'tune_characterize.json'}")


if __name__ == "__main__":
    main()
