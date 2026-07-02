"""Robocasa 임베딩에 클러스터링 방법론 적용 (post-hoc, cluster-then-label).

배경 (project_robocasa_posthoc_labeling):
  Robocasa의 주어진 task 라벨은 물리적 궤적 구조와 무관하므로 정답으로 쓰지 않는다.
  → 외부지표(ARI/NMI)는 '참고용'으로만 계산(주어진 라벨과 얼마나 무관한지 확인),
    평가는 내부지표(silhouette/CH/DB) 기준. 색은 예측 클러스터로 칠하고 사후 명명.

두 임베딩을 같은 샘플에 대해 비교:
  - raw_resnet  : 입력 ResNet34⊕state(⊕lang) 특징(정규화됨). 학습 전 raw feature 구조.
  - infom_latent: 학습된 inFOM 인코더 q(z|s,a)의 mean. '우리 실험 결과'.

방법론:
  자동 K 계열 (K 직접 미지정)  : HDBSCAN(+min_cluster_size 스윕), OPTICS
  k-스윕 계열 (교차검증)        : Agglomerative(ward) k=2..Kmax silhouette/CH/DB 곡선 → best-k
                                 Spectral을 best-k(및 HDBSCAN의 k)에서 실행
  Dendrogram (ward)             : merge-distance 점프로 자연스러운 k 눈으로 확인

메모리 안전: 전체 HDF5(≈11GB×2)를 로드하지 않고, 정규화 통계(mean/var)만 스트리밍으로
  전체 행에서 계산한 뒤 샘플링한 행만 읽는다.

출력: <run_dir>/plots/clustering_compare/ (또는 --outdir)
  {source}_ksweep.png, {source}_dendrogram.png, {source}_projection.png,
  metrics.csv, summary.json (클러스터별 medoid 인덱스 = 사후 렌더용)

Usage:
  python script/cluster_robocasa_embeddings.py \
      --run_dir exp/atomic_65_multimodal_precompute_3cam_lang_state_decoder/sd000_20260609_103625 \
      --epoch 500000 --n_samples 4000
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

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import HDBSCAN, OPTICS, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# encode_latents/ep_time 등은 env-agnostic(robocasa 모듈 공유); task 라벨만 dataset별.
from script.visualize_latent_robocasa import encode_latents  # noqa: E402

DATASET_DEFAULTS = {
    "robocasa": {
        "run_dir": "exp/atomic_65_multimodal_precompute_3cam_lang_state_decoder/sd000_20260609_103625",
        "viz_module": "script.visualize_latent_robocasa",
    },
    "libero": {
        "run_dir": "exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850",
        "viz_module": "script.visualize_latent_libero",
    },
}


def get_label_funcs(dataset):
    """dataset별 derive_task_labels / derive_task_family_labels 반환."""
    import importlib
    m = importlib.import_module(DATASET_DEFAULTS[dataset]["viz_module"])
    return m.derive_task_labels, m.derive_task_family_labels

try:
    import umap  # umap-learn
except ImportError:
    umap = None

import jax  # noqa: E402
import ml_collections  # noqa: E402

from agents import agents  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


# --------------------------------------------------------------------------- #
# 정규화 통계 (전체 행 스트리밍) — 학습 때의 obs_norm_type='normal' 재현.
# --------------------------------------------------------------------------- #
def streaming_obs_stats(h5_path: str, key: str, max_rows: int, chunk: int = 100_000):
    """(obs - mean)/sqrt(var + 1e-8) 재현용 mean/var를 전체 행에서 스트리밍 계산.

    utils/datasets.normalize_observations 와 동일하게 population(ddof=0) 통계.
    """
    with h5py.File(h5_path, "r") as f:
        n_total = f[key].shape[0]
        n = int(min(n_total, max_rows))
        d = f[key].shape[1]
        s = np.zeros(d, np.float64)
        ss = np.zeros(d, np.float64)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            x = f[key][i:j].astype(np.float64)
            s += x.sum(0)
            ss += (x * x).sum(0)
        mean = s / n
        var = ss / n - mean * mean
        var = np.maximum(var, 0.0)
    return mean.astype(np.float32), var.astype(np.float32), n_total


# --------------------------------------------------------------------------- #
# 최소 에이전트 빌드(데이터셋 전체 로드 없이) + 체크포인트 복원.
# --------------------------------------------------------------------------- #
def build_and_restore_agent(flags: dict, run_dir: str, epoch: int, obs_dim: int, act_dim: int):
    config = ml_collections.ConfigDict(flags["agent"])
    example_obs = np.zeros((1, obs_dim), np.float32)
    example_act = np.zeros((1, act_dim), np.float32)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(flags["seed"], example_obs, example_act, config)
    agent = restore_agent(agent, run_dir, epoch)
    return agent


# --------------------------------------------------------------------------- #
# 클러스터링 평가.
# --------------------------------------------------------------------------- #
def evaluate(X, labels, task_ref=None):
    """내부지표(노이즈 -1 제외) + 참고용 외부지표(주어진 task 라벨과의 일치도)."""
    out = {}
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out["n_clusters"] = int(n_clusters)
    out["n_noise"] = int((~mask).sum())
    out["noise_frac"] = float((~mask).mean())
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        out["silhouette"] = float(silhouette_score(Xm, lm))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
        out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
    else:
        out["silhouette"] = out["calinski_harabasz"] = out["davies_bouldin"] = float("nan")
    if task_ref is not None:
        # 참고용: 주어진 라벨과 얼마나 (무)관한지. Robocasa에선 낮게 나오는 게 정상.
        out["ARI_vs_task_ref"] = float(adjusted_rand_score(task_ref, labels))
        out["NMI_vs_task_ref"] = float(normalized_mutual_info_score(task_ref, labels))
    return out


# --------------------------------------------------------------------------- #
# k-스윕 (Agglomerative ward): silhouette/CH/DB 곡선 → best-silhouette k.
# --------------------------------------------------------------------------- #
def ward_ksweep(X, Z, kmax):
    ks, sil, ch, db = [], [], [], []
    for k in range(2, kmax + 1):
        lab = fcluster(Z, t=k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        ks.append(k)
        sil.append(silhouette_score(X, lab))
        ch.append(calinski_harabasz_score(X, lab))
        db.append(davies_bouldin_score(X, lab))
    ks = np.array(ks)
    sil = np.array(sil)
    best_k = int(ks[int(np.argmax(sil))]) if len(ks) else 2
    return ks, np.array(sil), np.array(ch), np.array(db), best_k


def plot_ksweep(source, ks, sil, ch, db, best_k, mcs_rows, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, y, name, better in zip(
        axes, [sil, ch, db],
        ["silhouette (↑)", "Calinski-Harabasz (↑)", "Davies-Bouldin (↓)"],
        ["max", "max", "min"],
    ):
        ax.plot(ks, y, "o-")
        ax.axvline(best_k, color="red", ls="--", alpha=0.6, label=f"best-sil k={best_k}")
        ax.set_xlabel("n_clusters (k)")
        ax.set_title(f"{name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    # HDBSCAN min_cluster_size 스윕 결과를 텍스트로 첨부.
    txt = "HDBSCAN min_cluster_size sweep:\n" + "\n".join(
        f"  mcs={r['min_cluster_size']:>4d}: k={r['n_clusters']:>2d} "
        f"noise={r['noise_frac']*100:4.1f}% sil={r['silhouette']:.3f}"
        for r in mcs_rows
    )
    fig.suptitle(
        f"[{source}] Agglomerative(ward) k-sweep  |  {txt}",
        fontsize=10, family="monospace",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[saved] {outpath}")


def plot_dendrogram(source, Z, outpath):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=40, leaf_rotation=90.0,
               leaf_font_size=6, show_contracted=True)
    ax.set_title(f"[{source}] dendrogram (ward linkage, last 40 merges)")
    ax.set_xlabel("sample / (cluster size)")
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[saved] {outpath}")


# --------------------------------------------------------------------------- #
# 2D projection (UMAP + PCA) — 예측 클러스터 색 + 주어진 라벨(참고).
# --------------------------------------------------------------------------- #
def _scatter(ax, XY, labels, title):
    lab = np.asarray(labels)
    noise = lab == -1
    if noise.any():
        ax.scatter(XY[noise, 0], XY[noise, 1], c="lightgray", s=6, marker="x", alpha=0.6)
    uniq = [u for u in sorted(set(lab[~noise]))]
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    for i, u in enumerate(uniq):
        m = lab == u
        ax.scatter(XY[m, 0], XY[m, 1], s=6, alpha=0.6, color=cmap(i % cmap.N))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_projection(source, XY_umap, XY_pca, labelings, outpath):
    """labelings: list of (title, labels). 행=UMAP/PCA, 열=labeling."""
    ncols = len(labelings)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 7.0))
    if ncols == 1:
        axes = axes[:, None]
    for c, (title, labels) in enumerate(labelings):
        _scatter(axes[0, c], XY_umap, labels, f"UMAP · {title}")
        _scatter(axes[1, c], XY_pca, labels, f"PCA · {title}")
    fig.suptitle(f"[{source}] 2D projection colored by predicted cluster "
                 f"(rightmost = given task label)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[saved] {outpath}")


# --------------------------------------------------------------------------- #
def cluster_medoids(Xr, labels):
    """클러스터별 medoid(클러스터 내 평균거리 최소 점)의 로컬 인덱스."""
    med = {}
    for u in sorted(set(labels)):
        if u == -1:
            continue
        idx = np.flatnonzero(labels == u)
        if len(idx) == 1:
            med[int(u)] = int(idx[0]); continue
        sub = Xr[idx]
        c = sub.mean(0, keepdims=True)
        d = np.linalg.norm(sub - c, axis=1)
        med[int(u)] = int(idx[int(np.argmin(d))])
    return med


def run_suite(source, X_full, task_ref, args, out_dir):
    """한 임베딩에 대해 전체 클러스터링 스위트 실행. metrics rows + medoid 정보 반환."""
    print(f"\n===== source={source}  shape={X_full.shape} =====")
    # PCA 축소 (고차원 거리 안정화). pca_dim=0 이면 원본 사용.
    if args.pca_dim and X_full.shape[1] > args.pca_dim:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        Xr = pca.fit_transform(X_full)
        evr = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA {X_full.shape[1]}→{args.pca_dim}d, explained var={evr:.3f}")
    else:
        Xr = X_full
        evr = 1.0
    Xr = Xr.astype(np.float64)

    rows = []
    labelings = []

    def record(method, labels):
        m = evaluate(Xr, labels, task_ref)
        m.update({"source": source, "method": method})
        rows.append(m)
        print(f"  {method:22s} k={m['n_clusters']:>2d} noise={m['noise_frac']*100:4.1f}% "
              f"sil={m['silhouette']:.3f} CH={m['calinski_harabasz']:.0f} "
              f"DB={m['davies_bouldin']:.3f} ARI_ref={m.get('ARI_vs_task_ref', float('nan')):.3f}")
        return m

    # --- 자동 K: HDBSCAN ---
    hdb = HDBSCAN(min_cluster_size=args.hdbscan_mcs, min_samples=args.hdbscan_ms)
    hdb_labels = hdb.fit_predict(Xr)
    record("HDBSCAN(auto)", hdb_labels)

    # --- HDBSCAN min_cluster_size 스윕 ---
    mcs_rows = []
    for mcs in [15, 25, 50, 100, 200]:
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=args.hdbscan_ms).fit_predict(Xr)
        mm = evaluate(Xr, lab)
        mm["min_cluster_size"] = mcs
        mcs_rows.append(mm)

    # --- 자동 K: OPTICS ---
    opt_labels = OPTICS(min_samples=args.optics_ms, xi=args.optics_xi,
                        min_cluster_size=args.optics_mcs).fit_predict(Xr)
    record("OPTICS(auto)", opt_labels)

    # --- Agglomerative(ward) k-sweep ---
    Z = linkage(Xr, method="ward")
    ks, sil, ch, db, best_k = ward_ksweep(Xr, Z, args.kmax)
    print(f"  ward k-sweep best-silhouette k*={best_k}")
    agglo_labels = fcluster(Z, t=best_k, criterion="maxclust")
    record(f"Agglomerative(ward,k={best_k})", agglo_labels)

    # --- Spectral at best_k ---
    try:
        spec_labels = SpectralClustering(
            n_clusters=best_k, affinity="nearest_neighbors",
            n_neighbors=args.spectral_nn, assign_labels="cluster_qr",
            random_state=args.seed,
        ).fit_predict(Xr)
        record(f"Spectral(k={best_k})", spec_labels)
    except Exception as e:  # noqa: BLE001
        print(f"  [Spectral skipped] {e}")
        spec_labels = None

    # --- 시각화 ---
    plot_ksweep(source, ks, sil, ch, db, best_k, mcs_rows,
                out_dir / f"{source}_ksweep.png")
    plot_dendrogram(source, Z, out_dir / f"{source}_dendrogram.png")

    # 2D projection 좌표 (축소된 Xr 기준).
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(Xr)
    if umap is not None:
        nn = max(2, min(30, len(Xr) - 1))
        umap2 = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                          random_state=args.seed).fit_transform(Xr)
    else:
        print("  [warn] umap-learn 없음 → UMAP 대신 PCA 재사용")
        umap2 = pca2

    labelings = [("HDBSCAN(auto)", hdb_labels),
                 (f"Agglo(k={best_k})", agglo_labels)]
    if spec_labels is not None:
        labelings.append((f"Spectral(k={best_k})", spec_labels))
    labelings.append(("given task (ref)", task_ref))
    plot_projection(source, umap2, pca2, labelings, out_dir / f"{source}_projection.png")

    # --- medoid (primary = HDBSCAN, 클러스터가 너무 적으면 Agglo) ---
    primary_name, primary_labels = "HDBSCAN(auto)", hdb_labels
    if len(set(hdb_labels[hdb_labels != -1])) < 2:
        primary_name, primary_labels = f"Agglomerative(ward,k={best_k})", agglo_labels
    medoids_local = cluster_medoids(Xr, primary_labels)

    summary = {
        "source": source, "n_samples": int(len(Xr)),
        "pca_dim": int(args.pca_dim), "pca_explained_var": evr,
        "ward_best_k": int(best_k),
        "ksweep": {"k": ks.tolist(), "silhouette": sil.tolist(),
                   "calinski_harabasz": ch.tolist(), "davies_bouldin": db.tolist()},
        "hdbscan_mcs_sweep": [
            {"min_cluster_size": r["min_cluster_size"], "n_clusters": r["n_clusters"],
             "noise_frac": r["noise_frac"], "silhouette": r["silhouette"]}
            for r in mcs_rows],
        "primary_labeling": primary_name,
        "medoids_local_idx": medoids_local,  # source-local (샘플 내) 인덱스
        "labels": {  # 사후 렌더/집계용
            "HDBSCAN": np.asarray(hdb_labels).tolist(),
            "Agglomerative": np.asarray(agglo_labels).tolist(),
        },
    }
    return rows, summary


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, default="robocasa", choices=["robocasa", "libero"])
    ap.add_argument("--run_dir", type=str, default="",
                    help="비우면 --dataset 기본 run 사용")
    ap.add_argument("--epoch", type=int, default=500000)
    ap.add_argument("--hdf5", type=str, default="",
                    help="pretrain HDF5 경로. 비우면 env_name에서 유추.")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--pca_dim", type=int, default=50, help="클러스터링 전 PCA 축소 차원(0=원본)")
    ap.add_argument("--kmax", type=int, default=30, help="k-sweep 최대 k")
    ap.add_argument("--sources", type=str, default="both",
                    choices=["raw_resnet", "infom_latent", "both"])
    # 클러스터러 하이퍼파라미터
    ap.add_argument("--hdbscan_mcs", type=int, default=25)
    ap.add_argument("--hdbscan_ms", type=int, default=5)
    ap.add_argument("--optics_ms", type=int, default=10)
    ap.add_argument("--optics_xi", type=float, default=0.05)
    ap.add_argument("--optics_mcs", type=float, default=0.03)
    ap.add_argument("--spectral_nn", type=int, default=15)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir or DATASET_DEFAULTS[args.dataset]["run_dir"])
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]
    derive_task_labels, derive_task_family_labels = get_label_funcs(args.dataset)

    # HDF5 경로 유추.
    if args.hdf5:
        h5_path = osp.expanduser(args.hdf5)
    else:
        import importlib
        eu = importlib.import_module(
            "envs.libero_utils" if args.dataset == "libero" else "envs.robocasa_utils")
        pre_name, _, _ = eu.parse_env_name(env_name)
        h5_path = osp.join(osp.expanduser(eu.DEFAULT_DATASET_DIR), f"{pre_name}.hdf5")
    stats_path = h5_path.replace(".hdf5", ".stats.json")
    print(f"Run dir : {run_dir}\nEpoch   : {args.epoch}\nHDF5    : {h5_path}")

    max_rows = flags.get("pretraining_size", np.inf) or np.inf

    # 1) 정규화 통계(전체 행 스트리밍).
    print("Computing obs normalization stats (streaming) ...")
    obs_mean, obs_var, n_total = streaming_obs_stats(h5_path, "observations", max_rows)
    n_used = int(min(n_total, max_rows))
    print(f"  n_total={n_total} n_used={n_used} obs_dim={obs_mean.shape[0]}")

    # 2) 샘플 인덱스(정렬; h5py fancy indexing 요건).
    rng = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(n_used, size=min(args.n_samples, n_used), replace=False))

    # 3) 샘플 행 읽기 + 정규화.
    with h5py.File(h5_path, "r") as f:
        obs_raw = f["observations"][idxs].astype(np.float32)
        act = f["actions"][idxs].astype(np.float32)
    obs_norm = (obs_raw - obs_mean) / np.sqrt(obs_var + 1e-8)
    del obs_raw

    # 4) 주어진 task 라벨. Robocasa=참고용; LIBERO=의미있는 정답(ARI/NMI 진짜 지표).
    task_ref, present_tasks = derive_task_labels(idxs, stats_path, n_used)
    fam_ref, _ = derive_task_family_labels(task_ref)
    print(f"Reference task labels: {len(present_tasks)} tasks present among samples")

    sources = {}
    want = {"raw_resnet", "infom_latent"} if args.sources == "both" else {args.sources}
    if "raw_resnet" in want:
        sources["raw_resnet"] = obs_norm  # 정규화된 입력 특징 = z-score된 raw feature
    if "infom_latent" in want:
        print("Building + restoring agent, encoding latents ...")
        agent = build_and_restore_agent(
            flags, str(run_dir), args.epoch, obs_norm.shape[1], act.shape[1])
        mean, std, _ = encode_latents(
            agent, {"observations": obs_norm, "actions": act}, jax.random.PRNGKey(args.seed))
        print(f"  latent dim={mean.shape[1]} μ[{mean.min():+.2f},{mean.max():+.2f}] "
              f"σ[{std.min():.2f},{std.max():.2f}]")
        sources["infom_latent"] = np.asarray(mean, np.float32)

    out_dir = Path(args.outdir) if args.outdir else run_dir / "plots" / "clustering_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, summaries = [], {}
    for name, X in sources.items():
        rows, summ = run_suite(name, X, task_ref, args, out_dir)
        all_rows.extend(rows)
        summaries[name] = summ

    # metrics.csv
    cols = ["source", "method", "n_clusters", "n_noise", "noise_frac", "silhouette",
            "calinski_harabasz", "davies_bouldin", "ARI_vs_task_ref", "NMI_vs_task_ref"]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[saved] {csv_path}")

    # summary.json — medoid의 글로벌 인덱스/task도 기록(사후 렌더용).
    for name, summ in summaries.items():
        med_global = {str(cl): {
            "global_idx": int(idxs[loc]),
            "given_task": str(task_ref[loc]),
            "given_family": str(fam_ref[loc]),
        } for cl, loc in summ["medoids_local_idx"].items()}
        summ["medoids"] = med_global
        summ.pop("medoids_local_idx", None)
    meta = {
        "run_dir": str(run_dir), "epoch": args.epoch, "env_name": env_name,
        "hdf5": h5_path, "n_samples": int(len(idxs)),
        "sample_global_idxs": idxs.tolist(),
        "reference_tasks_present": present_tasks,
        "note": "Robocasa given labels are NOT ground truth; ARI/NMI_vs_task_ref are reference-only.",
        "sources": summaries,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {out_dir/'summary.json'}")
    print("\n완료. 지표: silhouette↑ CH↑ DB↓ / noise_frac 병기 / ARI_vs_task_ref는 참고용(낮으면 라벨과 무관).")


if __name__ == "__main__":
    main()
