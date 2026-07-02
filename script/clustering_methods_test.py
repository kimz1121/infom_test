#!/usr/bin/env python
"""클러스터링 방법론 비교 테스트 (실험 데이터 적용 전 sanity check).

적용 방법론:
  - HDBSCAN                (sklearn.cluster.HDBSCAN, density-based, 노이즈/가변밀도)
  - AgglomerativeClustering (계층적, dendrogram 시각화 가능)
  - OPTICS                 (density-based, 가변밀도 reachability)
  - SpectralClustering     (그래프 라플라시안 기반, 비볼록 구조)

각 방법에 대해:
  1) 여러 종류의 합성 테스트 데이터에 적용
  2) 클러스터 라벨 산점도 시각화
  3) silhouette / Calinski-Harabasz / Davies-Bouldin / (라벨 있으면) ARI, NMI 평가
  4) 계층적 클러스터링은 dendrogram 별도 시각화

출력물은 --outdir (기본 script/clustering_test_out/) 에 PNG/CSV로 저장.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import (
    HDBSCAN,
    OPTICS,
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42


# --------------------------------------------------------------------------- #
# 테스트 데이터 생성
# --------------------------------------------------------------------------- #
def make_datasets(n=600):
    """서로 다른 구조/난이도의 합성 데이터 4종."""
    ds = {}

    # 1) 잘 분리된 등방성 blob (쉬움)
    X, y = make_blobs(n_samples=n, centers=4, cluster_std=0.60, random_state=RNG)
    ds["blobs_easy"] = (X, y)

    # 2) 밀도/크기가 제각각인 blob (가변 밀도 — HDBSCAN/OPTICS에 유리)
    X, y = make_blobs(
        n_samples=n,
        centers=[[-6, -6], [0, 0], [6, 6], [6, -6]],
        cluster_std=[0.4, 1.4, 0.8, 2.2],
        random_state=RNG,
    )
    ds["blobs_varied"] = (X, y)

    # 3) 두 개의 초승달 (비볼록 — Spectral/density에 유리, 중심기반엔 불리)
    X, y = make_moons(n_samples=n, noise=0.07, random_state=RNG)
    ds["moons"] = (X, y)

    # 4) 노이즈가 섞인 blob (background noise — density 계열이 노이즈로 분류)
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=0.7, random_state=RNG)
    rng = np.random.RandomState(RNG)
    noise = rng.uniform(X.min(axis=0) - 2, X.max(axis=0) + 2, size=(n // 6, 2))
    X = np.vstack([X, noise])
    y = np.concatenate([y, -np.ones(len(noise), dtype=int)])  # -1 = 노이즈 GT
    ds["blobs_noise"] = (X, y)

    return ds


# --------------------------------------------------------------------------- #
# 각 데이터에 맞는 클러스터러 구성
# --------------------------------------------------------------------------- #
def make_clusterers(n_clusters):
    """라벨 없이도 동작하는 density 계열 + k를 요구하는 계열."""
    return {
        "HDBSCAN": HDBSCAN(min_cluster_size=15, min_samples=5),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ),
        "OPTICS": OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05),
        "Spectral": SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="cluster_qr",
            random_state=RNG,
        ),
    }


# --------------------------------------------------------------------------- #
# 평가 지표
# --------------------------------------------------------------------------- #
def evaluate(X, labels, y_true=None):
    """내부(정답 불필요) + 외부(정답 필요) 지표. 노이즈(-1)는 내부지표에서 제외."""
    out = {}
    labels = np.asarray(labels)
    mask = labels != -1
    uniq = set(labels[mask])
    n_clusters = len(uniq)
    n_noise = int((~mask).sum())
    out["n_clusters"] = n_clusters
    out["n_noise"] = n_noise

    # 내부 지표: 노이즈 제외, 클러스터 2개 이상일 때만 정의됨
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        out["silhouette"] = float(silhouette_score(Xm, lm))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
        out["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
    else:
        out["silhouette"] = np.nan
        out["calinski_harabasz"] = np.nan
        out["davies_bouldin"] = np.nan

    # 외부 지표: 정답이 있으면 전체 라벨 기준으로 비교
    if y_true is not None:
        out["ARI"] = float(adjusted_rand_score(y_true, labels))
        out["NMI"] = float(normalized_mutual_info_score(y_true, labels))
    return out


# --------------------------------------------------------------------------- #
# 시각화: 데이터 × 방법 그리드 산점도
# --------------------------------------------------------------------------- #
def plot_grid(datasets, results, outpath):
    methods = ["HDBSCAN", "Agglomerative", "OPTICS", "Spectral"]
    names = list(datasets.keys())
    nrows, ncols = len(names), len(methods) + 1  # +1: ground truth 열

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for r, dname in enumerate(names):
        X, y = datasets[dname]
        Xs = StandardScaler().fit_transform(X)

        # 0열: Ground Truth
        ax = axes[r, 0]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=y, s=8, cmap="tab10")
        ax.set_title(f"{dname}\n(ground truth)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        for c, m in enumerate(methods, start=1):
            ax = axes[r, c]
            labels = results[dname][m]["labels"]
            met = results[dname][m]["metrics"]
            # 노이즈(-1) 회색, 나머지 컬러맵
            lab = np.asarray(labels)
            noise = lab == -1
            ax.scatter(Xs[noise, 0], Xs[noise, 1], c="lightgray", s=8, marker="x")
            ax.scatter(Xs[~noise, 0], Xs[~noise, 1], c=lab[~noise], s=8, cmap="tab10")
            sil = met.get("silhouette", np.nan)
            ari = met.get("ARI", np.nan)
            ax.set_title(
                f"{m}\nk={met['n_clusters']} noise={met['n_noise']}\n"
                f"sil={sil:.3f} ARI={ari:.3f}",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Clustering methods comparison (standardized features)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[saved] {outpath}")


# --------------------------------------------------------------------------- #
# 시각화: dendrogram (계층적 클러스터링)
# --------------------------------------------------------------------------- #
def plot_dendrograms(datasets, outpath, method="ward"):
    names = list(datasets.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    for ax, dname in zip(axes, names):
        X, _ = datasets[dname]
        Xs = StandardScaler().fit_transform(X)
        Z = linkage(Xs, method=method)
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=30,           # 마지막 30개 병합만 표시
            leaf_rotation=90.0,
            leaf_font_size=7,
            show_contracted=True,
        )
        ax.set_title(f"{dname}\ndendrogram ({method} linkage)", fontsize=10)
        ax.set_xlabel("sample / (cluster size)", fontsize=8)
        ax.set_ylabel("merge distance", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"[saved] {outpath}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "clustering_test_out"))
    ap.add_argument("--n", type=int, default=600, help="데이터셋당 샘플 수")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    datasets = make_datasets(args.n)
    results = {}
    rows = []

    for dname, (X, y) in datasets.items():
        Xs = StandardScaler().fit_transform(X)
        k_true = len(set(y[y != -1]))  # 노이즈 제외 GT 클러스터 수
        clusterers = make_clusterers(n_clusters=k_true)
        results[dname] = {}
        for mname, clu in clusterers.items():
            labels = clu.fit_predict(Xs)
            metrics = evaluate(Xs, labels, y_true=y)
            results[dname][mname] = {"labels": labels, "metrics": metrics}
            rows.append({"dataset": dname, "method": mname, **metrics})
            print(
                f"{dname:14s} {mname:14s} "
                f"k={metrics['n_clusters']} noise={metrics['n_noise']:3d} "
                f"sil={metrics['silhouette']:.3f} "
                f"CH={metrics['calinski_harabasz']:.1f} "
                f"DB={metrics['davies_bouldin']:.3f} "
                f"ARI={metrics.get('ARI', float('nan')):.3f} "
                f"NMI={metrics.get('NMI', float('nan')):.3f}"
            )

    # 시각화
    plot_grid(datasets, results, os.path.join(args.outdir, "clustering_comparison.png"))
    plot_dendrograms(datasets, os.path.join(args.outdir, "dendrograms.png"))

    # 지표 CSV 저장
    csv_path = os.path.join(args.outdir, "metrics.csv")
    cols = ["dataset", "method", "n_clusters", "n_noise", "silhouette",
            "calinski_harabasz", "davies_bouldin", "ARI", "NMI"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[saved] {csv_path}")

    print("\n지표 요약:")
    print("  - silhouette: 높을수록 좋음 (-1~1)")
    print("  - calinski_harabasz: 높을수록 좋음")
    print("  - davies_bouldin: 낮을수록 좋음")
    print("  - ARI / NMI: 정답 대비 일치도, 높을수록 좋음 (1=완벽)")


if __name__ == "__main__":
    main()
