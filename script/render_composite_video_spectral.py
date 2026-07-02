"""render_composite_video.py 변형(Robocasa): atomic-65 백드롭을 '주어진 task' 대신
**Spectral 클러스터 카테고리**로 색칠/라벨링해 composite 데모 영상을 다시 만든다.

render_libero_video_mm_spectral.py 의 Robocasa 판. render_composite_video.render_video
는 카테고리 이름에 무관하므로, 백드롭 per-point 라벨을 task→"S{c}"(Spectral 클러스터)로
바꾸면 KNN-투표 타임라인/범례가 Spectral 카테고리 기준이 된다.

원본(render_composite_video.py)은 건드리지 않고 헬퍼만 재사용. 출력은 별도 폴더
  <run_dir>/plots/composite_validation_spectral/ep<epoch>/  (기존 미덮어씀)

agent+백드롭+Spectral+백드롭 임베딩은 데모와 무관하므로 한 번만 계산하고 여러 composite에
재사용한다(원본은 composite마다 재빌드).

Usage:
  python script/render_composite_video_spectral.py \
      --run_dir exp/atomic_65_multimodal_precompute_3cam_lang_state_decoder/sd000_20260609_103625 \
      --n_clusters 30 \
      --composites loaddishwasher_3cam_lang,placeveggiesindrawer_3cam_lang,stackbowlscabinet_3cam_lang,startelectrickettle_3cam_lang
"""
from __future__ import annotations
import argparse, json, os, os.path as osp, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import jax, jax.numpy as jnp

from script.visualize_latent_robocasa import (
    build_agent_and_pretrain_dataset, encode_latents, _resolve_stats_path, distinct_color_map,
)
from script.visualize_composite_traj import load_composite_episode, knn_vote_labels
from script.render_composite_video import (
    composite_task_name_from_basename, find_composite_mp4, decode_all_frames,
    _pca_ref2, _align_to_ref, render_video,
)
from utils.flax_utils import restore_agent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default="exp/atomic_65_multimodal_precompute_3cam_lang_state_decoder/sd000_20260609_103625")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--composites", default="loaddishwasher_3cam_lang,placeveggiesindrawer_3cam_lang,stackbowlscabinet_3cam_lang,startelectrickettle_3cam_lang",
                    help="쉼표구분 composite basename 목록(각 --episode 렌더)")
    ap.add_argument("--category", default="composite", choices=["composite", "atomic"])
    ap.add_argument("--robocasa_dir", default="~/.robocasa/data")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera", default="robot0_agentview_left",
                    choices=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "both"])
    ap.add_argument("--num_atomic", type=int, default=4000)
    ap.add_argument("--max_atomic_plot", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--min_seg_frac", type=float, default=0.03)
    ap.add_argument("--max_T", type=int, default=0)
    # Spectral 설정 (분석 스크립트와 동일 config).
    ap.add_argument("--n_clusters", type=int, default=30)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--spectral_nn", type=int, default=15)
    ap.add_argument("--out_subdir", default="composite_validation_spectral")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags.get("env_name", "")
    assert env_name.startswith("robocasa_"), f"{env_name} is not robocasa_*"

    # ---- 데모 무관: 한 번만 계산 ----
    print("Building agent + atomic backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    agent = restore_agent(agent, str(run_dir), epoch)
    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Backdrop {len(a_mean)} latents, dim={a_mean.shape[1]}")

    # ---- Spectral(PCA-50) → per-point 카테고리 "S{c}" ----
    Xr = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(a_mean) \
        if a_mean.shape[1] > args.pca_dim else a_mean
    spec = SpectralClustering(
        n_clusters=args.n_clusters, affinity="nearest_neighbors",
        n_neighbors=args.spectral_nn, assign_labels="cluster_qr", random_state=args.seed,
    ).fit_predict(Xr)
    cat_labels = np.array([f"S{c:02d}" for c in spec], dtype=object)
    present = [f"S{i:02d}" for i in range(args.n_clusters) if (spec == i).any()]
    color_map = distinct_color_map(present)
    print(f"Spectral k={args.n_clusters} → {len(present)} non-empty clusters")

    # 플롯용 백드롭 서브샘플(클러스터별 quota).
    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present)))
    for c in present:
        idx_c = np.flatnonzero(cat_labels == c)
        sel.extend(rng_np.choice(idx_c, size=min(quota, len(idx_c)), replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot, a_labels_plot = a_mean[sel], cat_labels[sel]

    # 백드롭 임베딩 fit (데모 무관).
    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    backdrop2d = {}
    if "umap" in methods and umap is not None:
        m = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      random_state=args.seed, metric="euclidean").fit(a_mean_plot)
        backdrop2d["umap"] = (m.embedding_, ("umap", m))
    if "tsne" in methods:
        perp = min(30.0, max(5.0, (len(a_mean_plot) - 1) / 3.0))
        bd = TSNE(n_components=2, perplexity=perp, max_iter=1000, init="pca",
                  learning_rate="auto", random_state=args.seed).fit_transform(a_mean_plot)
        nn = NearestNeighbors(n_neighbors=10).fit(a_mean_plot)
        backdrop2d["tsne"] = (bd, ("tsne", (bd, nn)))
    ref2 = _pca_ref2(a_mean_plot)

    out_dir = run_dir / "plots" / args.out_subdir / f"ep{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- composite별 렌더 ----
    for cname in [c.strip() for c in args.composites.split(",") if c.strip()]:
        print(f"\n=== composite: {cname}  ep#{args.episode} ===")
        comp_hdf5 = osp.join(osp.expanduser(args.robocasa_dir), f"{cname}.hdf5")
        if not osp.exists(comp_hdf5):
            print(f"  [skip] not found: {comp_hdf5}"); continue
        ep = load_composite_episode(comp_hdf5, args.episode)
        T = ep["length"]
        assert ep["observations"].shape[1] == raw_obs.shape[1], \
            f"obs dim mismatch: comp {ep['observations'].shape[1]} vs run {raw_obs.shape[1]}"

        comp_obs_norm = pre_train.normalize_observations(observations=ep["observations"]).astype(np.float32)
        comp_act = np.clip(ep["actions"], -1.0 + 1e-5, 1.0 - 1e-5).astype(np.float32)
        comp_mean = np.asarray(agent.network.select("intention_encoder")(
            jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())

        voted, conf = knn_vote_labels(comp_mean, a_mean, cat_labels, k=args.knn)
        print(f"  T={T}  mean vote conf={conf.mean():.3f}  (Spectral k={args.n_clusters})")

        task_name = composite_task_name_from_basename(cname, args.category)
        mp4 = find_composite_mp4(task_name, args.episode, args.camera, args.category)
        frames = decode_all_frames(mp4)

        for method, (bd_layout, projector) in backdrop2d.items():
            if projector[0] == "umap":
                comp2d = projector[1].transform(comp_mean)
            else:
                _, nn = projector[1]
                _, nidx = nn.kneighbors(comp_mean)
                comp2d = bd_layout[nidx].mean(axis=1)
            emb_a, emb_c = _align_to_ref(bd_layout, comp2d, ref2)

            data = dict(run_dir=run_dir, epoch=epoch, T=T, comp_mean=comp_mean,
                        embeddings={method: (emb_a, emb_c)},
                        atomic_labels=a_labels_plot, present_tasks=present,
                        voted=voted, conf=conf, color_map=color_map)
            out_path = out_dir / f"composite_{cname}_ep{args.episode}_spectralk{args.n_clusters}_{method}.mp4"
            subtitle = (f"T={T} steps  •  epoch={epoch}  •  knn={args.knn}  •  "
                        f"mean_conf={conf.mean():.2f}  •  color=Spectral(k={args.n_clusters}) clusters")
            render_video(data, frames, out_path, emb_atomic=emb_a, emb_comp=emb_c,
                         embedding_name=method.upper(), fps=args.fps,
                         title=f"{task_name}  ep#{args.episode}  ({method.upper()})  [Spectral k={args.n_clusters}]",
                         subtitle=subtitle, max_T=args.max_T, min_seg_frac=args.min_seg_frac,
                         conf=conf)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
