"""render_libero_video_mm.py 변형: 백드롭을 '주어진 task' 대신 **Spectral 클러스터
카테고리**로 색칠/라벨링해 데모 영상을 다시 만든다.

동기: LIBERO latent 클러스터링에서 Spectral(k)로 나온 카테고리를 영상 위에서 보고 싶다.
render_composite_video.render_video 는 카테고리 이름에 무관하므로, 백드롭 per-point
라벨을 task→"S{c}"(Spectral 클러스터)로 갈아끼우면 KNN-투표 타임라인/범례가 자동으로
Spectral 카테고리 기준이 된다.

원본(render_libero_video_mm.py)은 건드리지 않고 헬퍼만 재사용한다. 출력은 별도 폴더
  <run_dir>/plots/latent_libero_spectral/   (기존 latent_libero/ 를 덮어쓰지 않음)

agent+백드롭+Spectral+백드롭 임베딩은 데모와 무관하므로 한 번만 계산하고 여러 task에
재사용한다(원본은 task마다 재빌드).

Usage:
  python script/render_libero_video_mm_spectral.py \
      --run_dir exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850 \
      --n_clusters 30 \
      --tasks turn_on_the_stove,push_the_plate_to_the_front_of_the_stove,put_the_wine_bottle_on_the_rack
"""
from __future__ import annotations
import argparse, json, os, sys
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

from data_gen_scripts.generate_libero_dataset import resolve_suite_dir, DEFAULT_STATE_KEYS
from script.visualize_composite_traj import knn_vote_labels
from script.render_composite_video import render_video
from script.visualize_latent_robocasa import distinct_color_map
from script.visualize_latent_libero import (
    build_agent_and_pretrain_dataset, encode_latents, _resolve_stats_path,
)
# 원본 모듈에서 데모 로더/정렬 헬퍼 재사용.
from script.render_libero_video_mm import load_demo_mm, _pca_ref2, _align_to_ref
from utils.flax_utils import restore_agent
from utils.visual_features import FrozenResNet34Extractor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", default="exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--raw_root", default="")
    ap.add_argument("--tasks", default="turn_on_the_stove,push_the_plate_to_the_front_of_the_stove,put_the_wine_bottle_on_the_rack",
                    help="쉼표구분 task 목록(각 demo0 렌더)")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--libero_dir", default="~/.libero/data")
    ap.add_argument("--lang_json", default="~/.libero/data/libero_goal_lang_embeddings.json")
    ap.add_argument("--camera", default="agentview", choices=["agentview", "eye_in_hand"])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "both"])
    ap.add_argument("--num_atomic", type=int, default=4000)
    ap.add_argument("--max_atomic_plot", type=int, default=2000)
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--min_seg_frac", type=float, default=0.03)
    ap.add_argument("--max_T", type=int, default=0)
    ap.add_argument("--no_conf", action="store_true")
    # Spectral 클러스터링 설정 (분석 스크립트와 동일 config).
    ap.add_argument("--n_clusters", type=int, default=30)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--spectral_nn", type=int, default=15)
    ap.add_argument("--out_subdir", default="latent_libero_spectral")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]; cfg = flags["agent"]
    img_feat_dim = int(cfg.get("image_feat_dim", 512))
    assert "multimodal" in env_name, f"{env_name} is not a multimodal run"

    # ---- 데모와 무관한 부분: 한 번만 계산 ----
    print("Building agent + libero backdrop ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    agent = restore_agent(agent, str(run_dir), epoch)
    target_dim = raw_obs.shape[1]
    is_lang = target_dim > img_feat_dim + 15

    rng_np = np.random.default_rng(args.seed)
    a_idxs = rng_np.integers(0, len(raw_obs), size=args.num_atomic)
    a_batch = pre_train.sample(args.num_atomic, idxs=a_idxs)
    a_mean, _, _ = encode_latents(agent, a_batch, jax.random.PRNGKey(args.seed))
    print(f"Backdrop {len(a_mean)} latents, dim={a_mean.shape[1]} (lang={is_lang})")

    # ---- Spectral 클러스터링(PCA-50) → per-point 카테고리 "S{c}" ----
    Xr = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(a_mean) \
        if a_mean.shape[1] > args.pca_dim else a_mean
    spec = SpectralClustering(
        n_clusters=args.n_clusters, affinity="nearest_neighbors",
        n_neighbors=args.spectral_nn, assign_labels="cluster_qr", random_state=args.seed,
    ).fit_predict(Xr)
    cat_labels = np.array([f"S{c:02d}" for c in spec], dtype=object)  # 백드롭 전체(4000) 라벨
    present = [f"S{i:02d}" for i in range(args.n_clusters) if (spec == i).any()]
    color_map = distinct_color_map(present)
    sizes = {c: int((cat_labels == c).sum()) for c in present}
    print(f"Spectral k={args.n_clusters} → {len(present)} non-empty clusters; sizes={sizes}")

    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    with open(stats_path) as f:
        stats = json.load(f)
    state_keys = stats.get("state_keys", list(DEFAULT_STATE_KEYS))
    image_size = int(stats.get("image_size", 128))

    # 플롯용 백드롭 서브샘플(클러스터별 quota) — 데모 무관.
    sel = []
    quota = max(1, args.max_atomic_plot // max(1, len(present)))
    for c in present:
        idx_c = np.flatnonzero(cat_labels == c)
        sel.extend(rng_np.choice(idx_c, size=min(quota, len(idx_c)), replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot, a_labels_plot = a_mean[sel], cat_labels[sel]

    # 백드롭 임베딩(umap/tsne) — 데모 무관, 한 번만 fit.
    methods = ["umap", "tsne"] if args.method == "both" else [args.method]
    backdrop2d = {}   # method -> (bd_layout, projector)
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

    ext = FrozenResNet34Extractor(device="cuda", batch_size=256)
    lang = json.load(open(os.path.expanduser(args.lang_json))) if is_lang else None
    suite_dir = resolve_suite_dir(args.suite, args.raw_root)
    out_dir = run_dir / "plots" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 데모별 렌더 ----
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"\n=== demo: {task} #{args.demo} ===")
        ep = load_demo_mm(suite_dir, task, args.demo, state_keys, args.camera,
                          image_size, flip=not args.no_flip)
        T = ep["length"]
        feat = ext.extract(ep["enc_imgs"][:T])
        demo_obs = np.concatenate([feat, ep["state"][:T]], axis=-1).astype(np.float32)
        if is_lang:
            emb = np.asarray(lang["embeddings"][task], dtype=np.float32)
            demo_obs = np.concatenate([demo_obs, np.broadcast_to(emb, (T, emb.shape[0]))], axis=-1)
        assert demo_obs.shape[1] == target_dim, (demo_obs.shape, target_dim)

        comp_obs_norm = pre_train.normalize_observations(observations=demo_obs).astype(np.float32)
        comp_act = np.clip(ep["actions"][:T], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
        comp_mean = np.asarray(agent.network.select("intention_encoder")(
            jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())

        voted, conf = knn_vote_labels(comp_mean, a_mean, cat_labels, k=args.knn)
        print(f"  T={T}  mean vote conf={conf.mean():.3f}  (categories=Spectral k={args.n_clusters})")

        for method, (bd_layout, projector) in backdrop2d.items():
            kind = projector[0]
            if kind == "umap":
                comp2d = projector[1].transform(comp_mean)
            else:
                _, nn = projector[1]
                _, nidx = nn.kneighbors(comp_mean)
                comp2d = bd_layout[nidx].mean(axis=1)
            emb_a, emb_c = _align_to_ref(bd_layout, comp2d, ref2)

            data = dict(run_dir=run_dir, epoch=epoch, T=T, comp_mean=comp_mean,
                        embeddings={method: (emb_a, emb_c)}, frames=ep["frames"],
                        instruction=ep["instruction"], atomic_labels=a_labels_plot,
                        present_tasks=present, voted=voted, conf=conf, color_map=color_map)
            out_path = out_dir / f"demo_{task}_ep{args.demo}_spectralk{args.n_clusters}_{method}.mp4"
            sub = (f"\"{ep['instruction']}\"  •  T={T}  •  epoch={epoch}  •  knn={args.knn}  •  "
                   f"mean_conf={conf.mean():.2f}  •  color=Spectral(k={args.n_clusters}) clusters")
            render_video(data, ep["frames"], out_path, emb_atomic=emb_a, emb_comp=emb_c,
                         embedding_name=method.upper(), fps=args.fps,
                         title=f"{task}  demo#{args.demo}  ({method.upper()})  [Spectral k={args.n_clusters}]",
                         subtitle=sub, max_T=args.max_T, min_seg_frac=args.min_seg_frac,
                         conf=(None if args.no_conf else conf))
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
