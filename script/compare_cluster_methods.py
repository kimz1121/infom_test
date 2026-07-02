"""Head-to-head clustering comparison: DINOv3-attnpool vs ResNet inFOM embeddings.

Purpose: the goal is ACTION discrimination -> fine-grained multi-cluster structure
matters, so we compare methods at MATCHED fixed k (not each method's own best-k)
plus auto-k HDBSCAN. For each embedding (raw + infom_latent of both runs) we run
Agglomerative(ward) / Spectral / KMeans at k in K_GRID and HDBSCAN(auto), and
report internal metrics (silhouette / Calinski-Harabasz / Davies-Bouldin).

Usage:
    python script/compare_cluster_methods.py \
        --dino_run exp/dinov3_attnpool/<run> --dino_epoch 40000 \
        --resnet_run exp/atomic_65_multimodal_precompute_3cam_lang_state_decoder/<run> \
        --resnet_epoch 500000 --n_samples 4000
"""
import argparse, json, os.path as osp, sys
from pathlib import Path
import numpy as np, jax

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, HDBSCAN
from sklearn.decomposition import PCA
from envs import robocasa_utils
from script.cluster_robocasa_embeddings import build_and_restore_agent, evaluate, streaming_obs_stats
from script.visualize_latent_robocasa import derive_task_labels, encode_latents
from utils.token_dataset import load_token_dataset

K_GRID = [5, 10, 20, 30, 40]


def encode_dino(run, epoch, n, seed):
    flags = json.load(open(Path(run) / "flags.json"))
    pre, _, _ = robocasa_utils.parse_env_name(flags["env_name"])
    dd = osp.expanduser(robocasa_utils.DEFAULT_DATASET_DIR)
    ds = load_token_dataset(dd, pre, max_size=flags.get("pretraining_size", np.inf) or np.inf)
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(ds.size, size=min(n, ds.size), replace=False))
    obs = ds._gather_obs(ds.obs_idx[idxs]).astype(np.float32)
    act = ds.actions[idxs].astype(np.float32)
    task_ref, _ = derive_task_labels(idxs, osp.join(dd, f"{pre}.stats.json"), ds.size)
    # raw = mean-pooled tokens + state (z-scored)
    nc, nt, fd = ds.n_cam, ds.n_tok, ds.feat
    tok = obs[:, :ds.token_dim].reshape(len(idxs), nc, nt, fd).mean(2).reshape(len(idxs), nc * fd)
    raw = np.concatenate([tok, obs[:, ds.token_dim:ds.token_dim + ds.state_dim]], -1)
    raw = ((raw - raw.mean(0)) / (raw.std(0) + 1e-8)).astype(np.float32)
    agent = build_and_restore_agent(flags, str(run), epoch, ds.obs_dim, act.shape[1])
    z, _, _ = encode_latents(agent, {"observations": obs, "actions": act}, jax.random.PRNGKey(seed))
    return {"dino_raw": raw, "dino_infom": np.asarray(z, np.float32)}, task_ref


def encode_resnet(run, epoch, n, seed):
    import h5py
    flags = json.load(open(Path(run) / "flags.json"))
    pre, _, _ = robocasa_utils.parse_env_name(flags["env_name"])
    dd = osp.expanduser(robocasa_utils.DEFAULT_DATASET_DIR)
    h5 = osp.join(dd, f"{pre}.hdf5")
    max_rows = flags.get("pretraining_size", np.inf) or np.inf
    mean, var, ntot = streaming_obs_stats(h5, "observations", max_rows)
    nused = int(min(ntot, max_rows))
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(nused, size=min(n, nused), replace=False))
    with h5py.File(h5, "r") as f:
        obs_raw = f["observations"][idxs].astype(np.float32)
        act = f["actions"][idxs].astype(np.float32)
    obs_norm = (obs_raw - mean) / np.sqrt(var + 1e-8)
    task_ref, _ = derive_task_labels(idxs, h5.replace(".hdf5", ".stats.json"), nused)
    agent = build_and_restore_agent(flags, str(run), epoch, obs_norm.shape[1], act.shape[1])
    z, _, _ = encode_latents(agent, {"observations": obs_norm, "actions": act}, jax.random.PRNGKey(seed))
    return {"resnet_raw": obs_norm.astype(np.float32), "resnet_infom": np.asarray(z, np.float32)}, task_ref


def sweep(name, X, task_ref, pca_dim, seed):
    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    X = X.astype(np.float64)
    rows = []
    def rec(method, labels):
        m = evaluate(X, labels, task_ref); m.update(source=name, method=method); rows.append(m); return m
    try:
        rec("HDBSCAN(auto)", HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(X))
    except Exception as e:
        print("  HDBSCAN failed", e)
    for k in K_GRID:
        rec(f"Agglo(ward,k={k})", AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X))
        rec(f"KMeans(k={k})", KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X))
        try:
            rec(f"Spectral(k={k})", SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                                       n_neighbors=15, random_state=seed,
                                                       assign_labels="cluster_qr").fit_predict(X))
        except Exception as e:
            print(f"  Spectral k={k} failed", e)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_run", required=True)
    ap.add_argument("--dino_epoch", type=int, required=True)
    ap.add_argument("--resnet_run", required=True)
    ap.add_argument("--resnet_epoch", type=int, default=500000)
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    print("Encoding DINOv3 ..."); dino, t_d = encode_dino(args.dino_run, args.dino_epoch, args.n_samples, args.seed)
    print("Encoding ResNet ..."); resn, t_r = encode_resnet(args.resnet_run, args.resnet_epoch, args.n_samples, args.seed)

    all_rows = []
    for name, X, tr in [("resnet_infom", resn["resnet_infom"], t_r),
                        ("dino_infom", dino["dino_infom"], t_d),
                        ("resnet_raw", resn["resnet_raw"], t_r),
                        ("dino_raw", dino["dino_raw"], t_d)]:
        print(f"\n### {name}  shape={X.shape}")
        all_rows += sweep(name, X, tr, args.pca_dim, args.seed)

    # print grid: method x source (silhouette / CH)
    srcs = ["resnet_infom", "dino_infom", "resnet_raw", "dino_raw"]
    methods = ["HDBSCAN(auto)"] + [f"{m}(k={k})".replace("(k", ",k") if False else f"{m}(k={k})"
                                   for k in K_GRID for m in ["Agglo(ward", "KMeans", "Spectral"]]
    idx = {(r["source"], r["method"]): r for r in all_rows}
    print("\n===== silhouette (higher=better) =====")
    print(f"{'method':22} " + " ".join(f"{s:>13}" for s in srcs))
    seen = []
    for r in all_rows:
        if r["method"] in seen: continue
        seen.append(r["method"])
    for m in seen:
        cells = []
        for s in srcs:
            rr = idx.get((s, m))
            cells.append(f"{rr['silhouette']:>13.3f}" if rr and rr['silhouette']==rr['silhouette'] else f"{'-':>13}")
        print(f"{m:22} " + " ".join(cells))
    print("\n===== calinski_harabasz (higher=better) =====")
    print(f"{'method':22} " + " ".join(f"{s:>13}" for s in srcs))
    for m in seen:
        cells = []
        for s in srcs:
            rr = idx.get((s, m))
            cells.append(f"{rr['calinski_harabasz']:>13.0f}" if rr and rr['calinski_harabasz']==rr['calinski_harabasz'] else f"{'-':>13}")
        print(f"{m:22} " + " ".join(cells))

    outdir = Path(args.outdir) if args.outdir else Path(args.dino_run) / "plots" / "method_compare"
    outdir.mkdir(parents=True, exist_ok=True)
    import csv
    with (outdir / "method_compare.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {outdir/'method_compare.csv'}")


if __name__ == "__main__":
    main()
