"""Rank DINOv3-attnpool checkpoints by fine-grained clustering quality.

Goal = action discrimination -> we score each checkpoint by how well its
intention-encoder latent forms MANY well-separated clusters. Obs are epoch-
independent, so we gather them once and only swap the restored agent per epoch.
For each epoch we cluster at fixed k in K_SCORE with Agglomerative/KMeans/Spectral
and report the per-k best silhouette + CH; the ranking key is the mean silhouette
over K_SCORE (fine-grained separability).

Usage:
    python script/sweep_dino_epochs.py --run_dir exp/dinov3_attnpool/<run> \
        --epochs 10000,20000,...,100000 --n_samples 4000
"""
import argparse, json, os.path as osp, sys
from pathlib import Path
import numpy as np, jax

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from envs import robocasa_utils
from script.cluster_robocasa_embeddings import build_and_restore_agent, evaluate
from script.visualize_latent_robocasa import derive_task_labels, encode_latents
from utils.token_dataset import load_token_dataset

K_SCORE = [10, 20, 30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", default="10000,20000,30000,40000,50000,60000,70000,80000,90000,100000")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(PROJECT_ROOT) / run_dir
    flags = json.load(open(run_dir / "flags.json"))
    pre, _, _ = robocasa_utils.parse_env_name(flags["env_name"])
    dd = osp.expanduser(robocasa_utils.DEFAULT_DATASET_DIR)

    ds = load_token_dataset(dd, pre, max_size=flags.get("pretraining_size", np.inf) or np.inf)
    rng = np.random.default_rng(args.seed)
    idxs = np.sort(rng.choice(ds.size, size=min(args.n_samples, ds.size), replace=False))
    obs = ds._gather_obs(ds.obs_idx[idxs]).astype(np.float32)
    act = ds.actions[idxs].astype(np.float32)
    task_ref, _ = derive_task_labels(idxs, osp.join(dd, f"{pre}.stats.json"), ds.size)
    print(f"gathered obs {obs.shape} once; sweeping epochs at k={K_SCORE}\n")

    epochs = [int(e) for e in args.epochs.split(",")]
    results = []
    for E in epochs:
        agent = build_and_restore_agent(flags, str(run_dir), E, obs.shape[1], act.shape[1])
        z, _, _ = encode_latents(agent, {"observations": obs, "actions": act}, jax.random.PRNGKey(args.seed))
        X = np.asarray(z, np.float32)
        if args.pca_dim and X.shape[1] > args.pca_dim:
            X = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(X)
        X = X.astype(np.float64)
        per_k = {}
        for k in K_SCORE:
            best = None
            for name, lab in [
                ("Agglo", AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)),
                ("KMeans", KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(X)),
                ("Spectral", SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                             n_neighbors=15, random_state=args.seed,
                             assign_labels="cluster_qr").fit_predict(X)),
            ]:
                m = evaluate(X, lab, task_ref)
                if best is None or m["silhouette"] > best[1]:
                    best = (name, m["silhouette"], m["calinski_harabasz"])
            per_k[k] = best
        mean_sil = float(np.mean([per_k[k][1] for k in K_SCORE]))
        mean_ch = float(np.mean([per_k[k][2] for k in K_SCORE]))
        results.append((E, mean_sil, mean_ch, per_k))
        cells = " ".join(f"k{k}:{per_k[k][1]:.3f}({per_k[k][0][:4]})" for k in K_SCORE)
        print(f"epoch {E:>6}: meanSil={mean_sil:.3f} meanCH={mean_ch:5.0f} | {cells}")

    results.sort(key=lambda r: r[1], reverse=True)
    print("\n=== ranked by mean silhouette over k=10/20/30 (fine-grained) ===")
    for E, ms, mc, _ in results:
        print(f"  epoch {E:>6}: meanSil={ms:.3f} meanCH={mc:5.0f}")
    print(f"\nBEST_EPOCH={results[0][0]}")


if __name__ == "__main__":
    main()
