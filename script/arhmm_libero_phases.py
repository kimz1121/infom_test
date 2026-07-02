"""AR-HMM으로 LIBERO inFOM intention latent 시퀀스의 'Action Phase' 추적.

동기: render_libero_video_mm_spectral.py 는 백드롭 latent을 정적 Spectral 클러스터로
나눈 뒤 데모 프레임마다 KNN-투표로 카테고리를 붙인다 — 시간 구조가 없어 라벨이
프레임 사이에 튀고 "phase가 언제 바뀌는지"를 원리적으로 못 짚는다.

여기서는 dynamax 의 LinearAutoregressiveHMM 을 에피소드별 intention latent 시퀀스
z_t = q(z|s_t,a_t).mean() 에 적합한다. 각 hidden state = 하나의 action phase =
하나의 선형 동역학 regime  z_t | z_{t-1}, k ~ N(A_k z_{t-1} + b_k, Q_k).  얻는 것:
  1) 시간적으로 일관된 phase segmentation (Viterbi) — KNN flicker 없음
  2) 전이 행렬 = phase 변화 그래프 (어느 phase→어느 phase, 얼마나 sticky)
  3) task별 phase-over-time 점유율 → bowl-placement family가 초기 phase를 공유하는지 등

원본 스크립트는 건드리지 않고 헬퍼만 재사용한다. 출력 폴더:
  <run_dir>/plots/latent_libero_arhmm/   (기존 latent_libero* 를 덮어쓰지 않음)

Usage:
  python script/arhmm_libero_phases.py \
      --run_dir exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850 \
      --num_states 10 --pca_dim 10 \
      --tasks turn_on_the_stove,push_the_plate_to_the_front_of_the_stove,put_the_wine_bottle_on_the_rack
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x
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
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

from data_gen_scripts.generate_libero_dataset import resolve_suite_dir, DEFAULT_STATE_KEYS
from script.render_composite_video import render_video
from script.visualize_latent_robocasa import distinct_color_map
from script.visualize_latent_libero import (
    build_agent_and_pretrain_dataset, encode_latents, _resolve_stats_path,
    derive_task_labels, TASK_FAMILY_MAP, LIBERO_GOAL_10,
)
from script.render_libero_video_mm import load_demo_mm, _pca_ref2, _align_to_ref
from utils.flax_utils import restore_agent
from utils.visual_features import FrozenResNet34Extractor


# ---------------------------------------------------------------------------
# 에피소드 추출 + 전체 인코딩
# ---------------------------------------------------------------------------
def episode_spans(terminals: np.ndarray) -> list[tuple[int, int]]:
    """terminals>0 위치로 [start,end] (양끝 포함) 에피소드 span 목록."""
    term = np.nonzero(terminals > 0)[0]
    if len(term) == 0:
        return [(0, len(terminals) - 1)]
    starts = np.concatenate([[0], term[:-1] + 1])
    return list(zip(starts.tolist(), term.tolist()))


def encode_all(agent, pre_train, n_rows: int, chunk: int, seed: int) -> np.ndarray:
    """전체 데이터셋 행을 intention latent mean 으로 인코딩 (512-d)."""
    rng = jax.random.PRNGKey(seed)
    out = []
    for s in range(0, n_rows, chunk):
        idxs = np.arange(s, min(s + chunk, n_rows))
        batch = pre_train.sample(len(idxs), idxs=idxs)
        mean, _, _ = encode_latents(agent, batch, rng)
        out.append(np.asarray(mean, dtype=np.float32))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# AR-HMM 적합 / 디코드
# ---------------------------------------------------------------------------
def build_hmm(num_states: int, dim: int, stickiness: float):
    return LinearAutoregressiveHMM(
        num_states=num_states, emission_dim=dim, num_lags=1,
        transition_matrix_stickiness=stickiness,
    )


def resample_seq(seq: np.ndarray, L: int) -> np.ndarray:
    """시퀀스를 선형보간으로 길이 L 로 리샘플(정규화 시간)."""
    if len(seq) == L:
        return seq.astype(np.float32)
    xp = np.linspace(0.0, 1.0, len(seq))
    x = np.linspace(0.0, 1.0, L)
    return np.stack([np.interp(x, xp, seq[:, d]) for d in range(seq.shape[1])],
                    axis=1).astype(np.float32)


def prepare_fit(seqs: list[np.ndarray], fit_len: int, jitter: float, seed: int):
    """적합용 전처리: 동일 길이 L 로 리샘플(패딩 퇴화 회피) + 미세 jitter(covariance
    floor — dynamax ARHMM emission 은 prior 가 없어 특이 covariance 시 NaN)."""
    lens = np.array([len(s) for s in seqs])
    L = fit_len if fit_len > 0 else int(np.median(lens))
    rs = [resample_seq(s, L) for s in seqs]
    stacked = np.concatenate(rs, axis=0)
    scale = float(stacked.std()) + 1e-8
    rng = np.random.default_rng(seed)
    rs = [s + (jitter * scale) * rng.standard_normal(s.shape).astype(np.float32)
          for s in rs]
    return rs, L


def fit_hmm(hmm, seqs: list[np.ndarray], num_iters: int, seed: int):
    """동일 길이 시퀀스 배치 EM 적합 (prepare_fit 로 전처리된 seqs 를 받는다)."""
    key = jax.random.PRNGKey(seed)
    stacked = np.concatenate(seqs, axis=0).astype(np.float32)
    try:
        params, props = hmm.initialize(key, method="kmeans", emissions=jnp.asarray(stacked))
    except Exception as e:  # pragma: no cover
        print(f"[warn] kmeans init failed ({e!r}); prior init")
        params, props = hmm.initialize(key, method="prior")
    E = np.stack(seqs).astype(np.float32)
    I = np.stack([np.asarray(hmm.compute_inputs(jnp.asarray(s)))
                  for s in seqs]).astype(np.float32)
    params, lls = hmm.fit_em(params, props, jnp.asarray(E), jnp.asarray(I),
                             num_iters=num_iters, verbose=False)
    return params, np.asarray(lls)


def decode_offline(hmm, params, seq: np.ndarray):
    """OFFLINE 참조: Viterbi(most_likely_states) + smoother posterior.
    둘 다 전체 시퀀스(미래 포함)를 봐야 하므로 **Online Action Detection 에는 부적합**.
    학습데이터 특성화/온라인 정확도 비교 기준으로만 사용."""
    s = jnp.asarray(seq.astype(np.float32))
    ins = hmm.compute_inputs(s)
    z = np.asarray(hmm.most_likely_states(params, s, ins))
    conf = np.asarray(hmm.smoother(params, s, ins).smoothed_probs).max(axis=-1)
    return z.astype(int), conf.astype(np.float32)


def decode_online(hmm, params, seq: np.ndarray):
    """ONLINE (causal): forward filter. phase_t = argmax p(z_t | y_{1:t}).
    filtered_probs[t] 는 프레임 t 까지만 의존(인과성 검증 완료) → OAD 에 적합.
    미래를 안 보므로 offline Viterbi 보다 경계에서 지연/떨림이 있을 수 있음."""
    s = jnp.asarray(seq.astype(np.float32))
    ins = hmm.compute_inputs(s)
    fp = np.asarray(hmm.filter(params, s, ins).filtered_probs)  # (T,K)
    return fp.argmax(axis=-1).astype(int), fp.max(axis=-1).astype(np.float32)


def decode_fixedlag(hmm, params, seq: np.ndarray, lag: int, left: int = 20):
    """CAUSAL fixed-lag smoothing: 프레임 τ 를 p(z_τ | y_{0:τ+lag}) 로 추정.
    'now'=τ+lag 까지만 관측 → 미래 안 봄(지연 lag 프레임). filter(lag=0)와 offline
    smoother(lag=∞) 사이의 중간. 윈도우 [τ-left, τ+lag] 에 smoother 를 돌려 근사
    (HMM forward/backward 영향이 기하 감쇠라 짧은 윈도우로 충분; 윈도우 첫 프레임의
    initial-prior 오차는 left 프레임 forward 로 씻김). 마지막 lag 프레임은 미래가 모자라
    가용 길이로 자동 축소 → 사실상 filter."""
    s = seq.astype(np.float32); T = len(s)
    states = np.empty(T, dtype=int); conf = np.empty(T, dtype=np.float32)
    for tau in range(T):
        end = min(T - 1, tau + lag)
        a = max(0, tau - left)
        win = jnp.asarray(s[a:end + 1])
        prev = None if a == 0 else jnp.asarray(s[a - 1:a])  # AR lag 이 윈도우 밖일 때 참 과거 주입
        ins = hmm.compute_inputs(win, prev)
        sp = np.asarray(hmm.smoother(params, win, ins).smoothed_probs)
        j = tau - a
        states[tau] = int(sp[j].argmax()); conf[tau] = float(sp[j].max())
    return states, conf


class OnlinePhaseDetector:
    """배포용 진짜 스트리밍 detector: PCA 적용된 latent z_t 를 하나씩 받아 belief 를
    갱신하고 phase 를 즉시 반환한다(미래 없음). dynamax filter() 배치 결과와 수치적으로
    동일함을 verify_online() 로 확인. 학습된 params 만 있으면 되고 dynamax 런타임 불필요.

    사용:
        det = OnlinePhaseDetector(params); det.reset()
        for z in stream:            # z: (pca_dim,) — PCA 변환된 latent mean
            phase, conf = det.step(z)
    """
    def __init__(self, params):
        self.A = np.asarray(params.transitions.transition_matrix, dtype=np.float64)
        self.init = np.asarray(params.initial.probs, dtype=np.float64)
        self.W = np.asarray(params.emissions.weights, dtype=np.float64)   # (K,d,d*lags)
        self.b = np.asarray(params.emissions.biases, dtype=np.float64)    # (K,d)
        covs = np.asarray(params.emissions.covs, dtype=np.float64)        # (K,d,d)
        self.K, self.d = self.b.shape
        self.chol = np.linalg.cholesky(covs)
        self.logdet = 2.0 * np.log(np.diagonal(self.chol, axis1=1, axis2=2)).sum(-1)
        self.reset()

    def reset(self):
        self.belief = None
        self.prev_y = None

    def _emission_ll(self, u, y):
        mean = self.W @ u + self.b                      # (K,d), num_lags=1 → u=y_{t-1}
        diff = (y[None, :] - mean)[..., None]           # (K,d,1)
        sol = np.linalg.solve(self.chol, diff)[..., 0]  # L_k x = diff_k
        maha = (sol ** 2).sum(-1)
        return -0.5 * (self.d * np.log(2 * np.pi) + self.logdet + maha)

    def step(self, y):
        y = np.asarray(y, dtype=np.float64)
        if self.belief is None:                 # t=0: 이전 관측 없음 → lag=0, prior 사용
            pred, u = self.init, np.zeros(self.d)
        else:                                    # predict: p(z_t|y_{<t}) = Aᵀ belief
            pred, u = self.A.T @ self.belief, self.prev_y
        log_post = np.log(pred + 1e-30) + self._emission_ll(u, y)
        log_post -= log_post.max()
        post = np.exp(log_post); post /= post.sum()
        self.belief, self.prev_y = post, y
        k = int(post.argmax())
        return k, float(post[k])

    # ---- 미래 예측 (전부 현재 belief b_t=p(z_t|y_{1:t}) 에서만 → causal 유지) ----
    def predict_phase(self, h: int):
        """h 프레임 뒤 phase 분포 p(z_{t+h}|y_{1:t}) = b_t A^h. (argmax, conf, dist) 반환."""
        assert self.belief is not None, "먼저 step() 을 호출해야 함"
        pi = self.belief.copy()
        for _ in range(h):
            pi = self.A.T @ pi
        k = int(pi.argmax())
        return k, float(pi[k]), pi

    def predict_horizon(self, H: int):
        """h=0..H 의 예측 phase 분포 (H+1,K) 와 argmax 경로 (H+1,) 를 한 번에."""
        assert self.belief is not None
        dists = np.empty((H + 1, self.K)); pi = self.belief.copy()
        dists[0] = pi
        for h in range(1, H + 1):
            pi = self.A.T @ pi; dists[h] = pi
        return dists, dists.argmax(axis=1).astype(int)

    def time_to_switch(self):
        """현재 예측 phase 의 self-transition p 로 근사한 기대 잔여 지속(프레임) = 1/(1-p).
        (memoryless geometric dwell 가정 — 대략적 anticipation 신호)."""
        assert self.belief is not None
        k = int(self.belief.argmax()); p = float(self.A[k, k])
        return k, 1.0 / max(1.0 - p, 1e-6)

    def predict_latent(self, h: int):
        """h 프레임 뒤 latent(=PCA emission) 평균 예측 ŷ_{t+h}. phase 혼합을 통한 AR 평균
        전파(정확 예측분포는 지수적 혼합 → 평균 근사). 현재 관측 prev_y 에서 롤아웃."""
        assert self.belief is not None and self.prev_y is not None
        y = self.prev_y.copy(); pi = self.belief.copy()
        for _ in range(h):
            pi = self.A.T @ pi                          # p(z_{t+j}|y_{1:t})
            means = np.einsum("kij,j->ki", self.W, y) + self.b   # (K,d): W_k y + b_k
            y = pi @ means                              # 혼합 평균
        return y

    def predict_latent_horizon(self, H: int):
        """h=0..H 의 예측 latent 궤적 (H+1,d). ŷ_0=현재 관측."""
        assert self.belief is not None and self.prev_y is not None
        out = np.empty((H + 1, self.d)); y = self.prev_y.copy(); pi = self.belief.copy()
        out[0] = y
        for h in range(1, H + 1):
            pi = self.A.T @ pi
            means = np.einsum("kij,j->ki", self.W, y) + self.b
            y = pi @ means; out[h] = y
        return out


def verify_online(det: "OnlinePhaseDetector", hmm, params, seq: np.ndarray) -> float:
    """증분 detector 의 프레임별 belief 가 dynamax filter() 와 일치하는지 max|Δ| 반환."""
    det.reset()
    beliefs = []
    for y in seq.astype(np.float64):
        det.step(y); beliefs.append(det.belief.copy())
    mine = np.stack(beliefs)
    s = jnp.asarray(seq.astype(np.float32))
    fp = np.asarray(hmm.filter(params, s, hmm.compute_inputs(s)).filtered_probs)
    return float(np.abs(mine - fp).max())


def online_vs_offline_diag(hmm, params, seqs: list[np.ndarray], n: int, seed: int,
                           decode_fn=decode_online) -> dict:
    """causal detector(decode_fn) vs offline(Viterbi) 비교: 프레임 일치도 + 검출 지연.

    검출 지연 = offline 이 a→b 로 바뀌는 각 경계 t0 에 대해, detector 가 같은 b 로
    바뀌는 첫 시점(그 offline b-구간 내)까지의 프레임 수. 미래를 못/덜 보므로 생기는
    OAD 고유의 지연을 정량화한다. decode_fn=decode_online(filter) 또는 fixed-lag."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    agree_num = agree_den = 0
    on_switches = off_switches = 0
    latencies = []
    for i in ids:
        seq = seqs[i]
        on, _ = decode_fn(hmm, params, seq)
        off, _ = decode_offline(hmm, params, seq)
        agree_num += int((on == off).sum()); agree_den += len(seq)
        on_switches += int((np.diff(on) != 0).sum())
        off_switches += int((np.diff(off) != 0).sum())
        bnd = np.nonzero(np.diff(off) != 0)[0] + 1  # offline 구간 시작점
        seg_end = np.append(bnd[1:], len(off))
        for t0, te in zip(bnd, seg_end):
            b = off[t0]
            hit = np.nonzero(on[t0:te] == b)[0]
            if len(hit):
                latencies.append(int(hit[0]))
    lat = np.array(latencies) if latencies else np.array([0])
    return {
        "n_episodes": int(len(ids)),
        "frame_agreement": agree_num / max(agree_den, 1),
        "mean_latency_frames": float(lat.mean()),
        "median_latency_frames": float(np.median(lat)),
        "p90_latency_frames": float(np.percentile(lat, 90)),
        "n_boundaries": int(len(latencies)),
        # flicker = online 이 offline 대비 얼마나 더 자주 스위치하나(100프레임당). OAD false detection 대리지표.
        "online_switch_per100": 100.0 * on_switches / max(agree_den, 1),
        "offline_switch_per100": 100.0 * off_switches / max(agree_den, 1),
    }


def forecast_diag(params, seqs: list[np.ndarray], H: int, n: int, seed: int) -> dict:
    """미래 상태 예측 진단(전부 causal — OnlinePhaseDetector 의 belief 에서만 롤아웃).
    각 시점 t 에서 h=1..H 앞 phase/latent 예측 → 실제 t+h 값과 비교.
      phase_acc[h] = P(argmax pred == 실제 t+h 실현 phase)  (실현=online filter 결과)
      latent_mse[h] = 예측 latent vs 실제 latent 의 per-dim MSE
      persistence_mse[h] = "현재값 유지" baseline MSE (모델이 이걸 이겨야 의미)"""
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    det = OnlinePhaseDetector(params)
    acc = np.zeros(H + 1); mse = np.zeros(H + 1); pmse = np.zeros(H + 1); cnt = np.zeros(H + 1)
    for i in ids:
        seq = seqs[i].astype(np.float64); T = len(seq)
        if T <= 1:
            continue
        det.reset()
        realized = np.empty(T, int)
        ph = np.empty((T, H + 1), int)
        yl = np.empty((T, H + 1, det.d))
        for t in range(T):
            realized[t], _ = det.step(seq[t])
            _, ph[t] = det.predict_horizon(H)
            yl[t] = det.predict_latent_horizon(H)
        for h in range(1, H + 1):
            m = T - h
            if m <= 0:
                continue
            acc[h] += int((ph[:m, h] == realized[h:h + m]).sum())
            mse[h] += float(((yl[:m, h] - seq[h:h + m]) ** 2).sum())
            pmse[h] += float(((seq[:m] - seq[h:h + m]) ** 2).sum())
            cnt[h] += m
    by_h = []
    for h in range(1, H + 1):
        c = max(cnt[h], 1.0)
        by_h.append({"h": h, "phase_acc": float(acc[h] / c),
                     "latent_mse": float(mse[h] / (c * det.d)),
                     "persistence_mse": float(pmse[h] / (c * det.d))})
    return {"H": H, "n_episodes": int(len(ids)), "by_h": by_h}


def bic(total_ll: float, n_frames: int, K: int, d: int) -> float:
    n_params = (K * (K - 1)) + (K - 1) + K * (d * d + d + d * (d + 1) // 2)
    return -2.0 * total_ll + n_params * np.log(max(n_frames, 1))


# ---------------------------------------------------------------------------
# 분석 플롯
# ---------------------------------------------------------------------------
def plot_forecast_curves(fc: dict, out_path: Path, fps: int = 20):
    """미래 예측: phase 정확도 곡선 + latent MSE 곡선(persistence baseline 병기), h/초 이중축."""
    hs = [r["h"] for r in fc["by_h"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(hs, [r["phase_acc"] for r in fc["by_h"]], "o-", color="tab:blue")
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("phase forecast accuracy")
    axes[0].set_title("Future phase accuracy vs horizon")
    axes[1].plot(hs, [r["latent_mse"] for r in fc["by_h"]], "o-", color="tab:blue",
                 label="AR-HMM forecast")
    axes[1].plot(hs, [r["persistence_mse"] for r in fc["by_h"]], "s--", color="tab:gray",
                 label="persistence (no-change)", alpha=0.8)
    axes[1].set_ylabel("latent MSE (per dim)"); axes[1].legend()
    axes[1].set_title("Future latent MSE vs horizon")
    for a in axes:
        a.set_xlabel("horizon h (frames)"); a.grid(alpha=0.3)
        a.secondary_xaxis("top", functions=(lambda x: x / fps, lambda x: x * fps)).set_xlabel("seconds")
    fig.suptitle(f"Causal future-state forecast (H={fc['H']}, n={fc['n_episodes']} eps)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_forecast_video(frames, realized, pred_paths, H, K, color_map, out_path,
                          fps, title, subtitle):
    """미래 예측 시각화 영상. 전부 causal(OnlinePhaseDetector belief 롤아웃).
      frames: (T,h,w,3) 로봇 RGB
      realized: (T,) 현재(online) phase, pred_paths: (T,H+1) 각 t 의 h=0..H 예측 phase
    구성: 로봇영상 | 텍스트(현재/+H 예측/전환예상) | 앞으로 H프레임 예측 리본 |
          전-에피소드 리본(위=realized, 아래=H프레임 전에 한 예측 → 나중에 맞았나 대조)."""
    T = len(realized)
    colarr = np.array([to_rgba(color_map[f"P{k:02d}"]) for k in range(K)])  # (K,4)
    gray = np.array([0.85, 0.85, 0.85, 1.0])
    # 전-에피소드 리본: row0=realized, row1=verified(τ 에서 H 전 예측=pred_paths[τ-H][H]).
    rib = np.empty((2, T, 4))
    rib[0] = colarr[realized]
    for tau in range(T):
        rib[1, tau] = colarr[pred_paths[tau - H][H]] if tau >= H else gray

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], height_ratios=[3.0, 1.1, 1.2],
                          hspace=0.55, wspace=0.15)
    ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis("off")
    ax_ahead = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1]); ax_text.axis("off")
    ax_rib = fig.add_subplot(gs[2, 1])

    vid_im = ax_img.imshow(frames[0])
    ax_img.set_title(title, fontsize=11)

    # 앞으로 H프레임 예측 리본 (0=now).
    ahead_im = ax_ahead.imshow(colarr[pred_paths[0]][None], aspect="auto",
                               extent=[0, H + 1, 0, 1], interpolation="nearest")
    ax_ahead.set_yticks([]); ax_ahead.set_xticks(np.arange(0, H + 1, max(1, H // 5)))
    ax_ahead.set_xlabel("frames ahead  (0 = now)", fontsize=9)
    ax_ahead.set_title(f"Predicted phase for next {H} frames ({H/fps:.2f}s)", fontsize=10)
    ahead_now = ax_ahead.axvline(0.5, color="k", lw=1.5)

    # 전-에피소드 리본.
    ax_rib.imshow(rib, aspect="auto", extent=[0, T, 0, 2], interpolation="nearest")
    ax_rib.set_yticks([0.5, 1.5]); ax_rib.set_yticklabels(["fcst(−H)", "realized"], fontsize=8)
    ax_rib.set_xlabel("episode frame", fontsize=9)
    rib_now = ax_rib.axvline(0, color="k", lw=1.5)
    rib_ahead = ax_rib.axvline(min(H, T), color="k", lw=1.0, ls=":")  # now+H

    txt = ax_text.text(0.02, 0.95, "", va="top", ha="left", fontsize=13, family="monospace",
                       transform=ax_text.transAxes)
    present = sorted(set(realized.tolist()) | set(pred_paths.reshape(-1).tolist()))
    ax_text.legend(handles=[Patch(color=colarr[k], label=f"P{k}") for k in present],
                   loc="lower left", bbox_to_anchor=(0.0, -0.15), ncol=min(6, len(present)),
                   fontsize=7, frameon=False)
    fig.suptitle(subtitle, fontsize=9, y=0.995)

    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2500,
                          extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"])
    print(f"Rendering forecast {T} frames @ {fps} fps → {out_path}")
    with writer.saving(fig, str(out_path), dpi=110):
        for t in tqdm(range(T)):
            vid_im.set_data(frames[t])
            ahead_im.set_data(colarr[pred_paths[t]][None])
            rib_now.set_xdata([t, t]); rib_ahead.set_xdata([min(t + H, T), min(t + H, T)])
            now_k = int(realized[t]); fut_k = int(pred_paths[t][H])
            same = "✓" if now_k == fut_k else "→"
            txt.set_text(f"now      : P{now_k}\n"
                         f"+{H}f({H/fps:.2f}s): P{fut_k}  {same}\n"
                         f"pred path: {'-'.join('P'+str(int(p)) for p in pred_paths[t])}")
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_transition_matrix(P: np.ndarray, out_path: Path):
    K = P.shape[0]
    fig, ax = plt.subplots(figsize=(0.55 * K + 3, 0.55 * K + 2.5))
    im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"P{k}" for k in range(K)], fontsize=8)
    ax.set_yticklabels([f"P{k}" for k in range(K)], fontsize=8)
    ax.set_xlabel("to phase"); ax.set_ylabel("from phase")
    ax.set_title("AR-HMM phase transition matrix")
    thresh = 0.5
    for i in range(K):
        for j in range(K):
            if P[i, j] >= 0.02:
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                        color="white" if P[i, j] < thresh else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(to | from)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phase_stats(occ: np.ndarray, dwell: np.ndarray, color_map: dict, out_path: Path):
    K = len(occ)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = [color_map[f"P{k:02d}"] for k in range(K)]
    axes[0].bar(range(K), occ, color=colors)
    axes[0].set_xticks(range(K)); axes[0].set_xticklabels([f"P{k}" for k in range(K)], fontsize=8)
    axes[0].set_ylabel("frame fraction"); axes[0].set_title("Phase occupancy")
    axes[1].bar(range(K), dwell, color=colors)
    axes[1].set_xticks(range(K)); axes[1].set_xticklabels([f"P{k}" for k in range(K)], fontsize=8)
    axes[1].set_ylabel("mean dwell (frames)"); axes[1].set_title("Mean phase duration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def phase_over_time(seqs_states: list[np.ndarray], K: int, n_bins: int) -> np.ndarray:
    """에피소드들의 Viterbi 라벨을 정규화-시간 n_bins 로 리샘플해 (n_bins,K) 점유율."""
    acc = np.zeros((n_bins, K), dtype=np.float64)
    for z in seqs_states:
        if len(z) == 0:
            continue
        pos = (np.linspace(0, len(z) - 1, n_bins)).round().astype(int)
        for b, p in enumerate(pos):
            acc[b, z[p]] += 1.0
    row = acc.sum(axis=1, keepdims=True)
    return acc / np.maximum(row, 1.0)


def plot_phase_over_time_by_task(task_to_states: dict, K: int, n_bins: int,
                                 color_map: dict, out_path: Path):
    tasks = list(task_to_states.keys())
    ncol = min(3, len(tasks))
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    x = np.linspace(0, 1, n_bins)
    colors = [color_map[f"P{k:02d}"] for k in range(K)]
    for i, task in enumerate(tasks):
        ax = axes[i // ncol][i % ncol]
        frac = phase_over_time(task_to_states[task], K, n_bins)  # (n_bins,K)
        ax.stackplot(x, frac.T, colors=colors, labels=[f"P{k}" for k in range(K)])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(task.replace("_", " ")[:38], fontsize=9)
        ax.set_xlabel("normalized episode time", fontsize=8)
        if i % ncol == 0:
            ax.set_ylabel("phase fraction", fontsize=8)
    for j in range(len(tasks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in range(K)]
    fig.legend(handles, [f"P{k}" for k in range(K)], loc="center left",
               bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=8, frameon=False,
               title="phase")
    fig.suptitle("Action-phase progression over normalized time, per task", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="exp/libero_goal_multimodal_lang_state_decoder/sd000_20260610_052850")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--raw_root", default="")
    # AR-HMM
    ap.add_argument("--num_states", type=int, default=10, help="phase 개수 K")
    ap.add_argument("--pca_dim", type=int, default=10, help="emission PCA 차원")
    ap.add_argument("--stickiness", type=float, default=5.0,
                    help="전이행렬 self-transition prior (클수록 sticky phase)")
    ap.add_argument("--infer", default="online", choices=["online", "offline", "fixedlag"],
                    help="phase 추론: online=causal forward filter(OAD, 기본), "
                         "fixedlag=causal fixed-lag smoothing(--lag 프레임 지연 허용), "
                         "offline=Viterbi+smoother(미래 사용, 특성화 참조용)")
    ap.add_argument("--lag", type=int, default=3,
                    help="fixedlag 모드의 지연 프레임 수 L (p(z_τ|y_{0:τ+L}))")
    ap.add_argument("--forecast_horizon", type=int, default=0,
                    help="미래 상태 예측 진단 지평선 H(프레임, 0=끔). causal — belief 롤아웃. "
                         "예: 10=0.5초@20fps. phase 정확도 + latent MSE 곡선 산출")
    ap.add_argument("--forecast_video", action="store_true",
                    help="예측 시각화 영상 렌더(--tasks, 기본 3개). 현재/+H 예측 phase + "
                         "look-ahead 리본 + 검증 리본. H 는 --forecast_horizon(0이면 10).")
    ap.add_argument("--em_iters", type=int, default=60)
    ap.add_argument("--fit_len", type=int, default=0,
                    help="적합용 리샘플 길이 (0=에피소드 길이 중앙값 자동)")
    ap.add_argument("--jitter", type=float, default=0.01,
                    help="적합 데이터 상대 jitter std (covariance floor; 0=끔)")
    ap.add_argument("--use_cache", action="store_true",
                    help="인코딩된 latent Z 캐시(Zcache_ep*.npy) 재사용")
    ap.add_argument("--max_fit_episodes", type=int, default=400,
                    help="적합에 쓸 에피소드 수 (task 균등 샘플)")
    ap.add_argument("--k_sweep", default="", help="쉼표구분 K 목록으로 스윕(예 4,6,8,10,12,15). "
                    "지정 시 각 K 를 적합해 BIC + OAD online 지표 계산 후 종료")
    ap.add_argument("--k_select", default="latency", choices=["latency", "bic", "agreement"],
                    help="k_sweep 에서 best K 선택 기준(기본 latency=online 검출 지연 최소)")
    ap.add_argument("--oad_diag_episodes", type=int, default=120,
                    help="OAD online vs offline 진단에 쓸 에피소드 수")
    ap.add_argument("--n_time_bins", type=int, default=24)
    ap.add_argument("--encode_chunk", type=int, default=8192)
    # 영상
    ap.add_argument("--tasks", default="turn_on_the_stove,push_the_plate_to_the_front_of_the_stove,put_the_wine_bottle_on_the_rack")
    ap.add_argument("--all_tasks", action="store_true",
                    help="LIBERO-goal 10개 task 전부 렌더(--tasks 무시)")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--no_analysis", action="store_true")
    ap.add_argument("--libero_dir", default="~/.libero/data")
    ap.add_argument("--lang_json", default="~/.libero/data/libero_goal_lang_embeddings.json")
    ap.add_argument("--camera", default="agentview", choices=["agentview", "eye_in_hand"])
    ap.add_argument("--no_flip", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--method", default="both", choices=["umap", "tsne", "both"])
    ap.add_argument("--max_backdrop_plot", type=int, default=2500)
    ap.add_argument("--min_seg_frac", type=float, default=0.03)
    ap.add_argument("--max_T", type=int, default=0)
    ap.add_argument("--out_subdir", default="latent_libero_arhmm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ckpts = sorted(run_dir.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    epoch = args.epoch if args.epoch is not None else int(ckpts[-1].stem.split("_")[1])
    with (run_dir / "flags.json").open() as f:
        flags = json.load(f)
    env_name = flags["env_name"]
    cfg = flags["agent"]
    img_feat_dim = int(cfg.get("image_feat_dim", 512))
    assert "multimodal" in env_name, f"{env_name} is not a multimodal run"

    out_dir = run_dir / "plots" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building agent + libero pretrain dataset ...")
    agent, pre_train, raw_obs = build_agent_and_pretrain_dataset(flags)
    agent = restore_agent(agent, str(run_dir), epoch)
    target_dim = raw_obs.shape[1]
    is_lang = target_dim > img_feat_dim + 15
    N = len(raw_obs)

    # ---- 전체 인코딩 + 에피소드 분해 ----
    terminals = np.asarray(pre_train["terminals"])
    spans = episode_spans(terminals)
    stats_path = _resolve_stats_path(env_name, args.libero_dir)
    ep_starts = np.array([s for s, _ in spans])
    ep_tasks, present_tasks = derive_task_labels(ep_starts, stats_path, N)
    cache_path = out_dir / f"Zcache_ep{epoch}.npy"
    if args.use_cache and cache_path.exists():
        Z_all = np.load(cache_path)
        print(f"{len(spans)} episodes, {len(present_tasks)} tasks; loaded Z cache {Z_all.shape}")
    else:
        print(f"{len(spans)} episodes, {len(present_tasks)} tasks; encoding {N} rows ...")
        Z_all = encode_all(agent, pre_train, N, args.encode_chunk, args.seed)  # (N,512)
        np.save(cache_path, Z_all)

    pca = PCA(n_components=args.pca_dim, random_state=args.seed).fit(Z_all)
    Zp_all = pca.transform(Z_all).astype(np.float32)
    print(f"PCA-{args.pca_dim} explains {pca.explained_variance_ratio_.sum():.3f} of latent variance")

    ep_seqs_p = [Zp_all[s:e + 1] for s, e in spans]      # emission space (10-d)
    ep_seqs_full = [Z_all[s:e + 1] for s, e in spans]    # 512-d (viz)

    # 적합용 에피소드: task 균등 샘플.
    rng = np.random.default_rng(args.seed)
    by_task: dict[str, list[int]] = {t: [] for t in present_tasks}
    for i, t in enumerate(ep_tasks):
        by_task[t].append(i)
    per_task_q = max(1, args.max_fit_episodes // max(1, len(present_tasks)))
    fit_ids = []
    for t, ids in by_task.items():
        take = min(per_task_q, len(ids))
        fit_ids.extend(rng.choice(ids, size=take, replace=False).tolist())
    fit_ids = sorted(fit_ids)
    fit_seqs_raw = [ep_seqs_p[i] for i in fit_ids]
    fit_seqs, fit_L = prepare_fit(fit_seqs_raw, args.fit_len, args.jitter, args.seed)
    print(f"Fitting on {len(fit_seqs)} episodes (~{per_task_q}/task), "
          f"resampled to L={fit_L}, jitter={args.jitter}")

    # ---- K-스윕: BIC + OAD online 지표. --k_select 로 선택 기준 지정(기본 latency). ----
    if args.k_sweep:
        Ks = [int(x) for x in args.k_sweep.split(",") if x.strip()]
        rows = []
        n_frames = int(sum(len(s) for s in fit_seqs))
        n_diag = min(args.oad_diag_episodes, len(ep_seqs_p))
        for K in Ks:
            hmm = build_hmm(K, args.pca_dim, args.stickiness)
            params, lls = fit_hmm(hmm, fit_seqs, args.em_iters, args.seed)
            total_ll = float(sum(
                float(hmm.marginal_log_prob(
                    params, jnp.asarray(s.astype(np.float32)),
                    hmm.compute_inputs(jnp.asarray(s.astype(np.float32)))))
                for s in fit_seqs))
            b = bic(total_ll, n_frames, K, args.pca_dim)
            oad = online_vs_offline_diag(hmm, params, ep_seqs_p, n=n_diag, seed=args.seed)
            rows.append({"K": K, "train_ll": total_ll, "bic": b, **oad})
            print(f"  K={K:3d}  BIC={b:11.0f}  lat(mean/p90)={oad['mean_latency_frames']:.2f}/"
                  f"{oad['p90_latency_frames']:.0f}  agree={oad['frame_agreement']:.3f}  "
                  f"flicker(on/off per100)={oad['online_switch_per100']:.2f}/"
                  f"{oad['offline_switch_per100']:.2f}")
        # 선택 기준. latency=평균 검출 지연 최소, 동률이면 flicker↓·agreement↑ 로 tie-break.
        keyf = {
            "latency": lambda r: (round(r["mean_latency_frames"], 3),
                                  r["online_switch_per100"], -r["frame_agreement"]),
            "bic": lambda r: r["bic"],
            "agreement": lambda r: -r["frame_agreement"],
        }[args.k_select]
        best = min(rows, key=keyf)
        with (out_dir / f"k_sweep_ep{epoch}_{args.k_select}.json").open("w") as f:
            json.dump({"epoch": epoch, "select_by": args.k_select, "rows": rows,
                       "best_K": best["K"]}, f, indent=2)
        # 2패널: online 검출 지연 + 일치도/flicker.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        Kx = [r["K"] for r in rows]
        axes[0].plot(Kx, [r["mean_latency_frames"] for r in rows], "o-", label="mean")
        axes[0].plot(Kx, [r["p90_latency_frames"] for r in rows], "s--", label="p90", alpha=0.7)
        axes[0].axvline(best["K"], color="red", ls=":", label=f"best K={best['K']}")
        axes[0].set_xlabel("num phases K"); axes[0].set_ylabel("online detection latency (frames)")
        axes[0].set_title("OAD latency vs K (lower=better)"); axes[0].legend(); axes[0].grid(alpha=0.3)
        ax2 = axes[1]; ax2b = ax2.twinx()
        l1 = ax2.plot(Kx, [r["frame_agreement"] for r in rows], "o-", color="tab:green",
                      label="frame agreement")
        l2 = ax2b.plot(Kx, [r["online_switch_per100"] for r in rows], "s--", color="tab:red",
                       label="online flicker /100f")
        l3 = ax2b.plot(Kx, [r["offline_switch_per100"] for r in rows], "^:", color="tab:orange",
                       label="offline switch /100f", alpha=0.7)
        ax2.axvline(best["K"], color="red", ls=":")
        ax2.set_xlabel("num phases K"); ax2.set_ylabel("online-offline frame agreement")
        ax2b.set_ylabel("switches per 100 frames"); ax2.set_title("agreement & flicker vs K")
        ax2.legend(handles=l1 + l2 + l3, loc="center right", fontsize=8); ax2.grid(alpha=0.3)
        fig.suptitle(f"AR-HMM phase-count selection (by {args.k_select})")
        fig.tight_layout()
        fig.savefig(out_dir / f"k_sweep_ep{epoch}_{args.k_select}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nBest K by {args.k_select} = {best['K']}  "
              f"(lat={best['mean_latency_frames']:.2f}f, agree={best['frame_agreement']:.3f}, "
              f"flicker={best['online_switch_per100']:.2f}/100f)  →  {out_dir}/k_sweep_ep{epoch}_{args.k_select}.png")
        return

    # ---- 메인 적합 ----
    K = args.num_states
    # 산출물 파일명 태그: epoch+K+infer(+lag) 를 넣어 서로 다른 config 가 절대 덮어쓰지 않게 한다.
    tag = f"ep{epoch}_K{K}_{args.infer}" + (f"L{args.lag}" if args.infer == "fixedlag" else "")
    hmm = build_hmm(K, args.pca_dim, args.stickiness)
    params, lls = fit_hmm(hmm, fit_seqs, args.em_iters, args.seed)
    print(f"EM done: LL {lls[0]:.1f} → {lls[-1]:.1f} over {len(lls)} iters")

    # 전 에피소드 디코드. OAD 는 causal 이어야 하므로 기본은 online(filter).
    # fixedlag = causal fixed-lag smoothing(지연 lag 프레임 허용, filter 와 offline 사이).
    if args.infer == "online":
        _decode = decode_online
    elif args.infer == "fixedlag":
        _decode = lambda h, p, s: decode_fixedlag(h, p, s, args.lag)
    else:
        _decode = decode_offline
    causal = args.infer in ("online", "fixedlag")
    all_states, all_conf = [], []
    task_to_states: dict[str, list[np.ndarray]] = {t: [] for t in present_tasks}
    for i, (seq, t) in enumerate(zip(ep_seqs_p, ep_tasks)):
        z, conf = _decode(hmm, params, seq)
        all_states.append(z); all_conf.append(conf)
        task_to_states[t].append(z)
    flat_states = np.concatenate(all_states)
    P = np.asarray(params.transitions.transition_matrix)

    # OAD 진단: 선택한 causal detector vs offline(Viterbi) 일치도 + 검출 지연.
    oad = online_vs_offline_diag(hmm, params, ep_seqs_p, n=min(120, len(ep_seqs_p)),
                                 seed=args.seed, decode_fn=_decode) if causal else None
    if oad:
        print(f"OAD {args.infer} vs offline: per-frame agree={oad['frame_agreement']:.3f}, "
              f"mean detection latency={oad['mean_latency_frames']:.1f} frames "
              f"(median={oad['median_latency_frames']:.0f}), "
              f"flicker={oad['online_switch_per100']:.2f}/100f")

    # 배포용 스트리밍 detector 가 filter() 와 동일한지 self-verify (online 모드만).
    if args.infer == "online":
        vok = verify_online(OnlinePhaseDetector(params), hmm, params, ep_seqs_p[0])
        print(f"OnlinePhaseDetector matches dynamax filter: max|Δp|={vok:.2e}")

    # 미래 상태 예측(causal) 진단: h=1..H 앞 phase 정확도 + latent MSE.
    fc = None
    if args.forecast_horizon > 0:
        fc = forecast_diag(params, ep_seqs_p, args.forecast_horizon,
                           n=min(150, len(ep_seqs_p)), seed=args.seed)
        h1, hH = fc["by_h"][0], fc["by_h"][-1]
        print(f"Forecast H={fc['H']}: phase acc h1={h1['phase_acc']:.3f} → "
              f"h{fc['H']}={hH['phase_acc']:.3f}; latent MSE h{fc['H']}={hH['latent_mse']:.4f} "
              f"vs persistence {hH['persistence_mse']:.4f}")

    present_phases = [f"P{k:02d}" for k in range(K) if (flat_states == k).any()]
    # 색맵은 전 K개에 대해 만든다(비어있는 phase도 plot 루프에서 참조하므로).
    color_map = distinct_color_map([f"P{k:02d}" for k in range(K)])

    # ---- 분석 플롯 저장 ----
    if not args.no_analysis:
        occ = np.array([(flat_states == k).mean() for k in range(K)])
        # 평균 dwell(연속 구간 길이).
        dwell = np.zeros(K)
        cnt = np.zeros(K)
        for z in all_states:
            if len(z) == 0:
                continue
            change = np.nonzero(np.diff(z) != 0)[0]
            segs = np.split(z, change + 1)
            for seg in segs:
                dwell[seg[0]] += len(seg); cnt[seg[0]] += 1
        dwell = dwell / np.maximum(cnt, 1)
        plot_transition_matrix(P, out_dir / f"transition_matrix_{tag}.png")
        plot_phase_stats(occ, dwell, color_map, out_dir / f"phase_stats_{tag}.png")
        # bowl-family를 위쪽에 모아서 비교하기 쉽게 정렬.
        fam_order = sorted(present_tasks,
                           key=lambda t: (TASK_FAMILY_MAP.get(t, "z"), t))
        ordered = {t: task_to_states[t] for t in fam_order}
        plot_phase_over_time_by_task(ordered, K, args.n_time_bins, color_map,
                                     out_dir / f"phase_over_time_by_task_{tag}.png")
        if fc is not None:
            plot_forecast_curves(fc, out_dir / f"forecast_curves_{tag}_H{fc['H']}.png",
                                 fps=args.fps)
        summary = {
            "epoch": epoch, "env_name": env_name, "num_states": K,
            "pca_dim": args.pca_dim, "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
            "stickiness": args.stickiness, "n_fit_episodes": len(fit_seqs),
            "n_total_episodes": len(spans), "final_em_ll": float(lls[-1]),
            "phase_occupancy": {f"P{k}": float(occ[k]) for k in range(K)},
            "phase_mean_dwell": {f"P{k}": float(dwell[k]) for k in range(K)},
            "mean_decode_conf": float(np.concatenate(all_conf).mean()),
            "infer_mode": args.infer,
            "oad_online_vs_offline": oad,
            "forecast": fc,
            "transition_matrix": P.tolist(),
        }
        with (out_dir / f"summary_{tag}.json").open("w") as f:
            json.dump(summary, f, indent=2)
        np.savez(out_dir / f"arhmm_{tag}.npz",
                 transition_matrix=P, initial_probs=np.asarray(params.initial.probs),
                 emission_weights=np.asarray(params.emissions.weights),
                 emission_biases=np.asarray(params.emissions.biases),
                 emission_covs=np.asarray(params.emissions.covs),
                 pca_components=pca.components_, pca_mean=pca.mean_)
        print(f"Analysis written → {out_dir}/*_{tag}.*  (mean decode conf={summary['mean_decode_conf']:.3f})")

    # ---- 미래 예측 시각화 영상 (causal) ----
    if args.forecast_video:
        Hf = args.forecast_horizon if args.forecast_horizon > 0 else 10
        with open(stats_path) as f:
            stats = json.load(f)
        state_keys = stats.get("state_keys", list(DEFAULT_STATE_KEYS))
        image_size = int(stats.get("image_size", 128))
        ext = FrozenResNet34Extractor(device="cuda", batch_size=256)
        lang = json.load(open(os.path.expanduser(args.lang_json))) if is_lang else None
        suite_dir = resolve_suite_dir(args.suite, args.raw_root)
        vinfer = args.infer + (f"L{args.lag}" if args.infer == "fixedlag" else "")
        task_list = list(LIBERO_GOAL_10) if args.all_tasks else \
            [t.strip() for t in args.tasks.split(",") if t.strip()]
        det = OnlinePhaseDetector(params)
        print(f"Rendering {len(task_list)} forecast video(s), H={Hf} ({Hf/args.fps:.2f}s)")
        for task in task_list:
            print(f"\n=== forecast demo: {task} #{args.demo} ===")
            ep = load_demo_mm(suite_dir, task, args.demo, state_keys, args.camera,
                              image_size, flip=not args.no_flip)
            T = ep["length"]
            feat = ext.extract(ep["enc_imgs"][:T])
            demo_obs = np.concatenate([feat, ep["state"][:T]], axis=-1).astype(np.float32)
            if is_lang:
                emb = np.asarray(lang["embeddings"][task], dtype=np.float32)
                demo_obs = np.concatenate([demo_obs, np.broadcast_to(emb, (T, emb.shape[0]))], axis=-1)
            comp_obs_norm = pre_train.normalize_observations(observations=demo_obs).astype(np.float32)
            comp_act = np.clip(ep["actions"][:T], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
            comp_mean = np.asarray(agent.network.select("intention_encoder")(
                jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())
            comp_p = pca.transform(comp_mean).astype(np.float32)
            det.reset()
            realized = np.empty(T, int); pred_paths = np.empty((T, Hf + 1), int)
            for t in range(T):
                realized[t], _ = det.step(comp_p[t])
                _, pred_paths[t] = det.predict_horizon(Hf)
            acc = (float(np.mean([pred_paths[t][Hf] == realized[t + Hf] for t in range(T - Hf)]))
                   if T > Hf else float("nan"))
            out_path = out_dir / f"forecast_{task}_ep{args.demo}_arhmmK{K}_{vinfer}_H{Hf}.mp4"
            render_forecast_video(
                ep["frames"], realized, pred_paths, Hf, K, color_map, out_path, args.fps,
                title=f"{task}  (K={K}, {vinfer})",
                subtitle=(f"\"{ep['instruction']}\"  •  causal forecast H={Hf} "
                          f"({Hf/args.fps:.2f}s)  •  this-demo +H phase acc={acc:.2f}"))
            print(f"  wrote {out_path}  (+{Hf}f phase acc={acc:.2f})")
        return

    if args.no_video:
        return

    # ---- 영상 렌더 (Viterbi phase 색칠) ----
    # 백드롭 = 적합 에피소드 프레임(512-d) + 그들의 Viterbi phase 라벨. phase별 quota 서브샘플.
    bd_z, bd_lab = [], []
    for i in fit_ids:
        bd_z.append(ep_seqs_full[i]); bd_lab.append(all_states[i])
    bd_z = np.concatenate(bd_z, axis=0)
    bd_lab_int = np.concatenate(bd_lab)
    bd_lab = np.array([f"P{c:02d}" for c in bd_lab_int], dtype=object)
    quota = max(1, args.max_backdrop_plot // max(1, len(present_phases)))
    sel = []
    for c in present_phases:
        idx_c = np.flatnonzero(bd_lab == c)
        sel.extend(rng.choice(idx_c, size=min(quota, len(idx_c)), replace=False).tolist())
    sel = np.asarray(sel)
    a_mean_plot, a_labels_plot = bd_z[sel], bd_lab[sel]

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

    with open(stats_path) as f:
        stats = json.load(f)
    state_keys = stats.get("state_keys", list(DEFAULT_STATE_KEYS))
    image_size = int(stats.get("image_size", 128))
    ext = FrozenResNet34Extractor(device="cuda", batch_size=256)
    lang = json.load(open(os.path.expanduser(args.lang_json))) if is_lang else None
    suite_dir = resolve_suite_dir(args.suite, args.raw_root)

    task_list = list(LIBERO_GOAL_10) if args.all_tasks else \
        [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Rendering {len(task_list)} task(s) × {len(backdrop2d)} method(s)")
    for task in task_list:
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
            jnp.asarray(comp_obs_norm), jnp.asarray(comp_act)).mean())  # (T,512)

        # phase + conf. 기본은 online(causal filter) — OAD 가 실제로 보는 것.
        comp_p = pca.transform(comp_mean).astype(np.float32)
        vstates, conf = _decode(hmm, params, comp_p)
        voted = np.array([f"P{c:02d}" for c in vstates], dtype=object)
        print(f"  T={T}  phases={sorted(set(int(v) for v in vstates))}  "
              f"mean posterior conf={conf.mean():.3f}")

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
                        present_tasks=present_phases, voted=voted, conf=conf,
                        color_map=color_map)
            # 파일명에 infer(+lag) 포함 — 안 그러면 online/fixedlag 영상이 서로 덮어씀.
            vinfer = args.infer + (f"L{args.lag}" if args.infer == "fixedlag" else "")
            out_path = out_dir / f"demo_{task}_ep{args.demo}_arhmmK{K}_{vinfer}_{method}.mp4"
            infer_lbl = args.infer + (f"(L={args.lag})" if args.infer == "fixedlag" else "")
            sub = (f"\"{ep['instruction']}\"  •  T={T}  •  epoch={epoch}  •  "
                   f"mean_conf={conf.mean():.2f}  •  color=AR-HMM phase (K={K}, {infer_lbl})")
            render_video(data, ep["frames"], out_path, emb_atomic=emb_a, emb_comp=emb_c,
                         embedding_name=method.upper(), fps=args.fps,
                         title=f"{task}  demo#{args.demo}  ({method.upper()})  [AR-HMM K={K}]",
                         subtitle=sub, max_T=args.max_T, min_seg_frac=args.min_seg_frac,
                         conf=conf)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
