"""Speed/VRAM benchmark for multimodal (image+state) inFOM embedding training.

Compares two strategies for incorporating images into the inFOM intention
encoder, using SYNTHETIC tensors (throughput depends on shapes/compute, not on
data values — so no dataset generation is needed):

  endtoend  : config.encoder='multimodal_resnet34'. resnet_34 runs fwd+bwd every
              training step on a batch of (B,H,W,3) images fused with state.
              Measures steps/sec + peak VRAM of agent.pretrain().
  precompute: config.encoder=None, observations are precomputed 528-d feature
              vectors (resnet feat 512 + state 16). This is the cheap path —
              identical to the existing state pipeline. Measures the downstream
              steps/sec (the resnet is NOT in the loop).
  extract   : one-time cost of the precompute path — torchvision resnet34
              (ImageNet-pretrained, frozen) forward-only throughput in frames/sec.

Run one (phase,res,batch) per process so jax and torch don't fight over GPU mem
and an OOM only kills that single config. A driver loops over configs.

Usage:
    python script/bench_multimodal.py --phase endtoend  --res 128 --batch 256
    python script/bench_multimodal.py --phase precompute --batch 256
    python script/bench_multimodal.py --phase extract    --res 256 --batch 256
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATE_DIM = 16
ACT_DIM = 12
RESNET_FEAT = 512


def bench_jax(phase, res, bs, steps, warmup):
    import numpy as np
    import jax
    from agents.infom import InFOMAgent, get_config

    multimodal = phase == "endtoend"
    cfg = get_config()
    cfg["batch_size"] = bs
    cfg["clip_flow_goals"] = False  # avoids obs_min/max shape coupling; ~0 compute
    cfg["encoder"] = "multimodal_resnet34" if multimodal else None

    def mk_obs(n):
        if multimodal:
            return {
                "image": np.random.randint(0, 256, (n, res, res, 3), dtype=np.uint8),
                "state": np.random.randn(n, STATE_DIM).astype(np.float32),
            }
        return np.random.randn(n, RESNET_FEAT + STATE_DIM).astype(np.float32)

    def mk_batch(n):
        return dict(
            observations=mk_obs(n),
            next_observations=mk_obs(n),
            actions=np.clip(np.random.randn(n, ACT_DIM), -1, 1).astype(np.float32),
            next_actions=np.clip(np.random.randn(n, ACT_DIM), -1, 1).astype(np.float32),
        )

    ex_obs = mk_obs(2)
    ex_act = np.clip(np.random.randn(2, ACT_DIM), -1, 1).astype(np.float32)
    agent = InFOMAgent.create(0, ex_obs, ex_act, cfg)

    batch = mk_batch(bs)
    info = None
    for _ in range(warmup):
        agent, info = agent.pretrain(batch)
    jax.block_until_ready(info)

    t0 = time.time()
    for _ in range(steps):
        agent, info = agent.pretrain(batch)
    jax.block_until_ready(info)
    dt = time.time() - t0

    try:
        peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        peak = float("nan")
    return dict(phase=phase, res=(res if multimodal else None), batch=bs,
                steps_per_sec=round(steps / dt, 2), peak_vram_gb=round(peak, 2))


def bench_extract(res, bs, steps, warmup):
    import torch
    import torchvision

    dev = "cuda"
    m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m = m.eval().to(dev)
    x = torch.randn(bs, 3, res, res, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            _ = m(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    fps = steps * bs / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    return dict(phase="extract", res=res, batch=bs,
                frames_per_sec=round(fps, 1), peak_vram_gb=round(peak, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["endtoend", "precompute", "extract"])
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    try:
        if args.phase == "extract":
            res = bench_extract(args.res, args.batch, args.steps, args.warmup)
        else:
            res = bench_jax(args.phase, args.res, args.batch, args.steps, args.warmup)
        print("RESULT " + json.dumps(res))
    except Exception as e:  # OOM or other — report, don't crash the driver
        import traceback
        traceback.print_exc()
        print("RESULT " + json.dumps(dict(phase=args.phase, res=args.res,
              batch=args.batch, error=type(e).__name__)))


if __name__ == "__main__":
    main()
