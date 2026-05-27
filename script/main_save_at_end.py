"""Drop-in replacement for main.py that ALWAYS saves a final checkpoint.

What it adds vs. main.py:
  * Patches every registered agent class's ``pretrain`` / ``finetune`` to
    record the returned new-agent reference into a module-level slot.
  * After ``main.main`` (the original training loop in ``main.py``) returns,
    saves the captured agent to ``<save_dir>/params_<final_step>.pkl``
    unless main's normal save_interval already produced that file.

Why a wrapper instead of editing main.py:
  Leaves ``main.py`` untouched. To use legacy behaviour run main.py as
  before; to guarantee an end-of-run checkpoint run this script.

Usage (CLI flags are identical to main.py):
    python script/main_save_at_end.py \\
        --env_name=cube-single-play-singletask-task1-v0 \\
        --pretraining_steps=1_000_000 --finetuning_steps=500_000 ...

For visual-* envs that also need run_memsafe.py's RAM patches, use the
combined launcher ``script/run_memsafe_save_at_end.py`` instead.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from absl import app, flags  # noqa: E402

# Import main module (defines its flags as a side effect, does NOT run main).
import main as _main_mod  # noqa: E402
from agents import agents as _AGENTS  # noqa: E402
from utils.flax_utils import save_agent  # noqa: E402

FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Capture the latest agent reference produced by pretrain / finetune.
#
# Agent classes are flax.struct.PyTreeNode subclasses whose pretrain /
# finetune methods are @jax.jit-wrapped and return ``(new_agent, info)``.
# Because they're frozen, in-place mutation isn't possible; the training
# loop in main.py rebinds a local ``agent`` each iteration. We mirror that
# by stashing the new agent in a module-level slot on every call, so when
# the training loop exits we can recover the final state.
# ---------------------------------------------------------------------------

_LATEST_AGENT = [None]


def _record_wrapper(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if isinstance(result, tuple) and result:
            _LATEST_AGENT[0] = result[0]
        return result

    wrapper.__wrapped__ = method
    wrapper.__name__ = getattr(method, "__name__", "wrapper")
    wrapper.__is_save_at_end_wrapped__ = True
    return wrapper


_patched = []
for _cls in _AGENTS.values():
    for _method_name in ("pretrain", "finetune"):
        _orig = _cls.__dict__.get(_method_name)
        if _orig is None or getattr(_orig, "__is_save_at_end_wrapped__", False):
            continue
        setattr(_cls, _method_name, _record_wrapper(_orig))
        _patched.append(f"{_cls.__name__}.{_method_name}")

print(
    f"[save_at_end] patched {len(_patched)} method(s) "
    f"to capture the latest agent reference: {_patched}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Wrap main() to save the final agent after the loop finishes.
# ---------------------------------------------------------------------------

_orig_main = _main_mod.main


def main(_):
    _orig_main(_)

    final_step = FLAGS.pretraining_steps + FLAGS.finetuning_steps
    save_dir = FLAGS.save_dir
    final_path = os.path.join(save_dir, f"params_{final_step}.pkl")

    if _LATEST_AGENT[0] is None:
        print("[save_at_end] WARNING: no agent reference captured — nothing to save.")
        return

    if os.path.exists(final_path):
        print(f"[save_at_end] final checkpoint already exists at {final_path}; skipping.")
        return

    save_agent(_LATEST_AGENT[0], save_dir, final_step)
    print(f"[save_at_end] saved final checkpoint to {final_path}", flush=True)


if __name__ == "__main__":
    app.run(main)
