"""Combines run_memsafe.py's RAM patches with main_save_at_end.py's final save.

Use this for visual-* envs when you also want a guaranteed end-of-training
checkpoint. CLI flags are the same as ``script/run_memsafe.py``.

Recommended invocation on a 32 GB-RAM host (mirrors run_memsafe.py example):

    python script/run_memsafe_save_at_end.py \\
        --env_name=visual-cube-single-play-singletask-task1-v0 \\
        --obs_norm_type=none \\
        --pretraining_steps=250_000 --pretraining_size=250_000 \\
        --finetuning_steps=100_000  --finetuning_size=100_000 \\
        --eval_interval=10_000 --save_interval=750_000 \\
        --p_aug=0.5 --frame_stack=3 \\
        --agent=agents/infom.py \\
        --agent.encoder=impala_small \\
        --agent.clip_flow_goals=False

With ``--save_interval`` greater than total training steps, the legacy
behaviour leaves no params_*.pkl behind. This launcher always writes
``params_<pretraining_steps + finetuning_steps>.pkl`` once training ends.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importing these as modules triggers their patching side effects. Neither
# fires its own ``if __name__ == '__main__'`` block on import.
import script.run_memsafe  # noqa: F401,E402  (applies memsafe patches)
import script.main_save_at_end as _save_at_end_mod  # noqa: E402

from absl import app  # noqa: E402


if __name__ == "__main__":
    app.run(_save_at_end_mod.main)
