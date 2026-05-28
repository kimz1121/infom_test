#!/usr/bin/env bash
# Portable, idempotent driver that rebuilds the canonical Robocasa datasets on
# any machine: download raw tars from Box, then convert to inFOM flat HDF5.
#
# Usage:
#   bash data_gen_scripts/build_robocasa.sh [group ...]
#
# Groups (default: all):
#   atomic-state      atomic_seen_state_pretrain + _ft_<FT_TASK>
#   atomic-image      atomic_seen_image_pretrain + _ft_<FT_TASK>   (heavy: video decode)
#   composite-state   loaddishwasher / placeveggiesindrawer / stackbowlscabinet / startelectrickettle
#   all               every group above
#
# Environment overrides:
#   ROBOCASA_ROOT   dataset root (default: ~/.robocasa). raw -> $ROOT/raw, out -> $ROOT/data
#   FT_TASK         finetune task for the _ft_ datasets (default: PickPlaceCounterToCabinet)
#   FORCE           1 = rebuild HDF5 even if the output already exists
#   PY              python interpreter (default: python)
#
# The conversion is deterministic (fixed val-split seed), so the resulting HDF5
# files are identical across environments given the same raw tars.
set -euo pipefail

ROOT="${ROBOCASA_ROOT:-$HOME/.robocasa}"
RAW="$ROOT/raw"
DATA="$ROOT/data"
FT_TASK="${FT_TASK:-PickPlaceCounterToCabinet}"
PY="${PY:-python}"
FORCE="${FORCE:-0}"

# Resolve repo's data_gen_scripts dir from this script's location so the driver
# works regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD="$SCRIPT_DIR/download_robocasa.py"
GENERATE="$SCRIPT_DIR/generate_robocasa_dataset.py"

COMPOSITE_TASKS=(LoadDishwasher PlaceVeggiesInDrawer StackBowlsCabinet StartElectricKettle)

GROUPS=("$@")
if [ "${#GROUPS[@]}" -eq 0 ]; then
  GROUPS=(all)
fi
want() {
  local g
  for g in "${GROUPS[@]}"; do
    [ "$g" = "all" ] && return 0
    [ "$g" = "$1" ] && return 0
  done
  return 1
}

# generate <name> <extra args...> — skips if output exists unless FORCE=1.
generate() {
  local name="$1"; shift
  if [ "$FORCE" != "1" ] && [ -f "$DATA/$name.hdf5" ]; then
    echo "[skip] $name.hdf5 already exists (FORCE=1 to rebuild)"
    return 0
  fi
  echo "[gen ] $name"
  "$PY" "$GENERATE" --raw_root "$RAW" --out_dir "$DATA" --name "$name" "$@"
}

echo "ROBOCASA_ROOT=$ROOT  FT_TASK=$FT_TASK  groups=${GROUPS[*]}"

# --- 1. Download raw tars from Box --------------------------------------------
if want atomic-state || want atomic-image; then
  echo "== download: target/atomic (18 seen tasks) =="
  "$PY" "$DOWNLOAD" --split target --category atomic --out_dir "$RAW"
fi
if want composite-state; then
  echo "== download: pretrain/composite =="
  "$PY" "$DOWNLOAD" --split pretrain --category composite \
    --tasks "${COMPOSITE_TASKS[@]}" --out_dir "$RAW"
fi

# --- 2. Convert raw -> flat HDF5 ----------------------------------------------
if want atomic-state; then
  generate atomic_seen_state_pretrain \
    --split target --category atomic --tasks atomic_seen_18 \
    --modality state --relabel_reward 0 --val_fraction 0.05
  generate "atomic_seen_state_ft_${FT_TASK}" \
    --split target --category atomic --tasks "$FT_TASK" \
    --modality state --relabel_reward 1 --val_fraction 0.1
fi

if want atomic-image; then
  generate atomic_seen_image_pretrain \
    --split target --category atomic --tasks atomic_seen_18 \
    --modality image --image_size 64 --relabel_reward 0 \
    --val_fraction 0.1 --max_episodes_per_task 50
  generate "atomic_seen_image_ft_${FT_TASK}" \
    --split target --category atomic --tasks "$FT_TASK" \
    --modality image --image_size 64 --relabel_reward 1 \
    --val_fraction 0.1 --max_episodes_per_task 50
fi

if want composite-state; then
  for task in "${COMPOSITE_TASKS[@]}"; do
    name="$(echo "$task" | tr '[:upper:]' '[:lower:]')_state"
    generate "$name" \
      --split pretrain --category composite --tasks "$task" \
      --modality state --relabel_reward 0 --val_fraction 0
  done
fi

echo "All requested datasets are in $DATA"
