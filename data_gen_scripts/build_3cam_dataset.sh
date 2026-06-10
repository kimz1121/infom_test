#!/usr/bin/env bash
# Build the 3-CAMERA multimodal-precompute inFOM datasets (frozen resnet34).
# DATA GENERATION ONLY — no training.
#
# Produces (under ~/.robocasa/data):
#   atomic_65_multimodal_precompute_3cam_state_pretrain.hdf5  (+_val, +.stats.json)
#   <composite_lowercase>_3cam.hdf5                           (flat, for validation viz)
#
# Observation layout (camera concat order = the inference/viz contract):
#   [ resnet34(agentview_left) 512 | agentview_right 512 | eye_in_hand 512 | proprio 16 ] = 1552-d
#   => image_feat_dim = 1536.  When TRAINING on this data, pass
#      --agent.image_feat_dim=1536  (the state-decoder slices proprio = obs[1536:]).
#
# Usage:
#   bash data_gen_scripts/build_3cam_dataset.sh                  # build all (skip existing)
#   FORCE=1 bash data_gen_scripts/build_3cam_dataset.sh         # rebuild even if present
#   BATCH=128 DEVICE=cuda bash data_gen_scripts/build_3cam_dataset.sh
#   PRETRAIN_ONLY=1 ...   /   COMPOSITE_ONLY=1 ...              # build just one stage
set -euo pipefail
cd "$(dirname "$0")/.."

CAMERAS="${CAMERAS:-robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH="${BATCH:-256}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-$HOME/.robocasa/data}"
COMPOSITES=(LoadDishwasher PlaceVeggiesInDrawer StackBowlsCabinet StartElectricKettle)

EXTRACT=(python data_gen_scripts/extract_mm_features.py
         --cameras "$CAMERAS" --image_size "$IMAGE_SIZE"
         --batch_size "$BATCH" --device "$DEVICE")

echo "cameras   : $CAMERAS"
echo "image_size: $IMAGE_SIZE | batch: $BATCH | device: $DEVICE"
echo "data dir  : $DATA_DIR"
echo

skip_if_exists() {  # $1 = output hdf5 path ; returns 0 (skip) if present and !FORCE
  if [[ -f "$1" && "${FORCE:-0}" != "1" ]]; then
    echo "   [skip] exists: $(basename "$1")  (FORCE=1 to rebuild)"
    return 0
  fi
  return 1
}

if [[ "${COMPOSITE_ONLY:-0}" != "1" ]]; then
  echo "==> [1] pretrain atomic (all 65 tasks), 3 cameras"
  if ! skip_if_exists "$DATA_DIR/atomic_65_multimodal_precompute_3cam_state_pretrain.hdf5"; then
    "${EXTRACT[@]}" --split pretrain --category atomic --tasks all \
      --name atomic_65_multimodal_precompute_3cam_state
  fi
  echo
fi

if [[ "${PRETRAIN_ONLY:-0}" != "1" ]]; then
  echo "==> [2] composite validation set (flat), 3 cameras"
  for T in "${COMPOSITES[@]}"; do
    low="$(echo "$T" | tr '[:upper:]' '[:lower:]')"
    if ! skip_if_exists "$DATA_DIR/${low}_3cam.hdf5"; then
      echo "   - $T -> ${low}_3cam.hdf5"
      "${EXTRACT[@]}" --split pretrain --category composite --tasks "$T" --flat \
        --name "${low}_3cam"
    fi
  done
  echo
fi

if [[ "${WITH_LANG:-1}" == "1" && "${PRETRAIN_ONLY:-0}" != "1" && "${COMPOSITE_ONLY:-0}" != "1" ]]; then
  echo "==> [3] language append (SBERT 384-d) -> lang datasets  obs=[1536|16|384]=1936-d"
  LANG_JSON="${LANG_JSON:-$DATA_DIR/task_lang_embeddings.json}"
  base="$DATA_DIR/atomic_65_multimodal_precompute_3cam_state_pretrain"
  langp="$DATA_DIR/atomic_65_multimodal_precompute_3cam_lang_state_pretrain"
  if ! skip_if_exists "${langp}.hdf5"; then
    python data_gen_scripts/append_language.py --obs_hdf5 "${base}.hdf5" \
      --stats "${base}.stats.json" --which train --lang "$LANG_JSON" --out "${langp}.hdf5"
    # append_language doesn't emit stats; the row/task structure is identical to
    # the state dataset, so reuse its stats (needed for task labels in analysis).
    cp "${base}.stats.json" "${langp}.stats.json"
  fi
  if [[ -f "${base}_val.hdf5" ]] && ! skip_if_exists "${langp}_val.hdf5"; then
    python data_gen_scripts/append_language.py --obs_hdf5 "${base}_val.hdf5" \
      --stats "${base}.stats.json" --which val --lang "$LANG_JSON" --out "${langp}_val.hdf5"
  fi
  for T in "${COMPOSITES[@]}"; do
    low="$(echo "$T" | tr '[:upper:]' '[:lower:]')"
    if [[ -f "$DATA_DIR/${low}_3cam.hdf5" ]] && ! skip_if_exists "$DATA_DIR/${low}_3cam_lang.hdf5"; then
      python data_gen_scripts/append_language.py --obs_hdf5 "$DATA_DIR/${low}_3cam.hdf5" \
        --which composite --composite_task "$T" --lang "$LANG_JSON" --out "$DATA_DIR/${low}_3cam_lang.hdf5"
    fi
  done
  echo
fi

echo "DONE."
echo "  state datasets: obs_dim=1552 (image_feat_dim=1536, +16 state)"
echo "  lang  datasets: obs_dim=1936 ([1536|16|384])  (set WITH_LANG=0 to skip)"
echo "Train:"
echo "  state: --env_name robocasa_atomic_65_multimodal_precompute_3cam_state       --agent.image_feat_dim=1536"
echo "  lang : --env_name robocasa_atomic_65_multimodal_precompute_3cam_lang_state  --agent.image_feat_dim=1536 --agent.state_dim=16"
