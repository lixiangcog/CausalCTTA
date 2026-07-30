#!/bin/bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${DATASET_ROOT:-${project_root}/example/Fundus}"
model_root="${MODEL_ROOT:-${project_root}/example/models}"
log_root="${LOG_ROOT:-${project_root}/logs}"

source_dataset="${SOURCE_DATASET:-RIM_ONE_r3}"
optimizer="${OPTIMIZER:-Adam}"
learning_rate="${LEARNING_RATE:-0.05}"
memory_size="${MEMORY_SIZE:-40}"
neighbor="${NEIGHBOR:-16}"
prompt_alpha="${PROMPT_ALPHA:-0.01}"
warm_n="${WARM_N:-5}"

cd "${project_root}/example"
python c2tta.py \
  --dataset_root "${dataset_root}" \
  --model_root "${model_root}" \
  --path_save_log "${log_root}" \
  --use_prompt \
  --use_AdaBN \
  --Source_Dataset "${source_dataset}" \
  --optimizer "${optimizer}" \
  --lr "${learning_rate}" \
  --memory_size "${memory_size}" \
  --neighbor "${neighbor}" \
  --prompt_alpha "${prompt_alpha}" \
  --warm_n "${warm_n}"
