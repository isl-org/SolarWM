#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${VENV_DIR:-"$repo_root/.venv-wan22-xpu"}
model_root=${SOLAR_MODEL_ROOT:-/home/ssheorey/models/SolarWM}
data_root=${SOLAR_DATA_ROOT:-/home/ssheorey/data/SolarWM-Data/releases-v1}
output_root=${SOLAR_OUTPUT_ROOT:-"$repo_root/outputs"}
run_id=${SOLAR_RUN_ID:-wan22-ti2v-5b-stage2-xpu}
config_path="$repo_root/configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
base_path="$model_root/SolarWM-5B-base"
checkpoint_path="$model_root/SolarWM-5B-sgf-stage2-81f"
checkpoint_file="$checkpoint_path/model.pt"
test_index="$data_root/recipes/clean-81f/raw-wds/test-index.jsonl.gz"

for required_path in "$venv_dir/bin/python" "$config_path" "$base_path/text_encoder/models_t5_umt5-xxl-enc-bf16.pth" \
    "$base_path/tokenizer" "$base_path/vae/Wan2.2_VAE.pth" "$checkpoint_file" "$test_index"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'Missing required inference asset: %s\n' "$required_path" >&2
        exit 1
    fi
done

exec "$venv_dir/bin/python" -m solarwm infer \
    --config "$config_path" \
    --set "model.base_path=$base_path" \
    --set "checkpoint.path=$checkpoint_path" \
    --set "data.index_root=$data_root" \
    --set "data.transport.root=$data_root" \
    --set "data.test_index=$test_index" \
    --set "inference.device=xpu" \
    --set "inference.run_id=$run_id" \
    --set "runtime.output_dir=$output_root/wan22-ti2v-5b-stage2-xpu"