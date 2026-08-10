#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lz/anaconda3/envs/mamba/bin/python}"
EXP_ROOT="${EXP_ROOT:-/var/tmp/mpdm-uils-paired}"
MAX_STEPS="${MAX_STEPS:-100001}"

mkdir -p "${EXP_ROOT}"
cd "${ROOT_DIR}"

run_experiment() {
    local experiment_name="$1"
    local config_path="$2"
    local stdout_log="${EXP_ROOT}/${experiment_name}.stdout.log"

    printf 'Starting %s with %s\n' "${experiment_name}" "${config_path}"
    "${PYTHON_BIN}" train.py \
        --config "${config_path}" \
        --exp_folder "${EXP_ROOT}" \
        --exp_name "${experiment_name}" \
        --max_steps "${MAX_STEPS}" \
        >"${stdout_log}" 2>&1
    printf 'Completed %s\n' "${experiment_name}"
}

run_experiment \
    uils_mpdm_candidate_mini_seed1234 \
    recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-candidate-mini.yaml
run_experiment \
    uils_mpdm_baseline_mini_seed1234 \
    recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-baseline-mini.yaml
