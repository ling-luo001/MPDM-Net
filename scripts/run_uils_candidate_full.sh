#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lz/anaconda3/envs/mamba/bin/python}"
EXP_ROOT="${EXP_ROOT:-/var/tmp/mpdm-uils-full}"
EXPERIMENT_NAME="uils_mpdm_candidate_full_v1"
CONFIG_PATH="recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-candidate-full.yaml"
STDOUT_LOG="${EXP_ROOT}/${EXPERIMENT_NAME}.stdout.log"

mkdir -p "${EXP_ROOT}"
cd "${ROOT_DIR}"

set +e
{
    printf '\n[%s] Launching %s with %s\n' \
        "$(date -Iseconds)" "${EXPERIMENT_NAME}" "${CONFIG_PATH}"
    "${PYTHON_BIN}" train.py \
        --config "${CONFIG_PATH}" \
        --exp_folder "${EXP_ROOT}" \
        --exp_name "${EXPERIMENT_NAME}"
    status=$?
    printf '[%s] %s exited with status %d\n' \
        "$(date -Iseconds)" "${EXPERIMENT_NAME}" "${status}"
} >>"${STDOUT_LOG}" 2>&1
set -e

exit "${status}"
