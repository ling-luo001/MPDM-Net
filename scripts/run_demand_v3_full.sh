#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rd_asymmetric_polar_demand_v3_full_seed1234}"
CONFIG_PATH="${CONFIG_PATH:-recipes/RD-Asymmetric-Polar-Demand-V3/RD-Asymmetric-Polar-Demand-V3.yaml}"
EPOCHS="${EPOCHS:-400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        /home/lz/anaconda3/envs/mamba/bin/python \
        /home/g515528/software/anaconda3/envs/mambavision/bin/python; do
        if [[ -x "${candidate}" ]]; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python environment not found. Set PYTHON_BIN explicitly.\n' >&2
    exit 1
fi

if [[ -z "${EXP_ROOT:-}" ]]; then
    if mountpoint -q /media/lz/WZZ; then
        EXP_ROOT="/media/lz/WZZ/mpdm_full_experiments/rd_asymmetric_polar_demand_v3"
    else
        EXP_ROOT="${HOME}/mpdm_full_experiments/rd_asymmetric_polar_demand_v3"
    fi
fi

RUN_DIR="${EXP_ROOT}/${EXPERIMENT_NAME}"
STDOUT_LOG="${EXP_ROOT}/${EXPERIMENT_NAME}.stdout.log"
PID_FILE="${EXP_ROOT}/${EXPERIMENT_NAME}.pid"
mkdir -p "${EXP_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(cat "${PID_FILE}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        printf '%s is already running as PID %s.\n' \
            "${EXPERIMENT_NAME}" "${existing_pid}"
        exit 0
    fi
fi

matching_pid="$(pgrep -f "train.py.*--exp_name[[:space:]]+${EXPERIMENT_NAME}" | head -n 1 || true)"
if [[ -n "${matching_pid}" ]]; then
    printf '%s is already running as PID %s.\n' \
        "${EXPERIMENT_NAME}" "${matching_pid}"
    printf '%s\n' "${matching_pid}" >"${PID_FILE}"
    exit 0
fi

if find "${RUN_DIR}" -maxdepth 1 -type f -name 'g_????????.pth' -print -quit \
    2>/dev/null | grep -q . && [[ "${ALLOW_RESUME:-0}" != "1" ]]; then
    printf 'Existing V3 checkpoints found in %s. Set ALLOW_RESUME=1 explicitly.\n' \
        "${RUN_DIR}" >&2
    exit 1
fi

cd "${ROOT_DIR}"
printf '\n[%s] Starting %s\n' "$(date -Iseconds)" "${EXPERIMENT_NAME}" >>"${STDOUT_LOG}"
printf 'Code: %s\nConfig: %s\nOutput: %s\n' \
    "${ROOT_DIR}" "${CONFIG_PATH}" "${RUN_DIR}" >>"${STDOUT_LOG}"

nohup env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -B -u train.py \
        --config "${CONFIG_PATH}" \
        --exp_folder "${EXP_ROOT}" \
        --exp_name "${EXPERIMENT_NAME}" \
        --epochs "${EPOCHS}" \
        >>"${STDOUT_LOG}" 2>&1 &

pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
sleep 3
if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'Launch failed. Last log lines:\n' >&2
    tail -n 80 "${STDOUT_LOG}" >&2 || true
    exit 1
fi

printf 'Started %s as PID %s.\n' "${EXPERIMENT_NAME}" "${pid}"
printf 'Log: %s\nRun directory: %s\n' "${STDOUT_LOG}" "${RUN_DIR}"
