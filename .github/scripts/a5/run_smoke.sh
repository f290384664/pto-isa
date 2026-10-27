#!/usr/bin/env bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -euo pipefail

repo_root=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
  pwd
)

# Failure diagnostics go into a per-run directory so a later checkout, task
# submission or initialization failure can never upload an earlier run's
# leftovers as this run's result. a5-smoke.yml passes run id/attempt/SHA.
diag_run_dir="${repo_root}/a5_smoke_diag/${A5_SMOKE_RUN_ID:-local}-${A5_SMOKE_RUN_ATTEMPT:-0}"
case_diag_dir="${diag_run_dir}/cases"
max_diag_bytes=$((512 * 1024 * 1024))

source_if_exists() {
  local script_path="$1"
  shift || true
  if [[ -f "${script_path}" ]]; then
    set +eu
    # shellcheck disable=SC1090
    source "${script_path}" "$@"
    set -eu
    return 0
  fi

  return 1
}

source_runner_env() {
  local conda_activate="${A5_CONDA_ACTIVATE_PATH:-}"
  local cann_setenv="${A5_CANN_SETENV_PATH:-}"

  if [[ -n "${conda_activate}" ]]; then
    source_if_exists "${conda_activate}" || true
  fi

  if [[ -n "${cann_setenv}" ]]; then
    source_if_exists "${cann_setenv}" || true
  fi
}

resolve_ascend_home() {
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -d "${ASCEND_HOME_PATH}" ]]; then
      printf '%s\n' "${ASCEND_HOME_PATH}"
      return 0
    fi
    echo "ERROR: Explicitly set ASCEND_HOME_PATH='${ASCEND_HOME_PATH}' is not a valid directory." >&2
    return 1
  fi

  local candidates=()
  local glob_candidate=""

  candidates+=(
    "/usr/local/Ascend/cann"
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit"
    "${HOME}/Ascend/cann"
  )
  shopt -s nullglob
  for glob_candidate in /usr/local/Ascend/cann-*; do
    candidates+=("${glob_candidate}")
  done
  shopt -u nullglob

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

source_ascend_env() {
  local ascend_home="$1"
  local env_scripts=(
    "${ascend_home}/bin/setenv.bash"
    "${ascend_home}/set_env.sh"
    "${ascend_home}/ascend-toolkit/set_env.sh"
  )

  local env_script=""
  for env_script in "${env_scripts[@]}"; do
    if source_if_exists "${env_script}"; then
      return 0
    fi
  done

  return 1
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
}

cleanup_a5_root_artifacts() {
  local a5_build_dir="${repo_root}/tests/npu/a5/src/st/build"
  set +e
  if [[ -d "${a5_build_dir}" ]]; then
    echo "Cleaning root-owned A5 build directory: ${a5_build_dir}"
    rm -rf "${a5_build_dir}"
  fi
}

# Archive only what explains the failure: core files first (irreplaceable),
# then generated inputs/goldens/outputs, plus small device/version snapshots.
# Collection is bounded by a size cap and an overall deadline, runs before the
# build cleanup trap, reports whether it actually succeeded, and never changes
# the run's exit code.
preserve_diagnostics() {
  local rc="$1"
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi
  set +e
  umask 022
  local a5_build_dir="${repo_root}/tests/npu/a5/src/st/build"
  local archive="${diag_run_dir}/diagnostics.tar"
  local deadline=$((SECONDS + 300))
  local written=0
  local status="ok"
  local entry="" bytes="" core_file="" version_file=""

  if ! mkdir -p "${diag_run_dir}"; then
    echo "WARNING: cannot create ${diag_run_dir}; diagnostics not collected" >&2
    set -e
    return 0
  fi
  # Hand the directory to the uploading runner user right away so the next
  # run's non-root cleanup can always remove it.
  if [[ "$(id -u)" -eq 0 && -n "${A5_SMOKE_DIAG_OWNER_UID:-}" ]]; then
    chown "${A5_SMOKE_DIAG_OWNER_UID}:${A5_SMOKE_DIAG_OWNER_GID:-${A5_SMOKE_DIAG_OWNER_UID}}" \
      "${repo_root}/a5_smoke_diag" "${diag_run_dir}" 2>/dev/null || true
  fi

  # Cores are the irreplaceable evidence: archive them before any case
  # directory can consume the budget.
  if [[ -d "${a5_build_dir}/bin" ]]; then
    for core_file in "${a5_build_dir}"/bin/core*; do
      [[ -f "${core_file}" ]] || continue
      if ((SECONDS >= deadline)); then
        echo "diagnostics: collection deadline reached while adding cores" >&2
        status="truncated"
        break
      fi
      bytes=$(du -sb "${core_file}" 2>/dev/null | cut -f1)
      bytes=${bytes:-0}
      if ((written + bytes > max_diag_bytes)); then
        echo "diagnostics: size cap reached, skipping core ${core_file}" >&2
        status="truncated"
        continue
      fi
      tar -rf "${archive}" -C "${a5_build_dir}/bin" "$(basename "${core_file}")" 2>/dev/null || status="partial"
      written=$((written + bytes))
    done
  fi
  if [[ -d "${a5_build_dir}" ]]; then
    for entry in "${a5_build_dir}"/*/; do
      [[ -d "${entry}" ]] || continue
      compgen -G "${entry}*.bin" >/dev/null || continue
      if ((SECONDS >= deadline)); then
        echo "diagnostics: collection deadline reached while adding case data" >&2
        status="truncated"
        break
      fi
      bytes=$(du -sb "${entry}" 2>/dev/null | cut -f1)
      bytes=${bytes:-0}
      if ((written + bytes > max_diag_bytes)); then
        echo "diagnostics: size cap reached, skipping ${entry}" >&2
        status="truncated"
        continue
      fi
      tar -rf "${archive}" -C "${a5_build_dir}" "$(basename "${entry}")" 2>/dev/null || status="partial"
      written=$((written + bytes))
    done
  fi
  if [[ -s "${archive}" ]]; then
    gzip -f "${archive}" 2>/dev/null || {
      echo "WARNING: failed to compress the diagnostics archive" >&2
      status="partial"
    }
  else
    rm -f "${archive}"
  fi

  if command -v dmesg >/dev/null 2>&1; then
    timeout 15 dmesg 2>/dev/null | tail -n 100 >"${diag_run_dir}/dmesg.tail" || true
  fi
  if command -v npu-smi >/dev/null 2>&1; then
    timeout 30 npu-smi info -t health >"${diag_run_dir}/npu-smi-health.txt" 2>/dev/null || true
  fi
  {
    echo "uname: $(uname -a 2>/dev/null)"
    for version_file in "${ASCEND_HOME_PATH:-/nonexistent}/version.cfg" \
      "${ASCEND_HOME_PATH:-/nonexistent}/version.info" \
      /usr/local/Ascend/driver/version.info; do
      if [[ -f "${version_file}" ]]; then
        echo "== ${version_file} =="
        cat "${version_file}" 2>/dev/null
      fi
    done
  } >"${diag_run_dir}/versions.txt" 2>/dev/null || true

  {
    echo "exit_code=${rc}"
    echo "run_id=${A5_SMOKE_RUN_ID:-unknown}"
    echo "run_attempt=${A5_SMOKE_RUN_ATTEMPT:-unknown}"
    echo "target_sha=${A5_SMOKE_TARGET_SHA:-unknown}"
    echo "started_at=${A5_SMOKE_STARTED_AT:-unknown}"
    echo "finished_at=$(date -Is)"
    echo "collection_status=${status}"
    echo "collected_bytes=${written}"
  } >"${diag_run_dir}/metadata.txt" 2>/dev/null

  # The uploader is the non-root runner user; hand it ownership and read
  # access when this script runs as root.
  if [[ "$(id -u)" -eq 0 && -n "${A5_SMOKE_DIAG_OWNER_UID:-}" ]]; then
    chown -R "${A5_SMOKE_DIAG_OWNER_UID}:${A5_SMOKE_DIAG_OWNER_GID:-${A5_SMOKE_DIAG_OWNER_UID}}" \
      "${diag_run_dir}" 2>/dev/null || true
  fi
  chmod -R a+rX "${diag_run_dir}" 2>/dev/null || true

  set -e
  if [[ "${status}" == "ok" ]]; then
    echo "A5 smoke diagnostics preserved under ${diag_run_dir} (${written} bytes)" >&2
  else
    echo "WARNING: A5 smoke diagnostics only partially preserved under ${diag_run_dir} (status=${status})" >&2
  fi
}

main() {
  source_runner_env
  # shellcheck disable=SC1091
  source "${repo_root}/.github/scripts/a5/setup_runner_env.sh"
  a5_bootstrap_runner_env

  local ascend_home=""
  if ! ascend_home="$(resolve_ascend_home)"; then
    echo "ERROR: ASCEND_HOME_PATH is not set and no default Ascend installation was found." >&2
    exit 1
  fi

  export ASCEND_HOME_PATH="${ascend_home}"
  if ! source_ascend_env "${ASCEND_HOME_PATH}"; then
    echo "WARNING: no Ascend environment script found under ${ASCEND_HOME_PATH}; continuing with current shell environment." >&2
  fi

  require_command python3
  require_command cmake
  require_command git
  require_command make

  cd "${repo_root}"
  # Keep everything this root-run script creates readable by the non-root
  # uploader, and give the per-run directory to the runner user immediately so
  # the next attempt can always clean it up.
  umask 022
  rm -rf "${diag_run_dir}"
  mkdir -p "${diag_run_dir}"
  if [[ "$(id -u)" -eq 0 && -n "${A5_SMOKE_DIAG_OWNER_UID:-}" ]]; then
    chown "${A5_SMOKE_DIAG_OWNER_UID}:${A5_SMOKE_DIAG_OWNER_GID:-${A5_SMOKE_DIAG_OWNER_UID}}" \
      "${repo_root}/a5_smoke_diag" "${diag_run_dir}" 2>/dev/null || true
  fi
  export A5_SMOKE_STARTED_AT="${A5_SMOKE_STARTED_AT:-$(date -Is)}"
  # Let run_st.py snapshot a failing testcase's inputs the moment it fails,
  # instead of relying on a stable seed or on files surviving until the end.
  export PTOAS_ST_DIAG_DIR="${case_diag_dir}"
  trap 'rc=$?; preserve_diagnostics "${rc}"; cleanup_a5_root_artifacts; exit "${rc}"' EXIT

  echo "Repository root: ${repo_root}"
  echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
  python3 --version
  cmake --version | head -n 1
  git --version
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
  fi

  local a5_build_dir="${repo_root}/tests/npu/a5/src/st/build"
  if [[ -d "${a5_build_dir}" ]]; then
    echo "Removing stale A5 build directory: ${a5_build_dir}"
    rm -rf "${a5_build_dir}"
  fi

  chmod +x ./build.sh ./tests/run_st.sh
  bash ./build.sh --run_simple --a5 --npu
}

main "$@"
