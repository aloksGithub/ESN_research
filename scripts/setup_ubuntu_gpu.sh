#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
CUDA_VARIANT="auto"
INSTALL_SYSTEM_PACKAGES=1

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_ubuntu_gpu.sh [options]

Create a Python 3.12 environment, install the repository dependencies, and
verify that JAX can execute on an NVIDIA GPU.

Options:
  --cuda auto|12|13          Select the JAX CUDA wheel (default: auto).
  --venv PATH                Virtual environment path (default: .venv).
  --skip-system-packages     Do not install Ubuntu packages with apt.
  -h, --help                 Show this help text.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --cuda)
            (($# >= 2)) || die "--cuda requires auto, 12, or 13"
            CUDA_VARIANT="$2"
            shift 2
            ;;
        --venv)
            (($# >= 2)) || die "--venv requires a path"
            VENV_DIR="$2"
            shift 2
            ;;
        --skip-system-packages)
            INSTALL_SYSTEM_PACKAGES=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${CUDA_VARIANT}" in
    auto|12|13) ;;
    *) die "--cuda must be auto, 12, or 13" ;;
esac

[[ "$(uname -s)" == "Linux" ]] || die "this setup script must run on Linux"
case "$(uname -m)" in
    x86_64|aarch64) ;;
    *) die "JAX CUDA wheels are not available for architecture $(uname -m)" ;;
esac

cd -- "${REPO_ROOT}"
[[ -f requirements.txt ]] || die "requirements.txt was not found in ${REPO_ROOT}"

if ((INSTALL_SYSTEM_PACKAGES)); then
    if ! command -v apt-get >/dev/null 2>&1; then
        die "apt-get is unavailable; rerun with --skip-system-packages after installing curl and build tools"
    fi

    if [[ "${EUID}" -eq 0 ]]; then
        APT_PREFIX=()
    elif command -v sudo >/dev/null 2>&1; then
        APT_PREFIX=(sudo)
    else
        die "sudo is unavailable; install ca-certificates, curl, and build-essential, then use --skip-system-packages"
    fi

    printf '\n==> Installing Ubuntu system packages\n'
    "${APT_PREFIX[@]}" apt-get update
    "${APT_PREFIX[@]}" apt-get install -y ca-certificates curl build-essential
fi

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable; install/attach the NVIDIA driver first"

printf '\n==> NVIDIA devices\n'
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
[[ -n "${DRIVER_VERSION}" ]] || die "could not determine the NVIDIA driver version"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
[[ "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]] || die "could not parse NVIDIA driver version ${DRIVER_VERSION}"

CUDA13_GPU_OK=1
COMPUTE_CAPABILITIES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
if [[ -z "${COMPUTE_CAPABILITIES}" ]]; then
    CUDA13_GPU_OK=0
    printf 'Warning: GPU compute capability could not be queried; auto-selection will use CUDA 12.\n'
else
    while IFS= read -r compute_capability; do
        compute_capability="${compute_capability//[[:space:]]/}"
        capability_major="${compute_capability%%.*}"
        capability_minor="${compute_capability#*.}"
        if [[ ! "${capability_major}" =~ ^[0-9]+$ || ! "${capability_minor}" =~ ^[0-9]+$ ]]; then
            CUDA13_GPU_OK=0
        elif ((capability_major < 7 || (capability_major == 7 && capability_minor < 5))); then
            CUDA13_GPU_OK=0
        fi
    done <<< "${COMPUTE_CAPABILITIES}"
fi

if [[ "${CUDA_VARIANT}" == "auto" ]]; then
    if ((DRIVER_MAJOR >= 580 && CUDA13_GPU_OK)); then
        CUDA_VARIANT=13
    else
        CUDA_VARIANT=12
    fi
fi

if [[ "${CUDA_VARIANT}" == "13" ]]; then
    ((DRIVER_MAJOR >= 580)) || die "CUDA 13 JAX requires NVIDIA driver 580 or newer (found ${DRIVER_VERSION})"
    ((CUDA13_GPU_OK)) || die "CUDA 13 JAX requires every visible GPU to have compute capability 7.5 or newer"
else
    ((DRIVER_MAJOR >= 525)) || die "CUDA 12 JAX requires NVIDIA driver 525 or newer (found ${DRIVER_VERSION})"
fi

printf '\n==> Selected JAX CUDA %s wheel\n' "${CUDA_VARIANT}"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    printf '\n==> Installing uv\n'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="${HOME}/.local/bin/uv"
fi
[[ -x "${UV_BIN}" ]] || die "uv installation failed"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON_VERSION="$(${VENV_DIR}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${VENV_PYTHON_VERSION}" == "3.12" ]] || die "${VENV_DIR} uses Python ${VENV_PYTHON_VERSION}; remove it or choose another path with --venv"
else
    printf '\n==> Creating Python 3.12 environment at %s\n' "${VENV_DIR}"
    "${UV_BIN}" venv --python 3.12 "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
SELECTED_REQUIREMENTS="${VENV_DIR}/requirements.cuda${CUDA_VARIANT}.txt"
sed -E "s/^jax\[cuda(12|13)\]/jax[cuda${CUDA_VARIANT}]/" requirements.txt > "${SELECTED_REQUIREMENTS}"

printf '\n==> Installing Python dependencies\n'
"${UV_BIN}" pip install --python "${VENV_PYTHON}" -r "${SELECTED_REQUIREMENTS}"

# Pip-provided CUDA libraries should take precedence over a system CUDA toolkit.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    printf 'Warning: unsetting LD_LIBRARY_PATH for the JAX check so it cannot override pip-provided CUDA libraries.\n'
    unset LD_LIBRARY_PATH
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf '\n==> Checking repository imports\n'
"${VENV_PYTHON}" -c 'import experiments.esnas_gestures; print("Repository imports passed")'

printf '\n==> Running JAX GPU computation\n'
"${VENV_PYTHON}" - <<'PY'
import sys

import jax
import jax.numpy as jnp
import jaxlib

print(f"Python: {sys.version.split()[0]}")
print(f"JAX: {jax.__version__}")
print(f"jaxlib: {jaxlib.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if not gpu_devices:
    raise SystemExit("ERROR: JAX did not detect an NVIDIA GPU")

with jax.default_device(gpu_devices[0]):
    matrix = jnp.ones((2048, 2048), dtype=jnp.float32)
    result = (matrix @ matrix).block_until_ready()

if result.device.platform != "gpu":
    raise SystemExit(f"ERROR: test computation ran on {result.device}, not a GPU")

print(f"GPU computation passed on {result.device}")
PY

cat <<EOF

Setup completed successfully.

Activate the environment with:
  source "${VENV_DIR}/bin/activate"

Before running an experiment, use:
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
EOF
