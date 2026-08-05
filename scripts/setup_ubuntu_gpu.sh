#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE_PATH="${REPO_ROOT}/containers/python312-bookworm.sif"
IMAGE_SOURCE="docker://python:3.12-bookworm"
CUDA_VARIANT="auto"
WAIT_FOR_JOB=0
INSIDE_CONTAINER=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_ubuntu_gpu.sh [options]

Prepare the project for the cluster's Singularity/Slurm GPU workflow. The
default invocation loads Singularity, pulls a Python 3.12 image if necessary,
and submits a GPU job that installs the Python dependencies and verifies JAX.

Options:
  --cuda auto|12|13    Select the JAX CUDA wheel (default: auto).
  --image PATH         Location of the Singularity image.
  --image-source URI   Source used when the image does not exist.
  --wait               Wait for the Slurm setup job to finish.
  -h, --help           Show this help text.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

initialize_modules() {
    if command -v module >/dev/null 2>&1; then
        return
    fi

    local init_script
    for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
        if [[ -r "${init_script}" ]]; then
            # shellcheck source=/dev/null
            source "${init_script}"
            break
        fi
    done
}

while (($# > 0)); do
    case "$1" in
        --cuda)
            (($# >= 2)) || die "--cuda requires auto, 12, or 13"
            CUDA_VARIANT="$2"
            shift 2
            ;;
        --image)
            (($# >= 2)) || die "--image requires a path"
            IMAGE_PATH="$2"
            shift 2
            ;;
        --image-source)
            (($# >= 2)) || die "--image-source requires a URI"
            IMAGE_SOURCE="$2"
            shift 2
            ;;
        --wait)
            WAIT_FOR_JOB=1
            shift
            ;;
        --inside-container)
            INSIDE_CONTAINER=1
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

if ((INSIDE_CONTAINER == 0)); then
    cd -- "${REPO_ROOT}"
    [[ -f requirements.txt ]] || die "requirements.txt was not found in ${REPO_ROOT}"
    [[ -f scripts/setup_ubuntu_gpu.sbatch ]] || die "scripts/setup_ubuntu_gpu.sbatch is missing"

    initialize_modules
    if command -v module >/dev/null 2>&1; then
        printf '\n==> Loading Singularity module\n'
        module load singularity
    fi

    command -v singularity >/dev/null 2>&1 || die "Singularity is unavailable after 'module load singularity'"
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; this script must run on a Slurm login node"

    if [[ "${IMAGE_PATH}" != /* ]]; then
        IMAGE_PATH="${REPO_ROOT}/${IMAGE_PATH}"
    fi

    mkdir -p -- "${REPO_ROOT}/containers" "${REPO_ROOT}/logs"

    if [[ ! -f "${IMAGE_PATH}" ]]; then
        mkdir -p -- "$(dirname -- "${IMAGE_PATH}")"
        printf '\n==> Pulling %s\n' "${IMAGE_SOURCE}"
        singularity pull "${IMAGE_PATH}" "${IMAGE_SOURCE}"
    else
        printf '\n==> Reusing image %s\n' "${IMAGE_PATH}"
    fi

    printf '\n==> Submitting the GPU setup and JAX validation job\n'
    SBATCH_OPTIONS=(--parsable)
    if ((WAIT_FOR_JOB)); then
        SBATCH_OPTIONS+=(--wait)
    fi

    JOB_ID="$(sbatch "${SBATCH_OPTIONS[@]}" scripts/setup_ubuntu_gpu.sbatch "${REPO_ROOT}" "${IMAGE_PATH}" "${CUDA_VARIANT}")"
    JOB_ID="${JOB_ID%%;*}"

    cat <<EOF

Submitted Slurm setup job ${JOB_ID}.

Monitor it with:
  squeue -j ${JOB_ID}
  tail -f "${REPO_ROOT}/logs/setup-gpu.${JOB_ID}.out"

The setup is complete when that log ends with "GPU setup completed successfully".
After it succeeds, submit the experiment with:
  sbatch scripts/run_esnas_gestures.sbatch
EOF
    exit 0
fi

# Everything below runs inside the Singularity container on a Slurm GPU node.
cd -- "${REPO_ROOT}"
[[ -f requirements.txt ]] || die "requirements.txt is not visible inside the container"
command -v python >/dev/null 2>&1 || die "the container does not provide Python"
command -v nvidia-smi >/dev/null 2>&1 || die "Singularity --nv did not expose nvidia-smi inside the container"

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || die "the container provides Python ${PYTHON_VERSION}, but Python 3.12 is required"

printf '\n==> NVIDIA allocation inside Singularity\n'
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
[[ -n "${DRIVER_VERSION}" ]] || die "could not determine the NVIDIA driver version"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
[[ "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]] || die "could not parse NVIDIA driver version ${DRIVER_VERSION}"

CUDA13_GPU_OK=1
COMPUTE_CAPABILITIES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
if [[ -z "${COMPUTE_CAPABILITIES}" ]]; then
    CUDA13_GPU_OK=0
    printf 'Warning: GPU compute capability is unavailable; auto-selection will use CUDA 12.\n'
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
    ((CUDA13_GPU_OK)) || die "CUDA 13 JAX requires compute capability 7.5 or newer"
else
    ((DRIVER_MAJOR >= 525)) || die "CUDA 12 JAX requires NVIDIA driver 525 or newer (found ${DRIVER_VERSION})"
fi

printf '\n==> Selected JAX CUDA %s wheel\n' "${CUDA_VARIANT}"

VENV_DIR="${REPO_ROOT}/venv"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON_VERSION="$(${VENV_DIR}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${VENV_PYTHON_VERSION}" == "3.12" ]] || die "${VENV_DIR} uses Python ${VENV_PYTHON_VERSION}; remove it before retrying"
else
    printf '\n==> Creating container environment at %s\n' "${VENV_DIR}"
    python -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
SELECTED_REQUIREMENTS="${VENV_DIR}/requirements.cuda${CUDA_VARIANT}.txt"
sed -E "s/^jax\[cuda(12|13)\]/jax[cuda${CUDA_VARIANT}]/" requirements.txt > "${SELECTED_REQUIREMENTS}"

printf '\n==> Installing Python dependencies\n'
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install -r "${SELECTED_REQUIREMENTS}"

# Singularity --nv exposes the host driver through this path. Do not unset it.
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

cat <<'EOF'

GPU setup completed successfully.
Use /workspace/venv whenever the repository is bound to /workspace.
Submit the experiment from the repository root with:
  sbatch scripts/run_esnas_gestures.sbatch
EOF
