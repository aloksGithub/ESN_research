#!/bin/bash -l

# This job only installs dependencies and validates JAX. The experiment's
# 20-CPU allocation is configured separately in run_esnas_gestures.sbatch.
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --output=logs/setup-gpu.%j.out
#SBATCH --error=logs/setup-gpu.%j.err
#SBATCH --job-name=esn-gpu-setup
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -Eeuo pipefail

REPO_ROOT="${1:?repository path argument is required}"
IMAGE_PATH="${2:?Singularity image path argument is required}"
CUDA_VARIANT="${3:-auto}"

module load singularity

singularity exec \
    --nv \
    --bind "${REPO_ROOT}:/workspace" \
    "${IMAGE_PATH}" \
    bash /workspace/scripts/setup_ubuntu_gpu.sh \
        --inside-container \
        --cuda "${CUDA_VARIANT}"
