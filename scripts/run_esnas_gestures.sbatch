#!/bin/bash -l

# Keep this equal to n_jobs=20 in experiments/esnas_gestures.py.
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --output=logs/esnas-gestures.%j.out
#SBATCH --error=logs/esnas-gestures.%j.err
#SBATCH --job-name=esnas-gestures
#SBATCH --time=12:00:00
#SBATCH --mem=32G

set -Eeuo pipefail

module load singularity

REPO_ROOT="${SLURM_SUBMIT_DIR}"
IMAGE_PATH="${REPO_ROOT}/containers/python312-bookworm.sif"

[[ -f "${IMAGE_PATH}" ]] || {
    printf 'ERROR: Singularity image not found: %s\n' "${IMAGE_PATH}" >&2
    exit 1
}
[[ -x "${REPO_ROOT}/venv/bin/python" ]] || {
    printf 'ERROR: Run bash scripts/setup_ubuntu_gpu.sh and wait for it to finish first.\n' >&2
    exit 1
}

singularity exec \
    --nv \
    --bind "${REPO_ROOT}:/workspace" \
    "${IMAGE_PATH}" \
    bash -lc '
        set -Eeuo pipefail
        cd /workspace
        source venv/bin/activate

        export XLA_PYTHON_CLIENT_PREALLOCATE=false

        echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
        echo "ESNAS n_jobs is configured manually as 20"
        nvidia-smi

        python -u -c \
            "from experiments.esnas_gestures import run_esnas_gestures; run_esnas_gestures()"
    '
