#!/bin/bash
#SBATCH --job-name=periodicity_sweep
#SBATCH --output=slurm_logs/periodicity_sweep_%A_%a.out
#SBATCH --error=slurm_logs/periodicity_sweep_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-15

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p slurm_logs

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if ! python -c "import ti" 2>/dev/null; then
    pip install -r requirements.txt
    pip install -e .
fi

export CUDA_VISIBLE_DEVICES=0

FLAGS=""
IDX=${SLURM_ARRAY_TASK_ID}

# A: success-conditioned alpha annealing (bit 0)
if (( IDX & 1 )); then
    FLAGS="${FLAGS} --use-alpha-anneal"
fi
# C: two-critic decomposition (bit 1)
if (( IDX & 2 )); then
    FLAGS="${FLAGS} --use-two-critic"
fi
# B: episode-gated intrinsic (bit 2)
if (( IDX & 4 )); then
    FLAGS="${FLAGS} --use-alpha-gate"
fi
# D: intrinsic normalization/clipping (bit 3)
if (( IDX & 8 )); then
    FLAGS="${FLAGS} --use-int-norm"
fi

BASE_ARGS=${SWEEP_BASE_ARGS:-""}
EXTRA_ARGS=${STUDY_ARGS:-""}
OUT_DIR="periodicity_study/outputs_sweep/comb_${IDX}"

echo "Sweep flags: ${FLAGS}"
python -m periodicity_study.run_study --device cuda:0 --output-dir "${OUT_DIR}" ${BASE_ARGS} ${FLAGS} ${EXTRA_ARGS}
