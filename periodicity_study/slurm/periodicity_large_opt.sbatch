#!/bin/bash
#SBATCH --job-name=periodicity_large_opt
#SBATCH --output=slurm_logs/periodicity_large_opt_%j.out
#SBATCH --error=slurm_logs/periodicity_large_opt_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

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

RUNS_ROOT=${RUNS_ROOT:-"periodicity_study/outputs_runs"}

ENV_IDS=${ENV_IDS:-"periodicity_large"}
OUT_DIR=${OUT_DIR:-"${RUNS_ROOT}/periodicity_large/opt_run"}
POLICY_INPUTS=${POLICY_INPUTS:-"rep"}
JL_MODE=${JL_MODE:-"randproj_l2"}
JL_DIM=${JL_DIM:-8}
JL_SEED=${JL_SEED:-0}
EXTRA_FLAGS=${EXTRA_FLAGS:-""}
FAST=${FAST:-0}

read -r -a extra <<< "$EXTRA_FLAGS"

cmd=(python -m periodicity_study.run_study
    --device cuda:0
    --envs "$ENV_IDS"
    --policy-inputs "$POLICY_INPUTS"
    --output-dir "$OUT_DIR"
    --handcrafted-proj-mode "$JL_MODE"
    --handcrafted-proj-dim "$JL_DIM"
)

if [ "$JL_SEED" -ne 0 ]; then
    cmd+=("--handcrafted-proj-seed" "$JL_SEED")
fi
if [ "$FAST" -eq 1 ]; then
    cmd+=("--fast")
fi
if [ "${#extra[@]}" -gt 0 ]; then
    cmd+=("${extra[@]}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
