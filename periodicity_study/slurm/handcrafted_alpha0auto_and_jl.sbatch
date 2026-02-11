#!/bin/bash
#SBATCH --job-name=periodicity_runs
#SBATCH --output=slurm_logs/periodicity_runs_%j.out
#SBATCH --error=slurm_logs/periodicity_runs_%j.err
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

ENV_IDS=${ENV_IDS:-"periodicity,periodicity_large"}
OUT_BASE=${OUT_BASE:-"${RUNS_ROOT}/handcrafted_alpha0auto"}
POLICY_INPUTS=${POLICY_INPUTS:-"rep"}
FAST=${FAST:-0}
EXTRA_FLAGS_1=${EXTRA_FLAGS_1:-""}
EXTRA_FLAGS_2=${EXTRA_FLAGS_2:-""}

read -r -a extra_1 <<< "$EXTRA_FLAGS_1"
read -r -a extra_2 <<< "$EXTRA_FLAGS_2"

cmd_noopt=(python -m periodicity_study.run_study
    --device cuda:0
    --envs "$ENV_IDS"
    --policy-inputs "$POLICY_INPUTS"
    --handcrafted-only
    --alpha0-auto
    --output-dir "${OUT_BASE}/no_opt"
)
if [ "$FAST" -eq 1 ]; then
    cmd_noopt+=("--fast")
fi
if [ "${#extra_1[@]}" -gt 0 ]; then
    cmd_noopt+=("${extra_1[@]}")
fi
"${cmd_noopt[@]}"

cmd_A=(python -m periodicity_study.run_study
    --device cuda:0
    --envs "$ENV_IDS"
    --policy-inputs "$POLICY_INPUTS"
    --handcrafted-only
    --alpha0-auto
    --output-dir "${OUT_BASE}/A"
    --use-alpha-anneal
)
if [ "$FAST" -eq 1 ]; then
    cmd_A+=("--fast")
fi
if [ "${#extra_1[@]}" -gt 0 ]; then
    cmd_A+=("${extra_1[@]}")
fi
"${cmd_A[@]}"

ENV_IDS_2=${ENV_IDS_2:-"periodicity_large"}
OUT_DIR_2=${OUT_DIR_2:-"${RUNS_ROOT}/periodicity_large/exp2_allreps_jl"}
POLICY_INPUTS_2=${POLICY_INPUTS_2:-"rep"}
JL_MODE=${JL_MODE:-"randproj_l2"}
JL_DIM=${JL_DIM:-8}
JL_SEED=${JL_SEED:-0}

cmd_allreps=(python -m periodicity_study.run_study
    --device cuda:0
    --envs "$ENV_IDS_2"
    --policy-inputs "$POLICY_INPUTS_2"
    --output-dir "$OUT_DIR_2"
    --handcrafted-proj-mode "$JL_MODE"
    --handcrafted-proj-dim "$JL_DIM"
)
if [ "$JL_SEED" -ne 0 ]; then
    cmd_allreps+=("--handcrafted-proj-seed" "$JL_SEED")
fi
if [ "$FAST" -eq 1 ]; then
    cmd_allreps+=("--fast")
fi
if [ "${#extra_2[@]}" -gt 0 ]; then
    cmd_allreps+=("${extra_2[@]}")
fi
"${cmd_allreps[@]}"
