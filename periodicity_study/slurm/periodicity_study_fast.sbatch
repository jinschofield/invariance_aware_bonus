#!/bin/bash
#SBATCH --job-name=periodicity_study_fast
#SBATCH --output=slurm_logs/periodicity_study_fast_%j.out
#SBATCH --error=slurm_logs/periodicity_study_fast_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python -m periodicity_study.run_study --fast --device cuda:0
