#!/bin/bash
#SBATCH --job-name=ti_rep_figures
#SBATCH --output=slurm_logs/01_rep_figures_%j.out
#SBATCH --error=slurm_logs/01_rep_figures_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_DIR=$(pwd)

mkdir -p outputs/figures outputs/tables outputs/runs outputs/cache slurm_logs

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

python scripts/run_figures.py --config configs/slurm_rep.yaml
