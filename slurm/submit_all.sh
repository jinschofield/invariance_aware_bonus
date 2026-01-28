#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

mkdir -p slurm_logs outputs/figures outputs/tables outputs/runs outputs/cache

DRY_RUN=false
SUBMIT_FIGURES=true
SUBMIT_OPTUNA=true

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --figures)
            SUBMIT_OPTUNA=false
            ;;
        --optuna)
            SUBMIT_FIGURES=false
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --figures    Submit only figure generation jobs (01-03)"
            echo "  --optuna     Submit only Optuna hyperparameter sweep jobs (04-06)"
            echo "  --dry-run    Show what would be submitted without actually submitting"
            echo "  --help, -h   Show this help message"
            exit 0
            ;;
    esac
done

submit_job() {
    local job_file="$1"
    local job_name="$2"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would submit: $job_file"
    else
        echo "Submitting: $job_name"
        sbatch "$job_file"
    fi
}

echo "Project directory: $PROJECT_DIR"

if [ "$SUBMIT_FIGURES" = true ]; then
    submit_job "slurm/01_rep_figures.sbatch" "01_rep_figures"
    submit_job "slurm/02_bonus_figures.sbatch" "02_bonus_figures"
    submit_job "slurm/03_online_figures.sbatch" "03_online_figures"
fi

if [ "$SUBMIT_OPTUNA" = true ]; then
    submit_job "slurm/04_optuna_periodicity.sbatch" "04_optuna_periodicity"
    submit_job "slurm/05_optuna_slippery.sbatch" "05_optuna_slippery"
    submit_job "slurm/06_optuna_teacup.sbatch" "06_optuna_teacup"
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No jobs were submitted."
else
    echo "All jobs submitted. Monitor with: squeue -u \$USER"
fi
