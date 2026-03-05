#!/bin/bash
#SBATCH --job-name=trl_summarize
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --time=12:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
export WANDB_PROJECT=trl
export WANDB_ENTITY=mila-language-drift
export SLURM_JOB_ID="${SLURM_JOB_ID:-${BEAKER_JOB_ID:0:8}}"
# source mila.sh
# tag with the git commit
export WANDB_TAGS=$(git rev-parse --short HEAD)
uv run accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=3 $@ --output_global_parent_dir results/

MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
MODEL_PATH_ARG="--model_name_or_path $MODEL_PATH"

CUDA_VISIBLE_DEVICES=0 uv run generate_gsm8k.py --config configs/generate_gsm8k.yml $MODEL_PATH_ARG

CUDA_VISIBLE_DEVICES=0 uv run eval_gsm8k.py --config configs/evaluate_gsm8k.yml $MODEL_PATH_ARG
