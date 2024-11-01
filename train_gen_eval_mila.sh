#!/bin/bash
#SBATCH --job-name=trl_summarize
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --time=8:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
source mila.sh
# tag with the git commit
export WANDB_TAGS=$(git rev-parse --short HEAD)
$@ --output_global_parent_dir $SCRATCH/trl_summarize/results

if [[ "$@" == *"bf16"* ]]; then
    DTYPE_ARG=" --torch_dtype bfloat16"
else
    DTYPE_ARG=""
fi

MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
MODEL_PATH_ARG="--model_name_or_path $MODEL_PATH"
python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG $DTYPE_ARG

if [[ "$MODEL_PATH" == *"pythia410m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH" == *"pythia1b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH" == *"pythia2.8b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi
python load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG
