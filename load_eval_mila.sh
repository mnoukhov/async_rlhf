#!/bin/bash
#SBATCH --job-name=trl_summarize_eval
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --time=8:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
source mila.sh
MODEL_PATH_ARG=$@

if [[ "$MODEL_PATH_ARG" == *"pythia410m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia1b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia2.8b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi

python load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG --torch_dtype float16
