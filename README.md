# Code for Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models

Note that the method of async RL for LLMs has been integrated in the the [open-instruct](https://github.com/allenai/open-instruct) RLHF library

## Setup

install dependencies
```
pip install -r requirements.txt
```

you can create the dataset that we use with `python relabel_with_rm.py --configs configs/relabel_rm.yml` or just use the dataset available on huggingface hub

## Train

Each algorithm is a command that is run with a config and command line args override the config

```
python online_dpo.py --config config/onlinedpo_pythia410m_tldr.yml --override_arg override_value
```

## Gold Eval

To evaluate with the gold model, we first generate completions with the trained model

```
python generate_for_eval.py --config/generate_tldr.yml --model_name_or_path PATH_TO_MODEL_CHECKPOINTS
```

Then we load the generated completions and evaluate them with our gold model
```
python load_and_eval.py --config/evaluate_tldr.yml --model_name_or_path PATH_TO_MODEL_CHECKPOINTS
```

By default generations are saved in `PATH_TO_MODEL_CHECKPOINTS/_generations` but you can save elsewhere with `--dataset_path`


## Slurm scripts

To make things easier, I provide slurm scripts for training, generation, and eval all-in-one

```bash
./train_generate_and_eval.sh command
```

For example to run online dpo

```bash
./train_generate_and_eval.sh python online_dpo.py --config configs/onlinedpo_pythia410m_tldr.yml --override_arg=override_value
```

Note: to facilitate passing the output directory to the eval scripts, the train script creates a symlink to the output dir called `output_dir`

## Multi-GPU training and eval 
By default, the script is single gpu. For multi-gpu, there is an annoying issue with vllm inference used in `generate_for_eval.py` 
In order to do evaluation, I need to load and unload vllm models. This is complicated https://github.com/vllm-project/vllm/issues/1908

The main solution I've found is to use `vllm==0.4.0post1` but this makes the environment deps sometimes angry.
So the workaround is either use two separate environments (one for training, one for eval) or just run training and eval separately. 
I suggest the latter and provide `train.sh` and then run `generate_and_eval.sh` single-gpu

Multi-gpu Training with 4 GPUs (3 for training, 1 for generation with vllm)
```
./train.sh accelerate launch --config_file configs/deepspeed_zero2.yaml --mixed_precision bf16 --num_processes 3 online_dpo.py --config configs/onlinedpo_pythia2.8b_tldr_vllm_bf16_4gpu.yml --output_dir onlinedpo_pythia2.8b_multigpu
```

Single-gpu Eval
```
./generate_and_eval.sh --model_name_or_path results/onlinedpo_pythia2.8b_multigpu --torch_dtype bfloat16
```

## Configs

All hyperparameters and model names are in corresponding configs in the `configs/` folder

- `python online_dpo.py --config configs/onlinedpo_*`
- `python ppo.py --config configs/ppo_*`
- `python rloo.py --config configs/rloo_*`
- `python rloo.py --config configs/bo2_*` for best-of-2 finetuning

If you want to create the sft and reward models instead of using mine on the huggingface hub

- `python sft.py --config configs/sft*`
- `python reward_modeling.py --config configs/rm*`

Notes

- configs with `vllm` are using vllm to generate, otherwise they use huggingface `generate`
- single-gpu `vllm` configs are placing an extra, vllm model on the gpu for generation. this uses more memory but can be worth it
- 4 gpu vllm configs assume 3 gpus for the training and 1 for vllm generation, adjust batch sizes if you do things differently
- if you're using older GPUs without bf16, add args `--fp16 --bf16 False --torch_dtype float16`
- `--wandb_run_id` sets the wandb run id or if set to `=slurm` it will default to `parent folder / slurm_id / output_dir` 

## Asynchronous Secret Sauce

The asynchronous learning leverages the code:

1. `src/vllm_utils.py` which tricks vllm into being on only the GPUs we tell it to be on
2. `src/online_dpo_vllm_trainer.py` which creates a separate thread for the training and the vllm inference and passes data between the two of them using queues

The majority of the asynchronous parts of the code were written by [Costa](https://github.com/vwxyzjn) but commits were lost when copying and patching so I want to give credit where it's due.

## Citation

```
@misc{noukhovitch_asynchronous_2024,
	title = {Asynchronous {RLHF}: {Faster} and {More} {Efficient} {Off}-{Policy} {RL} for {Language} {Models}},
	shorttitle = {Asynchronous {RLHF}},
	url = {http://arxiv.org/abs/2410.18252},
	publisher = {arXiv},
	author = {Noukhovitch, Michael and Huang, Shengyi and Xhonneux, Sophie and Hosseini, Arian and Agarwal, Rishabh and Courville, Aaron},
	month = oct,
	year = {2024},
}
```
