import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import ModelConfig

from src.gsm8k_utils import GSM8k_PROMPT, MathRewardModel, extract_answer
from src.ppo_single_vllm_trainer import PPOSingleVLLMTrainer, PPOVLLMConfig
from src.ppov2_trainer import PPOv2Trainer
from src.utils import TRLParser, WandbLogModelConfig


@dataclass
class ScriptArguments:
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_subset: str = field(default="main", metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    wandb_run_id: Optional[str] = field(default=None)
    vllm: bool = False


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        question_prompts = [GSM8k_PROMPT.format(question) for question in element["question"]]
        input_ids = tokenizer(
            question_prompts,
            padding=False,
        )["input_ids"]
        # answers = [ans.split("####")[1].strip() for ans in element["answer"]]
        answers = [extract_answer(ans) for ans in element["answer"]]
        answer_ids = tokenizer(
            answers,
            padding=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": [len(ids) for ids in input_ids], "labels": answer_ids}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
    )


if __name__ == "__main__":
    parser = TRLParser((ScriptArguments, PPOVLLMConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    if args.wandb_run_id == "slurm":
        run_id = os.environ["SLURM_JOB_ID"]
        config_name = os.path.basename(config.output_dir)
        # save to parent / slurm id / output_dir
        if args.output_global_parent_dir is not None:
            config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        config.sft_model_path,
        num_labels=1,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    reward_model = MathRewardModel(tokenizer=tokenizer)
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name, data_dir=args.dataset_subset)
    if config.sanity_check:
        config.push_to_hub = False
        config.report_to = ""
        config.save_strategy = "no"
        config.total_episodes = 2048
        config.per_device_train_batch_size = 2
        config.gradient_accumulation_steps = 4
        config.local_rollout_forward_batch_size = 8
        config.num_sample_generations = 0

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    if args.vllm:
        TrainerCls = PPOSingleVLLMTrainer
    else:
        TrainerCls = PPOv2Trainer

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)],
    )
    trainer.train()

    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()

        if trainer.accelerator.is_main_process:
            try:
                os.remove("output_dir")
            except OSError:
                pass

            os.symlink(config.output_dir, "output_dir")
