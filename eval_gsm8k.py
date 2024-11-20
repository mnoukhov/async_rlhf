import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import wandb
from src.gsm8k_utils import extract_answer
from src.utils import TRLParser


@dataclass
class EvalScriptArguments:
    model_name_or_path: str = None
    ref_model_name: Optional[str] = None
    sanity_check: Optional[bool] = False
    wandb_run_id: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="auto")
    batch_size: Optional[int] = field(default=16)
    temperature: float = None


def evaluate(args, all_prompts, all_reference, all_generations, all_episodes, log_to_wandb=False):
    state = PartialState()
    torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map={"": state.process_index},
    )

    ppl_pipeline = pipeline(
        task="perplexity",
        model=args.ref_model_name,
        model_kwargs=model_kwargs,
        prompt_template="\nSolution:",
    )
    # this hacks around the llamatokenizer starting \nSolution off with a weird empty token
    ppl_pipeline.prompt_template_tokens = ppl_pipeline.prompt_template_tokens[1:]

    if ppl_pipeline.tokenizer.pad_token_id is None:
        ppl_pipeline.tokenizer.pad_token_id = ppl_pipeline.tokenizer.eos_token_id

    step = 0
    for step_str, all_query_response in all_generations.items():
        results = []
        gen_ppls = []
        episode = all_episodes[step_str]
        # with state.split_between_processes(all_query_response) as query_response:
        for gen_response, ref_answer in zip(all_query_response, all_reference):
            result = extract_answer(gen_response) == ref_answer
            results.append(result)

        results = np.array(results)
        accuracy = results.mean()

        for out in tqdm(
            ppl_pipeline(all_query_response, batch_size=args.batch_size),
            total=len(all_query_response),
            disable=not state.is_local_main_process,
            desc=f"PPL Step {step_str}",
        ):
            gen_ppls += [r["ppl"] for r in out]

        gen_ppls = gather_object(gen_ppls)
        gen_ppls = np.array(gen_ppls)
        mean_ppl = gen_ppls.mean().item()

        # win_rate = (gen_rewards > ref_rewards).mean().item()
        # norm_reward = (gen_rewards - ref_rewards).mean().item()
        # mean_reward = gen_rewards.mean().item()

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            state.print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb and state.is_main_process:
            num_samples = 32
            sample_generations = wandb.Table(
                columns=["Prompt", "Solution", "Gen Response", "Correct"],
                rows=[
                    [prompt, solution, query_response[len(prompt) :], correct]
                    for prompt, solution, query_response, correct in zip(
                        prompts[:num_samples],
                        all_reference[:num_samples],
                        all_query_response[:num_samples],
                        results[:num_samples],
                    )
                ],
            )
            wandb.log(
                {
                    "eval/accuracy": accuracy,
                    "eval/ppl": mean_ppl,
                    "eval/samples": sample_generations,
                    "train/global_step": step,
                    "train/episode": episode,
                },
            )

        state.print(f"step {step}: accuracy {accuracy} ppl {mean_ppl}")


if __name__ == "__main__":
    parser = TRLParser([EvalScriptArguments])
    args = parser.parse_args_and_config()[0]

    generated_dataset_path = os.path.join(args.model_name_or_path, "_generations")
    dataset = load_from_disk(generated_dataset_path)

    with open(os.path.join(generated_dataset_path, "trainer_states.json"), "r") as f:
        trainer_states = json.load(f)

    prompts = dataset["question"]
    reference_answers = dataset.map(lambda example: {"answer": extract_answer(example["answer"])})["answer"]

    generations_cols = [name for name in dataset.column_names if name.startswith("generation")]
    generations = {}
    episodes = {}
    for col_name in generations_cols:
        # column name should be generations_{step name}
        checkpoint_name = col_name.split("_")[1]
        generations[checkpoint_name] = KeyDataset(dataset, col_name)
        if "episode" in trainer_states[checkpoint_name]:
            eps = trainer_states[checkpoint_name]["episode"]
        elif "dpo" in args.model_name_or_path:
            # assume offline dpo, which uses a pref dataset of 92858, although this is slightly off in practice
            eps = round(trainer_states[checkpoint_name]["epoch"] * 92858)
        else:
            # for sft and others
            eps = 0
        episodes[checkpoint_name] = eps

    if args.wandb_run_id == "slurm":
        # remove extra / at end
        normpath = os.path.normpath(args.model_name_or_path)
        path_parts = normpath.split("/")
        config_name = path_parts[-1]
        run_id = path_parts[-2]
        args.wandb_run_id = run_id + "_" + config_name

    if args.sanity_check:
        args.wandb_run_id = None
        first_ckpt = next(iter(generations.keys()))
        generations = {first_ckpt: generations[first_ckpt]}
        # generations[first_ckpt].dataset = generations[first_ckpt].dataset.select(range(100))
        # reference.dataset = reference.dataset.select(range(100))

    log_to_wandb = args.wandb_run_id is not None
    state = PartialState()
    if log_to_wandb and state.is_main_process:
        wandb.init(id=args.wandb_run_id, resume="allow")
        print(f"Logging to WandB {args.wandb_run_id}")

    evaluate(args, prompts, reference_answers, generations, episodes, log_to_wandb)
