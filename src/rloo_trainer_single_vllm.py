import gc
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.rloo_trainer import INVALID_LOGPROB
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    get_reward,
    print_rich_table,
    truncate_response,
)
from vllm import LLM, SamplingParams

from src.rloo_trainer import MyRLOOConfig, OnlineTrainerState
from src.utils import prepare_deepspeed


@dataclass
class RLOOVLLMConfig(MyRLOOConfig):
    sync: bool = True
    vllm_device: str = None
    "default will put it on accelerate.num_processes + 1"
    vllm_gpu_memory_utilization: float = 0.9


class RLOOSingleVLLMTrainer(Trainer):
    def __init__(
        self,
        config: RLOOVLLMConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator if data_collator is not None else DataCollatorWithPadding(tokenizer)
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = args.num_train_epochs * self.train_dataset_len
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        self.num_batches = exact_div(
            args.total_episodes,
            args.batch_size,
            f" total_episodes {args.total_episodes} should be divisible by batch_size {args.batch_size} ",
        )
        args.num_updates = self.num_batches * args.num_mini_batches
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, self.num_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size,
            args.rloo_k,
            "`local_batch_size` must be a multiple of rloo_k",
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(num_training_steps=self.num_batches)

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks,
            self.model,
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, config.fp16, config.bf16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, config.fp16, config.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        self.state.global_step = 0
        self.state.episode = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)

        model.train()
        self.state.max_steps = args.num_updates
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        sampling_params = SamplingParams(
            temperature=(args.temperature + 1e-7),
            top_p=args.top_p,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
            # logprobs=1,
        )

        if accelerator.is_main_process:
            if args.fp16:
                vllm_dtype = torch.float16
            elif args.bf16:
                vllm_dtype = torch.bfloat16
            else:
                vllm_dtype = torch.float32
            vllm_device = args.vllm_device or f"cuda:{accelerator.num_processes}"
            llm = LLM(
                model=args.sft_model_path,
                revision="main",
                tokenizer_revision="main",
                tensor_parallel_size=1,
                device=vllm_device,
                dtype=vllm_dtype,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            )
            accelerator.print(f"🔥🔥🔥 vllm loaded in {vllm_dtype}")

        if self.args.sync:
            next_queries = None
            next_labels = None
        else:
            # send first batch of data to actor
            data = next(iter_dataloader)
            next_queries = data["input_ids"].to(device)
            next_queries = next_queries.repeat(args.rloo_k, 1)
            next_labels = data.get("labels", None)
            if next_labels is not None:
                next_labels = next_labels.repeat(args.rloo_k, 1)
            next_g_queries_list = gather_object(next_queries.tolist())
            if accelerator.is_main_process:
                next_g_queries_list = [
                    [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]
                    for item in next_g_queries_list
                ]  # remove padding
                next_g_response_ids, next_g_response_lprobs = vllm_generate(
                    llm, sampling_params, None, next_g_queries_list, True
                )

        DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        for update in range(1, self.num_batches + 1):
            self.state.episode += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            vllm_responses = torch.zeros(
                (args.batch_size, args.response_length),
                device=accelerator.device,
                dtype=torch.long,
            )
            vllm_response_lprobs = torch.zeros(
                (args.batch_size, args.response_length),
                device=accelerator.device,
                dtype=torch.float,
            )
            if not args.sync:
                queries = next_queries
                labels = next_labels
                g_response_ids = next_g_response_ids
                g_padded_response_ids = [
                    list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                    for response in g_response_ids
                ]
                g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                vllm_responses[:] = g_padded_response_ids

                if next_g_response_lprobs is not None:
                    g_response_lprobs = next_g_response_lprobs
                    g_padded_response_lprobs = [
                        list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in g_response_lprobs
                    ]
                    g_padded_response_lprobs = torch.tensor(g_padded_response_lprobs, device=device)
                    vllm_response_lprobs[:] = g_padded_response_lprobs

            with torch.no_grad():
                next_queries = data["input_ids"].to(device)
                next_queries = next_queries.repeat(args.rloo_k, 1)
                next_labels = data.get("labels", None)
                if next_labels is not None:
                    next_labels = next_labels.repeat(args.rloo_k, 1)

                # with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                next_g_queries_list = gather_object(next_queries.tolist())
                if accelerator.is_main_process:
                    next_g_queries_list = [
                        [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]
                        for item in next_g_queries_list
                    ]  # remove padding

                    # send next queries to be generated
                    model_named_parameters = accelerator._get_named_parameters(model)
                    next_g_response_ids, next_g_response_lprobs = vllm_generate(
                        llm,
                        sampling_params,
                        model_named_parameters.items(),
                        next_g_queries_list,
                        log=(update % self.state.logging_steps == 0),
                    )

                    if args.sync:
                        queries = next_queries
                        labels = next_labels
                        g_response_ids = next_g_response_ids
                        g_padded_response_ids = [
                            list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                            for response in g_response_ids
                        ]
                        g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                        vllm_responses[:] = g_padded_response_ids

                        if next_g_response_lprobs is not None:
                            g_response_lprobs = next_g_response_lprobs
                            g_padded_response_lprobs = [
                                list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                                for response in g_response_lprobs
                            ]
                            g_padded_response_lprobs = torch.tensor(g_padded_response_lprobs, device=device)
                            vllm_response_lprobs[:] = g_padded_response_lprobs

                local_vllm_responses = vllm_responses[
                    accelerator.local_process_index * queries.shape[0] : (accelerator.local_process_index + 1)
                    * queries.shape[0]
                ]
                local_vllm_responses_lprobs = vllm_response_lprobs[
                    accelerator.local_process_index * queries.shape[0] : (accelerator.local_process_index + 1)
                    * queries.shape[0]
                ]

                query_responses = torch.cat((queries, local_vllm_responses), 1)
                context_length = queries.shape[1]
                responses = local_vllm_responses
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    if labels is not None:
                        label = labels[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    # use the logits during generation directly, instead of using the following
                    # all_logprob = F.log_softmax(logits, dim=-1)
                    # logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    # del logits, all_logprob
                    # torch.cuda.empty_cache()
                    # logprob = local_vllm_responses_lprobs[i : i + args.local_rollout_forward_batch_size]

                    # recalculate logprob for stability
                    output = forward(model, query_response, tokenizer.pad_token_id)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del output, logits, all_logprob
                    gc.collect()
                    torch.cuda.empty_cache()

                    ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    if label is not None:
                        score = reward_model(postprocessed_response, label)
                    else:
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                        )

                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                # responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = (-args.kl_coef * kl).sum(1)
                rlhf_reward = scores + non_score_reward

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            loss = pg_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.control.should_save:
                        self._save_checkpoint(model, trial=None, metrics=None)
                        self.control = self.callback_handler.on_save(self.args, self.state, self.control)
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        loss, prob_dist, entropy,
                        mb_advantage, mb_responses, mb_query_responses,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = generate(
                        unwrapped_model,
                        query,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = get_reward(
                    self.reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                )
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})


def vllm_generate(llm, sampling_params, model_named_parameters, g_queries_list, log=False):
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    if model_named_parameters is None and g_queries_list is None:
        print("model params and queries are None, exiting")
        return

    vllm_start_time = time.time()
    if model_named_parameters:
        # print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")
        llmp.load_weights(model_named_parameters)
        # if log:
        #     print(f"load weights took: {time.time() - vllm_start_time:.2f} seconds")

    outputs = llm.generate(prompt_token_ids=g_queries_list, sampling_params=sampling_params, use_tqdm=False)
    if log:
        print(
            f"🏃🏃🏃 load and gen of {len(g_queries_list)} prompts took: {time.time() - vllm_start_time:.2f} seconds"
        )

    return_logprobs = sampling_params.logprobs is not None and sampling_params.logprobs > 0
    response_token_ids = []
    response_token_logprobs = [] if return_logprobs else None
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        response_token_ids.append(token_ids)

        if return_logprobs:
            all_logprobs = output.outputs[0].logprobs
            token_logprobs = [logprob[token].logprob for logprob, token in zip(all_logprobs, token_ids)]
            response_token_logprobs.append(token_logprobs)

    return response_token_ids, response_token_logprobs
