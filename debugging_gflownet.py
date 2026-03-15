#!/usr/bin/env python3
"""
Minimal GFlowNet debugging runner for BBH5 tasks.

This script removes queue/offline-conditioning/m-step logic and keeps only:
1. fixed meta prompt + fixed few-shot context
2. prompt sampling from the policy
3. train accuracy reward measured with the repo-local BBH reasoning evaluator
4. Trajectory Balance update on the prompt policy
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_linear_schedule_with_warmup,
)
from vllm import LLM, SamplingParams

try:
    import wandb
except ImportError:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from junmo.bbh_eval_gfnpo import (
    ANSWER_EXTRACTION_STOPS,
    BBH5_TASKS,
    canonicalize_bbh_task,
    evaluate_prompts_chunked_bbh5_gfnpo,
    extract_bbh5_inputs_and_targets,
)
from junmo.dataset_utils import load_bigbench
from junmo.utils import JsonlLogger, load_eval_model_config, seed

SUPPORTED_TASKS = tuple(sorted(BBH5_TASKS))


class TopAccuracyTextsNoDuplicates:
    def __init__(self, max_size: int = 5):
        self.heap: List[tuple[float, int, str, int]] = []
        self.only_text: List[str] = []
        self.max_size = max_size

    def add(self, accuracy: float, text: str, step: int) -> bool:
        import heapq

        if text in self.only_text:
            return False
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (accuracy, len(text), text, step))
            self.only_text.append(text)
            return True
        if accuracy > self.heap[0][0]:
            removed_text = heapq.heappop(self.heap)[2]
            if removed_text in self.only_text:
                self.only_text.remove(removed_text)
            heapq.heappush(self.heap, (accuracy, len(text), text, step))
            self.only_text.append(text)
            return True
        return False

    def get_top_texts(self) -> List[tuple[float, str, int]]:
        return sorted([(acc, text, step) for acc, _, text, step in self.heap], reverse=True)


@dataclass
class TaskData:
    train_dataset: Any
    test_dataset: Any
    verbalizer: Any
    metrics: str
    task_prefix: str
    train_goals: List[str]
    train_final_targets: List[str]
    test_goals: List[str]
    test_final_targets: List[str]

def load_task_data(
    task_name: str,
    n_train_data: int,
    n_test_data: int,
    n_test_offset: int | None,
) -> TaskData:
    canonical_task = canonicalize_bbh_task(task_name)
    resolved_test_size = None if n_test_data is None or n_test_data < 0 else n_test_data
    metrics, train_dataset, test_dataset, verbalizer, task_prefix = load_bigbench(
        canonical_task,
        train_size=n_train_data,
        test_size=resolved_test_size,
        test_offset=n_test_offset,
    )
    train_goals, train_final_targets = extract_bbh5_inputs_and_targets(
        dataset=train_dataset,
        task_name=canonical_task,
        verbalizer=verbalizer,
    )
    test_goals, test_final_targets = extract_bbh5_inputs_and_targets(
        dataset=test_dataset,
        task_name=canonical_task,
        verbalizer=verbalizer,
    )
    return TaskData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        verbalizer=verbalizer,
        metrics=str(metrics),
        task_prefix=str(task_prefix),
        train_goals=list(train_goals),
        train_final_targets=list(train_final_targets),
        test_goals=list(test_goals),
        test_final_targets=list(test_final_targets),
    )


def choose_fewshot_indices(num_items: int, fewshot_count: int, strategy: str, seed_value: int, explicit: str) -> List[int]:
    if explicit:
        indices = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        for idx in indices:
            if idx < 0 or idx >= num_items:
                raise ValueError(f"few-shot index {idx} out of range 0..{num_items - 1}")
        return indices

    count = min(max(1, fewshot_count), num_items)
    if strategy == "first":
        return list(range(count))

    rng = random.Random(seed_value)
    return sorted(rng.sample(range(num_items), count))


def build_fixed_fewshot_block(goals: Sequence[str], final_targets: Sequence[str], indices: Sequence[int]) -> str:
    lines: List[str] = []
    for idx in indices:
        lines.append(str(goals[idx]).strip())
        lines.append(f"Output : {str(final_targets[idx]).strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def build_bbh_eval_prompt_payload(task_name: str, prompt_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    payload: Dict[str, Dict[str, str]] = {task_name: {}}
    for idx, row in enumerate(prompt_rows, start=1):
        payload[task_name][f"Debug-{idx:02d}"] = str(row["prompt"])
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

@torch.inference_mode()
def evaluate_prompts_with_bbh_eval(
    llm: Any,
    task_name: str,
    prompts: Sequence[str],
    dataset: Any,
    verbalizer: Any,
    reasoning_params: Any,
    answer_params: Any,
    chunk_size: int,
    reasoning_max_tokens: int,
    answer_max_tokens: int,
) -> torch.Tensor:
    return evaluate_prompts_chunked_bbh5_gfnpo(
        prompts=prompts,
        dataset=dataset,
        model=llm,
        task_name=canonicalize_bbh_task(task_name),
        verbalizer=verbalizer,
        chunk_size=chunk_size,
        reasoning_max_tokens=reasoning_max_tokens,
        answer_max_tokens=answer_max_tokens,
        reasoning_params=reasoning_params,
        answer_params=answer_params,
    )


class DebuggingGFlowNetRunner:
    def __init__(self, args: argparse.Namespace, task_name: str, run_root: Path):
        self.args = args
        self.task_name = task_name
        self.canonical_task_name = canonicalize_bbh_task(task_name)
        self.run_root = run_root
        self.task_root = run_root / task_name
        self.logs_root = self.task_root / "logs"
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.jsonl_logger = JsonlLogger(self.logs_root)
        self.queue = TopAccuracyTextsNoDuplicates(max_size=args.save_top_k)
        self.queue_prompt_metadata: Dict[str, Dict[str, Any]] = {}
        self.log_z_ema: float | None = None
        self.wandb_run = None
        self.policy_input_preview = ""

        self.task_data = load_task_data(
            task_name=task_name,
            n_train_data=args.bbh_train_size,
            n_test_data=args.bbh_test_size,
            n_test_offset=args.bbh_test_offset,
        )
        if args.beta is None:
            args.beta = 1.0 / float(len(self.task_data.train_goals))

        self.fewshot_indices = choose_fewshot_indices(
            num_items=len(self.task_data.train_goals),
            fewshot_count=args.num_example,
            strategy=args.fewshot_strategy,
            seed_value=args.fewshot_seed,
            explicit=args.fewshot_indices,
        )
        self.fewshot_block = build_fixed_fewshot_block(
            self.task_data.train_goals,
            self.task_data.train_final_targets,
            self.fewshot_indices,
        )
        self.fixed_context = self._build_fixed_context()
        self._init_wandb()

        self._init_policy_model()
        self._init_eval_model()
        self._save_run_config()

    def _build_fixed_context(self) -> str:
        parts = [str(self.args.meta_prompt).rstrip(), self.fewshot_block]
        return "\n".join(part for part in parts if part).strip()

    def _init_policy_model(self) -> None:
        self.device = self.args.agent_device if torch.cuda.is_available() else "cpu"
        config = AutoConfig.from_pretrained(self.args.agent_model)
        config.use_cache = True
        try:
            import flash_attn  # noqa: F401

            config._attn_implementation = "flash_attention_2"
        except Exception:
            config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.agent_model,
            cache_dir=self.args.cache_dir,
            torch_dtype=torch.bfloat16,
            config=config,
            device_map={"": self.device},
        )
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.agent_model, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        total_steps = self.args.train_steps * self.args.grad_acc_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            self.args.num_warmup_steps,
            total_steps,
        )

    def _init_eval_model(self) -> None:
        eval_model_config = AutoConfig.from_pretrained(self.args.eval_model_paths)
        try:
            gen_config = GenerationConfig.from_pretrained(self.args.eval_model_paths)
            eval_top_p = gen_config.top_p if gen_config.top_p is not None else 1.0
        except Exception:
            eval_top_p = 1.0

        llm_kwargs = {
            "model": self.args.eval_model_paths,
            "tokenizer": self.args.eval_model_paths,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "tensor_parallel_size": self.args.tp_size,
            "gpu_memory_utilization": self.args.eval_gpu_memory_utilization,
            "max_seq_len_to_capture": min(
                getattr(eval_model_config, "max_position_embeddings", 8192),
                getattr(self.args, "eval_max_seq_len_to_capture", 1024),
            ),
            "max_num_seqs": self.args.eval_max_num_seqs,
            "enable_prefix_caching": True,
        }
        while True:
            try:
                self.llm = LLM(**llm_kwargs)
                break
            except TypeError as exc:
                err = str(exc)
                if "unexpected keyword argument" not in err:
                    raise
                bad_key = err.split("'")[1]
                if bad_key not in llm_kwargs:
                    raise
                llm_kwargs.pop(bad_key, None)
                if bad_key == "max_seq_len_to_capture":
                    llm_kwargs["max_model_len"] = min(
                        getattr(eval_model_config, "max_position_embeddings", 8192),
                        getattr(self.args, "eval_max_seq_len_to_capture", 1024),
                    )

        self.reasoning_params = SamplingParams(
            temperature=0.0,
            top_p=eval_top_p,
            max_tokens=self.args.bbh_reasoning_max_tokens,
        )
        self.answer_params = SamplingParams(
            temperature=0.0,
            top_p=eval_top_p,
            max_tokens=self.args.bbh_answer_max_tokens,
            stop=ANSWER_EXTRACTION_STOPS,
        )

    def _init_wandb(self) -> None:
        if self.args.wandb_mode == "disabled":
            return
        if wandb is None:
            raise ImportError("wandb is not installed, but wandb logging is enabled.")

        wandb_config = dict(vars(self.args))
        wandb_config.update(
            {
                "task_name": self.task_name,
                "run_root": str(self.run_root),
                "task_root": str(self.task_root),
                "fewshot_indices": list(self.fewshot_indices),
                "train_size": len(self.task_data.train_goals),
                "test_size": len(self.task_data.test_goals),
            }
        )
        init_kwargs: Dict[str, Any] = {
            "project": self.args.wandb_project,
            "config": wandb_config,
            "name": f"{self.run_root.name}_{self.task_name}",
            "group": self.run_root.name,
            "mode": self.args.wandb_mode,
            "dir": str(self.task_root),
            "reinit": "finish_previous",
        }
        if self.args.wandb_entity:
            init_kwargs["entity"] = self.args.wandb_entity
        self.wandb_run = wandb.init(**init_kwargs)
        self.wandb_run.summary["task_name"] = self.task_name
        self.wandb_run.summary["run_root"] = str(self.run_root)

    def _save_run_config(self) -> None:
        chat_input = [
            {"role": "user", "content": self.fixed_context},
            {"role": "assistant", "content": "The Instruction is: "},
        ]
        self.policy_input_preview = self.tokenizer.apply_chat_template(
            chat_input,
            tokenize=False,
            add_generation_prompt=False,
        )
        config_payload = {
            "task_name": self.task_name,
            "args": vars(self.args),
            "fewshot_indices": self.fewshot_indices,
            "fewshot_block": self.fewshot_block,
            "fixed_context": self.fixed_context,
            "policy_input_preview": self.policy_input_preview,
            "train_size": len(self.task_data.train_goals),
            "test_size": len(self.task_data.test_goals),
            "task_prefix": self.task_data.task_prefix,
            "metrics": self.task_data.metrics,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(self.task_root / "config.json", config_payload)
        (self.task_root / "fixed_context.txt").write_text(self.fixed_context, encoding="utf-8")
        (self.task_root / "policy_input_preview.txt").write_text(self.policy_input_preview, encoding="utf-8")
        if self.wandb_run is not None:
            self.wandb_run.summary["fixed_context"] = self.fixed_context
            self.wandb_run.summary["policy_input_preview"] = self.policy_input_preview

    def _build_prompt_batch(self, num_samples: int) -> Dict[str, torch.Tensor]:
        chat_input = [
            {"role": "user", "content": self.fixed_context},
            {"role": "assistant", "content": "The Instruction is: "},
        ]
        encoded = self.tokenizer.apply_chat_template(
            chat_input,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(self.device)
        return {
            "input_ids": encoded.repeat(num_samples, 1),
            "attention_mask": torch.ones((num_samples, encoded.size(1)), dtype=torch.long, device=self.device),
        }

    @torch.inference_mode(True)
    def sample_prompts(self, prompt_batch: Dict[str, torch.Tensor]):
        if self.args.fixed_sampling_temp is not None:
            temp = float(self.args.fixed_sampling_temp)
        else:
            temp = random.uniform(self.args.temp_low, self.args.temp_high)

        prompt_responses = self.model.generate(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            do_sample=True,
            max_new_tokens=self.args.max_prompt_length,
            temperature=temp,
            top_p=self.args.sampling_top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = prompt_batch["input_ids"].size(1)
        only_responses = prompt_responses[:, prompt_len:]
        decoded_responses = self.tokenizer.batch_decode(only_responses, skip_special_tokens=True)
        decoded_responses = [str(text).strip() for text in decoded_responses]
        return prompt_responses, only_responses, decoded_responses, temp

    def get_logpf(self, prompt_batch: Dict[str, torch.Tensor], response_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_len = prompt_batch["input_ids"].size(1)
        concat_inputs = {
            key: torch.cat([prompt_batch[key], response_batch[key]], dim=1)
            for key in prompt_batch.keys()
        }
        outputs = self.model(**concat_inputs)
        logits = outputs.logits[:, prompt_len - 1 : -1]
        responses = response_batch["input_ids"]
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = torch.gather(log_prob, -1, responses.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob.masked_fill(response_batch["attention_mask"] == 0, 0.0)
        return torch.sum(log_prob, dim=1)

    @torch.inference_mode(True)
    def compute_log_acc_reward(self, acc: torch.Tensor) -> torch.Tensor:
        return torch.log(acc + self.args.reward_epsilon)

    def compute_tb_loss(self, log_z: torch.Tensor, sum_logpf: torch.Tensor, log_reward: torch.Tensor) -> torch.Tensor:
        return ((log_z + sum_logpf - log_reward) ** 2).mean()

    def _record_top_prompt(self, prompt: str, accuracy: float, step: int, log_reward: float) -> None:
        was_added = self.queue.add(accuracy, prompt, step)
        if was_added:
            self.queue_prompt_metadata[prompt] = {
                "train_acc": float(accuracy),
                "step": int(step),
                "log_reward": float(log_reward),
            }

    def _prepare_response_batch(
        self,
        input_ids: torch.Tensor,
        prompt_responses: torch.Tensor,
        only_responses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eos_tokens = torch.ones(input_ids.size(0), 1, dtype=torch.long, device=input_ids.device) * self.tokenizer.eos_token_id
        only_responses = torch.cat([only_responses, eos_tokens], dim=1)
        responses = only_responses.cpu()
        pad_mask = (responses == self.tokenizer.eos_token_id).cumsum(1) > 1
        response_lengths = torch.sum((~pad_mask).long(), dim=1)

        response_ids = []
        for idx in range(prompt_responses.size(0)):
            response_len = response_lengths[idx].item()
            response_ids.append(responses[idx, :response_len])

        response_mask = [torch.ones_like(x) for x in response_ids]
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id).to(self.device)
        response_mask = pad_sequence(response_mask, batch_first=True, padding_value=0).to(self.device)
        return {"input_ids": response_ids, "attention_mask": response_mask}

    def _evaluate_train_accs(self, prompts: Sequence[str]) -> torch.Tensor:
        return evaluate_prompts_with_bbh_eval(
            llm=self.llm,
            task_name=self.canonical_task_name,
            prompts=prompts,
            dataset=self.task_data.train_dataset,
            verbalizer=self.task_data.verbalizer,
            reasoning_params=self.reasoning_params,
            answer_params=self.answer_params,
            chunk_size=self.args.eval_chunk_size,
            reasoning_max_tokens=self.args.bbh_reasoning_max_tokens,
            answer_max_tokens=self.args.bbh_answer_max_tokens,
        )

    def _evaluate_test_prompt(self, prompt: str) -> float:
        if not self.task_data.test_goals:
            return 0.0
        acc = evaluate_prompts_with_bbh_eval(
            llm=self.llm,
            task_name=self.canonical_task_name,
            prompts=[prompt],
            dataset=self.task_data.test_dataset,
            verbalizer=self.task_data.verbalizer,
            reasoning_params=self.reasoning_params,
            answer_params=self.answer_params,
            chunk_size=self.args.eval_chunk_size,
            reasoning_max_tokens=self.args.bbh_reasoning_max_tokens,
            answer_max_tokens=self.args.bbh_answer_max_tokens,
        )
        return float(acc[0].item()) if len(acc) > 0 else 0.0

    def _export_prompt_payloads(self) -> None:
        top_rows = []
        for rank, (acc, prompt, step) in enumerate(self.queue.get_top_texts(), start=1):
            meta = self.queue_prompt_metadata.get(prompt, {})
            top_rows.append(
                {
                    "rank": rank,
                    "prompt": prompt,
                    "train_acc": float(acc),
                    "step": int(step),
                    "log_reward": meta.get("log_reward"),
                }
            )

        task_payload = build_bbh_eval_prompt_payload(self.task_name, top_rows)
        save_json(self.task_root / "bbh_eval_prompts.json", task_payload)
        save_json(
            self.task_root / "generated_meta_prompts" / f"{self.task_name}.json",
            {
                "task_name": self.task_name,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_config": {
                    "source": "debugging_gflownet",
                    "requested_num_prompts": len(top_rows),
                    "actual_num_prompts": len(top_rows),
                    "fewshot_indices": list(self.fewshot_indices),
                },
                "prompts": [
                    {
                        "name": f"Debug-{row['rank']:02d}",
                        "prompt": row["prompt"],
                    }
                    for row in top_rows
                ],
            },
        )
        save_json(self.task_root / "top_prompts.json", top_rows)

    def _log_wandb_step(
        self,
        global_step: int,
        summary: Dict[str, float],
        step_prompt_records: Sequence[Dict[str, Any]],
        top_prompts: Sequence[tuple[float, str, int]],
    ) -> None:
        if self.wandb_run is None:
            return

        self.wandb_run.log(dict(summary), step=global_step)
        if top_prompts:
            best_acc, best_prompt, best_step = top_prompts[0]
            self.wandb_run.summary["train/best_prompt"] = best_prompt
            self.wandb_run.summary["train/best_prompt_acc"] = float(best_acc)
            self.wandb_run.summary["train/best_prompt_step"] = int(best_step)

        if global_step % self.args.log_every != 0 or not step_prompt_records:
            return

        prompt_html = [
            f"<h3>{html.escape(self.task_name)} step {global_step}</h3>",
            "<details><summary>Fixed context</summary>",
            f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{html.escape(self.fixed_context)}</pre>",
            "</details>",
            "<table border='1'><tr><th>Micro</th><th>Idx</th><th>Prompt</th><th>Train Acc</th><th>Log Reward</th><th>LogPF</th><th>Temp</th><th>Best</th></tr>",
        ]
        for record in step_prompt_records:
            prompt_html.append(
                "<tr>"
                f"<td>{int(record['micro_step'])}</td>"
                f"<td>{int(record['prompt_idx'])}</td>"
                f"<td>{html.escape(str(record['prompt']))}</td>"
                f"<td>{float(record['train_acc']):.4f}</td>"
                f"<td>{float(record['log_reward']):.4f}</td>"
                f"<td>{float(record['sum_logpf']):.4f}</td>"
                f"<td>{float(record['sampling_temp']):.4f}</td>"
                f"<td>{'yes' if record['is_best_in_micro_step'] else 'no'}</td>"
                "</tr>"
            )
        prompt_html.append("</table>")
        self.wandb_run.log(
            {
                "sampling/prompts_html": wandb.Html("".join(prompt_html)),
            },
            step=global_step,
        )

    def _log_wandb_final(self, summary: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return

        for key, value in summary.items():
            self.wandb_run.summary[key] = value

        artifact = wandb.Artifact(
            name=f"{self.run_root.name}-{self.task_name}-results",
            type="debugging_gflownet_results",
        )
        artifact_paths = [
            self.task_root / "config.json",
            self.task_root / "fixed_context.txt",
            self.task_root / "policy_input_preview.txt",
            self.task_root / "summary.json",
            self.task_root / "top_prompts.json",
            self.task_root / "bbh_eval_prompts.json",
            self.task_root / "generated_meta_prompts" / f"{self.task_name}.json",
            self.logs_root / "metrics.jsonl",
            self.logs_root / "prompts.jsonl",
        ]
        for path in artifact_paths:
            if path.exists():
                artifact.add_file(str(path), name=str(path.relative_to(self.task_root)))
        self.wandb_run.log_artifact(artifact)

    def train(self) -> Dict[str, Any]:
        progress = tqdm(range(1, self.args.train_steps + 1), desc=f"{self.task_name}", dynamic_ncols=True)

        for global_step in progress:
            self.model.train()
            self.optimizer.zero_grad()
            batch_metrics: Dict[str, List[float]] = defaultdict(list)
            step_prompt_records: List[Dict[str, Any]] = []

            for micro_step in range(self.args.grad_acc_steps):
                prompt_batch = self._build_prompt_batch(self.args.batch_size)
                prompt_responses, only_responses, decoded_responses, sampling_temp = self.sample_prompts(prompt_batch)
                response_batch = self._prepare_response_batch(
                    input_ids=prompt_batch["input_ids"],
                    prompt_responses=prompt_responses,
                    only_responses=only_responses,
                )

                sum_logpf = self.get_logpf(prompt_batch, response_batch)
                train_acc = self._evaluate_train_accs(decoded_responses).to(sum_logpf.device)
                log_accuracy = self.compute_log_acc_reward(train_acc)
                log_reward = log_accuracy / self.args.beta

                log_z_batch = (log_reward - sum_logpf.detach()).mean()
                if self.log_z_ema is None:
                    self.log_z_ema = float(log_z_batch.item())
                else:
                    self.log_z_ema = (
                        self.args.ema_decay * self.log_z_ema + (1.0 - self.args.ema_decay) * float(log_z_batch.item())
                    )
                log_z = log_reward.new_full((log_reward.size(0),), self.log_z_ema)
                loss = self.compute_tb_loss(log_z, sum_logpf, log_reward)
                (loss / self.args.grad_acc_steps).backward()

                best_idx = int(train_acc.argmax().item())
                best_prompt = decoded_responses[best_idx]
                best_acc = float(train_acc[best_idx].item())
                best_log_reward = float(log_reward[best_idx].item())
                self._record_top_prompt(best_prompt, best_acc, global_step, best_log_reward)

                unique_prompt_ratio = float(len(set(decoded_responses)) / max(len(decoded_responses), 1))
                batch_metrics["train/batch_mean_acc"].append(float(train_acc.mean().item()))
                batch_metrics["train/batch_max_acc"].append(best_acc)
                batch_metrics["train/batch_mean_log_reward"].append(float(log_reward.mean().item()))
                batch_metrics["train/batch_mean_logpf"].append(float(sum_logpf.mean().item()))
                batch_metrics["train/batch_mean_log_accuracy"].append(float(log_accuracy.mean().item()))
                batch_metrics["train/batch_tb_loss"].append(float(loss.item()))
                batch_metrics["sampling/temp"].append(float(sampling_temp))
                batch_metrics["sampling/unique_prompt_ratio"].append(unique_prompt_ratio)
                batch_metrics["sampling/mean_prompt_chars"].append(
                    float(sum(len(x) for x in decoded_responses) / max(len(decoded_responses), 1))
                )

                for idx, prompt in enumerate(decoded_responses):
                    step_prompt_records.append(
                        {
                            "micro_step": micro_step,
                            "prompt_idx": idx,
                            "prompt": prompt,
                            "train_acc": float(train_acc[idx].item()),
                            "log_reward": float(log_reward[idx].item()),
                            "sum_logpf": float(sum_logpf[idx].item()),
                            "sampling_temp": float(sampling_temp),
                            "is_best_in_micro_step": bool(idx == best_idx),
                        }
                    )

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            self.optimizer.step()
            self.scheduler.step()

            summary = {key: sum(values) / float(len(values)) for key, values in batch_metrics.items()}
            top_prompts = self.queue.get_top_texts()
            summary["optim/grad_norm"] = float(grad_norm)
            summary["optim/lr"] = float(self.scheduler.get_last_lr()[0])
            summary["train/best_so_far_acc"] = float(top_prompts[0][0]) if top_prompts else 0.0
            summary["train/queue_size"] = float(len(top_prompts))
            summary["train/log_z_ema"] = float(self.log_z_ema if self.log_z_ema is not None else 0.0)

            run_test_eval = (
                self.args.test_eval_period > 0
                and self.task_data.test_goals
                and (global_step % self.args.test_eval_period == 0)
                and len(top_prompts) > 0
            )
            if run_test_eval:
                summary["eval/best_so_far_test_acc"] = self._evaluate_test_prompt(top_prompts[0][1])

            progress.set_postfix(
                train_best=f"{summary['train/best_so_far_acc']:.3f}",
                batch_max=f"{summary['train/batch_max_acc']:.3f}",
            )

            self.jsonl_logger.append(
                "metrics.jsonl",
                {
                    "global_step": int(global_step),
                    "metrics": summary,
                },
            )
            if global_step % self.args.log_every == 0:
                self.jsonl_logger.append(
                    "prompts.jsonl",
                    {
                        "global_step": int(global_step),
                        "fixed_context": self.fixed_context,
                        "prompts": step_prompt_records,
                    },
                )
            if global_step % self.args.export_every == 0:
                self._export_prompt_payloads()

            self._log_wandb_step(global_step, summary, step_prompt_records, top_prompts)

        self._export_prompt_payloads()

        top_prompts = self.queue.get_top_texts()
        summary = {
            "task_name": self.task_name,
            "best_train_acc": float(top_prompts[0][0]) if top_prompts else 0.0,
            "best_prompt": top_prompts[0][1] if top_prompts else "",
            "best_prompt_step": int(top_prompts[0][2]) if top_prompts else -1,
            "num_top_prompts": len(top_prompts),
        }
        if top_prompts and self.task_data.test_goals:
            summary["best_prompt_test_acc"] = self._evaluate_test_prompt(top_prompts[0][1])

        save_json(self.task_root / "summary.json", summary)
        self._log_wandb_final(summary)
        return summary

    def close(self) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
            self.wandb_run = None
        for attr_name in ["llm", "model", "tokenizer"]:
            if hasattr(self, attr_name):
                try:
                    delattr(self, attr_name)
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GFlowNet debugger for BBH5 prompt training")
    parser.add_argument("--task_names", type=str, nargs="+", default=["object_counting"])
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--eval_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--save_dir", type=str, default=str(REPO_ROOT / "debugging_gflownet_runs"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, choices=["disabled", "offline", "online"], default="online")
    parser.add_argument("--wandb_project", type=str, default="debugging_gflownet")
    parser.add_argument("--wandb_entity", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=4)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--num_warmup_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=150)
    parser.add_argument("--sampling_top_p", type=float, default=0.9)
    parser.add_argument("--temp_low", type=float, default=0.5)
    parser.add_argument("--temp_high", type=float, default=2.0)
    parser.add_argument("--fixed_sampling_temp", type=float, default=None)

    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--reward_epsilon", type=float, default=1e-8)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    parser.add_argument("--meta_prompt", type=str, default=(
        "I gave a friend an instruction and three inputs. "
        "The friend read the instruction and wrote an output for every one of the inputs. "
        "Here are the input-output pairs:"
    ))
    parser.add_argument("--num_example", type=int, default=5)
    parser.add_argument("--fewshot_strategy", type=str, choices=["first", "random"], default="first")
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--fewshot_indices", type=str, default="")

    parser.add_argument("--bbh_train_size", type=int, default=50)
    parser.add_argument("--bbh_test_size", type=int, default=100)
    parser.add_argument("--bbh_test_offset", type=int, default=None)
    parser.add_argument("--bbh_reasoning_max_tokens", type=int, default=50)
    parser.add_argument("--bbh_answer_max_tokens", type=int, default=1)
    parser.add_argument("--eval_chunk_size", type=int, default=64)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--agent_device", type=str, default="cuda:0")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--eval_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--eval_max_num_seqs", type=int, default=32)
    parser.add_argument("--eval_max_seq_len_to_capture", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--export_every", type=int, default=10)
    parser.add_argument("--test_eval_period", type=int, default=0)
    parser.add_argument("--save_top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_V1", "0")

    unknown_tasks = sorted({task for task in args.task_names if canonicalize_bbh_task(task) not in BBH5_TASKS})
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}. Supported: {sorted(SUPPORTED_TASKS)}")

    load_eval_model_config(args)
    seed(args.seed)

    run_name = args.run_name or f"debugging_gflownet_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.save_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    combined_prompt_payload: Dict[str, Dict[str, str]] = {}

    for task_name in args.task_names:
        task_args = argparse.Namespace(**vars(args))
        runner = DebuggingGFlowNetRunner(task_args, task_name=task_name, run_root=run_root)
        summary = runner.train()
        summaries.append(summary)

        payload_path = runner.task_root / "bbh_eval_prompts.json"
        with payload_path.open("r", encoding="utf-8") as f:
            task_payload = json.load(f)
        combined_prompt_payload.update(task_payload)
        runner.close()
        del runner

    save_json(run_root / "all_task_summaries.json", summaries)
    save_json(run_root / "bbh_eval_prompts_all_tasks.json", combined_prompt_payload)


if __name__ == "__main__":
    main()
