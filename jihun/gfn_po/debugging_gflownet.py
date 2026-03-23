#!/usr/bin/env python3
"""
Minimal GFlowNet debugging runner for BBH tasks.

This script removes queue/offline-conditioning/m-step logic and keeps only:
1. fixed meta prompt + fixed few-shot context
2. prompt sampling from the policy
3. train accuracy reward measured with bbh_vllm_eval-style evaluation
4. Trajectory Balance update on the prompt policy

Outputs include per-step metrics and prompt payloads that can be loaded by
bbh_vllm_eval/main.py via --meta_prompt_file.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import random
import re
import math
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
BBH_EVAL_ROOT = REPO_ROOT.parent / "bbh_vllm_eval"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from junmo.bbh_eval_gfnpo import canonicalize_bbh_task
from junmo.utils import JsonlLogger, load_eval_model_config, seed


def _load_bbh_eval_utils():
    module_path = BBH_EVAL_ROOT / "utils.py"
    spec = importlib.util.spec_from_file_location("bbh_eval_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load bbh_vllm_eval utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bbh_eval_utils = _load_bbh_eval_utils()
SUPPORTED_TASKS = tuple(bbh_eval_utils.SUPPORTED_TASKS)


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
    train_goals: List[str]
    train_final_targets: List[str]
    test_goals: List[str]
    test_final_targets: List[str]


def default_data_root() -> str:
    return str((BBH_EVAL_ROOT / "data" / "GreaTer_data" / "BBH").resolve())


def canonicalize_task_names(task_names: Sequence[str]) -> List[str]:
    return [canonicalize_bbh_task(task_name) for task_name in task_names]


def load_task_data(
    task_name: str,
    data_root: str,
    conversation_template: str,
    n_train_data: int,
    n_test_data: int,
) -> TaskData:
    data_file = Path(data_root) / f"{task_name}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing BBH task file: {data_file}")

    extractor_text = bbh_eval_utils.TASK_EXTRACTOR_TEXT[task_name]
    train_goals, _, test_goals, _, train_final_targets, test_final_targets = bbh_eval_utils.get_goals_and_targets(
        data_path=str(data_file),
        extractor_text=extractor_text,
        conversation_template_name=conversation_template,
        n_train_data=n_train_data,
        n_test_data=n_test_data,
    )
    return TaskData(
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


def _compose_user_content(goal: str, control: str) -> str:
    goal = str(goal)
    control = str(control).lstrip()
    if not control:
        return goal
    return f"{goal} {control}"


_LEADING_ROLE_ARTIFACT_RE = re.compile(
    r"^(?:<\|start_header_id\|>\s*)?(?:assistant|model)(?:\s*<\|end_header_id\|>)?\s*[:\-]?\s*",
    flags=re.IGNORECASE,
)


def strip_leading_role_artifacts(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    # Generated prompts should only contain the assistant continuation, but
    # some samples still echo the chat role prefix back as plain text.
    cleaned = (
        cleaned.replace("<|begin_of_text|>", " ")
        .replace("<|eot_id|>", " ")
        .replace("<|start_header_id|>", " ")
        .replace("<|end_header_id|>", " ")
        .strip()
    )
    while True:
        next_cleaned = _LEADING_ROLE_ARTIFACT_RE.sub("", cleaned, count=1).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned

    return cleaned


@torch.inference_mode()
def evaluate_prompts_with_bbh_eval(
    llm: Any,
    task_name: str,
    prompts: Sequence[str],
    goals: Sequence[str],
    final_targets: Sequence[str],
    reasoning_params: Any,
    answer_params: Any,
    chunk_size: int,
) -> torch.Tensor:
    num_prompts = len(prompts)
    if num_prompts == 0:
        return torch.zeros(0, dtype=torch.float32)

    dataset_len = len(goals)
    if dataset_len == 0:
        return torch.zeros(num_prompts, dtype=torch.float32)

    extractor_text = bbh_eval_utils.TASK_EXTRACTOR_TEXT[task_name]
    normalized_targets = [
        bbh_eval_utils.remove_parentheses_if_single_char(str(target).strip())
        for target in final_targets
    ]
    correct = torch.zeros(num_prompts, dtype=torch.long)
    total = num_prompts * dataset_len
    chunk_size = max(1, int(chunk_size))

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        stage1_prompts: List[str] = []
        pairs: List[tuple[int, int]] = []

        for global_idx in range(start, end):
            prompt_idx = global_idx // dataset_len
            data_idx = global_idx % dataset_len
            user_content = _compose_user_content(goals[data_idx], prompts[prompt_idx])
            stage1_prompts.append(bbh_eval_utils.render_llama3_user_prompt(user_content))
            pairs.append((prompt_idx, data_idx))

        reasoning_outputs = llm.generate(stage1_prompts, reasoning_params, use_tqdm=False)
        reasonings = [out.outputs[0].text if out.outputs else "" for out in reasoning_outputs]

        extractor_prompts = [
            bbh_eval_utils.render_extractor_prompt_llama3(stage1_prompt, reasoning, extractor_text)
            for stage1_prompt, reasoning in zip(stage1_prompts, reasonings)
        ]
        answer_outputs = llm.generate(extractor_prompts, answer_params, use_tqdm=False)

        for idx, output in enumerate(answer_outputs):
            prompt_idx, data_idx = pairs[idx]
            raw_answer = output.outputs[0].text if output.outputs else ""
            normalized_answer = bbh_eval_utils.remove_parentheses_if_single_char(str(raw_answer).strip())
            if normalized_answer == normalized_targets[data_idx]:
                correct[prompt_idx] += 1

    return correct.float() / float(dataset_len)


class DebuggingGFlowNetRunner:
    def __init__(self, args: argparse.Namespace, task_name: str, run_root: Path):
        self.args = args
        self.task_name = task_name
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
            data_root=args.data_root,
            conversation_template=args.conversation_template,
            n_train_data=args.bbh_train_size,
            n_test_data=args.bbh_test_size,
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
            stop=bbh_eval_utils.ANSWER_EXTRACTION_STOPS,
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
        self.policy_input_preview = self._build_policy_prefix_text(self.fixed_context)
        config_payload = {
            "task_name": self.task_name,
            "args": vars(self.args),
            "fewshot_indices": self.fewshot_indices,
            "fewshot_block": self.fewshot_block,
            "fixed_context": self.fixed_context,
            "policy_input_preview": self.policy_input_preview,
            "train_size": len(self.task_data.train_goals),
            "test_size": len(self.task_data.test_goals),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(self.task_root / "config.json", config_payload)
        (self.task_root / "fixed_context.txt").write_text(self.fixed_context, encoding="utf-8")
        (self.task_root / "policy_input_preview.txt").write_text(self.policy_input_preview, encoding="utf-8")
        if self.wandb_run is not None:
            self.wandb_run.summary["fixed_context"] = self.fixed_context
            self.wandb_run.summary["policy_input_preview"] = self.policy_input_preview

    def _build_policy_prefix_text(self, context_text: str) -> str:
        context_text = str(context_text).strip()
        if not context_text:
            context_text = ""

        if not context_text:
            return "The Instruction is: "
        return f"{context_text}\n\nThe Instruction is: "

    def _build_prompt_batch(self, num_samples: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self._build_policy_prefix_text(self.fixed_context),
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)
        return {
            "input_ids": encoded["input_ids"].repeat(num_samples, 1),
            "attention_mask": encoded["attention_mask"].repeat(num_samples, 1),
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
        decoded_responses = [strip_leading_role_artifacts(text) for text in decoded_responses]
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
        prompt = str(prompt).strip()
        if not prompt:
            return
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
            task_name=self.task_name,
            prompts=prompts,
            goals=self.task_data.train_goals,
            final_targets=self.task_data.train_final_targets,
            reasoning_params=self.reasoning_params,
            answer_params=self.answer_params,
            chunk_size=self.args.eval_chunk_size,
        )

    def _evaluate_test_prompt(self, prompt: str) -> float:
        if not self.task_data.test_goals:
            return 0.0
        acc = evaluate_prompts_with_bbh_eval(
            llm=self.llm,
            task_name=self.task_name,
            prompts=[prompt],
            goals=self.task_data.test_goals,
            final_targets=self.task_data.test_final_targets,
            reasoning_params=self.reasoning_params,
            answer_params=self.answer_params,
            chunk_size=self.args.eval_chunk_size,
        )
        return float(acc[0].item()) if len(acc) > 0 else 0.0

    def _evaluate_test_prompts(self, prompts: Sequence[str]) -> List[float]:
        if not self.task_data.test_goals or not prompts:
            return []
        acc = evaluate_prompts_with_bbh_eval(
            llm=self.llm,
            task_name=self.task_name,
            prompts=list(prompts),
            goals=self.task_data.test_goals,
            final_targets=self.task_data.test_final_targets,
            reasoning_params=self.reasoning_params,
            answer_params=self.answer_params,
            chunk_size=self.args.eval_chunk_size,
        )
        return [float(x.item()) for x in acc]

    def _build_test_eval_metrics(
        self,
        top_prompts: Sequence[tuple[float, str, int]],
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        if not top_prompts or not self.task_data.test_goals:
            return {}, []

        prompt_texts = [prompt for _, prompt, _ in top_prompts]
        test_accs = self._evaluate_test_prompts(prompt_texts)
        if not test_accs:
            return {}, []

        mean_test_acc = float(sum(test_accs) / len(test_accs))
        variance = float(sum((x - mean_test_acc) ** 2 for x in test_accs) / len(test_accs))
        eval_metrics: Dict[str, float] = {
            "eval/best_so_far_test_acc": float(test_accs[0]),
            "eval/top1_test_acc": float(test_accs[0]),
            "eval/top_prompts_count": float(len(test_accs)),
            "eval/top_prompts_mean_test_acc": mean_test_acc,
            "eval/top_prompts_max_test_acc": float(max(test_accs)),
            "eval/top_prompts_min_test_acc": float(min(test_accs)),
            "eval/top_prompts_std_test_acc": float(math.sqrt(max(variance, 0.0))),
        }
        for rank, test_acc in enumerate(test_accs, start=1):
            eval_metrics[f"eval/top{rank}_test_acc"] = float(test_acc)

        eval_prompt_records: List[Dict[str, Any]] = []
        for rank, ((train_acc, prompt, step), test_acc) in enumerate(zip(top_prompts, test_accs), start=1):
            eval_prompt_records.append(
                {
                    "rank": int(rank),
                    "prompt": prompt,
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "step": int(step),
                }
            )
        return eval_metrics, eval_prompt_records

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
        eval_prompt_records: Sequence[Dict[str, Any]] | None = None,
    ) -> None:
        if self.wandb_run is None:
            return

        monitor_metrics: Dict[str, Any] = {
            "monitor/global_step": int(global_step),
            "monitor/progress_fraction": float(global_step) / float(max(self.args.train_steps, 1)),
        }
        metric_aliases = {
            "train/batch_mean_acc": "monitor/train_batch_mean_acc",
            "train/batch_max_acc": "monitor/train_batch_max_acc",
            "train/mean_acc": "monitor/train_mean_acc",
            "train/max_acc": "monitor/train_max_acc",
            "train/best_so_far_acc": "monitor/train_best_so_far_acc",
            "train/batch_mean_log_reward": "monitor/train_batch_mean_log_reward",
            "train/batch_mean_logpf": "monitor/train_batch_mean_logpf",
            "optim/grad_norm": "monitor/grad_norm",
            "optim/lr": "monitor/lr",
            "sampling/temp": "monitor/sampling_temp",
            "sampling/mean_prompt_chars": "monitor/mean_prompt_chars",
            "sampling/unique_prompt_ratio": "monitor/unique_prompt_ratio",
            "train/queue_size": "monitor/queue_size",
            "train/log_z_ema": "monitor/log_z_ema",
            "eval/top_prompts_mean_test_acc": "monitor/eval_top_prompts_mean_test_acc",
            "eval/top_prompts_max_test_acc": "monitor/eval_top_prompts_max_test_acc",
            "eval/top_prompts_min_test_acc": "monitor/eval_top_prompts_min_test_acc",
            "eval/top_prompts_std_test_acc": "monitor/eval_top_prompts_std_test_acc",
        }
        for src_key, dst_key in metric_aliases.items():
            if src_key in summary:
                monitor_metrics[dst_key] = float(summary[src_key])
        if "eval/best_so_far_test_acc" in summary:
            monitor_metrics["monitor/eval_best_so_far_test_acc"] = float(summary["eval/best_so_far_test_acc"])

        self.wandb_run.log({**dict(summary), **monitor_metrics}, step=global_step)
        self.wandb_run.summary["monitor/last_step"] = int(global_step)
        if "monitor/train_best_so_far_acc" in monitor_metrics:
            self.wandb_run.summary["monitor/latest_train_best_so_far_acc"] = float(
                monitor_metrics["monitor/train_best_so_far_acc"]
            )
        if "monitor/eval_best_so_far_test_acc" in monitor_metrics:
            self.wandb_run.summary["monitor/latest_eval_best_so_far_test_acc"] = float(
                monitor_metrics["monitor/eval_best_so_far_test_acc"]
            )
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
        if not eval_prompt_records:
            return

        eval_html = [
            f"<h3>{html.escape(self.task_name)} step {global_step} eval</h3>",
            "<table border='1'><tr><th>Rank</th><th>Prompt</th><th>Train Acc</th><th>Test Acc</th><th>Step</th></tr>",
        ]
        for record in eval_prompt_records:
            eval_html.append(
                "<tr>"
                f"<td>{int(record['rank'])}</td>"
                f"<td>{html.escape(str(record['prompt']))}</td>"
                f"<td>{float(record['train_acc']):.4f}</td>"
                f"<td>{float(record['test_acc']):.4f}</td>"
                f"<td>{int(record['step'])}</td>"
                "</tr>"
            )
        eval_html.append("</table>")
        self.wandb_run.log(
            {
                "eval/top_prompts_html": wandb.Html("".join(eval_html)),
            },
            step=global_step,
        )

    def _log_wandb_final(self, summary: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return

        for key, value in summary.items():
            self.wandb_run.summary[key] = value
        if "best_train_acc" in summary:
            self.wandb_run.summary["monitor/final_best_train_acc"] = float(summary["best_train_acc"])
        if "best_prompt_test_acc" in summary:
            self.wandb_run.summary["monitor/final_best_prompt_test_acc"] = float(summary["best_prompt_test_acc"])
        self.wandb_run.summary["monitor/final_num_top_prompts"] = int(summary.get("num_top_prompts", 0))

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
            self.task_root / "top_prompts_eval.json",
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
                batch_metrics["train/batch_max_log_reward"].append(float(log_reward.max().item()))
                batch_metrics["train/batch_mean_logpf"].append(float(sum_logpf.mean().item()))
                batch_metrics["train/batch_mean_log_accuracy"].append(float(log_accuracy.mean().item()))
                batch_metrics["train/batch_max_log_accuracy"].append(float(log_accuracy.max().item()))
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
            summary["train/mean_acc"] = float(summary["train/batch_mean_acc"])
            summary["train/max_acc"] = float(summary["train/batch_max_acc"])
            summary["train/mean_log_reward"] = float(summary["train/batch_mean_log_reward"])
            summary["train/max_log_reward"] = float(summary["train/batch_max_log_reward"])
            summary["train/mean_log_accuracy"] = float(summary["train/batch_mean_log_accuracy"])
            summary["train/max_log_accuracy"] = float(summary["train/batch_max_log_accuracy"])
            summary["sampling/queue_size"] = float(len(top_prompts))
            summary["sampling/queue_best_acc"] = float(top_prompts[0][0]) if top_prompts else 0.0

            eval_prompt_records: List[Dict[str, Any]] = []
            run_test_eval = (
                self.args.test_eval_period > 0
                and self.task_data.test_goals
                and (global_step % self.args.test_eval_period == 0)
                and len(top_prompts) > 0
            )
            if run_test_eval:
                eval_summary, eval_prompt_records = self._build_test_eval_metrics(top_prompts)
                summary.update(eval_summary)

            progress.set_postfix(
                train_best=f"{summary['train/best_so_far_acc']:.3f}",
                mean_acc=f"{summary['train/mean_acc']:.3f}",
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

            self._log_wandb_step(global_step, summary, step_prompt_records, top_prompts, eval_prompt_records)

        self._export_prompt_payloads()

        top_prompts = self.queue.get_top_texts()
        summary = {
            "task_name": self.task_name,
            "best_train_acc": float(top_prompts[0][0]) if top_prompts else 0.0,
            "best_prompt": top_prompts[0][1] if top_prompts else "",
            "best_prompt_step": int(top_prompts[0][2]) if top_prompts else -1,
            "num_top_prompts": len(top_prompts),
        }
        final_eval_records: List[Dict[str, Any]] = []
        if top_prompts and self.task_data.test_goals:
            eval_summary, final_eval_records = self._build_test_eval_metrics(top_prompts)
            summary.update(eval_summary)
            summary["best_prompt_test_acc"] = float(eval_summary.get("eval/top1_test_acc", 0.0))

        if final_eval_records:
            save_json(self.task_root / "top_prompts_eval.json", final_eval_records)

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
    parser = argparse.ArgumentParser(description="Minimal GFlowNet debugger for BBH prompt training")
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
        "I gave a friend an instruction and five inputs. "
        "The friend read the instruction and wrote an output for every one of the inputs. "
        "Here are the input-output pairs:"
    ))
    parser.add_argument("--num_example", type=int, default=5)
    parser.add_argument("--fewshot_strategy", type=str, choices=["first", "random"], default="first")
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--fewshot_indices", type=str, default="")

    parser.add_argument("--data_root", type=str, default=default_data_root())
    parser.add_argument("--conversation_template", type=str, default="llama-3")
    parser.add_argument("--bbh_train_size", type=int, default=50)
    parser.add_argument("--bbh_test_size", type=int, default=100)
    parser.add_argument("--bbh_reasoning_max_tokens", type=int, default=1024)
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

    args.task_names = canonicalize_task_names(args.task_names)

    unknown_tasks = sorted(set(args.task_names) - set(SUPPORTED_TASKS))
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
