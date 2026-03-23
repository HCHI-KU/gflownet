#!/usr/bin/env python3
"""
Incremental variants of debugging_gflownet.py.

v1: add log_prior reward term
v2: add log_prior + train_buffer offline replay
v3: add log_prior + train_buffer replay + reference conditioning
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import debugging_gflownet as base
from junmo.utils import base_to_lora, load_eval_model_config, lora_to_base, seed


VARIANT_CONFIGS = {
    "v1_log_prior": {
        "enable_log_prior": True,
        "enable_buffer": False,
        "enable_conditioning": False,
    },
    "v2_log_prior_buffer": {
        "enable_log_prior": True,
        "enable_buffer": True,
        "enable_conditioning": False,
    },
    "v3_log_prior_buffer_condition": {
        "enable_log_prior": True,
        "enable_buffer": True,
        "enable_conditioning": True,
    },
}


class IncrementalFeatureRunner(base.DebuggingGFlowNetRunner):
    def __init__(
        self,
        args: argparse.Namespace,
        task_name: str,
        run_root: Path,
        *,
        enable_log_prior: bool,
        enable_buffer: bool,
        enable_conditioning: bool,
    ):
        self.enable_log_prior = bool(enable_log_prior)
        self.enable_buffer = bool(enable_buffer)
        self.enable_conditioning = bool(enable_conditioning)

        self.prompt_buffer: List[Dict[str, Any]] = []
        self.train_buffer: List[tuple[float, float, Dict[str, Any]]] = []
        self.condition_buffer: List[Dict[str, Any]] = []
        self.condition_buffer_created = False

        super().__init__(args=args, task_name=task_name, run_root=run_root)

        self.train_buffer_max_size = int(getattr(args, "train_buffer_max_size", 0) or 0)
        self.use_offline_sampling = self.enable_buffer and bool(getattr(args, "use_offline_sampling", False))
        self.online_ratio = float(getattr(args, "online_ratio", 1.0))
        self.offline_start_step = int(getattr(args, "offline_start_step", 0))
        self.use_condition_buffer = self.enable_conditioning and bool(getattr(args, "condition_buffer", False))
        self.fewshot_resample_each_step = bool(getattr(args, "fewshot_resample_each_step", False))
        self.fewshot_rng = random.Random(int(getattr(args, "fewshot_seed", getattr(args, "seed", 42))))

        if self.enable_buffer and getattr(args, "init_queue_prompt", None):
            seed_prompt = base.strip_leading_role_artifacts(str(args.init_queue_prompt))
            if seed_prompt:
                self.queue.add(0.0, seed_prompt, 0)

        train_buffer_path = getattr(args, "train_buffer_path", "")
        if self.enable_buffer and train_buffer_path:
            self._load_train_buffer(Path(train_buffer_path))

    def _compose_fixed_context(self, fewshot_block: str) -> str:
        parts = [str(self.args.meta_prompt).rstrip(), str(fewshot_block).strip()]
        return "\n".join(part for part in parts if part).strip()

    def _sample_step_fewshot_block(self) -> Dict[str, Any]:
        if not self.fewshot_resample_each_step:
            return {
                "fewshot_block": self.fewshot_block,
                "fewshot_indices": list(self.fewshot_indices),
                "fewshot_mode": "fixed",
            }

        train_size = len(self.task_data.train_goals)
        if train_size <= 0:
            return {
                "fewshot_block": "",
                "fewshot_indices": [],
                "fewshot_mode": "random_with_replacement",
            }

        shot = min(max(1, int(self.args.num_example)), train_size)
        sampled_indices: List[int] = []
        lines: List[str] = []
        for _ in range(shot):
            idx = self.fewshot_rng.randrange(train_size)
            sampled_indices.append(int(idx))
            lines.append(str(self.task_data.train_goals[idx]).strip())
            lines.append(f"Output : {str(self.task_data.train_final_targets[idx]).strip()}")
            lines.append("")

        return {
            "fewshot_block": "\n".join(lines).strip(),
            "fewshot_indices": sampled_indices,
            "fewshot_mode": "random_with_replacement",
        }

    def _build_step_context(self) -> Dict[str, Any]:
        fewshot_payload = self._sample_step_fewshot_block()
        fixed_context = self._compose_fixed_context(fewshot_payload["fewshot_block"])
        return {
            **fewshot_payload,
            "fixed_context": fixed_context,
            "fewshot_block_chars": len(fewshot_payload["fewshot_block"]),
            "fixed_context_chars": len(fixed_context),
        }

    def _build_prompt_batch_with_context(self, context_text: str, num_samples: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self._build_policy_prefix_text(context_text),
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)
        return {
            "input_ids": encoded["input_ids"].repeat(num_samples, 1),
            "attention_mask": encoded["attention_mask"].repeat(num_samples, 1),
        }

    def _build_prompt_context(self, global_step: int, base_context: str) -> Dict[str, Any]:
        full_context = base_context
        condition_text = ""
        reference_prompts: List[str] = []

        if self.enable_conditioning and global_step > (self.offline_start_step + 10):
            queue_texts = [item[1] for item in self.queue.get_top_texts()]
            num_from_queue = min(int(getattr(self.args, "condition_queue_samples", 1)), len(queue_texts))
            queue_samples = random.sample(queue_texts, num_from_queue) if num_from_queue > 0 else []

            if self.use_condition_buffer and self.condition_buffer:
                buffer_source = self.condition_buffer
            else:
                buffer_source = self.get_train_buffer_as_list()
            buffer_texts = [item["prompt"] for item in buffer_source]
            num_from_buffer = min(int(getattr(self.args, "condition_buffer_samples", 2)), len(buffer_texts))
            buffer_samples = random.sample(buffer_texts, num_from_buffer) if num_from_buffer > 0 else []

            seen = set()
            for prompt in queue_samples + buffer_samples:
                cleaned = base.strip_leading_role_artifacts(prompt)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    reference_prompts.append(cleaned)

            if reference_prompts:
                lines = ["Here are some reference instructions:"]
                lines.extend(f"{idx}. {prompt}" for idx, prompt in enumerate(reference_prompts, start=1))
                condition_text = "\n" + "\n".join(lines)
                full_context = f"{base_context}{condition_text}"

        return {
            "full_context": full_context,
            "condition_text": condition_text,
            "reference_prompts": reference_prompts,
            "reference_prompt_count": len(reference_prompts),
            "queue_sample_count": len(queue_samples) if "queue_samples" in locals() else 0,
            "buffer_sample_count": len(buffer_samples) if "buffer_samples" in locals() else 0,
            "condition_text_chars": len(condition_text),
        }

    def _prepare_response_batch_from_texts(
        self,
        prompt_batch: Dict[str, torch.Tensor],
        decoded_responses: Sequence[str],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            response_inputs = self.tokenizer(
                list(decoded_responses),
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = original_padding_side

        eos_tokens = torch.full(
            (response_inputs["input_ids"].size(0), 1),
            fill_value=self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        response_ids = torch.cat([response_inputs["input_ids"], eos_tokens], dim=1)
        response_mask = torch.cat([response_inputs["attention_mask"], torch.ones_like(eos_tokens)], dim=1)
        prompt_responses = torch.cat([prompt_batch["input_ids"], response_ids], dim=1)
        return prompt_responses, {"input_ids": response_ids, "attention_mask": response_mask}

    @torch.inference_mode(True)
    def get_log_prior(
        self,
        prompt_responses: torch.Tensor,
        prompt_len: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        only_responses = prompt_responses[:, prompt_len:]
        if only_responses.numel() == 0:
            return torch.zeros(prompt_responses.size(0), dtype=torch.float32, device=prompt_responses.device)

        pad_mask = (only_responses == self.tokenizer.pad_token_id).cumsum(1) > 1
        full_attention_mask = torch.cat([attention_mask, (~pad_mask).long()], dim=1)
        prior_chunk_size = int(getattr(self.args, "prior_chunk_size", 0) or 0)
        if prior_chunk_size <= 0:
            prior_chunk_size = prompt_responses.size(0)

        log_prior_chunks: List[torch.Tensor] = []
        lora_to_base(self.model)
        try:
            for start in range(0, prompt_responses.size(0), prior_chunk_size):
                end = min(start + prior_chunk_size, prompt_responses.size(0))
                chunk_inputs = prompt_responses[start:end]
                chunk_mask = full_attention_mask[start:end]
                chunk_pad_mask = pad_mask[start:end]

                outputs = self.model(
                    input_ids=chunk_inputs,
                    attention_mask=chunk_mask,
                )
                logits = outputs.logits[:, prompt_len - 1 : -1]
                labels = chunk_inputs[:, prompt_len:]

                token_log_prob = torch.gather(
                    F.log_softmax(logits, dim=-1),
                    -1,
                    labels.unsqueeze(-1),
                ).squeeze(-1)
                token_log_prob = torch.where(chunk_pad_mask, 0.0, token_log_prob)
                chunk_prior = torch.sum(token_log_prob, dim=1)

                if getattr(self.args, "prior_reduction", "sum") == "mean":
                    denom = (~chunk_pad_mask).sum(dim=1).clamp_min(1).to(chunk_prior.dtype)
                    chunk_prior = chunk_prior / denom
                log_prior_chunks.append(chunk_prior)
        finally:
            base_to_lora(self.model)

        return torch.cat(log_prior_chunks, dim=0)

    def add_to_train_buffer(self, samples: List[Dict[str, Any]]) -> None:
        if not self.enable_buffer or self.train_buffer_max_size <= 0:
            return

        for sample in samples:
            entry = (
                float(sample["accuracy"]),
                random.random(),
                sample,
            )
            if len(self.train_buffer) < self.train_buffer_max_size:
                heapq.heappush(self.train_buffer, entry)
                continue
            if entry[0] > self.train_buffer[0][0]:
                heapq.heapreplace(self.train_buffer, entry)

    def sample_from_train_buffer(self, batch_size: int) -> List[Dict[str, Any]]:
        if not self.train_buffer:
            return []
        sample_size = min(int(batch_size), len(self.train_buffer))
        return [item[2] for item in random.sample(self.train_buffer, sample_size)]

    def get_train_buffer_as_list(self) -> List[Dict[str, Any]]:
        return [item[2] for item in self.train_buffer]

    def copy_train_buffer(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.get_train_buffer_as_list())

    def _load_train_buffer(self, path: Path) -> None:
        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples", payload) if isinstance(payload, dict) else payload
        samples: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            prompt = base.strip_leading_role_artifacts(str(row.get("prompt", "")).strip())
            if not prompt:
                continue
            samples.append(
                {
                    "prompt": prompt,
                    "accuracy": float(row.get("accuracy", row.get("train_acc", 0.0))),
                    "log_reward": float(row.get("log_reward", 0.0)),
                    "log_prior": float(row.get("log_prior", 0.0)),
                }
            )
        self.add_to_train_buffer(samples)

    def _save_train_buffer_snapshot(self) -> None:
        if not (self.enable_buffer and getattr(self.args, "train_buffer_save", False)):
            return

        base.save_json(
            self.task_root / "train_buffer.json",
            {
                "task_name": self.task_name,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_samples": len(self.train_buffer),
                "samples": sorted(self.get_train_buffer_as_list(), key=lambda x: x.get("accuracy", 0.0), reverse=True),
            },
        )

    def _maybe_create_condition_buffer(self, global_step: int) -> None:
        if not (self.enable_conditioning and self.use_condition_buffer):
            return
        if self.condition_buffer_created:
            return
        if global_step < self.offline_start_step:
            return

        self.condition_buffer = self.copy_train_buffer()
        self.condition_buffer_created = True

    def _should_use_offline(self, global_step: int) -> bool:
        if not self.use_offline_sampling:
            return False
        if global_step < self.offline_start_step:
            return False
        if not self.train_buffer:
            return False
        return random.random() >= self.online_ratio

    def _record_top_prompt(
        self,
        prompt: str,
        accuracy: float,
        step: int,
        log_reward: float,
        *,
        log_prior: float | None = None,
        source: str = "online",
    ) -> None:
        super()._record_top_prompt(prompt, accuracy, step, log_reward)
        if prompt in self.queue_prompt_metadata:
            if log_prior is not None:
                self.queue_prompt_metadata[prompt]["log_prior"] = float(log_prior)
            self.queue_prompt_metadata[prompt]["source"] = str(source)

    def _log_wandb_step(
        self,
        global_step: int,
        summary: Dict[str, float],
        step_prompt_records: Sequence[Dict[str, Any]],
        top_prompts: Sequence[tuple[float, str, int]],
        eval_prompt_records: Sequence[Dict[str, Any]] | None = None,
        step_context: Dict[str, Any] | None = None,
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
            "train/batch_mean_log_prior": "monitor/train_batch_mean_log_prior",
            "train/batch_mean_logpf": "monitor/train_batch_mean_logpf",
            "optim/grad_norm": "monitor/grad_norm",
            "optim/lr": "monitor/lr",
            "sampling/temp": "monitor/sampling_temp",
            "sampling/mean_prompt_chars": "monitor/mean_prompt_chars",
            "sampling/max_prompt_chars": "monitor/max_prompt_chars",
            "sampling/full_meta_prompt_chars": "monitor/full_meta_prompt_chars",
            "sampling/fewshot_block_chars": "monitor/fewshot_block_chars",
            "sampling/unique_prompt_ratio": "monitor/unique_prompt_ratio",
            "sampling/use_offline": "monitor/use_offline",
            "sampling/buffer_size": "monitor/buffer_size",
            "conditioning/reference_prompt_count": "monitor/reference_prompt_count",
            "train/queue_size": "monitor/queue_size",
            "train/log_z_ema": "monitor/log_z_ema",
            "reward/prior_term_mean": "monitor/prior_term_mean",
            "reward/acc_term_mean": "monitor/acc_term_mean",
            "reward/prior_to_acc_abs_ratio": "monitor/prior_to_acc_abs_ratio",
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
            f"<h3>{self.task_name} step {global_step}</h3>",
            "<details><summary>Fixed context</summary>",
            f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{(step_context or {}).get('fixed_context', self.fixed_context)}</pre>",
            "</details>",
            "<table border='1'><tr><th>Micro</th><th>Idx</th><th>Source</th><th>Prompt</th><th>Train Acc</th><th>Log Reward</th><th>Log Prior</th><th>LogPF</th><th>Temp</th><th>Best</th></tr>",
        ]
        for record in step_prompt_records:
            prompt_html.append(
                "<tr>"
                f"<td>{int(record['micro_step'])}</td>"
                f"<td>{int(record['prompt_idx'])}</td>"
                f"<td>{str(record.get('source', 'online'))}</td>"
                f"<td>{str(record['prompt'])}</td>"
                f"<td>{float(record['train_acc']):.4f}</td>"
                f"<td>{float(record['log_reward']):.4f}</td>"
                f"<td>{float(record.get('log_prior', 0.0)):.4f}</td>"
                f"<td>{float(record['sum_logpf']):.4f}</td>"
                f"<td>{float(record['sampling_temp']):.4f}</td>"
                f"<td>{'yes' if record['is_best_in_micro_step'] else 'no'}</td>"
                "</tr>"
            )
        prompt_html.append("</table>")
        self.wandb_run.log({"sampling/prompts_html": base.wandb.Html("".join(prompt_html))}, step=global_step)
        if not eval_prompt_records:
            return

        eval_html = [
            f"<h3>{self.task_name} step {global_step} eval</h3>",
            "<table border='1'><tr><th>Rank</th><th>Prompt</th><th>Train Acc</th><th>Test Acc</th><th>Step</th></tr>",
        ]
        for record in eval_prompt_records:
            eval_html.append(
                "<tr>"
                f"<td>{int(record['rank'])}</td>"
                f"<td>{str(record['prompt'])}</td>"
                f"<td>{float(record['train_acc']):.4f}</td>"
                f"<td>{float(record['test_acc']):.4f}</td>"
                f"<td>{int(record['step'])}</td>"
                "</tr>"
            )
        eval_html.append("</table>")
        self.wandb_run.log({"eval/top_prompts_html": base.wandb.Html("".join(eval_html))}, step=global_step)

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
        self.wandb_run.summary["monitor/final_train_buffer_size"] = int(summary.get("train_buffer_size", 0))

        artifact = base.wandb.Artifact(
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
            self.task_root / "train_buffer.json",
            self.logs_root / "metrics.jsonl",
            self.logs_root / "prompts.jsonl",
        ]
        for path in artifact_paths:
            if path.exists():
                artifact.add_file(str(path), name=str(path.relative_to(self.task_root)))
        self.wandb_run.log_artifact(artifact)

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
                    "log_prior": meta.get("log_prior"),
                    "source": meta.get("source"),
                }
            )

        task_payload = base.build_bbh_eval_prompt_payload(self.task_name, top_rows)
        base.save_json(self.task_root / "bbh_eval_prompts.json", task_payload)
        base.save_json(
            self.task_root / "generated_meta_prompts" / f"{self.task_name}.json",
            {
                "task_name": self.task_name,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_config": {
                    "source": self.args.variant_name,
                    "requested_num_prompts": len(top_rows),
                    "actual_num_prompts": len(top_rows),
                    "fewshot_indices": list(self.fewshot_indices),
                    "fewshot_resample_each_step": bool(self.fewshot_resample_each_step),
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
        base.save_json(self.task_root / "top_prompts.json", top_rows)

    def train(self) -> Dict[str, Any]:
        progress = tqdm(range(1, self.args.train_steps + 1), desc=f"{self.task_name}", dynamic_ncols=True)

        for global_step in progress:
            self._maybe_create_condition_buffer(global_step)
            step_context = self._build_step_context()

            self.model.train()
            self.optimizer.zero_grad()
            batch_metrics: Dict[str, List[float]] = defaultdict(list)
            step_prompt_records: List[Dict[str, Any]] = []

            for micro_step in range(self.args.grad_acc_steps):
                prompt_ctx = self._build_prompt_context(global_step, step_context["fixed_context"])
                use_offline = self._should_use_offline(global_step)
                source = "offline" if use_offline else "online"

                if use_offline:
                    sampled = self.sample_from_train_buffer(self.args.batch_size)
                    if sampled:
                        decoded_responses = [base.strip_leading_role_artifacts(sample["prompt"]) for sample in sampled]
                        sampling_temp = 0.0
                        prompt_batch = self._build_prompt_batch_with_context(
                            prompt_ctx["full_context"],
                            num_samples=len(decoded_responses),
                        )
                        prompt_responses, response_batch = self._prepare_response_batch_from_texts(
                            prompt_batch=prompt_batch,
                            decoded_responses=decoded_responses,
                        )
                    else:
                        use_offline = False
                        source = "online"

                if not use_offline:
                    prompt_batch = self._build_prompt_batch_with_context(
                        prompt_ctx["full_context"],
                        num_samples=self.args.batch_size,
                    )
                    prompt_responses, only_responses, decoded_responses, sampling_temp = self.sample_prompts(prompt_batch)
                    response_batch = self._prepare_response_batch(
                        input_ids=prompt_batch["input_ids"],
                        prompt_responses=prompt_responses,
                        only_responses=only_responses,
                    )

                sum_logpf = self.get_logpf(prompt_batch, response_batch)
                train_acc = self._evaluate_train_accs(decoded_responses).to(sum_logpf.device)
                log_accuracy = self.compute_log_acc_reward(train_acc)

                if self.enable_log_prior:
                    log_prior = self.get_log_prior(
                        prompt_responses=prompt_responses,
                        prompt_len=prompt_batch["input_ids"].size(1),
                        attention_mask=prompt_batch["attention_mask"],
                    )
                else:
                    log_prior = torch.zeros_like(log_accuracy)

                log_reward = log_accuracy / self.args.beta
                if self.enable_log_prior:
                    log_reward = log_reward + (log_prior / self.args.gamma)
                    prior_term = log_prior / self.args.gamma
                else:
                    prior_term = torch.zeros_like(log_accuracy)
                acc_term = log_accuracy / self.args.beta

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
                best_log_prior = float(log_prior[best_idx].item())
                self._record_top_prompt(
                    best_prompt,
                    best_acc,
                    global_step,
                    best_log_reward,
                    log_prior=best_log_prior,
                    source=source,
                )

                if self.enable_buffer and not use_offline:
                    batch_samples = []
                    for idx, prompt_text in enumerate(decoded_responses):
                        batch_samples.append(
                            {
                                "prompt": prompt_text,
                                "accuracy": float(train_acc[idx].item()),
                                "log_reward": float(log_reward[idx].item()),
                                "log_prior": float(log_prior[idx].item()),
                                "source": "online",
                            }
                        )
                    self.add_to_train_buffer(batch_samples)

                unique_prompt_ratio = float(len(set(decoded_responses)) / max(len(decoded_responses), 1))
                batch_metrics["train/batch_mean_acc"].append(float(train_acc.mean().item()))
                batch_metrics["train/batch_max_acc"].append(best_acc)
                batch_metrics["train/batch_mean_log_reward"].append(float(log_reward.mean().item()))
                batch_metrics["train/batch_max_log_reward"].append(float(log_reward.max().item()))
                batch_metrics["train/batch_mean_logpf"].append(float(sum_logpf.mean().item()))
                batch_metrics["train/batch_mean_log_accuracy"].append(float(log_accuracy.mean().item()))
                batch_metrics["train/batch_max_log_accuracy"].append(float(log_accuracy.max().item()))
                batch_metrics["train/batch_mean_log_prior"].append(float(log_prior.mean().item()))
                batch_metrics["train/batch_max_log_prior"].append(float(log_prior.max().item()))
                batch_metrics["train/batch_tb_loss"].append(float(loss.item()))
                batch_metrics["sampling/temp"].append(float(sampling_temp))
                batch_metrics["sampling/unique_prompt_ratio"].append(unique_prompt_ratio)
                batch_metrics["sampling/mean_prompt_chars"].append(
                    float(sum(len(x) for x in decoded_responses) / max(len(decoded_responses), 1))
                )
                batch_metrics["sampling/max_prompt_chars"].append(
                    float(max((len(x) for x in decoded_responses), default=0))
                )
                batch_metrics["sampling/full_meta_prompt_chars"].append(float(len(prompt_ctx["full_context"])))
                batch_metrics["sampling/fewshot_block_chars"].append(float(step_context["fewshot_block_chars"]))
                batch_metrics["sampling/use_offline"].append(1.0 if use_offline else 0.0)
                batch_metrics["sampling/buffer_size"].append(float(len(self.train_buffer)))
                batch_metrics["conditioning/reference_prompt_count"].append(float(prompt_ctx["reference_prompt_count"]))
                batch_metrics["sampling/reference_prompt_count"].append(float(prompt_ctx["reference_prompt_count"]))
                batch_metrics["sampling/queue_sample_count"].append(float(prompt_ctx["queue_sample_count"]))
                batch_metrics["sampling/buffer_sample_count"].append(float(prompt_ctx["buffer_sample_count"]))
                batch_metrics["sampling/condition_text_chars"].append(float(prompt_ctx["condition_text_chars"]))
                batch_metrics["reward/prior_term_mean"].append(float(prior_term.mean().item()))
                batch_metrics["reward/acc_term_mean"].append(float(acc_term.mean().item()))
                batch_metrics["reward/prior_to_acc_abs_ratio"].append(
                    float(prior_term.abs().mean().item() / max(acc_term.abs().mean().item(), 1e-8))
                )

                for idx, prompt in enumerate(decoded_responses):
                    step_prompt_records.append(
                        {
                            "micro_step": micro_step,
                            "prompt_idx": idx,
                            "prompt": prompt,
                            "source": source,
                            "train_acc": float(train_acc[idx].item()),
                            "log_reward": float(log_reward[idx].item()),
                            "log_prior": float(log_prior[idx].item()),
                            "sum_logpf": float(sum_logpf[idx].item()),
                            "sampling_temp": float(sampling_temp),
                            "reference_prompt_count": int(prompt_ctx["reference_prompt_count"]),
                            "prompt_chars": int(len(prompt)),
                            "full_context_chars": int(len(prompt_ctx["full_context"])),
                            "fewshot_mode": str(step_context["fewshot_mode"]),
                            "fewshot_indices": list(step_context["fewshot_indices"]),
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
            summary["train/mean_log_prior"] = float(summary["train/batch_mean_log_prior"])
            summary["train/max_log_prior"] = float(summary["train/batch_max_log_prior"])
            summary["train/mean_log_accuracy"] = float(summary["train/batch_mean_log_accuracy"])
            summary["train/max_log_accuracy"] = float(summary["train/batch_max_log_accuracy"])
            summary["sampling/queue_size"] = float(len(top_prompts))
            summary["sampling/queue_best_acc"] = float(top_prompts[0][0]) if top_prompts else 0.0
            summary["sampling/condition_buffer_size"] = float(len(self.condition_buffer))

            eval_prompt_records: List[Dict[str, Any]] = []
            if (
                self.args.test_eval_period > 0
                and self.task_data.test_goals
                and (global_step % self.args.test_eval_period == 0)
                and top_prompts
            ):
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
                        "fixed_context": step_context["fixed_context"],
                        "fewshot_block": step_context["fewshot_block"],
                        "fewshot_indices": step_context["fewshot_indices"],
                        "fewshot_mode": step_context["fewshot_mode"],
                        "prompts": step_prompt_records,
                    },
                )
            if global_step % self.args.export_every == 0:
                self._export_prompt_payloads()
                self._save_train_buffer_snapshot()

            self._log_wandb_step(
                global_step,
                summary,
                step_prompt_records,
                top_prompts,
                eval_prompt_records,
                step_context=step_context,
            )

        self._export_prompt_payloads()
        self._save_train_buffer_snapshot()

        top_prompts = self.queue.get_top_texts()
        summary = {
            "task_name": self.task_name,
            "best_train_acc": float(top_prompts[0][0]) if top_prompts else 0.0,
            "best_prompt": top_prompts[0][1] if top_prompts else "",
            "best_prompt_step": int(top_prompts[0][2]) if top_prompts else -1,
            "num_top_prompts": len(top_prompts),
            "train_buffer_size": len(self.train_buffer),
        }
        final_eval_records: List[Dict[str, Any]] = []
        if top_prompts and self.task_data.test_goals:
            eval_summary, final_eval_records = self._build_test_eval_metrics(top_prompts)
            summary.update(eval_summary)
            summary["best_prompt_test_acc"] = float(eval_summary.get("eval/top1_test_acc", 0.0))

        if final_eval_records:
            base.save_json(self.task_root / "top_prompts_eval.json", final_eval_records)

        base.save_json(self.task_root / "summary.json", summary)
        self._log_wandb_final(summary)
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental GFlowNet debugger for BBH prompt training")
    parser.add_argument("--task_names", type=str, nargs="+", default=["object_counting"])
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--eval_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/huggingface"))
    parser.add_argument("--save_dir", type=str, default=str(base.REPO_ROOT / "debugging_gflownet_runs"))
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
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward_epsilon", type=float, default=1e-8)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--prior_reduction", type=str, choices=["sum", "mean"], default="sum")
    parser.add_argument("--prior_chunk_size", type=int, default=0)

    parser.add_argument("--meta_prompt", type=str, default=(
        "I gave a friend an instruction and five inputs. "
        "The friend read the instruction and wrote an output for every one of the inputs. "
        "Here are the input-output pairs:"
    ))
    parser.add_argument("--num_example", type=int, default=5)
    parser.add_argument("--fewshot_strategy", type=str, choices=["first", "random"], default="first")
    parser.add_argument("--fewshot_seed", type=int, default=42)
    parser.add_argument("--fewshot_indices", type=str, default="")
    parser.add_argument("--fewshot_resample_each_step", action="store_true")

    parser.add_argument("--data_root", type=str, default=base.default_data_root())
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

    parser.add_argument("--init_queue_prompt", type=str, default=None)
    parser.add_argument("--train_buffer_max_size", type=int, default=1000)
    parser.add_argument("--train_buffer_path", type=str, default="")
    parser.add_argument("--train_buffer_save", action="store_true")
    parser.add_argument("--use_offline_sampling", action="store_true")
    parser.add_argument("--online_ratio", type=float, default=0.5)
    parser.add_argument("--offline_start_step", type=int, default=100)
    parser.add_argument("--condition_buffer", action="store_true")
    parser.add_argument("--condition_queue_samples", type=int, default=1)
    parser.add_argument("--condition_buffer_samples", type=int, default=2)
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def run_variant(variant_name: str) -> None:
    if variant_name not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant_name}")

    args = parse_args()
    args.variant_name = variant_name
    os.environ.setdefault("VLLM_USE_V1", "0")

    args.task_names = base.canonicalize_task_names(args.task_names)

    unknown_tasks = sorted(set(args.task_names) - set(base.SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}. Supported: {sorted(base.SUPPORTED_TASKS)}")

    load_eval_model_config(args)
    seed(args.seed)

    run_name = args.run_name or f"{variant_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = Path(args.save_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    combined_prompt_payload: Dict[str, Dict[str, str]] = {}
    variant_config = VARIANT_CONFIGS[variant_name]

    for task_name in args.task_names:
        task_args = argparse.Namespace(**vars(args))
        runner = IncrementalFeatureRunner(
            task_args,
            task_name=task_name,
            run_root=run_root,
            **variant_config,
        )
        summary = runner.train()
        summaries.append(summary)

        payload_path = runner.task_root / "bbh_eval_prompts.json"
        with payload_path.open("r", encoding="utf-8") as f:
            task_payload = json.load(f)
        combined_prompt_payload.update(task_payload)
        runner.close()
        del runner

    base.save_json(run_root / "all_task_summaries.json", summaries)
    base.save_json(run_root / "bbh_eval_prompts_all_tasks.json", combined_prompt_payload)
