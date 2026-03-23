#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import wandb


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_monitor_metrics(step: int, summary: Dict[str, Any], train_steps: int) -> Dict[str, float]:
    monitor = {
        "monitor/global_step": float(step),
        "monitor/progress_fraction": float(step) / float(max(train_steps, 1)),
    }
    gfn_aliases: Dict[str, float] = {}
    if "train/mean_acc" in summary:
        gfn_aliases["train/mean_acc"] = float(summary["train/mean_acc"])
    elif "train/batch_mean_acc" in summary:
        gfn_aliases["train/mean_acc"] = float(summary["train/batch_mean_acc"])
    if "train/max_acc" in summary:
        gfn_aliases["train/max_acc"] = float(summary["train/max_acc"])
    elif "train/batch_max_acc" in summary:
        gfn_aliases["train/max_acc"] = float(summary["train/batch_max_acc"])
    if "train/best_so_far_acc" in summary:
        gfn_aliases["sampling/queue_best_acc"] = float(summary["train/best_so_far_acc"])
    if "train/queue_size" in summary:
        gfn_aliases["sampling/queue_size"] = float(summary["train/queue_size"])
    if "sampling/buffer_size" in summary:
        gfn_aliases["sampling/condition_buffer_size"] = float(summary["sampling/buffer_size"])
    if "train/batch_mean_log_reward" in summary:
        gfn_aliases["train/mean_log_reward"] = float(summary["train/batch_mean_log_reward"])
    if "train/batch_mean_log_prior" in summary:
        gfn_aliases["train/mean_log_prior"] = float(summary["train/batch_mean_log_prior"])
    if "train/batch_mean_log_accuracy" in summary:
        gfn_aliases["train/mean_log_accuracy"] = float(summary["train/batch_mean_log_accuracy"])
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
        "sampling/unique_prompt_ratio": "monitor/unique_prompt_ratio",
        "sampling/use_offline": "monitor/use_offline",
        "sampling/buffer_size": "monitor/buffer_size",
        "conditioning/reference_prompt_count": "monitor/reference_prompt_count",
        "train/queue_size": "monitor/queue_size",
        "train/log_z_ema": "monitor/log_z_ema",
        "eval/best_so_far_test_acc": "monitor/eval_best_so_far_test_acc",
        "eval/top_prompts_mean_test_acc": "monitor/eval_top_prompts_mean_test_acc",
        "eval/top_prompts_max_test_acc": "monitor/eval_top_prompts_max_test_acc",
        "eval/top_prompts_min_test_acc": "monitor/eval_top_prompts_min_test_acc",
        "eval/top_prompts_std_test_acc": "monitor/eval_top_prompts_std_test_acc",
    }
    for src_key, dst_key in metric_aliases.items():
        if src_key in summary:
            monitor[dst_key] = float(summary[src_key])
    return {**gfn_aliases, **monitor}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sidecar logger for extra W&B monitor metrics.")
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--watch-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_jsonl)
    entity = args.entity or None
    run = wandb.init(
        project=args.project,
        entity=entity,
        id=args.run_id,
        resume="allow",
        reinit="finish_previous",
        dir=str(metrics_path.parent.parent),
    )

    if run is None:
        raise RuntimeError("Failed to initialize wandb sidecar run.")

    seen_steps: set[int] = set()
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                seen_steps.add(int(row["global_step"]))
    idle_cycles = 0

    while True:
        new_step_logged = False
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    step = int(row["global_step"])
                    if step in seen_steps:
                        continue
                    summary = row.get("metrics", {})
                    run.log(build_monitor_metrics(step, summary, args.train_steps), step=step)
                    run.summary["monitor/last_step"] = step
                    if "train/best_so_far_acc" in summary:
                        run.summary["monitor/latest_train_best_so_far_acc"] = float(summary["train/best_so_far_acc"])
                    if "eval/best_so_far_test_acc" in summary:
                        run.summary["monitor/latest_eval_best_so_far_test_acc"] = float(summary["eval/best_so_far_test_acc"])
                    seen_steps.add(step)
                    new_step_logged = True

        if new_step_logged:
            idle_cycles = 0
        else:
            idle_cycles += 1

        if args.watch_pid and not process_alive(args.watch_pid) and idle_cycles >= 2:
            break

        time.sleep(max(args.poll_seconds, 1.0))

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
