#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from transformers import AutoConfig, GenerationConfig
from vllm import LLM, SamplingParams
import wandb

import debugging_gflownet as dg


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_latest_eval_step(metrics_path: Path) -> int:
    latest_step = -1
    if not metrics_path.exists():
        return latest_step
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            summary = row.get("metrics", {})
            if "eval/best_so_far_test_acc" in summary:
                latest_step = int(row["global_step"])
    return latest_step


def build_llm(config: Dict[str, Any]) -> LLM:
    model_path = config["eval_model_paths"]
    model_cfg = AutoConfig.from_pretrained(model_path)
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
        eval_top_p = gen_cfg.top_p if gen_cfg.top_p is not None else 1.0
    except Exception:
        eval_top_p = 1.0

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": int(config["tp_size"]),
        "gpu_memory_utilization": float(config["eval_gpu_memory_utilization"]),
        "max_seq_len_to_capture": min(
            getattr(model_cfg, "max_position_embeddings", 8192),
            int(config["eval_max_seq_len_to_capture"]),
        ),
        "max_num_seqs": int(config["eval_max_num_seqs"]),
        "enable_prefix_caching": True,
        "disable_log_stats": True,
    }
    while True:
        try:
            llm = LLM(**llm_kwargs)
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
                    getattr(model_cfg, "max_position_embeddings", 8192),
                    int(config["eval_max_seq_len_to_capture"]),
                )
    llm.eval_top_p = eval_top_p  # type: ignore[attr-defined]
    return llm


def build_sampling_params(llm: LLM, config: Dict[str, Any]) -> tuple[SamplingParams, SamplingParams]:
    eval_top_p = getattr(llm, "eval_top_p", 1.0)
    reasoning_params = SamplingParams(
        temperature=0.0,
        top_p=eval_top_p,
        max_tokens=int(config["bbh_reasoning_max_tokens"]),
    )
    answer_params = SamplingParams(
        temperature=0.0,
        top_p=eval_top_p,
        max_tokens=int(config["bbh_answer_max_tokens"]),
        stop=dg.bbh_eval_utils.ANSWER_EXTRACTION_STOPS,
    )
    return reasoning_params, answer_params


def evaluate_top_prompts(
    llm: LLM,
    config: Dict[str, Any],
    task_data: dg.TaskData,
    top_prompts_path: Path,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows = json.loads(top_prompts_path.read_text(encoding="utf-8"))
    prompts = [row["prompt"] for row in rows]
    if not prompts:
        return {}, []

    reasoning_params, answer_params = build_sampling_params(llm, config)
    acc = dg.evaluate_prompts_with_bbh_eval(
        llm=llm,
        task_name=config["task_names"][0],
        prompts=prompts,
        goals=task_data.test_goals,
        final_targets=task_data.test_final_targets,
        reasoning_params=reasoning_params,
        answer_params=answer_params,
        chunk_size=int(config["eval_chunk_size"]),
    )
    test_accs = [float(x.item()) for x in acc]
    mean_acc = float(sum(test_accs) / len(test_accs))
    variance = float(sum((x - mean_acc) ** 2 for x in test_accs) / len(test_accs))
    metrics: Dict[str, float] = {
        "eval/top1_test_acc": float(test_accs[0]),
        "eval/top_prompts_count": float(len(test_accs)),
        "eval/top_prompts_mean_test_acc": mean_acc,
        "eval/top_prompts_max_test_acc": float(max(test_accs)),
        "eval/top_prompts_min_test_acc": float(min(test_accs)),
        "eval/top_prompts_std_test_acc": float(max(variance, 0.0) ** 0.5),
    }
    records: List[Dict[str, Any]] = []
    for rank, (row, test_acc) in enumerate(zip(rows, test_accs), start=1):
        metrics[f"eval/top{rank}_test_acc"] = float(test_acc)
        records.append(
            {
                "rank": int(rank),
                "prompt": row["prompt"],
                "train_acc": float(row.get("train_acc", 0.0)),
                "test_acc": float(test_acc),
                "step": int(row.get("step", -1)),
                "source": row.get("source", ""),
            }
        )
    return metrics, records


def build_eval_html(task_name: str, step: int, records: Sequence[Dict[str, Any]]) -> wandb.Html:
    html_rows = [
        f"<h3>{task_name} step {step} eval sidecar</h3>",
        "<table border='1'><tr><th>Rank</th><th>Prompt</th><th>Train Acc</th><th>Test Acc</th><th>Step</th><th>Source</th></tr>",
    ]
    for record in records:
        html_rows.append(
            "<tr>"
            f"<td>{int(record['rank'])}</td>"
            f"<td>{record['prompt']}</td>"
            f"<td>{float(record['train_acc']):.4f}</td>"
            f"<td>{float(record['test_acc']):.4f}</td>"
            f"<td>{int(record['step'])}</td>"
            f"<td>{record['source']}</td>"
            "</tr>"
        )
    html_rows.append("</table>")
    return wandb.Html("".join(html_rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sidecar evaluator for top prompt test metrics.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--top-prompts-json", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default="")
    parser.add_argument("--watch-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--task-name", default="")
    args = parser.parse_args()

    config_payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    config = config_payload["args"]
    task_name = args.task_name or config_payload.get("task_name") or config["task_names"][0]
    metrics_path = Path(args.metrics_jsonl)
    top_prompts_path = Path(args.top_prompts_json)
    run_dir = str(top_prompts_path.parent)
    entity = args.entity or None

    task_data = dg.load_task_data(
        task_name=config["task_names"][0],
        data_root=config["data_root"],
        conversation_template=config["conversation_template"],
        n_train_data=int(config["bbh_train_size"]),
        n_test_data=int(config["bbh_test_size"]),
    )

    llm = build_llm(config)
    run = wandb.init(
        project=args.project,
        entity=entity,
        id=args.run_id,
        resume="allow",
        reinit="finish_previous",
        dir=run_dir,
    )
    if run is None:
        raise RuntimeError("Failed to initialize wandb eval sidecar run.")

    seen_eval_steps: set[int] = set()
    idle_cycles = 0

    while True:
        latest_step = load_latest_eval_step(metrics_path)
        new_eval_logged = False
        if latest_step >= 0 and latest_step not in seen_eval_steps and top_prompts_path.exists():
            eval_metrics, eval_records = evaluate_top_prompts(llm, config, task_data, top_prompts_path)
            if eval_metrics:
                payload = dict(eval_metrics)
                payload["eval/top_prompts_html"] = build_eval_html(task_name, latest_step, eval_records)
                run.log(payload, step=latest_step)
                run.summary["monitor/latest_eval_top_prompts_mean_test_acc"] = float(
                    eval_metrics["eval/top_prompts_mean_test_acc"]
                )
                run.summary["monitor/latest_eval_top1_test_acc_sidecar"] = float(eval_metrics["eval/top1_test_acc"])
                seen_eval_steps.add(latest_step)
                new_eval_logged = True

        if new_eval_logged:
            idle_cycles = 0
        else:
            idle_cycles += 1

        if args.watch_pid and not process_alive(args.watch_pid) and idle_cycles >= 2:
            break

        time.sleep(max(args.poll_seconds, 1.0))

    run.finish()
    try:
        llm.llm_engine.shutdown()  # type: ignore[attr-defined]
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
