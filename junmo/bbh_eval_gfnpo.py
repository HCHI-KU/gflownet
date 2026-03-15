from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

try:
    from vllm import SamplingParams
except Exception:  # pragma: no cover - allows lightweight local checks without vLLM installed
    SamplingParams = None


TASK_EXTRACTOR_TEXT: Dict[str, str] = {
    "tracking_shuffled_objects_five_objects": (
        "Therefore, the final answer (use exact format: '$A' or '$B' or '$C' or '$D' or '$E') is $"
    ),
    "object_counting": (
        "Therefore, the final answer (use exactly this format: $NUMBER$, where NUMBER is a positive integer) is $"
    ),
    "causal_judgement": (
        "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ "
    ),
    "movie_recommendation": (
        "Therefore, the final answer (use exact format: '$A' or '$B' or '$C' or '$D') is $"
    ),
    "hyperbaton": (
        "Therefore, the final answer (use exact format: '$A' or '$B') is $"
    ),
}

TASK_ALIASES: Dict[str, str] = {
    "causal_judgment": "causal_judgement",
}

TASK_SOURCE_DIR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "tracking_shuffled_objects_five_objects": (
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects",
    ),
    "object_counting": ("object_counting",),
    "causal_judgement": ("causal_judgment", "causal_judgement"),
    "movie_recommendation": ("movie_recommendation",),
    "hyperbaton": ("hyperbaton",),
}

BBH5_TASKS = set(TASK_EXTRACTOR_TEXT.keys())
ANSWER_EXTRACTION_STOPS = ["\n", "<|eot_id|>", "<end_of_turn>", "</s>"]


def canonicalize_bbh_task(task_name: str) -> str:
    return TASK_ALIASES.get(str(task_name), str(task_name))


def is_supported_bbh5_task(task_name: str) -> bool:
    return canonicalize_bbh_task(task_name) in BBH5_TASKS


def remove_parentheses_if_single_char(text: str) -> str:
    text = str(text).strip()
    if text.startswith("(") and text.endswith(")") and len(text) == 3:
        return text[1:-1]
    return text


def render_llama3_user_prompt(user_content: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def render_extractor_prompt_llama3(stage1_prompt: str, reasoning: str, extractor_text: str) -> str:
    separator = "" if (not reasoning or str(reasoning).endswith((" ", "\n"))) else " "
    return f"{stage1_prompt}{reasoning}{separator}{extractor_text}"


def _compose_user_content(control: str, task_input: str) -> str:
    control = str(control).strip()
    task_input = str(task_input).strip()
    if control:
        return f"{control}\n{task_input}\nOutput : "
    return f"{task_input}\nOutput : "


def _iter_dataset_rows(dataset: Any) -> Iterable[Dict[str, Any]]:
    for idx in range(len(dataset)):
        row = dataset[idx]
        if not isinstance(row, dict):
            raise TypeError(f"Expected dataset row dict, got {type(row)} at index {idx}")
        yield row


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_bbh_source_paths(task_name: str) -> Iterable[Path]:
    canonical_task = canonicalize_bbh_task(task_name)
    task_dirs = TASK_SOURCE_DIR_ALIASES.get(canonical_task, (canonical_task,))
    repo_root = _repo_root()
    for task_dir in task_dirs:
        yield repo_root / "junmo" / "automatic_prompt_engineer" / "data" / "bigbench-ii" / task_dir / "task.json"


def _load_bbh_source_examples(task_name: str) -> List[Dict[str, Any]] | None:
    for path in _iter_bbh_source_paths(task_name):
        if not path.exists():
            continue
        try:
            with path.open("r") as f:
                json_data = json.load(f)
        except Exception:
            continue
        examples = json_data.get("examples")
        if isinstance(examples, list) and len(examples) > 0:
            return examples
    return None


def _detect_bbh_source_path(task_name: str) -> str | None:
    for path in _iter_bbh_source_paths(task_name):
        if path.exists():
            return str(path)
    return None


def _select_dataset_split_examples(
    source_examples: Sequence[Dict[str, Any]],
    dataset_len: int,
) -> Sequence[Dict[str, Any]] | None:
    total_examples = len(source_examples)
    if dataset_len == total_examples:
        return source_examples

    if total_examples >= 50 and dataset_len == 50:
        return source_examples[:50]
    if total_examples >= 150 and dataset_len == 100:
        return source_examples[50:150]

    train_size = 30
    if total_examples >= train_size and dataset_len == train_size:
        return source_examples[:train_size]
    if total_examples >= train_size and dataset_len == total_examples - train_size:
        return source_examples[train_size:]
    return None


def _build_task_input_from_source_example(example: Dict[str, Any]) -> str:
    task_input = f"Input : {str(example['input']).strip()}"
    target_scores = example.get("target_scores")
    if isinstance(target_scores, dict) and len(target_scores) > 0:
        task_input += "\n Choices : \n"
        for idx, choice_text in enumerate(target_scores.keys()):
            task_input += f"{_index_to_letter(idx)} : {str(choice_text).strip()}\n"
    elif "Options:" in task_input:
        option_block = task_input.split("Options:", 1)[1]
        choices: List[str] = []
        for line in option_block.splitlines():
            line = line.strip()
            match = re.match(r"^\(([A-Z])\)\s*(.+)$", line)
            if match:
                choices.append(match.group(2).strip())
        if choices:
            task_input = task_input.split("Options:", 1)[0].rstrip()
            task_input += "\n Choices : \n"
            for idx, choice_text in enumerate(choices):
                task_input += f"{_index_to_letter(idx)} : {choice_text}\n"
    return task_input


def _parse_choice_texts(text: str) -> List[str]:
    if "\n Choices :" not in text:
        return []
    raw_choices = text.split("\n Choices :", 1)[1]
    choices = []
    for line in raw_choices.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^[A-Z]\s*:\s*(.+)$", line)
        if match:
            choices.append(match.group(1).strip())
    return choices


def _label_to_index(label: Any) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.item())
    return int(label)


def _index_to_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def _lookup_verbalizer_token(verbalizer: Any, label_index: int) -> str | None:
    if verbalizer is None:
        return None
    idx = int(label_index)
    if isinstance(verbalizer, dict):
        if idx in verbalizer:
            return str(verbalizer[idx]).strip()
        idx_key = str(idx)
        if idx_key in verbalizer:
            return str(verbalizer[idx_key]).strip()
        values = list(verbalizer.values())
        if 0 <= idx < len(values):
            return str(values[idx]).strip()
        return None
    values = list(verbalizer)
    if 0 <= idx < len(values):
        return str(values[idx]).strip()
    return None


def _resolve_final_target(task_name: str, text: str, label: Any, verbalizer: Any = None) -> str:
    canonical_task = canonicalize_bbh_task(task_name)
    if canonical_task == "object_counting":
        return str(label).strip()

    label_index = _label_to_index(label)
    verbalizer_token = _lookup_verbalizer_token(verbalizer, label_index)
    fallback_letter = _index_to_letter(label_index)

    if canonical_task == "causal_judgement":
        choice_texts = _parse_choice_texts(text)
        if 0 <= label_index < len(choice_texts):
            choice = choice_texts[label_index].strip().lower()
            if choice in {"yes", "no"}:
                return choice.capitalize()
        if verbalizer_token is not None:
            return verbalizer_token.strip()
        return fallback_letter

    if canonical_task in {
        "tracking_shuffled_objects_five_objects",
        "movie_recommendation",
        "hyperbaton",
    }:
        if verbalizer_token is not None:
            token = verbalizer_token.strip()
            if len(token) == 1 and token.isalpha():
                return token.upper()
        return fallback_letter

    raise ValueError(f"Unsupported BBH5 task for strict evaluator: {task_name}")


def _resolve_final_target_from_source_example(task_name: str, example: Dict[str, Any]) -> str:
    canonical_task = canonicalize_bbh_task(task_name)

    if canonical_task == "object_counting":
        target = example.get("target")
        if isinstance(target, list) and len(target) > 1:
            return str(target[1]).strip()
        return str(target).strip()

    target_scores = example.get("target_scores")
    if isinstance(target_scores, dict) and len(target_scores) > 0:
        choices = list(target_scores.keys())
        correct_idx = 0
        for idx, (_, score) in enumerate(target_scores.items()):
            if float(score) > 0.5:
                correct_idx = idx
                break
    else:
        target_text = str(example.get("target", "")).strip()
        match = re.search(r"([A-Z])", target_text)
        if not match:
            raise ValueError(f"Source example for '{task_name}' is missing target_scores/target label.")
        correct_idx = ord(match.group(1)) - ord("A")
        choices = []

    correct_choice = str(choices[correct_idx]).strip() if choices else ""
    if canonical_task == "causal_judgement" and correct_choice.lower() in {"yes", "no"}:
        return correct_choice.capitalize()
    if canonical_task in {
        "tracking_shuffled_objects_five_objects",
        "movie_recommendation",
        "hyperbaton",
    }:
        return _index_to_letter(correct_idx)
    return correct_choice


def extract_bbh5_inputs_and_targets(
    dataset: Any,
    task_name: str,
    verbalizer: Any = None,
) -> Tuple[List[str], List[str]]:
    canonical_task = canonicalize_bbh_task(task_name)
    if canonical_task not in BBH5_TASKS:
        raise ValueError(f"Task '{task_name}' is not one of supported BBH5 tasks: {sorted(BBH5_TASKS)}")

    dataset_len = len(dataset)
    source_examples = _load_bbh_source_examples(canonical_task)
    aligned_examples = None
    if source_examples is not None:
        aligned_examples = _select_dataset_split_examples(source_examples, dataset_len)

    task_inputs: List[str] = []
    final_targets: List[str] = []
    if aligned_examples is not None:
        for example in aligned_examples:
            task_inputs.append(_build_task_input_from_source_example(example))
            final_targets.append(_resolve_final_target_from_source_example(canonical_task, example))
        return task_inputs, final_targets

    for row in _iter_dataset_rows(dataset):
        text = row["text"] if "text" in row else row["sentence"]
        label = row["label"]
        task_input = str(text).strip()
        if not task_input.startswith("Input :"):
            task_input = f"Input : {task_input}"
        task_inputs.append(task_input)
        final_targets.append(_resolve_final_target(canonical_task, task_input, label, verbalizer=verbalizer))
    return task_inputs, final_targets


def _build_reasoning_params(reasoning_max_tokens: int):
    if SamplingParams is None:
        raise ImportError("vllm is required to build SamplingParams for BBH evaluator.")
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(reasoning_max_tokens),
    )


def _build_answer_params(answer_max_tokens: int = 1):
    if SamplingParams is None:
        raise ImportError("vllm is required to build SamplingParams for BBH evaluator.")
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(answer_max_tokens),
        stop=ANSWER_EXTRACTION_STOPS,
    )


@torch.inference_mode()
def evaluate_prompts_chunked_bbh5_gfnpo(
    prompts: Sequence[str],
    dataset: Any,
    model: Any,
    task_name: str,
    verbalizer: Any = None,
    chunk_size: int = 128,
    reasoning_max_tokens: int = 50,
    answer_max_tokens: int = 1,
    reasoning_params: Any = None,
    answer_params: Any = None,
    debug_writer: Any = None,
    debug_context: Dict[str, Any] | None = None,
    debug_max_prompts: int = 0,
    debug_max_samples: int = 0,
) -> torch.Tensor:
    canonical_task = canonicalize_bbh_task(task_name)
    if canonical_task not in BBH5_TASKS:
        raise ValueError(f"Task '{task_name}' is not supported by gfn_po-style BBH evaluator.")

    num_prompts = len(prompts)
    if num_prompts == 0:
        return torch.zeros(0, dtype=torch.float32)

    task_inputs, final_targets = extract_bbh5_inputs_and_targets(
        dataset=dataset,
        task_name=canonical_task,
        verbalizer=verbalizer,
    )
    dataset_len = len(task_inputs)
    if dataset_len == 0:
        return torch.zeros(num_prompts, dtype=torch.float32)

    if reasoning_params is None:
        reasoning_params = _build_reasoning_params(reasoning_max_tokens)
    if answer_params is None:
        answer_params = _build_answer_params(answer_max_tokens)

    normalized_targets = [remove_parentheses_if_single_char(str(target).strip()) for target in final_targets]
    extractor_text = TASK_EXTRACTOR_TEXT[canonical_task]
    correct = torch.zeros(num_prompts, dtype=torch.long)
    debug_counts = [0 for _ in range(num_prompts)]
    debug_source_path = _detect_bbh_source_path(canonical_task)

    total = num_prompts * dataset_len
    chunk_size = max(1, int(chunk_size))

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        stage1_prompts: List[str] = []
        pd_pairs: List[Tuple[int, int]] = []

        for global_idx in range(start, end):
            prompt_idx = global_idx // dataset_len
            data_idx = global_idx % dataset_len
            user_content = _compose_user_content(prompts[prompt_idx], task_inputs[data_idx])
            stage1_prompts.append(render_llama3_user_prompt(user_content))
            pd_pairs.append((prompt_idx, data_idx))

        stage1_outputs = model.generate(stage1_prompts, reasoning_params, use_tqdm=False)
        reasonings = [out.outputs[0].text if out.outputs else "" for out in stage1_outputs]

        extractor_prompts = [
            render_extractor_prompt_llama3(stage1_prompt, reasoning, extractor_text)
            for stage1_prompt, reasoning in zip(stage1_prompts, reasonings)
        ]
        answer_outputs = model.generate(extractor_prompts, answer_params, use_tqdm=False)

        for i, output in enumerate(answer_outputs):
            prompt_idx, data_idx = pd_pairs[i]
            raw_answer = output.outputs[0].text if output.outputs else ""
            normalized_answer = remove_parentheses_if_single_char(str(raw_answer).strip())
            is_correct = normalized_answer == normalized_targets[data_idx]
            if is_correct:
                correct[prompt_idx] += 1
            if (
                debug_writer is not None
                and prompt_idx < max(0, int(debug_max_prompts))
                and debug_counts[prompt_idx] < max(0, int(debug_max_samples))
            ):
                record = dict(debug_context or {})
                record.update({
                    "task_name": canonical_task,
                    "source_path": debug_source_path,
                    "prompt_idx": int(prompt_idx),
                    "data_idx": int(data_idx),
                    "prompt": str(prompts[prompt_idx]),
                    "task_input": str(task_inputs[data_idx]),
                    "reasoning": str(reasonings[i]),
                    "raw_answer": str(raw_answer),
                    "normalized_answer": str(normalized_answer),
                    "target": str(normalized_targets[data_idx]),
                    "correct": bool(is_correct),
                })
                debug_writer(record)
                debug_counts[prompt_idx] += 1

    return correct.float() / float(dataset_len)
