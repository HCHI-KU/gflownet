#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def summarize(name, values):
    if not values:
        print(f"{name}: n=0")
        return
    mean = sum(values) / len(values)
    print(
        f"{name}: n={len(values)} mean={mean:.4f} "
        f"p50={percentile(values, 0.5):.4f} "
        f"p90={percentile(values, 0.9):.4f} "
        f"min={min(values):.4f} max={max(values):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze relative scale of prior and accuracy terms from train.jsonl.")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--train_size", type=int, default=None,
                        help="If beta is omitted, uses 1/train_size.")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--reward_epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    if args.beta is None:
        if not args.train_size:
            raise ValueError("Provide --beta or --train_size.")
        beta = 1.0 / args.train_size
    else:
        beta = args.beta

    rows = load_jsonl(Path(args.log_dir) / "train.jsonl")

    prior_terms = []
    acc_terms = []
    reward_terms = []
    dominance_ratios = []

    for row in rows:
        for prompt in row.get("prompts", []):
            acc = float(prompt["train_acc"])
            log_prior = float(prompt["log_prior"])
            log_accuracy = math.log(acc + args.reward_epsilon)
            prior_term = log_prior / args.gamma
            acc_term = log_accuracy / beta
            reward_term = float(prompt["log_reward"])
            prior_terms.append(abs(prior_term))
            acc_terms.append(abs(acc_term))
            reward_terms.append(abs(reward_term))
            if abs(acc_term) > 0:
                dominance_ratios.append(abs(prior_term) / abs(acc_term))

    print(f"log_dir={args.log_dir}")
    print(f"beta={beta:.8f}")
    print(f"gamma={args.gamma:.4f}")
    summarize("|prior_term|", prior_terms)
    summarize("|acc_term|", acc_terms)
    summarize("|log_reward|", reward_terms)
    summarize("|prior_term| / |acc_term|", dominance_ratios)


if __name__ == "__main__":
    main()
