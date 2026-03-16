#!/usr/bin/env python3
import argparse
import statistics

from junmo.dataset_utils import load_bigbench


def _summarize_lengths(dataset):
    lengths = [len(item["text"]) for item in dataset]
    if not lengths:
        return {"mean": 0.0, "max": 0, "min": 0}
    return {
        "mean": statistics.mean(lengths),
        "max": max(lengths),
        "min": min(lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect BBH split/examples used by GFLOWPO.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--test_offset", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()

    resolved_test_size = None if args.test_size is None or args.test_size < 0 else args.test_size
    metrics, train_dataset, test_dataset, verbalizer, task_prefix = load_bigbench(
        args.task,
        train_size=args.train_size,
        test_size=resolved_test_size,
        test_offset=args.test_offset,
    )

    print(f"task={args.task}")
    print(f"metric={metrics}")
    print(f"task_prefix={repr(task_prefix)}")
    print(f"train_size={len(train_dataset)}")
    print(f"test_size={len(test_dataset)}")
    print(f"verbalizer={verbalizer}")

    train_len = _summarize_lengths(train_dataset)
    test_len = _summarize_lengths(test_dataset)
    print(
        "train_text_len(mean/min/max)="
        f"{train_len['mean']:.1f}/{train_len['min']}/{train_len['max']}"
    )
    print(
        "test_text_len(mean/min/max)="
        f"{test_len['mean']:.1f}/{test_len['min']}/{test_len['max']}"
    )

    for split_name, dataset in (("train", train_dataset), ("test", test_dataset)):
        print(f"\n[{split_name} samples]")
        for idx in range(min(args.num_samples, len(dataset))):
            sample = dataset[idx]
            print(f"- {split_name}[{idx}] label={sample['label']}")
            print(sample["text"])
            print("---")


if __name__ == "__main__":
    main()
