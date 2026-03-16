import math
import random
import re
import string
from collections import Counter
from typing import Any

TASK_TO_METRIC = {
    'object_counting': 'f1',
    'movie_recommendation': 'em',
    'hyperbaton': 'em',
    'causal_judgment': 'em',
    'tracking_shuffled_objects_five_objects': 'em',
}
default_metric = 'em'


def load_ii_data(task: str, seed: int = 42):
    raise NotImplementedError('This repository is BBH-only. II tasks are not included.')


def normalize_prediction(prediction: str, lowercase: bool = True) -> str:
    prediction = str(prediction).replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip().split('\n')[0].split('.')[0]
    if lowercase:
        prediction = prediction.lower()
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))
    return prediction


def get_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def get_em_score(prediction: str, ground_truth: str) -> bool:
    return normalize_prediction(prediction, lowercase=True) == normalize_prediction(ground_truth, lowercase=True)


def get_exact_set_score(prediction: str, ground_truth: str) -> int:
    prediction_normalized = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction: str, ground_truth: str) -> int:
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1
    return 0


def got_example_ii(dataset: Any, shot: int = 5):
    raise NotImplementedError('This repository is BBH-only. II example formatting is not included.')


def got_example_bbh(dataset, dataset_dict, shot=5, label_key='label', metrics='multiple_choice_grade'):
    examples = ''
    if len(dataset) == 0:
        return examples
    for _ in range(shot):
        idx = random.randint(0, len(dataset) - 1)
        example = dataset[idx]
        if example[label_key] == -1:
            continue
        if 'text' not in example:
            continue
        if metrics == 'multiple_choice_grade':
            output = dataset_dict[example[label_key]]
        else:
            output = example['label']
        examples += example['text'] + '\nOutput : ' + str(output) + '\n'
    return examples
