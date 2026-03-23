import heapq
import random


class TopAccuracyTextsNoDuplicates:
    def __init__(self, max_size=5):
        self.heap = []
        self.text_map = {}
        self.max_size = max_size
        self.only_text = []

    def add(self, accuracy, text, ep):
        if text in self.only_text:
            return False
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (accuracy, len(text), text, ep))
            self.text_map[text] = (len(self.heap) - 1, ep)
            self.only_text.append(text)
            return True
        if accuracy > self.heap[0][0]:
            removed_text = heapq.heappop(self.heap)[2]
            if removed_text in self.text_map:
                self.text_map.pop(removed_text)
            if removed_text in self.only_text:
                self.only_text.remove(removed_text)
            heapq.heappush(self.heap, (accuracy, len(text), text, ep))
            self.text_map[text] = (len(self.heap) - 1, ep)
            self.only_text.append(text)
            return True
        return False

    def get_top_texts(self):
        return sorted([(accuracy, text, ep) for accuracy, _, text, ep in self.heap], reverse=True)


def got_example(dataset, dataset_dict, shot=5, label_key='label'):
    examples = ''
    if len(dataset) == 0:
        return examples
    for _ in range(shot):
        idx = random.randint(0, len(dataset) - 1)
        example = dataset[idx]
        if example[label_key] == -1:
            continue
        if 'text' in example:
            examples += example['text'] + '\nOutput : ' + str(dataset_dict[example[label_key]]) + '\n'
    return examples


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


def got_example_mmlu(dataset, dataset_dict, shot=5, label_key='label'):
    return got_example(dataset, dataset_dict, shot=shot, label_key=label_key)
