import numpy as np
from collections import Counter, OrderedDict


def pre_deal(crowd_file, truth_file):

    gt2t = {}
    f_truth_open = open(truth_file, 'r')
    reader = f_truth_open.readlines()
    reader = [line.strip("\n") for line in reader]
    for line in reader:
        task, gt = line.split('\t')
        if gt not in gt2t:
            gt2t[gt] = []
        gt2t[gt].append(task)

    f_truth_open.close()

    gt_sum = len(gt2t.keys())


    gt2t = sorted(gt2t.items(), key=lambda item: item[0], reverse=False)

    classification_task = []

    for i in range(len(gt2t)):
        classification_task.append(gt2t[i][1])



    data_list = []
    f_crowd_open = open(crowd_file, 'r')
    reader = f_crowd_open.readlines()
    reader = [line.strip("\n") for line in reader]
    for i in range(gt_sum):
        data = []
        for line in reader:
            task, worker, label = line.split('\t')
            if task in classification_task[i]:
                data.append(int(label))
        data_list.append(data)

    f_crowd_open.close()

    return data_list


def all_list(arr):
    x = {}
    for i in set(arr):
        x[i] = arr.count(i)
    x = sorted(x.items(), key=lambda item: item[0], reverse=False)
    result = []
    for i in range(len(x)):
        result.append(x[i][1])
    return result

def calculate_kl(x, y):
    # 归一化
    x = all_list(x)
    y = all_list(y)
    print(x)
    print(y)
    px = x/np.sum(x)
    py = y/np.sum(y)
    kl = 0.0
    for i in range(len(x)):
        kl += px[i] * np.log(px[i] / py[i])
    return kl


def idx_to_word(x, vocab, eos_idx, pad_idx):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
        if i == eos_idx or i == pad_idx:
            break
    words = " ".join(words)
    return words


def get_batch(batch):
    annotator_id = batch["annotator_id"]
    answer = batch["answer"]
    sentences = batch["input"]
    target = batch["target"]
    sentences_length = batch["length"]
    return annotator_id, answer, sentences, target, sentences_length


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)