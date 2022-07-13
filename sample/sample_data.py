import random

import numpy as np


class sampling:
    def __init__(self, crowd_file, number):
        # initialise datafile
        self.crowd_file = crowd_file
        self.number = number
        t2wl, w2tl, task_set, label_set = self.get_info_data()
        self.t2wl = t2wl
        self.w2tl = w2tl
        self.task_set = task_set
        self.label_set = label_set
        self.num_workers = len(w2tl.keys())
        self.num_tasks = len(t2wl.keys())
        self.num_labels = len(label_set)
    def get_info_data(self):
        t2wl = {}
        w2tl = {}
        task_set = set()
        label_set = set()
        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]
        for line in reader:
            task, worker, label = line.split('\t')
            if task not in t2wl:
                t2wl[task] = {}
            t2wl[task][worker] = label    # {t0:{w0：l0, w1:l1, ... }}
            if worker not in w2tl:
                w2tl[worker] = {}
            w2tl[worker][task] = label    # {w0:{t0：l0, t1:l1, ... }}
            if task not in task_set:
                task_set.add(task)
            if label not in label_set:
                label_set.add(label)
        return t2wl, w2tl, task_set, label_set

    def run(self, sample_file):
        task_total = list(self.t2wl.keys())
        sample_task = random.sample(task_total, self.number)
        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(sample_file, 'w')
        workers_list = []
        for line in reader:
            task, worker, label = line.split('\t')
            if task in sample_task:
                f_save.write(task + '\t' + worker + '\t' + label + '\n')
            if worker not in workers_list:
                workers_list.append(worker)
        f_open.close()
        f_save.close()
        return len(workers_list)
    def run_fixed_task(self):
        task_total = list(self.t2wl.keys())
        exist_task = random.sample(task_total, self.number)
        return exist_task

    def run_fixed_annotator(self):
        annotator_total = list(self.w2tl.keys())
        exist_annotator = random.sample(annotator_total, self.number)
        return exist_annotator
