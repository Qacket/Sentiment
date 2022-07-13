import csv
import random

import numpy as np


class MV_generate:

    def __init__(self, crowd_file, truth_file, **kwargs):

        # initialise datafile
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        t2wl, task_set, label_set = self.get_info_data()
        self.t2wl = t2wl
        self.task_set = task_set
        self.num_tasks = len(task_set)
        self.label_set = label_set
        self.num_labels = len(label_set)

    def get_info_data(self):
        t2wl = {}
        task_set = set()
        label_set = set()

        # f = open(self.datafile, 'r')
        # reader = csv.reader(f)
        # next(reader)

        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            task, worker, label = line.split('\t')
            if task not in t2wl:
                t2wl[task] = {}

            t2wl[task][worker] = label  # {t0:{w0：l0, w1:l1, ... }}

            if task not in task_set:
                task_set.add(task)

            if label not in label_set:
                label_set.add(label)

        return t2wl, task_set, label_set

    def get_accuracy(self):
        if not hasattr(self, 'truth_file'):
            raise BaseException('There is no truth file!')
        if not hasattr(self, 't2a'):
            raise BaseException('There is no aggregated answers!')

        count = []

        # f = open(self.truth_file, 'r')
        # reader = csv.reader(f)
        # next(reader)

        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            task, truth = line.split('\t')
            if task not in self.t2a:
                print('task %s in not found in the list of answers' % task)
            elif self.t2a[task] == truth:
                count.append(1)
            else:
                count.append(0)

        return sum(count) / len(count)

    def run(self):
        # initialization
        count = {}  # {t:{l0:0, l1:0 , l2:0, ...}}
        for task in self.task_set:
            count[task] = {}
            for label in self.label_set:
                count[task][label] = 0

        # compute
        for task in self.task_set:
            for worker in self.t2wl[task]:
                label = self.t2wl[task][worker]
                count[task][label] += 1


        # compute_p
        t2p = count.copy()  # {t:{l0:p0, l1:p1 , l2:p2, ...}}

        for task in self.task_set:
            sum = len(self.t2wl[task].keys())
            for label in self.label_set:
                t2p[task][label] = t2p[task][label]/sum
        self.t2p = t2p


    def generate(self, sample_file, generate_file):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()
        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:

            task, worker, label = line.split('\t')
            l2p = {}
            for label in self.label_set:
                if label == e2truth[task]:
                    l2p[label] = self.t2p[task][e2truth[task]]
                else:
                    l2p[label] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
            p = np.array(list(l2p.values()))
            label = np.random.choice(list(l2p.keys()), p=p.ravel())
            f_save.write(task+'\t'+worker+'\t'+label+'\n')

        f_open.close()
        f_save.close()


    def generate_fixed_annotator(self, exist_annotator, generate_file):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()



        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:

            task, worker, label = line.split('\t')
            if worker in exist_annotator:
                f_save.write(task + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label in self.label_set:
                    if label == e2truth[task]:
                        l2p[label] = self.t2p[task][e2truth[task]]
                    else:
                        l2p[label] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(task+'\t'+worker+'\t'+label+'\n')

        f_open.close()
        f_save.close()

    def generate_fixed_task(self, exist_task, generate_file):
        # generate
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()
        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            task, worker, label = line.split('\t')
            if task in exist_task:
                f_save.write(task + '\t' + worker + '\t' + label + '\n')
            else:
                l2p = {}
                for label in self.label_set:
                    if label == e2truth[task]:
                        l2p[label] = self.t2p[task][e2truth[task]]
                    else:
                        l2p[label] = (1 - self.t2p[task][e2truth[task]])/(len(self.label_set) - 1)
                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_save.write(task+'\t'+worker+'\t'+label+'\n')

        f_open.close()
        f_save.close()
