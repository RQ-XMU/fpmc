from __future__ import division

import numpy as np
import scipy.sparse as ssp
import os.path
import random
import operator
import collections


class Evaluator(object):
    def __init__(self, dataset, k=10):
        super(Evaluator, self).__init__()
        self.instances = []
        self.dataset = dataset
        self.k = k

        self.metrics = {'sps': self.short_term_prediction_success,
             'recall': self.average_recall,
             'precision': self.average_precision,
             'ndcg': self.average_ndcg,
             'item_coverage': self.item_coverage,
             'user_coverage': self.user_coverage,
             'assr': self.assr,
             'blockbuster_share': self.blockbuster_share}

    def add_instance(self, goal, predictions):
        self.instances.append([goal, predictions])

    def average_precision(self):
        precision = 0
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                precision += float(len(set(goal) & set(prediction[:min(len(prediction), self.k)]))) / min(len(prediction), self.k)

        return precision / len(self.instances)

    def average_recall(self):
        recall = 0
        for goal, prediction in self.instances:
            if len(goal) > 0:
                recall += float(len(set(goal) & set(prediction[:min(len(prediction), self.k)]))) / len(goal)

        return recall / len(self.instances)

    def average_ndcg(self):
        ndcg = 0
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                dcg = 0
                max_dcg = 0
                for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
                    if i < len(goal):
                        max_dcg += 1. / np.log2(2 +i)

                    if p in goal:
                        dcg = 1. /np.log2(2 + i)
                ndcg += dcg/max_dcg

        return ndcg / len(self.instances)

    def user_coverage(self):
        score = 0
        for goal, prediction in self.instances:
            score += int(len(set(goal) & set(prediction[:min(len(prediction), self.k)])) > 0)

        return score / len(self.instances)

    def get_correct_predictions(self):
        correct_predictions = []
        for goal, prediction in self.instances:
            correct_predictions.extend(list(set(goal) & set(prediction[:min(len(prediction), self.k)])))
        return correct_predictions

    def item_coverage(self):
        return len(set(self.get_correct_predictions()))