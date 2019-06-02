from __future__ import division

import torch


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        assert pred.size(0) == true.size(0)

        self.correct += pred.data.eq(true.data).sum().item()
        self.count += pred.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)
