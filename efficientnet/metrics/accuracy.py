import torch
from torch import Tensor

from .metric import Metric


class Accuracy(Metric):
    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct = 0
        self.count = 0

    def update(self, output: Tensor, target: Tensor) -> float:
        assert output.size(0) == target.size(0)

        with torch.no_grad():
            _, pred = output.topk(self.top_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[: self.top_k].view(-1).float().sum(0, keepdim=True).item()

        self.correct += correct_k
        self.count += output.size(0)

    @property
    def value(self) -> float:
        return 100 * self.correct / self.count

    def __str__(self) -> str:
        return f"{self.value:.2f}%"
