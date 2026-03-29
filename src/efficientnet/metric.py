from torch import Tensor


class MeanMetric:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += value * weight
        self.count += weight

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class Accuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, outputs: Tensor, targets: Tensor) -> None:
        _, preds = outputs.max(1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total
