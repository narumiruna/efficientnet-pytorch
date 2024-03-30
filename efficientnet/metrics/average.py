from .metric import Metric


class Average(Metric):
    def __init__(self) -> None:
        self.sum = 0
        self.count = 0

    def update(self, value: float, number: int = 1) -> None:
        self.sum += value * number
        self.count += number

    @property
    def value(self) -> float:
        if self.count == 0:
            return float("inf")
        else:
            return self.sum / self.count

    def __str__(self) -> str:
        return f"{self.value:.4f}"
