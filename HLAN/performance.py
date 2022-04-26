from typing import NamedTuple


class ModelPerformance(NamedTuple):
    loss: float
    precision: float
    recall: float
    jaccard_index: float

    def __add__(self, other):
        return ModelPerformance(
            self.loss + other.loss,
            self.precision + other.precision,
            self.recall + other.recall,
            self.jaccard_index + other.jaccard_index,
        )

    def __truediv__(self, scalar):
        return ModelPerformance(
            self.loss / scalar,
            self.precision / scalar,
            self.recall / scalar,
            self.jaccard_index / scalar,
        )


class RunningModelPerformance:
    def __init__(self, performance: ModelPerformance, count: int):
        self.performance_sum = performance
        self.count = count

    @staticmethod
    def empty():
        return RunningModelPerformance(ModelPerformance(0, 0, 0, 0), 0)

    def __add__(self, performance: ModelPerformance):
        return RunningModelPerformance(
            self.performance_sum + performance, self.count + 1
        )

    def average(self) -> ModelPerformance:
        return self.performance_sum / self.count
