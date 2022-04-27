from typing import NamedTuple

import numpy as np
from sklearn import metrics


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


class ModelOutputs(NamedTuple):
    logits: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray


class SummaryPerformance(NamedTuple):
    micro_f1_score: float
    micro_roc_auc_score: float


def compute_summary_performance(model_outputs: ModelOutputs) -> SummaryPerformance:
    micro_roc_auc_score = metrics.roc_auc_score(
        model_outputs.labels, sigmoid(model_outputs.logits), average="micro"
    )
    micro_f1_score = metrics.f1_score(
        model_outputs.labels, model_outputs.predictions, average="micro"
    )
    return SummaryPerformance(
        micro_roc_auc_score=micro_roc_auc_score, micro_f1_score=micro_f1_score
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
