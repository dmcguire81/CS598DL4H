import sys
from typing import Iterable, Tuple

import pandas as pd
from scipy import stats


def performance_with_confidence_interval(
    performance: pd.DataFrame, confidence: float = 0.95
) -> Iterable[Tuple[str, float, float]]:
    # find the critical t-value for a 2-tailed Gaussian confidence interval
    parameters = 1
    tails = 2
    samples = len(performance)
    degrees_of_freedom = samples - parameters
    significance = 1 - confidence
    t = stats.t.ppf(1 - (significance / tails), degrees_of_freedom)

    for column in performance.columns:
        yield column, performance[column].mean(), performance[column].sem() * t


def main(performance_csv: str):
    for metric, measure, standard_error in performance_with_confidence_interval(
        pd.read_csv(performance_csv)
    ):
        print(f"{metric}: {measure * 100:.1f}+/-{standard_error * 100:.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
