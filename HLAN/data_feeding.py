from typing import Any, Iterable, Mapping

import numpy as np

from HLAN.HAN_model_dynamic import HAN


def feed_data(
    model: HAN, X: np.ndarray, Y: np.ndarray, batch_size: int, dropout_rate: float = 0.0
) -> Iterable[Mapping[Any, Any]]:
    n = len(X)
    b = batch_size
    assert len(Y) == n
    # trim the final slice to *exactly* the size of the input, rather than overshooting
    batches = [slice(i, i + b) for i in range(0, n, b)][:-1] + [slice(n - (n % b), n)]

    for batch in batches:
        batch_X = X[batch]
        batch_Y = Y[batch]
        yield {
            model.input_x: batch_X,
            model.input_y: batch_Y,
            model.dropout_rate: dropout_rate,
        }
