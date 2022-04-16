from pathlib import Path
from typing import List

import numpy as np

from HLAN.HAN_train import load_data


def test_load_data(
    word2vec_model_path: Path,
    caml_dataset_paths: List[Path],
    sequence_length,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    (num_classes, (trainX, trainY), (validX, validY), (testX, testY)) = load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )

    assert num_classes == 50

    assert isinstance(trainX, np.ndarray)
    assert isinstance(trainY, np.ndarray)
    assert isinstance(validX, np.ndarray)
    assert isinstance(validY, np.ndarray)
    assert isinstance(testX, np.ndarray)
    assert isinstance(testY, np.ndarray)

    assert len(trainX) == len(trainY)
    assert len(validX) == len(validY)
    assert len(testX) == len(testY)
    assert len(trainY) == 8066
    assert len(validY) == 1573
    assert len(testY) == 1729

    assert np.mean(trainX) == 1078.8257977188198
    assert np.mean(trainY) == 0.11385817009670221
    assert np.mean(validX) == 1440.640283534647
    assert np.mean(validY) == 0.11802924348378893
    assert np.mean(testX) == 1465.9599076923078
    assert np.mean(testY) == 0.12119144013880856
