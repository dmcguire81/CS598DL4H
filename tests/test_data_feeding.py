from pathlib import Path
from typing import List

import pytest
from gensim.models import Word2Vec
from pytest_mock import MockerFixture

from HLAN import data_feeding, data_loading


def test_feed_data(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    batch_size: int,
    sequence_length: int,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    monkeypatch.setattr(
        data_loading.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    _, training_data_split = data_loading.load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )
    monkeypatch.undo()

    mock_model = mocker.MagicMock()

    (trainX, trainY) = training_data_split.training
    (validX, validY) = training_data_split.validation
    (testX, testY) = training_data_split.testing

    for (X, Y, dropout_rate) in [
        (trainX, trainY, 0.5),
        (validX, validY, 0.0),
        (testX, testY, 0.0),
    ]:
        n = len(X)
        b = batch_size
        batches = list(
            data_feeding.feed_data(mock_model, X, Y, batch_size, dropout_rate)
        )
        assert len(batches) == int(n / b) + 1 if (n % b) else 0
        assert len(batches[-1][mock_model.input_x]) == len(X) % batch_size
        assert len(batches[-1][mock_model.input_y]) == len(Y) % batch_size
        assert all(
            [len(batch[mock_model.input_x]) == batch_size for batch in batches[:-1]]
        )
        assert all(
            [len(batch[mock_model.input_y]) == batch_size for batch in batches[:-1]]
        )
        assert all(
            [batch[mock_model.dropout_rate] == dropout_rate for batch in batches]
        )
