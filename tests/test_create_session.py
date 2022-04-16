from pathlib import Path
from typing import List

import numpy as np
import pytest
from gensim.models import Word2Vec
from pytest_mock.plugin import MockerFixture

from HLAN import create_session as cs
from HLAN import load_data as ld


def test_assign_pretrained_word_embedding(
    word2vec_model_path: Path,
):
    embedding: np.ndarray = cs.assign_pretrained_word_embedding(
        word2vec_model_path,
    )

    assert embedding.mean() == pytest.approx(0.000983558)


def test_assign_pretrained_label_embedding_not_per_label(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    label_embedding_model_path: Path,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    monkeypatch.setattr(
        ld.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = ld.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )
    monkeypatch.undo()

    w_projection: np.ndarray = cs.assign_pretrained_label_embedding(
        onehot_encoding.reverse.labels,
        label_embedding_model_path,
    )

    assert w_projection.mean() == pytest.approx(-0.0017410218)


def test_assign_pretrained_label_embedding_per_label(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    label_embedding_model_path_per_label: Path,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    monkeypatch.setattr(
        ld.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = ld.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )
    monkeypatch.undo()

    context_vector_per_label = cs.assign_pretrained_label_embedding(
        onehot_encoding.reverse.labels,
        label_embedding_model_path_per_label,
    )

    assert context_vector_per_label.mean() == pytest.approx(-0.001482437)
