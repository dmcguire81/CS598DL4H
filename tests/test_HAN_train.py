from pathlib import Path
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from gensim.models import Word2Vec
from pytest_mock import MockerFixture

from HLAN import load_data as ld
from HLAN.HAN_model_dynamic import HAN
from HLAN.HAN_train import create_session, load_data


def test_load_data(
    word2vec_model_path: Path,
    caml_dataset_paths: List[Path],
    sequence_length,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    onehot_encoding, training_data_split = load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )

    assert len(onehot_encoding.forward.labels) == 50
    assert len(onehot_encoding.forward.words) == 150855

    assert len(onehot_encoding.reverse.labels) == 50
    assert len(onehot_encoding.reverse.words) == 150855

    (trainX, trainY) = training_data_split.training
    (validX, validY) = training_data_split.validation
    (testX, testY) = training_data_split.testing

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


def test_create_session_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    batch_size: int,
    per_label_attention: bool,
    per_label_sent_only: bool,
    sequence_length: int,
    ckpt_dir: Path,
    remove_ckpts_before_train: bool,
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

    model = HAN(
        num_classes=onehot_encoding.num_classes,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=onehot_encoding.vocab_size,
        embed_size=100,
        hidden_size=100,
        per_label_attention=per_label_attention,
        per_label_sent_only=per_label_sent_only,
    )

    with create_session(
        model=model,
        ckpt_dir=ckpt_dir,
        remove_ckpts_before_train=remove_ckpts_before_train,
        per_label_attention=None,  # type: ignore
        per_label_sent_only=None,  # type: ignore
        reverse_embedding=None,  # type: ignore
        word2vec_model_path=None,  # type: ignore
        label_embedding_model_path=None,  # type: ignore
        label_embedding_model_path_per_label=None,  # type: ignore
    ) as session:
        assert session

        assert model.Embedding.eval().mean() == pytest.approx(-8.406006963923573e-05)
        assert model.W_projection.eval().mean() == pytest.approx(-0.0108663635)
        if not per_label_sent_only:
            assert model.context_vector_word_per_label.eval().mean() == pytest.approx(
                -0.009843622334301472
            )
        assert model.context_vector_sentence_per_label.eval().mean() == pytest.approx(
            -0.0023771661799401045
        )


def test_create_session_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    batch_size: int,
    per_label_attention: bool,
    per_label_sent_only: bool,
    sequence_length: int,
    empty_ckpt_dir: Path,
    remove_ckpts_before_train: bool,
    label_embedding_model_path: Path,
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

    # The TensorFlow session appears to be process global, so this and the above
    # model definitions are colliding because they (re)-define the same variables
    # in the session without explicitly declaring reuse=True or reuse=tf.AUTO_REUSE.
    # Now, because the original implementation, for which we have a checkpoint, was
    # *not* scoped, the checkpoint can't be loaded to a scope, so using the same scope
    # and allowing reuse isn't an option. However, because we don't actually care
    # where the session thinks the variables in the model live, so long as it lets
    # us assert on them, so we scope *only the from-scratch* load to keep the two
    # from colliding in a single test process.
    with tf.compat.v1.variable_scope("test_create_session_from_scratch"):

        model = HAN(
            num_classes=onehot_encoding.num_classes,
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=onehot_encoding.vocab_size,
            embed_size=100,
            hidden_size=100,
            per_label_attention=per_label_attention,
            per_label_sent_only=per_label_sent_only,
        )

        with create_session(
            model=model,
            ckpt_dir=empty_ckpt_dir,
            remove_ckpts_before_train=remove_ckpts_before_train,
            per_label_attention=per_label_attention,
            per_label_sent_only=per_label_sent_only,
            reverse_embedding=onehot_encoding.reverse,
            word2vec_model_path=word2vec_model_path,
            label_embedding_model_path=label_embedding_model_path,
            label_embedding_model_path_per_label=label_embedding_model_path_per_label,
        ) as session:
            assert session

            assert model.Embedding.eval().mean() == pytest.approx(0.000983558)
            assert model.W_projection.eval().mean() == pytest.approx(-0.0017410218)
            if not per_label_sent_only:
                assert (
                    model.context_vector_word_per_label.eval().mean()
                    == pytest.approx(-0.001482437)
                )
            assert (
                model.context_vector_sentence_per_label.eval().mean()
                == pytest.approx(-0.001482437)
            )
