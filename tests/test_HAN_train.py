from pathlib import Path
from typing import List, Type

import numpy as np
import pytest
import tensorflow as tf
from gensim.models import Word2Vec
from pytest_mock import MockerFixture

from HLAN import load_data as ld
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN
from HLAN.HAN_train import create_session, feed_data, load_data


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
    sequence_length: int,
    num_sentences: int,
    ckpt_dir: Path,
    remove_ckpts_before_train: bool,
    tmp_path: Path,
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

    model = HLAN(
        num_classes=onehot_encoding.num_classes,
        learning_rate=0.01,
        batch_size=batch_size,
        decay_steps=6000,
        decay_rate=1.0,
        sequence_length=sequence_length,
        num_sentences=num_sentences,
        vocab_size=onehot_encoding.vocab_size,
        embed_size=100,
        hidden_size=100,
        log_dir=tmp_path,
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
        assert model.context_vector_word_per_label.eval().mean() == pytest.approx(
            -0.009843622334301472
        )


@pytest.mark.parametrize(
    ("per_label_attention", "per_label_sent_only", "model_class"),
    [
        (False, False, HAN),
        (True, True, HA_GRU),
        (True, False, HLAN),
    ],
)
def test_create_session_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    batch_size: int,
    sequence_length: int,
    num_sentences: int,
    empty_ckpt_dir: Path,
    remove_ckpts_before_train: bool,
    label_embedding_model_path: Path,
    label_embedding_model_path_per_label: Path,
    tmp_path: Path,
    per_label_attention: bool,
    per_label_sent_only: bool,
    model_class: Type,
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

    # NOTE: TensorFlow sessions appear to be process global, so this and the above
    # model definitions are colliding because they (re)-define the same variables
    # in the session without explicitly declaring reuse=True or reuse=tf.AUTO_REUSE.
    # Now, because the original implementation, for which we have a checkpoint, was
    # *not* scoped, the checkpoint can't be loaded to a scope, so using the same scope
    # and allowing reuse isn't an option. However, because we don't actually care
    # where the session thinks the variables in the model live, so long as it lets
    # us assert on them, so we scope *only the from-scratch* load to keep the two
    # from colliding in a single test process.
    with tf.compat.v1.variable_scope(
        f"test_create_session_from_scratch-{per_label_attention}-{per_label_sent_only}"
    ):

        model = model_class(
            num_classes=onehot_encoding.num_classes,
            learning_rate=0.01,
            batch_size=batch_size,
            decay_steps=6000,
            decay_rate=1.0,
            sequence_length=sequence_length,
            num_sentences=num_sentences,
            vocab_size=onehot_encoding.vocab_size,
            embed_size=100,
            hidden_size=100,
            logs_dir=tmp_path,
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


def test_feed_data(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
    word2vec_model: Word2Vec,
    batch_size: int,
    num_sentences: int,
    sequence_length: int,
    tmp_path: Path,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    monkeypatch.setattr(
        ld.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding, training_data_split = load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )
    monkeypatch.undo()

    # See NOTE on scoping in test above
    with tf.compat.v1.variable_scope("test_feed_data"):
        model = HLAN(
            num_classes=onehot_encoding.num_classes,
            learning_rate=0.01,
            batch_size=batch_size,
            decay_steps=6000,
            decay_rate=1.0,
            sequence_length=sequence_length,
            num_sentences=num_sentences,
            vocab_size=onehot_encoding.vocab_size,
            embed_size=100,
            hidden_size=100,
            log_dir=tmp_path,
        )

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
            batches = list(feed_data(model, X, Y, batch_size, dropout_rate))
            assert len(batches) == int(n / b) + 1 if (n % b) else 0
            assert len(batches[-1][model.input_x]) == len(X) % batch_size
            assert len(batches[-1][model.input_y]) == len(Y) % batch_size
            assert all(
                [len(batch[model.input_x]) == batch_size for batch in batches[:-1]]
            )
            assert all(
                [len(batch[model.input_y]) == batch_size for batch in batches[:-1]]
            )
            assert all([batch[model.dropout_rate] == dropout_rate for batch in batches])
