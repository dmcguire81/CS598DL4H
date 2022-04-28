from pathlib import Path
from typing import List, Type

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
from gensim.models import Word2Vec
from pytest_mock import MockerFixture

from HLAN import data_loading, session_creation
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN


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
    mock_hlan_class: Type,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    monkeypatch.setattr(
        data_loading.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = data_loading.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )
    monkeypatch.undo()

    with tf.variable_scope(""):
        model = mock_hlan_class(
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

        with session_creation.create_session(
            model=model,
            ckpt_dir=ckpt_dir,
            remove_ckpts_before_train=remove_ckpts_before_train,
            per_label_attention=None,  # type: ignore
            per_label_sent_only=None,  # type: ignore
            reverse_embedding=None,  # type: ignore
            word2vec_model_path=None,  # type: ignore
            label_embedding_model_path=None,  # type: ignore
            label_embedding_model_path_per_label=None,  # type: ignore
            use_label_embedding=True,
        ) as session:
            assert session

            assert model.Embedding.eval().mean() == pytest.approx(
                -8.406006963923573e-05
            )
            assert model.W_projection.eval().mean() == pytest.approx(-0.0108663635)
            assert model.context_vector_word_per_label.eval().mean() == pytest.approx(
                -0.009843622334301472
            )


@pytest.mark.slow()
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
        data_loading.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = data_loading.load_encoding_data(
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
    with tf.variable_scope(
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
            log_dir=tmp_path,
            initializer=tf.random_normal_initializer(stddev=0.1, seed=0),
        )

        with session_creation.create_session(
            model=model,
            ckpt_dir=empty_ckpt_dir,
            remove_ckpts_before_train=remove_ckpts_before_train,
            per_label_attention=per_label_attention,
            per_label_sent_only=per_label_sent_only,
            reverse_embedding=onehot_encoding.reverse,
            word2vec_model_path=word2vec_model_path,
            label_embedding_model_path=label_embedding_model_path,
            label_embedding_model_path_per_label=label_embedding_model_path_per_label,
            use_label_embedding=True,
        ) as session:
            assert session

            assert model.Embedding.eval().mean() == pytest.approx(0.000983558)
            assert model.W_projection.eval().mean() == pytest.approx(-0.0017410218)
            if per_label_attention:
                if not per_label_sent_only:
                    assert not hasattr(model, "context_vector_word")
                    assert (
                        model.context_vector_word_per_label.eval().mean()
                        == pytest.approx(-0.001482437)
                    )
                else:
                    assert model.context_vector_word.eval().mean() == pytest.approx(
                        -0.004016303
                    )
                    assert not hasattr(model, "context_vector_word_per_label")

                assert not hasattr(model, "context_vector_sentence")
                assert (
                    model.context_vector_sentence_per_label.eval().mean()
                    == pytest.approx(-0.001482437)
                )
            else:
                assert model.context_vector_word.eval().mean() == pytest.approx(
                    -0.004016303
                )
                assert not hasattr(model, "context_vector_word_per_label")

                assert model.context_vector_sentence.eval().mean() == pytest.approx(
                    -0.004016303
                )
                assert not hasattr(model, "context_vector_sentence_per_label")


def test_assign_pretrained_word_embedding(
    word2vec_model_path: Path,
):
    embedding: np.ndarray = session_creation.assign_pretrained_word_embedding(
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
        data_loading.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = data_loading.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )
    monkeypatch.undo()

    w_projection: np.ndarray = session_creation.assign_pretrained_label_embedding(
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
        data_loading.Word2Vec, "load", mocker.MagicMock(return_value=word2vec_model)
    )
    onehot_encoding = data_loading.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )
    monkeypatch.undo()

    context_vector_per_label = session_creation.assign_pretrained_label_embedding(
        onehot_encoding.reverse.labels,
        label_embedding_model_path_per_label,
    )

    assert context_vector_per_label.mean() == pytest.approx(-0.001482437)
