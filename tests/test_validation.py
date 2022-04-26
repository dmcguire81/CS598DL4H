from pathlib import Path
from typing import List, Type

import numpy as np
import pytest
import tensorflow as tf
from gensim.models import Word2Vec
from pytest_mock import MockerFixture
from sklearn import metrics

from HLAN import data_feeding, data_loading, session_creation, validation
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN


def intersect_size(yhat, y, axis):
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:, i].sum() > 0:  # if the label has a true instance in the data
            fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], yhat_raw[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = metrics.auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    # macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels["auc_%d" % i])
    roc_auc["auc_macro"] = np.mean(aucs)

    # micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    return roc_auc


@pytest.mark.slow()
@pytest.mark.parametrize(
    ("per_label_attention", "per_label_sent_only", "model_class"),
    [
        (False, False, HAN),
        (True, True, HA_GRU),
        (True, False, HLAN),
    ],
)
def test_validation_performance(
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
    (onehot_encoding, training_data_split,) = data_loading.load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length=sequence_length,
    )

    with tf.compat.v1.variable_scope(
        f"test_validation_performance-{per_label_attention}-{per_label_sent_only}"
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
        )

        with session_creation.create_session(
            model=model,
            ckpt_dir=empty_ckpt_dir,
            remove_ckpts_before_train=remove_ckpts_before_train,
            per_label_attention=False,
            per_label_sent_only=False,
            reverse_embedding=onehot_encoding.reverse,
            word2vec_model_path=word2vec_model_path,
            label_embedding_model_path=label_embedding_model_path,
            label_embedding_model_path_per_label=label_embedding_model_path_per_label,
            use_label_embedding=True,
        ) as session:
            epoch = 0

            all_predictions, all_labels = validation.validate(
                session,
                model,
                epoch,
                onehot_encoding.num_classes,
                lambda: data_feeding.feed_data(
                    model, *training_data_split.validation, batch_size=batch_size
                ),
            )

        sklearn_micro_f1_score = metrics.f1_score(
            all_labels, all_predictions, average="micro"
        )
        original_micro_f1_score = micro_f1(all_predictions.ravel(), all_labels.ravel())

        assert sklearn_micro_f1_score == original_micro_f1_score

        sklearn_micro_roc_auc_score = metrics.roc_auc_score(
            all_labels, all_predictions, average="micro"
        )
        original_micro_roc_auc_score = auc_metrics(
            all_predictions, all_labels, all_labels.ravel()
        )["auc_micro"]

        assert sklearn_micro_roc_auc_score == original_micro_roc_auc_score


def test_update_validation_performance_saves_checkpoint_on_improvement(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    empty_ckpt_dir: Path,
):
    mock_model = mocker.MagicMock()
    mock_session = mocker.MagicMock()

    mock_roc_auc_score = mocker.MagicMock()
    monkeypatch.setattr(validation.metrics, "roc_auc_score", mock_roc_auc_score)

    # performance has improved
    best_micro_f1_score = 0
    mock_f1_score = mocker.MagicMock(return_value=1)
    monkeypatch.setattr(validation.metrics, "f1_score", mock_f1_score)

    mock_saver = mocker.MagicMock()
    mock_saver_constructor = mocker.MagicMock(return_value=mock_saver)
    monkeypatch.setattr(validation.tf.compat.v1.train, "Saver", mock_saver_constructor)

    epoch = 0
    all_predictions = np.empty((0, 50))
    all_labels = np.empty((0, 50))

    best_micro_f1_score, _ = validation.update_performance(
        empty_ckpt_dir,
        mock_model,
        mock_session,
        best_micro_f1_score,
        epoch,
        all_predictions,
        all_labels,
    )

    # updated
    assert best_micro_f1_score == 1

    mock_roc_auc_score.assert_called_once_with(
        all_labels,
        all_predictions,
        average="micro",
    )

    mock_f1_score.assert_called_once_with(
        all_labels,
        all_predictions,
        average="micro",
    )

    mock_saver.save.assert_called_once_with(
        mock_session,
        str(empty_ckpt_dir / "model.ckpt"),
        global_step=epoch,
    )


def test_update_validation_performance_halves_learning_rate_on_degredation(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    empty_ckpt_dir: Path,
):
    mock_model = mocker.MagicMock()
    mock_session = mocker.MagicMock()

    mock_roc_auc_score = mocker.MagicMock()
    monkeypatch.setattr(validation.metrics, "roc_auc_score", mock_roc_auc_score)

    # performance has degraded
    best_micro_f1_score = 1
    mock_f1_score = mocker.MagicMock(return_value=0)
    monkeypatch.setattr(validation.metrics, "f1_score", mock_f1_score)

    epoch = 0
    all_predictions = np.empty((0, 50))
    all_labels = np.empty((0, 50))

    best_micro_f1_score, _ = validation.update_performance(
        empty_ckpt_dir,
        mock_model,
        mock_session,
        best_micro_f1_score,
        epoch,
        all_predictions,
        all_labels,
    )

    # not updated
    assert best_micro_f1_score == 1

    mock_roc_auc_score.assert_called_once_with(
        all_labels,
        all_predictions,
        average="micro",
    )

    mock_f1_score.assert_called_once_with(
        all_labels,
        all_predictions,
        average="micro",
    )

    mock_session.run.assert_has_calls(
        [
            mocker.call(mock_model.learning_rate),
            mocker.call([mock_model.learning_rate_decay_half_op]),
            mocker.call(mock_model.learning_rate),
        ],
        any_order=False,
    )
