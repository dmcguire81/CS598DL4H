from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

import HLAN.validation as valid


def test_update_validation_performance_saves_checkpoint_on_improvement(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    empty_ckpt_dir: Path,
):
    mock_model = mocker.MagicMock()
    mock_session = mocker.MagicMock()

    mock_roc_auc_score = mocker.MagicMock()
    monkeypatch.setattr(valid.metrics, "roc_auc_score", mock_roc_auc_score)

    # performance has improved
    best_micro_f1_score = 0
    mock_f1_score = mocker.MagicMock(return_value=1)
    monkeypatch.setattr(valid.metrics, "f1_score", mock_f1_score)

    mock_saver = mocker.MagicMock()
    mock_saver_constructor = mocker.MagicMock(return_value=mock_saver)
    monkeypatch.setattr(valid.tf.train, "Saver", mock_saver_constructor)

    epoch = 0
    all_predictions = np.empty((0, 50))
    all_labels = np.empty((0, 50))

    best_micro_f1_score = valid.update_validation_performance(
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
    monkeypatch.setattr(valid.metrics, "roc_auc_score", mock_roc_auc_score)

    # performance has degraded
    best_micro_f1_score = 1
    mock_f1_score = mocker.MagicMock(return_value=0)
    monkeypatch.setattr(valid.metrics, "f1_score", mock_f1_score)

    epoch = 0
    all_predictions = np.empty((0, 50))
    all_labels = np.empty((0, 50))

    best_micro_f1_score = valid.update_validation_performance(
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
