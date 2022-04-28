from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from HLAN import validation
from HLAN.performance import ModelOutputs, SummaryPerformance


def test_update_validation_performance_saves_checkpoint_on_improvement(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    empty_ckpt_dir: Path,
):
    mock_model = mocker.MagicMock()
    mock_session = mocker.MagicMock()

    # performance has improved
    best_micro_f1_score = 0
    mock_summary_performance = SummaryPerformance(
        micro_f1_score=1, micro_roc_auc_score=1
    )
    mock_compute_summary_performance = mocker.MagicMock(
        return_value=mock_summary_performance
    )
    monkeypatch.setattr(
        validation, "compute_summary_performance", mock_compute_summary_performance
    )

    mock_saver = mocker.MagicMock()
    mock_saver_constructor = mocker.MagicMock(return_value=mock_saver)
    monkeypatch.setattr(validation.tf.train, "Saver", mock_saver_constructor)

    epoch = 0
    model_outputs = ModelOutputs(
        logits=np.empty((0, 50)),
        predictions=np.empty((0, 50)),
        labels=np.empty((0, 50)),
    )

    best_micro_f1_score, _ = validation.update_performance(
        empty_ckpt_dir,
        mock_model,
        mock_session,
        best_micro_f1_score,
        epoch,
        model_outputs,
    )

    # updated
    assert best_micro_f1_score == 1

    mock_compute_summary_performance.assert_called_once_with(model_outputs)

    mock_saver.save.assert_called_once_with(
        mock_session,
        (empty_ckpt_dir / "model.ckpt").as_posix(),
        global_step=epoch,
    )


def test_update_validation_performance_halves_learning_rate_on_degredation(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    empty_ckpt_dir: Path,
):
    mock_model = mocker.MagicMock()
    mock_session = mocker.MagicMock()

    # performance has degraded
    best_micro_f1_score = 1
    mock_summary_performance = SummaryPerformance(
        micro_f1_score=0, micro_roc_auc_score=0
    )
    mock_compute_summary_performance = mocker.MagicMock(
        return_value=mock_summary_performance
    )
    monkeypatch.setattr(
        validation, "compute_summary_performance", mock_compute_summary_performance
    )

    epoch = 0
    model_outputs = ModelOutputs(
        logits=np.empty((0, 50)),
        predictions=np.empty((0, 50)),
        labels=np.empty((0, 50)),
    )

    best_micro_f1_score, _ = validation.update_performance(
        empty_ckpt_dir,
        mock_model,
        mock_session,
        best_micro_f1_score,
        epoch,
        model_outputs,
    )

    # not updated
    assert best_micro_f1_score == 1

    mock_compute_summary_performance.assert_called_once_with(model_outputs)

    mock_session.run.assert_has_calls(
        [
            mocker.call(mock_model.learning_rate),
            mocker.call([mock_model.learning_rate_decay_half_op]),
            mocker.call(mock_model.learning_rate),
        ],
        any_order=False,
    )
