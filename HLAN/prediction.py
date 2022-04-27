import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import tensorflow as tf

from HLAN.HAN_model_dynamic import HAN
from HLAN.performance import (
    ModelOutputs,
    ModelPerformance,
    RunningModelPerformance,
    SummaryPerformance,
    compute_summary_performance,
)


def predict(
    session: tf.compat.v1.Session,
    model: HAN,
    num_classes: int,
    ckpt_dir: Path,
    feeder: Callable[[], Iterable[Mapping[Any, Any]]],
) -> ModelOutputs:
    logger = logging.getLogger("prediction")

    ckpt_file = ckpt_dir / "checkpoint"
    if ckpt_file.exists():
        logger.info(
            "For best prediction performance, restoring from checkpoint at %s",
            ckpt_file,
        )
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        saver.restore(session, tf.train.latest_checkpoint(ckpt_dir))

    running_performance = RunningModelPerformance.empty()
    all_logits = np.empty((0, num_classes))
    all_predictions = np.empty((0, num_classes))
    all_labels = np.empty((0, num_classes))

    for step, feed_dict in enumerate(feeder()):
        (
            loss,
            precision,
            recall,
            jaccard_index,
            logits,
            predictions,
            labels,
        ) = session.run(
            [
                model.loss,
                model.precision,
                model.recall,
                model.jaccard_index,
                model.logits,
                model.predictions,
                model.input_y,
            ],
            feed_dict,
        )

        performance = ModelPerformance(
            loss=loss, precision=precision, recall=recall, jaccard_index=jaccard_index
        )
        logger.debug("Current performance: %s", performance)

        running_performance = running_performance + performance
        assert running_performance.count == step + 1

        if step % 50 == 0:
            logger.info(
                "Average performance (step %s): %s",
                step,
                running_performance.average(),
            )

        all_logits = np.concatenate((all_logits, logits), axis=0)
        all_predictions = np.concatenate((all_predictions, predictions), axis=0)
        all_labels = np.concatenate((all_labels, labels), axis=0)

    return ModelOutputs(
        logits=all_logits, predictions=all_predictions, labels=all_labels
    )


def calculate_performance(model_outputs: ModelOutputs) -> SummaryPerformance:
    logger = logging.getLogger("calculate_performance")
    summary_performance = compute_summary_performance(model_outputs)

    logger.info("Micro ROC-AUC score is %s", summary_performance.micro_roc_auc_score)
    logger.info("Micro F1 score is %s", summary_performance.micro_f1_score)

    return summary_performance
