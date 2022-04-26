import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import tensorflow as tf
from sklearn import metrics

from HLAN.HAN_model_dynamic import HAN
from HLAN.performance import ModelPerformance, RunningModelPerformance


def predict(
    session: tf.compat.v1.Session,
    model: HAN,
    num_classes: int,
    ckpt_dir: Path,
    feeder: Callable[[], Iterable[Mapping[Any, Any]]],
):
    logger = logging.getLogger("prediction")

    ckpt_file = ckpt_dir / "checkpoint"
    if ckpt_file.exists():
        logger.info(
            "For best prediction performance, restoring from checkpoint at %s",
            ckpt_file,
        )
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(session, tf.train.latest_checkpoint(ckpt_dir))

    running_performance = RunningModelPerformance.empty()
    all_predictions = np.empty((0, num_classes))
    all_labels = np.empty((0, num_classes))

    for step, feed_dict in enumerate(feeder()):
        (loss, precision, recall, jaccard_index, predictions, labels,) = session.run(
            [
                model.loss,
                model.precision,
                model.recall,
                model.jaccard_index,
                model.predictions,
                model.input_y,
            ],
            feed_dict,
        )

        performance = ModelPerformance(
            loss=loss, precision=precision, recall=recall, jaccard_index=jaccard_index
        )
        logger.debug("Current prediction performance: %s", performance)

        running_performance = running_performance + performance
        assert running_performance.count == step + 1

        if step % 50 == 0:
            logger.info(
                "Average prediction performance (step %s): %s",
                step,
                running_performance.average(),
            )

        all_predictions = np.concatenate((all_predictions, predictions), axis=0)
        all_labels = np.concatenate((all_labels, labels), axis=0)

    return all_predictions, all_labels


def calculate_performance(all_predictions, all_labels):
    logger = logging.getLogger("calculate_prediction_performance")

    micro_roc_auc_score = metrics.roc_auc_score(
        all_labels, all_predictions, average="micro"
    )
    logger.info("Micro ROC-AUC score is %s", micro_roc_auc_score)

    micro_f1_score = metrics.f1_score(all_labels, all_predictions, average="micro")
    logger.info("Micro F1 score is %s", micro_f1_score)

    return micro_f1_score, micro_roc_auc_score
