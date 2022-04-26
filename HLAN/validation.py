import logging
from typing import Any, Callable, Iterable, Mapping, Tuple

import numpy as np
import tensorflow as tf
from sklearn import metrics

from HLAN.HAN_model_dynamic import HAN
from HLAN.performance import ModelPerformance, RunningModelPerformance


def validate(
    session: tf.compat.v1.Session,
    model: HAN,
    epoch: int,
    num_classes: int,
    feeder: Callable[[], Iterable[Mapping[Any, Any]]],
) -> Tuple[np.ndarray, np.ndarray]:
    logger = logging.getLogger("validation")
    running_performance = RunningModelPerformance.empty()
    all_predictions = np.empty((0, num_classes))
    all_labels = np.empty((0, num_classes))

    for step, feed_dict in enumerate(feeder()):
        (
            validation_loss_per_batch,
            validation_loss_per_epoch,
            loss,
            precision,
            recall,
            jaccard_index,
            predictions,
            labels,
        ) = session.run(
            [
                model.validation_loss_per_batch,
                model.validation_loss_per_epoch,
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
        logger.debug("Current validation performance: %s", performance)

        running_performance = running_performance + performance
        assert running_performance.count == step + 1

        if step % 50 == 0:
            logger.info(
                "Average validation performance (epoch %s, step %s): %s",
                epoch,
                step,
                running_performance.average(),
            )

        model.writer.add_summary(validation_loss_per_batch, step)

        if step == 0:  # epoch rolled over
            model.writer.add_summary(validation_loss_per_epoch, epoch)

        all_predictions = np.concatenate((all_predictions, predictions), axis=0)
        all_labels = np.concatenate((all_labels, labels), axis=0)

    return all_predictions, all_labels


def update_performance(
    ckpt_dir, model, session, best_micro_f1_score, epoch, all_predictions, all_labels
):
    logger = logging.getLogger("update_performance")

    micro_roc_auc_score = metrics.roc_auc_score(
        all_labels, all_predictions, average="micro"
    )
    logger.info("Micro ROC-AUC score is %s", micro_roc_auc_score)

    micro_f1_score = metrics.f1_score(all_labels, all_predictions, average="micro")

    if micro_f1_score >= best_micro_f1_score:
        logger.info(
            "Micro F1 score improved from %s to %s", best_micro_f1_score, micro_f1_score
        )
        saver = tf.train.Saver(max_to_keep=1)
        save_path = ckpt_dir / "model.ckpt"
        logger.info("Saving model checkpoint to %s", save_path)
        saver.save(session, str(save_path), global_step=epoch)
        best_micro_f1_score = micro_f1_score
    else:
        logger.info(
            "Micro F1 score degraded from %s to %s", best_micro_f1_score, micro_f1_score
        )
        current_learning_rate = session.run(model.learning_rate)
        _ = session.run([model.learning_rate_decay_half_op])
        new_learning_rate = session.run(model.learning_rate)
        logger.info(
            "Updated learning rate from %s to %s",
            current_learning_rate,
            new_learning_rate,
        )

    return best_micro_f1_score
