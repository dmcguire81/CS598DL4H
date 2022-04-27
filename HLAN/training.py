import logging
from typing import Any, Callable, Iterable, Mapping

import tensorflow as tf

from HLAN.HAN_model_dynamic import HAN
from HLAN.performance import ModelPerformance, RunningModelPerformance


def train(
    session: tf.compat.v1.Session,
    model: HAN,
    epoch: int,
    feeder: Callable[[], Iterable[Mapping[Any, Any]]],
):
    logger = logging.getLogger("training")
    running_performance = RunningModelPerformance.empty()

    for step, feed_dict in enumerate(feeder()):

        (
            training_loss_per_batch,
            training_loss_per_epoch,
            loss,
            precision,
            recall,
            jaccard_index,
            _,
        ) = session.run(
            [
                model.training_loss_per_batch,
                model.training_loss_per_epoch,
                model.loss,
                model.precision,
                model.recall,
                model.jaccard_index,
                model.train_op,
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
                "Average performance (epoch %s, step %s): %s",
                epoch,
                step,
                running_performance.average(),
            )

        model.writer.add_summary(training_loss_per_batch, step)

        if step == 0:  # epoch rolled over
            model.writer.add_summary(training_loss_per_epoch, epoch)

    logger.info(
        "Average performance (epoch %s, step %s): %s",
        epoch,
        step,
        running_performance.average(),
    )
