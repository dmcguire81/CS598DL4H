import logging

import tensorflow as tf
from sklearn import metrics


def update_validation_performance(
    ckpt_dir, model, session, best_micro_f1_score, epoch, all_predictions, all_labels
):
    logger = logging.getLogger("update_validation_performance")
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
