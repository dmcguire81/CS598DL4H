import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, NamedTuple, Tuple, cast

import click
import numpy as np
import tensorflow as tf

import HLAN.create_session as cs
import HLAN.load_data as ld
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN

LOG_LEVEL = logging._nameToLevel[os.getenv("LOG_LEVEL", "INFO")]
logging.basicConfig(level=LOG_LEVEL)

tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.CRITICAL)


class ModelPerformance(NamedTuple):
    precision: float
    recall: float
    jaccard_index: float

    def __add__(self, other):
        return ModelPerformance(
            self.precision + other.precision,
            self.recall + other.recall,
            self.jaccard_index + other.jaccard_index,
        )

    def __truediv__(self, scalar):
        return ModelPerformance(
            self.precision / scalar,
            self.recall / scalar,
            self.jaccard_index / scalar,
        )


class RunningModelPerformance:
    def __init__(self, performance: ModelPerformance, count: int):
        self.performance_sum = performance
        self.count = count

    @staticmethod
    def empty():
        return RunningModelPerformance(ModelPerformance(0, 0, 0), 0)

    def __add__(self, performance: ModelPerformance):
        return RunningModelPerformance(
            self.performance_sum + performance, self.count + 1
        )

    def average(self) -> ModelPerformance:
        return self.performance_sum / self.count


def load_data(
    word2vec_model_path: Path,
    validation_data_path: Path,
    training_data_path: Path,
    testing_data_path: Path,
    sequence_length: int,
) -> Tuple[ld.OnehotEncoding, ld.TrainingDataSplit]:
    onehot_encoding = ld.load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )

    training_data_split = ld.load_training_data(
        onehot_encoding.forward,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )

    return onehot_encoding, training_data_split


@contextmanager
def create_session(
    model: HAN,
    ckpt_dir: Path,
    remove_ckpts_before_train: bool,
    per_label_attention: bool,
    per_label_sent_only: bool,
    reverse_embedding: ld.ReverseOnehotEncoding,
    word2vec_model_path: Path,
    label_embedding_model_path: Path,
    label_embedding_model_path_per_label: Path,
    use_embedding: bool = True,
    use_label_embedding: bool = True,
) -> Iterator[tf.compat.v1.Session]:
    logger = logging.getLogger("create_session")
    logger.debug("creating session")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False

    with tf.compat.v1.Session(config=config) as session:
        saver = tf.train.Saver(max_to_keep=1)

        if remove_ckpts_before_train and ckpt_dir.exists():
            shutil.rmtree(str(ckpt_dir))
            logger.info("Removed checkpoint at %s", ckpt_dir)

        ckpt_file = ckpt_dir / "checkpoint"
        if ckpt_file.exists():
            logger.info("Restoring from checkpoint at %s", ckpt_file)
            saver.restore(session, tf.train.latest_checkpoint(ckpt_dir))
        else:
            logger.info("Initializing without checkpoint")
            session.run(tf.global_variables_initializer())

            if use_embedding:
                word_embedding = cs.assign_pretrained_word_embedding(
                    word2vec_model_path,
                )
                result = session.run(
                    tf.assign(
                        model.Embedding, tf.constant(word_embedding, dtype=tf.float32)
                    )
                )
                logger.info("Variable %s assigned %s", model.Embedding, result)
            if use_label_embedding:
                label_embedding_transposed = cs.assign_pretrained_label_embedding(
                    reverse_embedding.labels,
                    label_embedding_model_path,
                ).transpose()
                result = session.run(
                    tf.assign(
                        model.W_projection,
                        tf.constant(label_embedding_transposed, dtype=tf.float32),
                    )
                )
                logger.info("Variable %s assigned %s", model.W_projection, result)
                if per_label_attention:
                    label_embedding = cs.assign_pretrained_label_embedding(
                        reverse_embedding.labels,
                        label_embedding_model_path_per_label,
                    )
                    label_embedding_tensor = tf.constant(
                        label_embedding, dtype=tf.float32
                    )
                    if not per_label_sent_only:
                        result = session.run(
                            tf.assign(
                                cast(HLAN, model).context_vector_word_per_label,
                                label_embedding_tensor,
                            )
                        )
                        logger.info(
                            "Variable %s assigned %s",
                            cast(HLAN, model).context_vector_word_per_label,
                            result,
                        )
                    result = session.run(
                        tf.assign(
                            cast(HA_GRU, model).context_vector_sentence_per_label,
                            label_embedding_tensor,
                        )
                    )
                    logger.info(
                        "Variable %s assigned %s",
                        cast(HA_GRU, model).context_vector_sentence_per_label,
                        result,
                    )

        yield session
        session.close()


def feed_data(
    model: HAN, X: np.ndarray, Y: np.ndarray, batch_size: int, dropout_rate: float = 0.0
) -> Iterable[Mapping[Any, Any]]:
    n = len(X)
    b = batch_size
    assert len(Y) == n
    # trim the final slice to *exactly* the size of the input, rather than overshooting
    batches = [slice(i, i + b) for i in range(0, n, b)][:-1] + [slice(n - (n % b), n)]

    for batch in batches:
        batch_X = X[batch]
        batch_Y = Y[batch]
        yield {
            model.input_x: batch_X,
            model.input_y: batch_Y,
            model.dropout_rate: dropout_rate,
        }


def training(
    session: tf.compat.v1.Session,
    model: HAN,
    step: int,
    epoch: int,
    feed_dict: Mapping[Any, Any],
    running_performance: RunningModelPerformance,
) -> RunningModelPerformance:
    logger = logging.getLogger("training")
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

    performance = ModelPerformance(precision, recall, jaccard_index)
    logger.debug("Current training performance: %s", performance)

    running_performance = running_performance + performance
    assert running_performance.count == step + 1

    if step % 50 == 0:
        logger.info("Average training performance: %s", running_performance.average())

    model.writer.add_summary(training_loss_per_batch, step)

    if step == 0:  # epoch rolled over
        model.writer.add_summary(training_loss_per_epoch, epoch)

    return running_performance


def validation(
    session: tf.compat.v1.Session,
    model: HAN,
    step: int,
    epoch: int,
    feed_dict: Mapping[Any, Any],
):
    pass


def prediction(session: tf.compat.v1.Session, model: HAN, feed_dict: Mapping[Any, Any]):
    pass


def process_options(
    batch_size: int,
    per_label_attention: bool,
    per_label_sent_only: bool,
    num_epochs: int,
    report_rand_pred: bool,
    running_times: int,
    early_stop_lr: float,
    remove_ckpts_before_train: bool,
    ckpt_dir: Path,
    use_sent_split_padded_version: bool,
    marking_id: str,
    gpu: bool,
    dataset_paths: List[Path],
    word2vec_model_path: Path,
) -> Tuple[Path, Path, Path]:
    logger = logging.getLogger("process_options")
    logger.debug(f"Training on {[str(path) for path in dataset_paths]}")

    validation_data_path, testing_data_path, training_data_path = sorted(
        dataset_paths, key=str
    )
    logger.info(f"Using {training_data_path} for training")
    logger.info(f"Using {validation_data_path} for validation")
    logger.info(f"Using {testing_data_path} for testing")

    logger.info(f"Using batch size {batch_size}")

    if per_label_attention:
        if not per_label_sent_only:
            logger.info("Running HLAN variant")
        else:
            logger.info("Running HA-GRU variant")
    else:
        logger.info("Running HAN variant")

    logger.info(f"Using num epochs {num_epochs}")

    if report_rand_pred:
        raise NotImplementedError("Random prediction reporting not implemented")

    if running_times != 1:
        raise NotImplementedError(
            "Multiple runs with a static split is not implemented"
        )

    logger.info(f"Using early stop LR {early_stop_lr}")

    logger.info(
        f"{'Removing' if remove_ckpts_before_train else 'Not removing'} checkpoints before training"
    )

    logger.info(f"Using checkpoint path {ckpt_dir}")

    if use_sent_split_padded_version:
        raise NotImplementedError(
            "Sentence splitting instead of chunking (HLAN plus sent split) not implemented"
        )

    logger.info(f"Tagging outputs with marking ID {marking_id}")

    if not gpu:
        raise NotImplementedError(
            "Forcibly disabling static GPUs not implemented (just change environment)"
        )

    logger.info(f"Loading training data word embedding from {word2vec_model_path}")

    return validation_data_path, training_data_path, testing_data_path


@click.command()
@click.option(
    "--dataset",
    help="Pass a (single-quoted) glob pattern to expand into dev/test/train file set",
    required=True,
)
@click.option(
    "--batch_size",
    required=True,
    type=int,
)
@click.option(
    "--per_label_attention",
    required=True,
    type=bool,
)
@click.option(
    "--per_label_sent_only",
    required=True,
    type=bool,
)
@click.option(
    "--num_epochs",
    required=True,
    type=int,
)
@click.option(
    "--report_rand_pred",
    required=True,
    type=bool,
    help="report random prediction for qualitative analysis",
)
@click.option(
    "--running_times",
    required=True,
    type=int,
    help="running the model for a number of times to get averaged results. This is only applied if using pre-defined data split (kfold=0)",
)
@click.option(
    "--early_stop_lr",
    required=True,
    type=float,
    help="early stop point when learning rate is below this threshold",
)
@click.option(
    "--remove_ckpts_before_train",
    required=True,
    type=bool,
)
@click.option(
    "--use_label_embedding",
    required=True,
    type=bool,
)
@click.option(
    "--ckpt_dir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--use_sent_split_padded_version",
    required=True,
    type=bool,
)
@click.option(
    "--marking_id",
    required=True,
    type=str,
)
@click.option(
    "--gpu",
    required=True,
    type=bool,
)
@click.option(
    "--log_dir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--word2vec_model_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=False, path_type=Path),
    help="gensim word2vec's vocabulary and vectors",
)
@click.option(
    "--sequence_length",
    required=False,
    default=2500,
    help="as in Mullenbach et al., 2018",
)
@click.option(
    "--num_sentences",
    required=False,
    default=100,
    help="break sentences up as blocks of N characters",
)
@click.option(
    "--label_embedding_model_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=False, path_type=Path),
    help="pre-trained model from mimic3-ds labels for label embedding initialisation: final projection matrix, self.W_projection.",
)
@click.option(
    "--label_embedding_model_path_per_label",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=False, path_type=Path),
    help="pre-trained model from mimic3-ds labels for label embedding initialisation: per label context matrices, self.context_vector_word_per_label and self.context_vector_sentence_per_label",
)
def main(
    dataset: str,
    batch_size: int,
    per_label_attention: bool,
    per_label_sent_only: bool,
    num_epochs: int,
    report_rand_pred: bool,
    running_times: int,
    early_stop_lr: float,
    remove_ckpts_before_train: bool,
    use_label_embedding: bool,
    ckpt_dir: Path,
    use_sent_split_padded_version: bool,
    marking_id: str,
    gpu: bool,
    log_dir: Path,
    word2vec_model_path: Path,
    sequence_length: int,
    num_sentences: int,
    label_embedding_model_path: Path,
    label_embedding_model_path_per_label: Path,
):
    last_separator = dataset.rindex(os.sep)
    root, glob = dataset[:last_separator], dataset[last_separator + 1 :]
    dataset_paths = list(Path(root).glob(glob))

    validation_data_path, training_data_path, testing_data_path = process_options(
        batch_size,
        per_label_attention,
        per_label_sent_only,
        num_epochs,
        report_rand_pred,
        running_times,
        early_stop_lr,
        remove_ckpts_before_train,
        ckpt_dir,
        use_sent_split_padded_version,
        marking_id,
        gpu,
        dataset_paths,
        word2vec_model_path,
    )
    (onehot_encoding, training_data_split,) = load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length=sequence_length,
    )

    kwargs = dict(
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
        log_dir=log_dir,
    )

    if not per_label_attention:
        model = HAN(**kwargs)
    else:
        if per_label_sent_only:
            model = HA_GRU(**kwargs)
        else:
            model = HLAN(**kwargs)

    with create_session(
        model=model,
        ckpt_dir=ckpt_dir,
        remove_ckpts_before_train=remove_ckpts_before_train,
        per_label_attention=per_label_attention,
        per_label_sent_only=per_label_sent_only,
        reverse_embedding=onehot_encoding.reverse,
        word2vec_model_path=word2vec_model_path,
        label_embedding_model_path=label_embedding_model_path,
        label_embedding_model_path_per_label=label_embedding_model_path_per_label,
    ) as session:
        for epoch in range(0, num_epochs):

            running_training_performance = RunningModelPerformance.empty()

            for step, feed_dict in enumerate(
                feed_data(
                    model,
                    *training_data_split.training,
                    batch_size=batch_size,
                    dropout_rate=0.5,
                )
            ):
                running_training_performance = training(
                    session, model, step, epoch, feed_dict, running_training_performance
                )

            for step, feed_dict in enumerate(
                feed_data(model, *training_data_split.validation, batch_size=batch_size)
            ):
                validation(session, model, step, epoch, feed_dict)

            click.echo("Incrementing epoch counter in session")
            session.run(model.epoch_increment)

        for feed_dict in feed_data(
            model, *training_data_split.testing, batch_size=batch_size
        ):
            click.echo(feed_dict)
            prediction(session, model, feed_dict)


if __name__ == "__main__":
    main()
