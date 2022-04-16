import logging
import os
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
from tflearn import data_utils

import HLAN.load_data as ld

LOG_LEVEL = logging._nameToLevel[os.getenv("LOG_LEVEL", "INFO")]
logging.basicConfig(level=LOG_LEVEL)


def load_data(
    word2vec_model_path: Path,
    validation_data_path: Path,
    training_data_path: Path,
    testing_data_path: Path,
    sequence_length: int,
) -> Tuple[
    int,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    logger = logging.getLogger("load_data")

    (
        vocabulary_word2index_label,
        vocabulary_index2word_label,
    ) = ld.create_vocabulary_label_pre_split(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        testing_data_path=testing_data_path,
    )

    num_classes = len(vocabulary_word2index_label)
    label_sim_mat = np.random.rand(num_classes, num_classes)
    label_sub_mat = np.zeros((num_classes, num_classes))

    logger.debug(
        "display the first two labels: %s %s",
        vocabulary_index2word_label[0],
        vocabulary_index2word_label[1],
    )

    vocabulary_word2index, _vocabulary_index2word = ld.create_vocabulary(
        word2vec_model_path
    )

    # check sim and sub relations
    logger.debug("label_sim_mat: %s", label_sim_mat.shape)
    logger.debug("label_sim_mat[0]: %s", label_sim_mat[0])
    logger.debug("label_sub_mat: %s", label_sub_mat.shape)
    logger.debug("label_sub_mat[0]: %s", label_sub_mat[0])
    logger.debug("label_sub_mat_sum: %s", np.sum(label_sub_mat))

    vocab_size = len(vocabulary_word2index)
    logger.debug("vocab_size: %s", vocab_size)

    trainX, trainY = ld.load_data_multilabel_pre_split(
        vocabulary_word2index,
        vocabulary_word2index_label,
        data_path=training_data_path,
    )
    validX, validY = ld.load_data_multilabel_pre_split(
        vocabulary_word2index,
        vocabulary_word2index_label,
        data_path=validation_data_path,
    )
    testX, testY = ld.load_data_multilabel_pre_split(
        vocabulary_word2index,
        vocabulary_word2index_label,
        data_path=testing_data_path,
    )

    trainX = data_utils.pad_sequences(trainX, maxlen=sequence_length, value=0.0)
    validX = data_utils.pad_sequences(validX, maxlen=sequence_length, value=0.0)
    testX = data_utils.pad_sequences(testX, maxlen=sequence_length, value=0.0)

    return (
        num_classes,
        (np.array(trainX), np.array(trainY)),
        (np.array(validX), np.array(validY)),
        (np.array(testX), np.array(testY)),
    )


def create_session():
    pass


def feed_data():
    pass


def training():
    pass


def validation():
    pass


def prediction():
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
    use_label_embedding: bool,
    ckpt_dir: Path,
    use_sent_split_padded_version: bool,
    marking_id: str,
    gpu: bool,
    dataset_paths: List[Path],
    word2vec_model_path: Path,
) -> Tuple[Path, Path, Path]:
    click.echo(f"Training on {[str(path) for path in dataset_paths]}")

    validation_data_path, testing_data_path, training_data_path = sorted(
        dataset_paths, key=str
    )
    click.echo(f"Using {training_data_path} for training")
    click.echo(f"Using {validation_data_path} for validation")
    click.echo(f"Using {testing_data_path} for testing")

    click.echo(f"Using batch size {batch_size}")

    if not per_label_attention:
        raise NotImplementedError("HAN variant not implemented")

    if per_label_sent_only:
        raise NotImplementedError("HA-GRU varient not implemented")

    click.echo("Running HLAN variant")
    click.echo(f"Using num epochs {num_epochs}")

    if report_rand_pred:
        raise NotImplementedError("Random prediction reporting not implemented")

    if running_times != 1:
        raise NotImplementedError(
            "Multiple runs with a static split is not implemented"
        )

    click.echo(f"Using early stop LR {early_stop_lr}")

    click.echo(
        f"{'Removing' if remove_ckpts_before_train else 'Not removing'} checkpoints before training"
    )

    if not use_label_embedding:
        raise NotImplementedError(
            "Omitting label embedding (HLAN minus LE) not implemented"
        )

    click.echo(f"Using checkpoint path {ckpt_dir}")

    if use_sent_split_padded_version:
        raise NotImplementedError(
            "Sentence splitting instead of chunking (HLAN plus sent split) not implemented"
        )

    click.echo(f"Tagging outputs with marking ID {marking_id}")

    if not gpu:
        raise NotImplementedError(
            "Forcibly disabling static GPUs not implemented (just change environment)"
        )

    click.echo(f"Loading training data word embedding from {word2vec_model_path}")

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
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=False),
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
    "--word2vec_model_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=False),
    help="gensim word2vec's vocabulary and vectors",
)
@click.option(
    "--sequence_length",
    required=False,
    default=2500,
    help="as in Mullenbach et al., 2018",
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
    word2vec_model_path: Path,
    sequence_length: int,
):
    last_separator = dataset.rindex(os.sep)
    root, glob = dataset[:last_separator], dataset[last_separator + 1 :]
    dataset_paths = Path(root).glob(glob)

    validation_data_path, training_data_path, testing_data_path = process_options(
        batch_size,
        per_label_attention,
        per_label_sent_only,
        num_epochs,
        report_rand_pred,
        running_times,
        early_stop_lr,
        remove_ckpts_before_train,
        use_label_embedding,
        ckpt_dir,
        use_sent_split_padded_version,
        marking_id,
        gpu,
        list(dataset_paths),
        word2vec_model_path,
    )
    (num_classes, (trainX, trainY), (validX, validY), (testX, testY)) = load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length=sequence_length,
    )
    click.echo(num_classes)
    click.echo(trainX)
    click.echo(trainY)
    click.echo(validX)
    click.echo(validY)
    click.echo(testX)
    click.echo(testY)

    create_session()
    feed_data()
    training()
    validation()
    prediction()


if __name__ == "__main__":
    main()
