import logging
import os
from pathlib import Path
from typing import List, Tuple

import click

from HLAN import (
    data_feeding,
    data_loading,
    prediction,
    session_creation,
    training,
    validation,
)
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN

LOG_LEVEL = logging._nameToLevel[os.getenv("LOG_LEVEL", "INFO")]
logging.basicConfig(level=LOG_LEVEL)

tf_logger = logging.getLogger("tensorflow")
tf_logger.setLevel(logging.CRITICAL)


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
    (onehot_encoding, training_data_split,) = data_loading.load_data(
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

    with session_creation.create_session(
        model=model,
        ckpt_dir=ckpt_dir,
        remove_ckpts_before_train=remove_ckpts_before_train,
        per_label_attention=per_label_attention,
        per_label_sent_only=per_label_sent_only,
        reverse_embedding=onehot_encoding.reverse,
        word2vec_model_path=word2vec_model_path,
        label_embedding_model_path=label_embedding_model_path,
        label_embedding_model_path_per_label=label_embedding_model_path_per_label,
        use_label_embedding=use_label_embedding,
    ) as session:
        best_micro_f1_score = 0

        for epoch in range(0, num_epochs):

            training.train(
                session,
                model,
                epoch,
                lambda: data_feeding.feed_data(
                    model,
                    *training_data_split.training,
                    batch_size=batch_size,
                    dropout_rate=0.5,
                ),
            )

            all_predictions, all_labels = validation.validate(
                session,
                model,
                epoch,
                onehot_encoding.num_classes,
                lambda: data_feeding.feed_data(
                    model, *training_data_split.validation, batch_size=batch_size
                ),
            )

            best_micro_f1_score = validation.update_performance(
                ckpt_dir,
                model,
                session,
                best_micro_f1_score,
                epoch,
                all_predictions,
                all_labels,
            )

            click.echo("Incrementing epoch counter in session")
            session.run(model.epoch_increment)

        all_predictions, all_labels = prediction.predict(
            session,
            model,
            onehot_encoding.num_classes,
            ckpt_dir,
            lambda: data_feeding.feed_data(
                model, *training_data_split.testing, batch_size=batch_size
            ),
        )

        micro_f1_score, micro_roc_auc_score = prediction.calculate_performance(
            all_predictions,
            all_labels,
        )

        click.echo("Final results --")
        click.echo(f"Micro F1 Score: {micro_f1_score}")
        click.echo(f"Micro ROC-AUC Score: {micro_roc_auc_score}")


if __name__ == "__main__":
    main()
