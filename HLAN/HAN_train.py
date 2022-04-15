import os
from pathlib import Path

import click


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
def train(
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
):
    last_separator = dataset.rindex(os.sep)
    root, glob = dataset[:last_separator], dataset[last_separator + 1 :]
    dataset_files = Path(root).glob(glob)
    click.echo(f"Training on {[str(file) for file in dataset_files]}")

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


if __name__ == "__main__":
    train()
