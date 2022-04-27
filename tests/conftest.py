import os
from pathlib import Path
from typing import List

import pytest
from gensim.models import Word2Vec


@pytest.fixture
def caml_dataset():
    return "caml-mimic/mimicdata/mimic3/*_50.csv"


@pytest.fixture
def caml_dataset_paths(caml_dataset) -> List[Path]:
    last_separator = caml_dataset.rindex(os.sep)
    root, glob = caml_dataset[:last_separator], caml_dataset[last_separator + 1 :]
    return list(Path(root).glob(glob))


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def per_label_attention():
    return True


@pytest.fixture
def per_label_sent_only():
    return False


@pytest.fixture
def num_epochs():
    return 100


@pytest.fixture
def report_rand_pred():
    False


@pytest.fixture
def running_times():
    return 1


@pytest.fixture
def early_stop_lr():
    return 0.00002


@pytest.fixture
def remove_ckpts_before_train():
    return False


@pytest.fixture
def use_label_embedding():
    return True


@pytest.fixture
def ckpt_dir():
    return Path(
        "Explainable-Automated-Medical-Coding/checkpoints/checkpoint_HAN_50_per_label_bs32_LE/"
    )


@pytest.fixture
def empty_ckpt_dir():
    return Path(
        "Explainable-Automated-Medical-Coding/checkpoints/no_checkpoint_here_for_sure/"
    )


@pytest.fixture
def use_sent_split_padded_version():
    return False


@pytest.fixture
def gpu():
    return True


@pytest.fixture(scope="session")
def word2vec_model_path():
    return Path("Explainable-Automated-Medical-Coding/embeddings/processed_full.w2v")


@pytest.fixture(scope="session")
def word2vec_model(word2vec_model_path: Path):
    return Word2Vec.load(str(word2vec_model_path))


@pytest.fixture()
def label_embedding_model_path():
    return Path(
        "Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-400.model"
    )


@pytest.fixture()
def label_embedding_model_path_per_label():
    return Path(
        "Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-200.model"
    )


@pytest.fixture
def sequence_length():
    return 2500


@pytest.fixture
def num_sentences():
    return 100
