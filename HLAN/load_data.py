import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec


def create_vocabulary_label_pre_split(
    training_data_path: Path,
    validation_data_path: Path,
    testing_data_path: Path,
) -> Tuple[Mapping[str, int], Mapping[int, str]]:
    logger = logging.getLogger("create_vocabulary_label_pre_split")

    label_counter: Counter = Counter()

    for data_path in [training_data_path, validation_data_path, testing_data_path]:
        df = pd.read_csv(str(data_path))

        for labels_list in df["LABELS"].tolist():
            label_counter += Counter(labels_list.split(";"))

    ordered_labels = [label for label, _count in label_counter.most_common()]

    logger.debug("length of ordered labels %s", len(ordered_labels))

    top10_cumulative_sum = sum([label_counter[label] for label in ordered_labels[:10]])
    logger.info("top 10 cumulative sum: %s", top10_cumulative_sum)

    word2index = {word: index for index, word in enumerate(ordered_labels)}
    index2word = {index: word for index, word in enumerate(ordered_labels)}

    return word2index, index2word


def create_vocabulary(
    word2vec_model_path: Path, name_scope: str = ""
) -> Tuple[Mapping[str, int], Mapping[int, str]]:
    model = Word2Vec.load(str(word2vec_model_path))
    all_features = ["PAD_ID"] + list(model.wv.vocab.keys())

    word2index = {word: index for index, word in enumerate(all_features)}
    index2word = {index: word for index, word in enumerate(all_features)}

    return word2index, index2word


def load_data_multilabel_pre_split(
    vocabulary_word2index: Mapping[str, int],
    vocabulary_word2index_label: Mapping[str, int],
    data_path: Path,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    df = pd.read_csv(str(data_path))
    word2index: Dict[str, int] = defaultdict(int)
    word2index.update(**vocabulary_word2index)
    word2index_label: Dict[str, int] = defaultdict(int)
    word2index_label.update(**vocabulary_word2index_label)
    label_dimension = len(word2index_label)

    text_lines = df["TEXT"].tolist()
    labels_lines = df["LABELS"].tolist()

    X = []
    Y = []

    for text, labels in zip(text_lines, labels_lines):
        x = [word2index[word] for word in text.split(" ")]
        y_index = [word2index_label[label] for label in labels.split(";")]
        y_one_hot = tf.keras.utils.to_categorical(
            y_index, num_classes=label_dimension, dtype=int
        )
        y_multi_hot = np.sum(y_one_hot, axis=0)

        X.append(x)
        Y.append(y_multi_hot)

    return X, Y
