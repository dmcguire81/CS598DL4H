import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from tflearn import data_utils

ForwardOnehot = Mapping[str, int]
ReverseOnehot = Mapping[int, str]


@dataclass
class ForwardOnehotEncoding:
    words: ForwardOnehot
    labels: ForwardOnehot


@dataclass
class ReverseOnehotEncoding:
    words: ReverseOnehot
    labels: ReverseOnehot


@dataclass
class OnehotEncoding:
    num_classes: int
    vocab_size: int
    forward: ForwardOnehotEncoding
    reverse: ReverseOnehotEncoding


TrainingData = Tuple[np.ndarray, np.ndarray]


@dataclass
class TrainingDataSplit:
    validation: TrainingData
    training: TrainingData
    testing: TrainingData


def load_data(
    word2vec_model_path: Path,
    validation_data_path: Path,
    training_data_path: Path,
    testing_data_path: Path,
    sequence_length: int,
) -> Tuple[OnehotEncoding, TrainingDataSplit]:
    onehot_encoding = load_encoding_data(
        word2vec_model_path, validation_data_path, training_data_path, testing_data_path
    )

    training_data_split = load_training_data(
        onehot_encoding.forward,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )

    return onehot_encoding, training_data_split


def load_encoding_data(
    word2vec_model_path: Path,
    validation_data_path: Path,
    training_data_path: Path,
    testing_data_path: Path,
) -> OnehotEncoding:
    logger = logging.getLogger("load_data.load_encoding_data")

    (
        vocabulary_word2index_label,
        vocabulary_index2word_label,
    ) = create_vocabulary_label_pre_split(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        testing_data_path=testing_data_path,
    )

    logger.debug(
        "display the first two labels: %s %s",
        vocabulary_index2word_label[0],
        vocabulary_index2word_label[1],
    )

    vocabulary_word2index, vocabulary_index2word = create_vocabulary(
        word2vec_model_path
    )

    vocab_size = len(vocabulary_word2index)
    logger.debug("vocab_size: %s", vocab_size)

    return OnehotEncoding(
        num_classes=len(vocabulary_index2word_label),
        vocab_size=len(vocabulary_index2word),
        forward=ForwardOnehotEncoding(
            words=vocabulary_word2index, labels=vocabulary_word2index_label
        ),
        reverse=ReverseOnehotEncoding(
            words=vocabulary_index2word, labels=vocabulary_index2word_label
        ),
    )


def create_vocabulary_label_pre_split(
    training_data_path: Path,
    validation_data_path: Path,
    testing_data_path: Path,
) -> Tuple[ForwardOnehot, ReverseOnehot]:
    logger = logging.getLogger("load_data.create_vocabulary_label_pre_split")

    label_counter: Counter = Counter()

    for data_path in [training_data_path, validation_data_path, testing_data_path]:
        df = pd.read_csv(data_path.as_posix())

        for labels_list in df["LABELS"].tolist():
            label_counter += Counter(labels_list.split(";"))

    ordered_labels = [label for label, _count in label_counter.most_common()]

    logger.debug("length of ordered labels %s", len(ordered_labels))

    top10_cumulative_sum = sum([label_counter[label] for label in ordered_labels[:10]])
    logger.info("top 10 cumulative sum: %s", top10_cumulative_sum)

    word2index = {word: index for index, word in enumerate(ordered_labels)}
    index2word = {index: word for index, word in enumerate(ordered_labels)}

    return word2index, index2word


def create_vocabulary(word2vec_model_path: Path) -> Tuple[ForwardOnehot, ReverseOnehot]:
    model = Word2Vec.load(word2vec_model_path.as_posix())
    all_features = ["PAD_ID"] + list(model.wv.vocab.keys())

    word2index = {word: index for index, word in enumerate(all_features)}
    index2word = {index: word for index, word in enumerate(all_features)}

    return word2index, index2word


def load_training_data(
    vocabulary_embedding: ForwardOnehotEncoding,
    validation_data_path: Path,
    training_data_path: Path,
    testing_data_path: Path,
    sequence_length: int,
) -> TrainingDataSplit:
    trainX, trainY = load_data_multilabel_pre_split(
        vocabulary_embedding,
        data_path=training_data_path,
    )
    validX, validY = load_data_multilabel_pre_split(
        vocabulary_embedding,
        data_path=validation_data_path,
    )
    testX, testY = load_data_multilabel_pre_split(
        vocabulary_embedding,
        data_path=testing_data_path,
    )

    trainX = data_utils.pad_sequences(trainX, maxlen=sequence_length, value=0.0)
    validX = data_utils.pad_sequences(validX, maxlen=sequence_length, value=0.0)
    testX = data_utils.pad_sequences(testX, maxlen=sequence_length, value=0.0)

    return TrainingDataSplit(
        training=(trainX, trainY),
        validation=(validX, validY),
        testing=(testX, testY),
    )


def load_data_multilabel_pre_split(
    vocabulary_embedding: ForwardOnehotEncoding,
    data_path: Path,
) -> TrainingData:
    df = pd.read_csv(data_path.as_posix())
    word2index: Dict[str, int] = defaultdict(int)
    word2index.update(**vocabulary_embedding.words)
    word2index_label: Dict[str, int] = defaultdict(int)
    word2index_label.update(**vocabulary_embedding.labels)
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

    # dtype is object because these are "ragged nested sequences" prior to padding
    return np.array(X, dtype=object), np.array(Y, dtype=object)
