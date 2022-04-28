from pathlib import Path
from typing import List

import numpy as np
from gensim.models import Word2Vec

from HLAN import data_loading


def test_load_data(
    word2vec_model_path: Path,
    caml_dataset_paths: List[Path],
    sequence_length,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    onehot_encoding, training_data_split = data_loading.load_data(
        word2vec_model_path,
        validation_data_path,
        training_data_path,
        testing_data_path,
        sequence_length,
    )

    assert len(onehot_encoding.forward.labels) == 50
    assert len(onehot_encoding.forward.words) == 150855

    assert len(onehot_encoding.reverse.labels) == 50
    assert len(onehot_encoding.reverse.words) == 150855

    (trainX, trainY) = training_data_split.training
    (validX, validY) = training_data_split.validation
    (testX, testY) = training_data_split.testing

    assert isinstance(trainX, np.ndarray)
    assert isinstance(trainY, np.ndarray)
    assert isinstance(validX, np.ndarray)
    assert isinstance(validY, np.ndarray)
    assert isinstance(testX, np.ndarray)
    assert isinstance(testY, np.ndarray)

    assert len(trainX) == len(trainY)
    assert len(validX) == len(validY)
    assert len(testX) == len(testY)
    assert len(trainY) == 8066
    assert len(validY) == 1573
    assert len(testY) == 1729

    assert np.mean(trainX) == 1078.8257977188198
    assert np.mean(trainY) == 0.11385817009670221
    assert np.mean(validX) == 1440.640283534647
    assert np.mean(validY) == 0.11802924348378893
    assert np.mean(testX) == 1465.9599076923078
    assert np.mean(testY) == 0.12119144013880856


def test_create_vocabulary_label_pre_split(
    caml_dataset_paths: List[Path],
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    (
        vocabulary_word2index_label,
        vocabulary_index2word_label,
    ) = data_loading.create_vocabulary_label_pre_split(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        testing_data_path=testing_data_path,
    )

    assert len(vocabulary_word2index_label) == len(vocabulary_index2word_label)
    assert len(vocabulary_word2index_label) == 50
    assert set(vocabulary_index2word_label.keys()) == set(
        vocabulary_word2index_label.values()
    )
    assert set(vocabulary_index2word_label.values()) == set(
        vocabulary_word2index_label.keys()
    )


def test_create_vocabulary(
    word2vec_model_path: Path,
):
    (vocabulary_word2index, vocabulary_index2word,) = data_loading.create_vocabulary(
        word2vec_model_path=word2vec_model_path,
    )

    model = Word2Vec.load(word2vec_model_path.as_posix())
    dimension = len(model.wv.vocab) + 1

    assert len(vocabulary_word2index) == len(vocabulary_index2word)
    assert len(vocabulary_word2index) == dimension
    assert set(vocabulary_index2word.keys()) == set(vocabulary_word2index.values())
    assert set(vocabulary_index2word.values()) == set(vocabulary_word2index.keys())


def test_load_data_multilabel_pre_split(
    caml_dataset_paths: List[Path],
    word2vec_model_path: Path,
):
    validation_data_path, testing_data_path, training_data_path = sorted(
        caml_dataset_paths, key=str
    )

    (
        vocabulary_word2index_label,
        vocabulary_index2word_label,
    ) = data_loading.create_vocabulary_label_pre_split(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        testing_data_path=testing_data_path,
    )

    vocabulary_word2index, _vocabulary_index2word = data_loading.create_vocabulary(
        word2vec_model_path
    )

    forward_embedding = data_loading.ForwardOnehotEncoding(
        words=vocabulary_word2index, labels=vocabulary_word2index_label
    )

    trainX, trainY = data_loading.load_data_multilabel_pre_split(
        forward_embedding,
        data_path=training_data_path,
    )
    validX, validY = data_loading.load_data_multilabel_pre_split(
        forward_embedding,
        data_path=validation_data_path,
    )
    testX, testY = data_loading.load_data_multilabel_pre_split(
        forward_embedding,
        data_path=testing_data_path,
    )

    assert len(trainX) == len(trainY)
    assert len(validX) == len(validY)
    assert len(testX) == len(testY)
    assert len(trainY) == 8066
    assert len(validY) == 1573
    assert len(testY) == 1729

    trainY0 = np.zeros_like(trainY[0])
    trainY0[[6, 37, 48]] = 1
    assert (trainY[0] == trainY0).all()

    validY0 = np.zeros_like(validY[0])
    validY0[[0, 3, 4, 10, 26]] = 1
    assert (validY[0] == validY0).all()

    testY0 = np.zeros_like(testY[0])
    testY0[[8, 10, 11]] = 1
    assert (testY[0] == testY0).all()
