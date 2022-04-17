import logging
from pathlib import Path
from typing import Mapping

import numpy as np
from gensim.models import Word2Vec


def assign_pretrained_word_embedding(
    word2vec_model_path: Path,
    embed_size: int = 100,
) -> np.ndarray:
    logger = logging.getLogger("create_session.assign_pretrained_word_embedding")
    logger.info(
        "Using pre-trained word emebedding at %s",
        word2vec_model_path,
    )
    word2vec_model = Word2Vec.load(str(word2vec_model_path))
    return np.array(
        [np.zeros(embed_size)]
        + [word2vec_model.wv[word] for word in word2vec_model.wv.vocab]
    )


def assign_pretrained_label_embedding(
    vocabulary_index2word_label: Mapping[int, str],
    label_embedding_model_path: Path,
) -> np.ndarray:
    logger = logging.getLogger("create_session.assign_pretrained_label_embedding")
    logger.info("Using pre-trained label emebedding at %s", label_embedding_model_path)

    word2vec_model_labels = Word2Vec.load(str(label_embedding_model_path))
    label_embeddings = [
        word2vec_model_labels.wv[vocabulary_index2word_label[i]]
        for i in range(len(vocabulary_index2word_label))
    ]
    return np.array(
        [
            embedding / float(np.linalg.norm(embedding) + 1e-6)
            for embedding in label_embeddings
        ]
    )
