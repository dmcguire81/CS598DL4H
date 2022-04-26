import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, cast

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

from HLAN import data_loading
from HLAN.HAN_model_dynamic import HA_GRU, HAN, HLAN


@contextmanager
def create_session(
    model: HAN,
    ckpt_dir: Path,
    remove_ckpts_before_train: bool,
    per_label_attention: bool,
    per_label_sent_only: bool,
    reverse_embedding: data_loading.ReverseOnehotEncoding,
    word2vec_model_path: Path,
    label_embedding_model_path: Path,
    label_embedding_model_path_per_label: Path,
    use_label_embedding: bool,
) -> Iterator[tf.compat.v1.Session]:
    logger = logging.getLogger("create_session")
    logger.debug("creating session")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False

    with tf.compat.v1.Session(config=config) as session:
        saver = tf.compat.v1.train.Saver(max_to_keep=1)

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

            word_embedding = assign_pretrained_word_embedding(
                word2vec_model_path,
            )
            result = session.run(
                tf.assign(
                    model.Embedding, tf.constant(word_embedding, dtype=tf.float32)
                )
            )
            logger.info("Variable %s assigned %s", model.Embedding, result)
            if use_label_embedding:
                label_embedding_transposed = assign_pretrained_label_embedding(
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
                    label_embedding = assign_pretrained_label_embedding(
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
