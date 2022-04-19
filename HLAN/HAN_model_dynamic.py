import tensorflow as tf


class HAN:
    def __init__(
        self,
        num_classes,
        batch_size,
        sequence_length,
        vocab_size,
        embed_size,
        hidden_size,
        per_label_attention=False,
        per_label_sent_only=False,
        initializer=tf.random_normal_initializer(stddev=0.1),
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.per_label_attention = per_label_attention
        self.per_label_sent_only = per_label_sent_only

        self.input_x = tf.placeholder(
            tf.int32, [None, self.sequence_length], name="input_x"
        )
        self.input_y = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y"
        )
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1))
        )

        self.instantiate_weights()

    def instantiate_weights(
        self,
    ):
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable(
                "Embedding",
                shape=[self.vocab_size, self.embed_size],
                initializer=self.initializer,
            )
            self.W_projection = tf.get_variable(
                "W_projection",
                shape=[self.hidden_size * 4, self.num_classes],
                initializer=self.initializer,
            )

        with tf.name_scope("gru_weights_word_level"):
            if self.per_label_attention:
                if not self.per_label_sent_only:
                    self.context_vector_word_per_label = tf.get_variable(
                        "what_is_the_informative_word_per_label",
                        shape=[self.num_classes, self.hidden_size * 2],
                        initializer=self.initializer,
                    )
                else:
                    self.context_vector_word = tf.get_variable(
                        "what_is_the_informative_word",
                        shape=[self.hidden_size * 2],
                        initializer=self.initializer,
                    )
                self.context_vector_sentence_per_label = tf.get_variable(
                    "what_is_the_informative_sentence_per_label",
                    shape=[self.num_classes, self.hidden_size * 2],
                    initializer=self.initializer,
                )
            else:
                self.context_vector_word = tf.get_variable(
                    "what_is_the_informative_word",
                    shape=[self.hidden_size * 2],
                    initializer=self.initializer,
                )
                self.context_vector_sentence = tf.get_variable(
                    "what_is_the_informative_sentence",
                    shape=[self.hidden_size * 2],
                    initializer=self.initializer,
                )
