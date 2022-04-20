import logging

import tensorflow as tf


class HAN:
    def __init__(
        self,
        num_classes,
        learning_rate,
        batch_size,
        decay_steps,
        decay_rate,
        sequence_length,
        num_sentences,
        vocab_size,
        embed_size,
        hidden_size,
        log_dir,
        initializer=tf.random_normal_initializer(stddev=0.1),
        clip_gradients=5.0,
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.sentence_length = int(self.sequence_length / self.num_sentences)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients

        self.learning_rate = tf.Variable(
            learning_rate, trainable=False, name="learning_rate"
        )

        self.input_x = tf.placeholder(
            tf.int32, [None, self.sequence_length], name="input_x"
        )
        self.input_y = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y"
        )
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1))
        )
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss = self.loss_function()
        self.train_op = self.train()

        self.training_loss_per_batch = tf.summary.scalar(
            "train_loss_per_batch", self.loss
        )
        self.training_loss_per_epoch = tf.summary.scalar(
            "train_loss_per_epoch", self.loss
        )
        self.writer = tf.summary.FileWriter(str(log_dir))

    def instantiate_weights(
        self,
    ):
        logger = logging.getLogger("HAN.instantiate_weights")

        with tf.name_scope("embedding_projection"):
            logger.debug("embedding projection")
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
            self.b_projection = tf.get_variable(
                "b_projection", shape=[self.num_classes]
            )

        with tf.name_scope("gru_weights_word_level"):
            logger.debug("GRU word-level weights (update gate)")
            self.W_z = tf.get_variable(
                "W_z",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.U_z = tf.get_variable(
                "U_z",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])

            logger.debug("GRU word-level weights (reset gate)")
            self.W_r = tf.get_variable(
                "W_r",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.U_r = tf.get_variable(
                "U_r",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable(
                "W_h",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.U_h = tf.get_variable(
                "U_h",
                shape=[self.embed_size, self.hidden_size],
                initializer=self.initializer,
            )
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_sentence_level"):
            logger.debug("GRU sentence-level weights (update gate)")
            self.W_z_sentence = tf.get_variable(
                "W_z_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.U_z_sentence = tf.get_variable(
                "U_z_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.b_z_sentence = tf.get_variable(
                "b_z_sentence", shape=[self.hidden_size * 2]
            )

            logger.debug("GRU sentence-level weights (reset gate)")
            self.W_r_sentence = tf.get_variable(
                "W_r_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.U_r_sentence = tf.get_variable(
                "U_r_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.b_r_sentence = tf.get_variable(
                "b_r_sentence", shape=[self.hidden_size * 2]
            )

            self.W_h_sentence = tf.get_variable(
                "W_h_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.U_h_sentence = tf.get_variable(
                "U_h_sentence",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.b_h_sentence = tf.get_variable(
                "b_h_sentence", shape=[self.hidden_size * 2]
            )

        with tf.name_scope("attention"):
            logger.debug("Word-level attention")
            self.W_w_attention_word = tf.get_variable(
                "W_w_attention_word",
                shape=[self.hidden_size * 2, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.W_b_attention_word = tf.get_variable(
                "W_b_attention_word", shape=[self.hidden_size * 2]
            )

            logger.debug("Sentence-level attention")
            self.W_w_attention_sentence = tf.get_variable(
                "W_w_attention_sentence",
                shape=[self.hidden_size * 4, self.hidden_size * 2],
                initializer=self.initializer,
            )
            self.W_b_attention_sentence = tf.get_variable(
                "W_b_attention_sentence", shape=[self.hidden_size * 2]
            )

            logger.debug("Informative word across all labels")
            self.context_vector_word = tf.get_variable(
                "what_is_the_informative_word",
                shape=[self.hidden_size * 2],
                initializer=self.initializer,
            )
            logger.debug("Informative sentence across all labels")
            self.context_vector_sentence = tf.get_variable(
                "what_is_the_informative_sentence",
                shape=[self.hidden_size * 2],
                initializer=self.initializer,
            )

    def inference(self):
        """main computation graph here: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier"""
        logger = logging.getLogger("HAN.inference")

        logger.info("1 Word Encoder")

        logger.debug("1.1 embedding of words")
        input_x = tf.stack(tf.split(self.input_x, self.num_sentences, axis=1), axis=1)
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, input_x)
        embedded_words_reshaped = tf.reshape(
            self.embedded_words, shape=[-1, self.sentence_length, self.embed_size]
        )

        logger.debug("1.2 forward gru")
        hidden_state_forward_list = self.gru_word_level(embedded_words_reshaped)

        logger.debug("1.3 backward gru")
        hidden_state_backward_list = self.gru_word_level(
            embedded_words_reshaped, reverse=True
        )

        logger.debug("1.4 concat forward hidden state and backward hidden state")
        self.hidden_state = [
            tf.concat([h_forward, h_backward], axis=1)
            for h_forward, h_backward in zip(
                hidden_state_forward_list, hidden_state_backward_list
            )
        ]

        logger.info("2 Word Attention")
        sentence_representation = self.attention_word_level(self.hidden_state)
        sentence_representation = tf.reshape(
            sentence_representation,
            shape=self.word_level_attention_shape(),
        )

        logger.info("3 Sentence Encoder")

        logger.debug("3.1 forward gru for sentence")
        hidden_state_forward_sentences = self.gru_sentence_level(
            sentence_representation
        )

        logger.debug("3.2 backward gru for sentence")
        hidden_state_backward_sentences = self.gru_sentence_level(
            sentence_representation, reverse=True
        )

        logger.debug("3.3 concat forward hidden state and backward hidden state")
        self.hidden_state_sentence = [
            tf.concat([h_forward, h_backward], axis=self.sentence_encoding_axis())
            for h_forward, h_backward in zip(
                hidden_state_forward_sentences, hidden_state_backward_sentences
            )
        ]

        logger.info("4 Sentence Attention")

        document_representation = self.attention_sentence_level(
            self.hidden_state_sentence
        )
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(document_representation, rate=self.dropout_rate)

        logger.info("5 Logits (linear layer) and Predictions (argmax)")
        with tf.name_scope("output"):
            logits = self.linear_layer()

        return logits

    def word_level_attention_shape(self):
        return [-1, self.num_sentences, self.hidden_size * 2]

    def sentence_encoding_shape(self):
        return [-1, self.hidden_size * 4]

    def sentence_encoding_axis(self):
        return len(self.sentence_encoding_shape()) - 1

    def linear_layer(self):
        return tf.matmul(self.h_drop, self.W_projection) + self.b_projection

    def loss_function(self, l2_lambda=0.0001):
        logger = logging.getLogger("loss_function")

        with tf.name_scope("loss"):
            logger.debug("Cross-entropy Loss")
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input_y, logits=self.logits
            )
            self.cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(losses, axis=1))
            logger.debug("L2 Regularization Loss")
            self.l2_regularization_loss = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if "bias" not in v.name
                    ]
                )
                * l2_lambda
            )
            logger.debug("Total Loss")
            loss = self.cross_entropy_loss + self.l2_regularization_loss
        return loss

    def gru_word_level(self, embedded_words, reverse=False):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        embedded_words_split = tf.split(embedded_words, self.sentence_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_split]
        if reverse:
            embedded_words_squeeze.reverse()
        h_t = tf.ones_like(embedded_words_squeeze[0])
        h_t_list = []
        for Xt in embedded_words_squeeze:
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        return h_t_list

    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        z_t = tf.nn.sigmoid(
            tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z
        )
        r_t = tf.nn.sigmoid(
            tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r
        )
        h_t_candiate = tf.nn.tanh(
            tf.matmul(Xt, self.W_h)
            + r_t * (tf.matmul(h_t_minus_1, self.U_h))
            + self.b_h
        )
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def attention_word_level(self, hidden_state):
        """
        input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
        :return:representation.shape:[num_classes,batch_size*num_sentences,hidden_size*2] or [batch_size*num_sentences,hidden_size*2]
        """
        hidden_state_ = tf.stack(hidden_state, axis=1)
        hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, self.hidden_size * 2])
        hidden_representation = tf.nn.tanh(
            tf.matmul(hidden_state_2, self.W_w_attention_word) + self.W_b_attention_word
        )
        hidden_representation = tf.reshape(
            hidden_representation,
            shape=[-1, self.sentence_length, self.hidden_size * 2],
        )
        hidden_state_context_similarity = tf.multiply(
            self.word_hidden_representation(hidden_representation),
            self.context_vector_word_level(),
        )
        attention_logits = tf.reduce_sum(
            hidden_state_context_similarity,
            axis=self.word_level_attention_logits_axis(),
        )
        attention_logits_max = tf.reduce_max(
            attention_logits,
            axis=(self.word_level_attention_logits_axis() - 1),
            keep_dims=True,
        )
        self.p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(
            self.p_attention, axis=self.word_level_attention_logits_axis()
        )
        sentence_representation = tf.multiply(p_attention_expanded, hidden_state_)
        sentence_representation = tf.reduce_sum(
            sentence_representation, axis=(self.word_level_attention_logits_axis() - 1)
        )
        return sentence_representation

    def word_hidden_representation(self, hidden_representation):
        return hidden_representation

    def context_vector_word_level(self):
        return self.context_vector_word

    def word_level_attention_logits_axis(self):
        return 2

    def gru_sentence_level(self, sentence_representation, reverse=False):
        """
        :param sentence_representation: [num_classes, batch_size,num_sentences,hidden_size*2] or [batch_size,num_sentences,hidden_size*2]
        :return: hidden state: a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size] or [batch_size,hidden_size]
        """
        sentence_representation_split = tf.split(
            sentence_representation, self.num_sentences, axis=self.sentence_axis()
        )
        sentence_representation_squeeze = [
            tf.squeeze(x, axis=self.sentence_axis())
            for x in sentence_representation_split
        ]
        if reverse:
            sentence_representation_squeeze.reverse()
        h_t = tf.ones_like(sentence_representation_squeeze[0])
        h_t_list = []
        for Xt in sentence_representation_squeeze:
            h_t = self.gru_single_step_sentence_level(Xt, h_t)
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        return h_t_list

    def sentence_axis(self):
        return 1

    def gru_single_step_sentence_level(self, Xt, h_t_minus_1):
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        W_z_sentence, U_z_sentence = self.gru_update_gate_weights()
        W_r_sentence, U_r_sentence = self.gru_reset_gate_weights()
        W_h_sentence, U_h_sentence = self.gru_hidden_state_weights()
        z_t = tf.nn.sigmoid(
            tf.matmul(Xt, W_z_sentence)
            + tf.matmul(h_t_minus_1, U_z_sentence)
            + self.b_z_sentence
        )
        r_t = tf.nn.sigmoid(
            tf.matmul(Xt, W_r_sentence)
            + tf.matmul(h_t_minus_1, U_r_sentence)
            + self.b_r_sentence
        )
        h_t_candiate = tf.nn.tanh(
            tf.matmul(Xt, W_h_sentence)
            + r_t * (tf.matmul(h_t_minus_1, U_h_sentence))
            + self.b_h_sentence
        )
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t

    def gru_update_gate_weights(self):
        return (self.W_z_sentence, self.U_z_sentence)

    def gru_reset_gate_weights(self):
        return (self.W_r_sentence, self.U_r_sentence)

    def gru_hidden_state_weights(self):
        return (self.W_h_sentence, self.U_h_sentence)

    def attention_sentence_level(self, hidden_state_sentence):
        """
        input1: hidden_state_sentence: a list,len:num_sentence,element:[num_classes,None,hidden_size*4] or [None,hidden_size*4]
        input2 (in self): sentence level context vector:[num_classes,self.hidden_size*2] or [self.hidden_size*2]
        :return:representation.shape:[num_classes,None,hidden_size*4] or [None,hidden_size*4]
        """
        hidden_state_ = tf.stack(
            hidden_state_sentence, axis=self.sentence_encoding_axis()
        )
        hidden_state_2 = tf.reshape(hidden_state_, shape=self.sentence_encoding_shape())
        hidden_representation = tf.nn.tanh(
            tf.matmul(hidden_state_2, self.sentence_level_attention_weights())
            + self.W_b_attention_sentence
        )
        hidden_representation = tf.reshape(
            hidden_representation,
            shape=self.word_level_attention_shape(),
        )
        hidden_representation = self.sentence_hidden_representation(
            hidden_representation
        )
        hidden_state_context_similarity = tf.multiply(
            hidden_representation, self.context_vector_sentence_level()
        )
        attention_logits = tf.reduce_sum(
            hidden_state_context_similarity,
            axis=self.sentence_level_attention_logits_axis(),
        )
        attention_logits_max = tf.reduce_max(
            attention_logits,
            axis=(self.sentence_level_attention_logits_axis() - 1),
            keep_dims=True,
        )
        self.p_attention_sent = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(
            self.p_attention_sent, axis=self.sentence_level_attention_logits_axis()
        )
        document_representation = tf.multiply(p_attention_expanded, hidden_state_)
        document_representation = tf.reduce_sum(
            document_representation,
            axis=(self.sentence_level_attention_logits_axis() - 1),
        )
        return document_representation

    def sentence_level_attention_weights(self):
        return self.W_w_attention_sentence

    def sentence_hidden_representation(self, hidden_representation):
        return hidden_representation

    def context_vector_sentence_level(self):
        return self.context_vector_sentence

    def sentence_level_attention_logits_axis(self):
        return 2

    def train(self):
        """based on the loss, use SGD to update parameter"""
        logger = logging.getLogger("train")

        logger.info("Apply exponential decay to LR")
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True,
        )

        logger.info("Optimize SGD with Adam and gradient clipping")
        train_op = tf.contrib.layers.optimize_loss(
            self.loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer="Adam",
            clip_gradients=self.clip_gradients,
        )

        logger.info("Training configured")
        return train_op


class HA_GRU(HAN):
    def instantiate_weights(
        self,
    ):
        super(HA_GRU, self).instantiate_weights()
        logger = logging.getLogger("HA_GRU.instantiate_weights")

        with tf.name_scope("attention"):
            logger.debug("Sentence-level attention")
            logger.debug("Informative sentence per label")
            self.context_vector_sentence_per_label = tf.get_variable(
                "what_is_the_informative_sentence_per_label",
                shape=[self.num_classes, self.hidden_size * 2],
                initializer=self.initializer,
            )

    def linear_layer(self):
        h_drop_transposed = tf.transpose(self.h_drop, perm=[1, 2, 0])
        logits = tf.multiply(h_drop_transposed, self.W_projection)
        return tf.reduce_sum(logits, axis=1) + self.b_projection

    def context_vector_sentence_level(self):
        return tf.expand_dims(
            tf.expand_dims(self.context_vector_sentence_per_label, axis=1), axis=1
        )

    def sentence_hidden_representation(self, hidden_representation):
        return tf.expand_dims(hidden_representation, axis=0)

    def sentence_level_attention_logits_axis(self):
        return 3


class HLAN(HA_GRU):
    def instantiate_weights(
        self,
    ):
        super(HLAN, self).instantiate_weights()
        logger = logging.getLogger("HLAN.instantiate_weights")

        with tf.name_scope("attention"):
            logger.debug("Word-level attention")
            logger.debug("Informative word per label")
            self.context_vector_word_per_label = tf.get_variable(
                "what_is_the_informative_word_per_label",
                shape=[self.num_classes, self.hidden_size * 2],
                initializer=self.initializer,
            )

    def word_level_attention_shape(self):
        return [self.num_classes, -1, self.num_sentences, self.hidden_size * 2]

    def sentence_encoding_shape(self):
        return [self.num_classes, -1, self.hidden_size * 4]

    def sentence_encoding_axis(self):
        return len(self.sentence_encoding_shape()) - 1

    def sentence_level_attention_weights(self):
        return tf.tile(
            tf.expand_dims(self.W_w_attention_sentence, axis=0),
            [self.num_classes, 1, 1],
        )

    def word_hidden_representation(self, hidden_representation):
        return tf.expand_dims(hidden_representation, axis=0)

    def context_vector_word_level(self):
        return tf.expand_dims(
            tf.expand_dims(self.context_vector_word_per_label, axis=1), axis=1
        )

    def word_level_attention_logits_axis(self):
        return 3

    def sentence_hidden_representation(self, hidden_representation):
        return hidden_representation

    def sentence_axis(self):
        return 2

    def gru_update_gate_weights(self):
        dimension = [self.num_classes, 1, 1]
        return (
            tf.tile(tf.expand_dims(self.W_z_sentence, axis=0), dimension),
            tf.tile(tf.expand_dims(self.U_z_sentence, axis=0), dimension),
        )

    def gru_reset_gate_weights(self):
        dimension = [self.num_classes, 1, 1]
        return (
            tf.tile(tf.expand_dims(self.W_r_sentence, axis=0), dimension),
            tf.tile(tf.expand_dims(self.U_r_sentence, axis=0), dimension),
        )

    def gru_hidden_state_weights(self):
        dimension = [self.num_classes, 1, 1]
        return (
            tf.tile(tf.expand_dims(self.W_h_sentence, axis=0), dimension),
            tf.tile(tf.expand_dims(self.U_h_sentence, axis=0), dimension),
        )
