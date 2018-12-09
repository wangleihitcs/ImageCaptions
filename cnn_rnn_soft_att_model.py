import tensorflow as tf
from utils import nn
from nets import vgg

class Model(object):
    def __init__(self, is_training=True):
        self.batch_size = 32
        self.vocabulary_size = 14643     # 4698 word + '<S>' '</S>' '<EOS>'

        self.lstm_units = 512
        self.embedding_size = 512
        self.initial_learning_rate = 1e-4
        self.image_size = 224
        self.decay_rate = 0.9
        self.decay_epochs = 5 * (82783 / self.batch_size)

        self.max_caption_length = 20
        self.lstm_drop_rate = 0.3
        self.dropout_rate = 0.5
        self.is_training = is_training

        self.images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.sentences = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.max_caption_length])
        self.masks = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_caption_length])

        self.build_cnn()
        self.build_rnn()
        self.build_metrics()
        if self.is_training:
            self.build_optimizer()
            self.build_summary()
        print('---model built.---')

    def build_cnn(self):
        _, end_points = vgg.vgg_19(self.images, num_classes=1000, is_training=self.is_training)

        visual_feats = end_points['vgg_19/conv5/conv5_4']  # [batch_size, 14, 14, 512]
        self.visual_feats = tf.reshape(visual_feats, [self.batch_size, 196, 512])

        print('cnn built.')

    def build_rnn(self):
        with tf.variable_scope("word_embedding"):
            word_embedding_matrix = tf.get_variable(
                name='weights',
                shape=[self.vocabulary_size, self.embedding_size],
                initializer=nn.kernel_initializer(),
                regularizer=nn.kernel_regularizer(),
                trainable=True)

        # 1. build Word LSTM
        WordLSTM = tf.nn.rnn_cell.LSTMCell(
            self.lstm_units,
            initializer=nn.kernel_initializer())
        if self.is_training:
            WordLSTM = tf.nn.rnn_cell.DropoutWrapper(
                WordLSTM,
                input_keep_prob=1.0 - self.lstm_drop_rate,
                output_keep_prob=1.0 - self.lstm_drop_rate,
                state_keep_prob=1.0 - self.lstm_drop_rate)

        # 2. initialize word lstm
        with tf.variable_scope("word_lstm_initialize"):
            context = tf.reduce_mean(self.visual_feats, axis = 1)
            context_dropout = nn.dropout(context, self.dropout_rate, self.is_training, name='drop_c')
            initial_memory = nn.dense(context_dropout, units=self.lstm_units, activation=None, name='fc_m')
            initial_output = nn.dense(context_dropout, self.lstm_units, activation=None, name='fc_o')

        WordLSTM_last_state = initial_memory, initial_output
        WordLSTM_last_output = initial_output
        WordLSTM_last_word = tf.zeros([self.batch_size], tf.int32)  # tf.zeros() means the '<S>' token

        predictions = []  # store predict word
        prediction_corrects = []  # store correct predict to compute accuracy
        cross_entropies = []  # store cross entropy loss
        alphas = []

        # 3. generate word step by step
        for id in range(self.max_caption_length):
            with tf.variable_scope("word_embedding"):
                word_embedding = tf.nn.embedding_lookup(word_embedding_matrix, WordLSTM_last_word)

            with tf.variable_scope("attend", reuse=tf.AUTO_REUSE):
                alpha = self.attend(self.visual_feats, WordLSTM_last_output)
                context = tf.reduce_sum(self.visual_feats * tf.expand_dims(alpha, axis=2), axis=1)
                if self.is_training:
                    titled_masks = tf.tile(tf.expand_dims(self.masks[:, id], axis=1), [1, 196])
                    masked_alpha = alpha * titled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            with tf.variable_scope('WordLSTM'):
                inputs = tf.concat([context, word_embedding], axis=1)
                WordLSTM_current_output, WordLSTM_current_state = WordLSTM(inputs, WordLSTM_last_state)

            with tf.variable_scope('decode'):
                expanded_output = tf.concat([context, word_embedding, WordLSTM_current_output], axis=1)
                expanded_output_dropout = nn.dropout(expanded_output, self.dropout_rate, self.is_training, name='drop')
                logits = nn.dense(expanded_output_dropout, units=self.vocabulary_size, activation=None, name='fc')
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            tf.get_variable_scope().reuse_variables()

            WordLSTM_last_state = WordLSTM_current_state
            # use teacher policy
            if self.is_training:
                WordLSTM_last_word = self.sentences[:, id]
            else:
                WordLSTM_last_word = prediction

            # compute loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.sentences[:, id],
                logits=logits)
            masked_cross_entropy = cross_entropy * self.masks[:, id]
            cross_entropies.append(masked_cross_entropy)

            # compute accuracy
            ground_truth = tf.cast(self.sentences[:, id], tf.int64)
            prediction_correct = tf.where(
                tf.equal(prediction, ground_truth),
                tf.cast(self.masks[:, id], tf.float32),
                tf.cast(tf.zeros_like(prediction), tf.float32)
            )
            prediction_corrects.append(prediction_correct)

        # 4. compute accuracy
        prediction_corrects = tf.stack(prediction_corrects, axis=1)
        accuracy = tf.reduce_sum(prediction_corrects) / tf.reduce_sum(self.masks)

        self.predictions = predictions
        self.cross_entropies = cross_entropies
        self.alphas = alphas
        self.accuracy = accuracy
        print('rnn built.')

    def build_metrics(self):
        cross_entropies = tf.stack(self.cross_entropies, axis=1)
        self.cross_entropy_text = tf.reduce_sum(cross_entropies) / tf.reduce_sum(self.masks)

        self.reg_loss = tf.losses.get_regularization_loss()

        if self.is_training:
            alphas = tf.stack(self.alphas, axis=1)
            alphas = tf.reshape(alphas, [self.batch_size, 196, -1])
            attentions = tf.reduce_sum(alphas, axis=2)
            diffs = tf.ones_like(attentions) - attentions
            self.attention_loss = 0.01 * tf.nn.l2_loss(diffs) / (self.batch_size * 196)
            self.loss = self.cross_entropy_text + self.attention_loss + self.reg_loss

        else:
            self.loss = self.cross_entropies + self.reg_loss

        print('metrics built.')

    def build_optimizer(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.constant(self.initial_learning_rate)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=self.decay_epochs,
                decay_rate=self.decay_rate,
                staircase=True
            )

        learning_rate_decay_fn = _learning_rate_decay_fn
        # learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-6
            )

            self.step_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=self.global_step,
                learning_rate=learning_rate,
                optimizer=optimizer,
                clip_gradients=5.0,
                learning_rate_decay_fn=learning_rate_decay_fn
            )
        print('optimizer built.')

    def build_summary(self):
        with tf.name_scope("metrics"):
            tf.summary.scalar('cross entropy', self.cross_entropy_text)
            tf.summary.scalar('attention loss', self.attention_loss)
            tf.summary.scalar('reg loss', self.reg_loss)
            tf.summary.scalar('acc', self.accuracy)

        self.summary = tf.summary.merge_all()
        print('summary built.')

    # contexts = [batch, 196, 512], output = [batch, 512]
    def attend(self, contexts, output):
        alpha_list = []
        contexts = nn.dropout(contexts, self.dropout_rate, self.is_training, name='drop_c')
        output = nn.dropout(output, self.dropout_rate, self.is_training, name='drop_o')

        for i in range(self.batch_size):
            context = contexts[i]           # shape = [196, 512]
            logits_context = nn.dense(context, units=196, activation=None, use_bias=False, name='fc_lc')   # shape = [196, 196]

            output_i = tf.reshape(output[i], shape=[512, 1])
            ones = tf.ones([1, 196], tf.float32)
            logits_temp = tf.matmul(output_i, ones)  # shape = [512, 196]
            logits_temp = tf.transpose(logits_temp)  # shape = [196, 512]
            logits_output = nn.dense(logits_temp, units=196, activation=None, use_bias=False, name='fc_lo')  # shape = [196, 196]

            logit_tanh = tf.tanh(logits_context + logits_output)
            alpha = nn.dense(logit_tanh, units=1, activation=None, use_bias=False, name='fc_alpha')          # shape = [196, 1]
            alpha = tf.reshape(alpha, shape=[196])
            alpha_list.append(alpha)

        alpha_batch = tf.stack(alpha_list, axis=0)  # shape = [batch_size, 196]
        alpha = tf.nn.softmax(alpha_batch)  # shape = [batch_size, 196]

        return alpha