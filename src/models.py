from src.constants import BATCH_SIZE, NUM_EPOCH
from src.utilities import PrepUtility
import tensorflow as tf
import numpy as np
import math


class SkipGramModel:

    def __init__(self, session, word_dict, voc_size, lr=1e-3, embedding_size=50, sample_size=16):
        self.voc_size = voc_size
        self.word_dict = word_dict
        self.embedding_size = embedding_size

        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])

        # word2vec Model
        self.embeddings = tf.Variable(tf.random_uniform([self.voc_size, embedding_size], -1.0, 1.0))
        self.selected_embed = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        # weight and bias for nce_loss() function
        nce_weights = tf.Variable(tf.random_uniform([self.voc_size, embedding_size], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([self.voc_size]))

        self.cost_op = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, self.labels, self.selected_embed, sample_size, self.voc_size))

        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost_op)
        self.saver = tf.train.Saver()
        self.session = session
        self.trained_embeddings = None

    def train(self, num_epoch, total_batch, dataset_iterator, display_interval=50):
        # global NUM_EPOCH
        data = dataset_iterator.get_next()
        print('[INFO]: Total batch: %d' % total_batch)
        init = tf.global_variables_initializer()

        self.session.run(init)
        self.session.run(dataset_iterator.initializer)
        for epoch in range(num_epoch):
            for i in range(total_batch):
                batch_inputs, batch_labels = self.session.run(data)

                # BATCH_SIZE
                if batch_inputs.shape[0] != BATCH_SIZE:
                    continue

                self.session.run(self.train_op, feed_dict={self.inputs: batch_inputs, self.labels: batch_labels})

                if i % display_interval == 0:
                    # calculate the cost/accuracy of the current wv_model
                    cost = self.session.run(self.cost_op, feed_dict={self.inputs: batch_inputs,
                                                                     self.labels: batch_labels})
                    print("[INFO]: Epoch %02d/%02d, batch_iter: %03d/%d, Cost=%.4f" % (
                        epoch, NUM_EPOCH - 1, i, total_batch, cost))

    def save_model(self, model_path):
        path = self.saver.save(self.session, model_path)
        print('[INFO]: Model saved to %s' % path)

    def load_model(self, model_path):
        self.saver.restore(self.session, model_path)
        print('[INFO]: Model restored from: %s' % model_path)

    def word2vect(self, word):
        if self.trained_embeddings is None:
            self.trained_embeddings = self.embeddings.eval(session=self.session)
        if word not in self.word_dict:
            return None
        return self.trained_embeddings[self.word_dict[word]]


class Seq2SeqModel:
    def __init__(self, session, word_dict_a, embedding_size, max_len_q, lr=1e-3, hidden_size=128):
        self.session = session

        self.word_dict_a = word_dict_a
        self.vect_to_answer_dict = {v: k for k, v in self.word_dict_a.items()}

        self.input_max_len = max_len_q
        self.n_input = embedding_size
        self.n_class = len(self.word_dict_a)
        print('[INFO]: Seq2Seq enc input word vector size:', self.n_input)
        print('[INFO]: Seq2Seq dec input word vector size:', self.n_class)

        # Neural Network Model
        # encoder/decoder shape = [batch size, time steps, input size]
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.n_input])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.n_class])
        self.targets = tf.placeholder(tf.int64, [None, None])

        scope_prefix = 'model_airdialogue'
        with tf.variable_scope(scope_prefix):
            # Encoder Cell
            with tf.variable_scope('encode'):
                enc_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
                enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
                outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input,
                                                        dtype=tf.float32)
            # Decoder Cell
            with tf.variable_scope('decode'):
                dec_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
                dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
                outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input,
                                                        initial_state=enc_states,
                                                        dtype=tf.float32)
            self.model = tf.layers.dense(outputs, self.n_class, activation=None)

            self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.model, labels=self.targets))
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.cost)

        self.saver = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_prefix),
            max_to_keep=None,
        )

    def train(self, total_samples, dataset_iterator, validate_iterator, num_epoch=5000, display_interval=50):
        # BATCH_SIZE
        total_batch = 1
        print('[INFO]: Total batch: %d' % total_batch)

        data = dataset_iterator.get_next()

        init = tf.global_variables_initializer()
        self.session.run(init)

        iter = 0
        for ep in range(num_epoch):
            self.session.run(dataset_iterator.initializer)

            for i in range(total_batch):
                # Generate a batch data
                input_batch, output_batch, target_batch = self.session.run(data)
                if input_batch.shape[0] != 64:
                    continue

                feed_dict = {self.enc_input: input_batch, self.dec_input: output_batch, self.targets: target_batch}
                _, loss = self.session.run([self.optimizer, self.cost], feed_dict)
                if (iter + 1) % display_interval == 0:
                    print('[INFO]: Epoch: %04d batch_iter: %2d/%d - cost = %.6f' %
                          (ep + 1, i + 1, total_batch, loss))
                iter += 1
            if (ep + 1) % 100 == 0:
                self.eval(validate_iterator, total_samples)

        print('[INFO]: Training completed')

    def eval(self, validate_iterator, total_samples):
        test_data = validate_iterator.get_next()
        self.session.run(validate_iterator.initializer)

        correct = 0.
        for i in range(math.ceil(total_samples / 128)):
            # Generate a batch data
            input_batch, output_batch, target_batch = self.session.run(test_data)
            prediction = tf.argmax(self.model, 2)

            result = self.session.run(prediction, feed_dict={self.enc_input: input_batch, self.dec_input: output_batch,
                                                             self.targets: target_batch})
            # convert index number to actual token
            for batch in range(result.shape[0]):
                if np.array_equal(result[batch], target_batch[batch]):
                    correct += 1.

        print('[INFO]: eval: correctness = {}'.format(correct / float(total_samples)))

    # Answer the question using the trained seq2seq model
    def answer(self, sentence, wv_model):
        qa_seq = (PrepUtility.preprocess_text(sentence), '_U_')
        input_batch, output_batch, target_batch = PrepUtility.sequence_to_vectors(
            qa_seq, self.input_max_len, wv_model, self.word_dict_a)
        prediction = tf.argmax(self.model, 2)

        result = self.session.run(prediction, feed_dict={self.enc_input: [input_batch], self.dec_input: [output_batch],
                                                         self.targets: [target_batch]})

        # convert index number to actual token
        decoded = [self.vect_to_answer_dict[i] for i in result[0]]

        # Remove anything after '_E_'
        return ' '.join(decoded[:decoded.index('_E_')]) if "_E_" in decoded else ' '.join(decoded[:])

    def save_model(self, model_path):
        path = self.saver.save(self.session, model_path)
        print('[INFO]: Model saved to %s' % path)

    def load_model(self, model_path):
        self.saver.restore(self.session, model_path)
        print('[INFO]: Model restored from: %s' % model_path)