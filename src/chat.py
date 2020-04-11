
# -*- coding: utf-8 -*-

"""A chat/shoutbox using Sijax."""

import os
import hmac
from hashlib import sha1
import flask

from flask import Flask, g, render_template, abort, request, jsonify
from flask import session as f_session
from pydrive import drive
from werkzeug.security import safe_str_cmp
import flask_sijax

# dependencies and prerequisites
import nltk
import pickle
import fileinput
import numpy as np
import pandas as pd
import os, re, math
import tensorflow as tf
import datefinder
import mechanize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
from geograpy import extraction
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datetime import date

path = os.path.join('.', os.path.dirname(__file__), 'static/js/sijax/')
app = Flask(__name__)

app.config['SIJAX_STATIC_PATH'] = path
app.config['SIJAX_JSON_URI'] = '../static/js/sijax/json2.js'
flask_sijax.Sijax(app)


'''------------------------------------------------------------------------------------------------------------------------------------'''


class FileUtils:
    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def gdrive_download_file(filename, id):
        file = drive.CreateFile({'id': id})
        file.GetContentFile(filename)

    @staticmethod
    def gdrive_download_dir(path, id):
        self = FileUtils
        self.mkdir(path)
        files = drive.ListFile({'q': "'%s' in parents" % id}).GetList()
        for f in files:
            file_path = '%s/%s' % (path, f['title'])
            mimeType = f['mimeType']
            file_id = f['id']
            if mimeType == 'application/vnd.google-apps.folder':
                self.gdrive_download_dir(file_path, file_id)
            else:
                self.gdrive_download_file(file_path, file_id)



class PrepUtils:
    @staticmethod
    def safe_read(l, idx):
        if idx < 0 or idx > len(l) - 1:
            return []
        return [l[idx]]

    @staticmethod
    # relax common english contractions
    def handle_contractions(text):
        # fix strength apostrophe
        text = text.replace('’', "'")
        text = text.replace('”', '"')

        # common English contractions from Lab05
        contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                            "could've": "could have",
                            "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                            "hadn't": "had not",
                            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
                            "he's": "he is", "how'd": "how did",
                            "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                            "I'd've": "I would have",
                            "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                            "i'd": "i would",
                            "i'd've": "i would have",
                            "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                            "isn't": "is not",
                            "it'd": "it would",
                            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                            "let's": "let us",
                            "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                            "mightn't've": "might not have",
                            "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                            "needn't": "need not", "needn't've": "need not have",
                            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                            "shan't": "shall not", "sha'n't": "shall not",
                            "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                            "she'll": "she will", "she'll've": "she will have",
                            "she's": "she is", "should've": "should have", "shouldn't": "should not",
                            "shouldn't've": "should not have", "so've": "so have",
                            "so's": "so as", "this's": "this is", "that'd": "that would",
                            "that'd've": "that would have",
                            "that's": "that is", "there'd": "there would",
                            "there'd've": "there would have", "there's": "there is", "here's": "here is",
                            "they'd": "they would", "they'd've": "they would have",
                            "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                            "they've": "they have", "to've": "to have", "wasn't": "was not",
                            "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                            "we'll've": "we will have",
                            "we're": "we are", "we've": "we have",
                            "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                            "what're": "what are", "what's": "what is", "what've": "what have",
                            "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                            "where've": "where have", "who'll": "who will",
                            "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                            "why've": "why have", "will've": "will have",
                            "won't": "will not", "won't've": "will not have", "would've": "would have",
                            "wouldn't": "would not", "wouldn't've": "would not have",
                            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                            "y'all're": "you all are", "y'all've": "you all have",
                            "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                            "you'll've": "you will have", "you're": "you are", "you've": "you have"}
        for c in contraction_dict:
            if c in text:
                text = text.replace(c, contraction_dict[c])
        return text

    @staticmethod
    def preprocess_text(text):
        lemmatizer = WordNetLemmatizer()

        # decapitalize
        text = text.lower()
        # replace contractions:
        text = PrepUtils.handle_contractions(text)
        # remove symbols
        text = re.sub(r'[^\w\s]', '', text)
        # remove numbers
        text = re.sub(r'[0-9]', '', text)
        # tokenize
        tokens = word_tokenize(text)

        # lemmatizer
        tokens = [lemmatizer.lemmatize(tk) for tk in tokens]
        return tokens

    @staticmethod
    def tokens_to_ids(tokens, lookup):
        ids = []
        for token in tokens:
            if token in lookup:
                ids.append(lookup[token])
            else:
                ids.append(lookup['_U_'])
        return ids

    @staticmethod
    def get_question_vector(tokens, max_len, wv_model):
        # append auxiliary tags for padding and unknown word
        WV_TAG_U = np.zeros((wv_model.embedding_size), dtype=np.float32)
        WV_TAG_P = np.ones((wv_model.embedding_size), dtype=np.float32)

        word_vectors = []
        # replace unknown words with pre-defined U tag vector
        for token in tokens:
            vector = wv_model.word2vect(token)
            if vector is None:
                vector = WV_TAG_U
            word_vectors.append(vector)

        # pad up to max length with pre-defined P tag vectors
        diff = max_len - len(word_vectors)
        [word_vectors.append(WV_TAG_P) for _ in range(diff)]
        return np.array(word_vectors)

    @staticmethod
    def get_answer_vector(answer, lookup, convert_index=True):
        # lookup one-hot idx from word_dict
        seq_idx = PrepUtils.tokens_to_ids(answer, lookup)
        if convert_index:
            # idx to one-hot vector
            return np.eye(len(lookup), dtype=np.float32)[seq_idx]
        return np.array(seq_idx)

    @staticmethod
    def sequence_to_vectors(seq, max_len, wv_model, word_dict_a):
        (question, answer) = seq
        q = PrepUtils.get_question_vector(question, max_len, wv_model)
        a_out = PrepUtils.get_answer_vector(['_B_'] + [answer], word_dict_a)
        a_target = PrepUtils.get_answer_vector([answer] + ['_E_'], word_dict_a, convert_index=False)
        return q, a_out, a_target


class AirdialogueDataset:
    DATASET_ID_AIRDIALOGUE = '1yGaZzxQ6Nz4wrBd5Eo0wuzoYZEBlhWED'

    DATASET_FILENAME_AIRDIALOGUE = 'airdialogue.tsv'

    @staticmethod
    def download():
        self = AirdialogueDataset
        if not os.path.exists('./' + self.DATASET_FILENAME_AIRDIALOGUE):
            FileUtils.gdrive_download_file(self.DATASET_FILENAME_AIRDIALOGUE,
                                           self.DATASET_ID_AIRDIALOGUE)
        # if not os.path.exists('./' + self.DATASET_FILENAME_COMIC):
        #     FileUtils.gdrive_download_file(self.DATASET_FILENAME_COMIC,
        #                                    self.DATASET_ID_COMIC)
        # if not os.path.exists('./' + self.DATASET_FILENAME_PROFESSIONAL):
        #     FileUtils.gdrive_download_file(self.DATASET_FILENAME_PROFESSIONAL,
        #                                    self.DATASET_ID_PROFESSIONAL)
        return self.read_dataset()

    @staticmethod
    def read_dataset():
        self = AirdialogueDataset
        print("debug in AirdialogueDataset")
        data_airdialogue = pd.read_csv(self.DATASET_FILENAME_AIRDIALOGUE, sep="\t")
        return data_airdialogue

    @staticmethod
    def preprocess(datasets):
        # preprocess input data
        qa_sequences = list()
        corpusQ, corpusA = set(), set()
        for index, row in datasets.iterrows():
            question = row[0]
            answer = row[1]
            q_tokens = PrepUtils.preprocess_text(question)
            corpusQ.update(q_tokens)
            corpusA.add(answer)
            qa_sequences.append((q_tokens, answer))
        datasets = (qa_sequences, list(corpusA))
        return datasets

class SkipGramDatasetReader(AirdialogueDataset):
    def __init__(self, word_dict_path, window_size=1):
        self.word_dict_path = word_dict_path

        # download datasets
        datasets = super().download()

        #
        #
        #
        #
        # ing datasets
        sequences = []
        total = []
        for index, row in datasets.iterrows():
            processed_q = PrepUtils.preprocess_text(row[0])
            processed_a = PrepUtils.preprocess_text(row[1])
            sequences.append(processed_q)
            sequences.append(processed_a)
            total.extend(processed_q)
            total.extend(processed_a)
        self.bow = []
        [self.bow.append(i) for i in total if not self.bow.count(i)]

        # generate word dict for question
        self.word_dict = {w: i for i, w in enumerate(self.bow)}
        with open(self.word_dict_path, 'wb') as file:
            print("[INFO]: Saving new word dictionary record (one-hot) for question tokens")
            pickle.dump(self.word_dict, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO]: Total word counts: {}".format(len(self.word_dict)))

        self.skip_grams = []
        for i, seq in enumerate(sequences):
            for j, word in enumerate(seq):
                context = []
                for ws in range(window_size):
                    context.extend(PrepUtils.safe_read(seq, j - ws - 1))
                    context.extend(PrepUtils.safe_read(seq, j + ws + 1))
                for c in context:
                    self.skip_grams.append([self.word_dict[word], self.word_dict[c]])
        self.num_samples = len(self.skip_grams)
        self.skip_grams = np.array(self.skip_grams)
        self.vocab_size = len(self.bow)

    def prepare(self, num_epoch, batch_size, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.skip_grams[:, 0], np.expand_dims(self.skip_grams[:, 1], 1)))
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat(num_epoch)
        return dataset.make_initializable_iterator()

    @staticmethod
    def resume_word_dict(path):
        print("huiwen debug path", path)
        assert os.path.isfile(path), True
        with open(path, 'rb') as file:
            print("[INFO]: Resumed from previous word dictionary record (one-hot) for question tokens")
            word_dict = pickle.load(file)
        return word_dict


class Seq2SeqDatasetReader(AirdialogueDataset):
    def __init__(self, word_dict_a_path, wv_model):
        self.wv_model = wv_model
        self.word_dict_a_path = word_dict_a_path

        # section 1: preprocessing: word preprocessing
        datasets = super().download()
        datasets = super().preprocess(datasets)

        # preprocessing: words to vectors and build answer lookup dict
        #self.datasets = {}
        #self.word_dict_a = {}
        # self.max_len_q = {}
        # self.num_seq = {}
        print("huiwen debug type of dataset in seq2seqreader", type(datasets))
        (qa_seqs, unique_answers) = datasets
        # append auxiliary tags for begin and end
        unique_answers.append('_B_')
        unique_answers.append('_E_')
        unique_answers.append('_U_')

        # generate answer word dict
        word_dict_path = word_dict_a_path
        self.word_dict_a = {n: i for i, n in enumerate(unique_answers)}
        with open(word_dict_path, 'wb') as file:
            print("[INFO]: Saving new word dictionary record (one-hot) for answers")
            pickle.dump(self.word_dict_a, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO]: Total answers counts: {}".format(len(self.word_dict_a)))

        self.num_seq = len(qa_seqs)
        self.max_len_q= max([len(q) for (q, a) in qa_seqs])
        input_batch, output_batch, target_batch = [], [], []
        for seq in qa_seqs:
            q_in, a_out, a_target = PrepUtils.sequence_to_vectors(
                seq, self.max_len_q, self.wv_model, self.word_dict_a)
            input_batch.append(q_in)
            if len(q_in.shape) < 2:
                print(seq, q_in)
            output_batch.append(a_out)
            target_batch.append(a_target)
        self.datasets = (input_batch, output_batch, target_batch)
        print("[INFO] Max question length:", self.max_len_q)

    def prepare(self, num_epoch=1000, batch_size=64, shuffle=False):
        iterators = {}
        dataset = self.datasets
        batch_in = np.array(dataset[0])
        batch_out = np.array(dataset[1])
        batch_target = np.array(dataset[2])
        dataset = tf.data.Dataset.from_tensor_slices((batch_in, batch_out, batch_target))
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat(num_epoch)
        iterators = dataset.make_initializable_iterator()
        return iterators

    @staticmethod
    def resume_answer_dict():
        global WORD_DICT_A_PATH
        word_dict_a = {}
        # for p in PersonalityDataset.PERSONALITIES:
        #     file_path = '%s_%s' % (WORD_DICT_A_PATH, p)
        assert os.path.isfile(WORD_DICT_A_PATH), True
        with open(WORD_DICT_A_PATH, 'rb') as file:
            print("[INFO]: Resumed from previous word dictionary record (one-hot) for answers")
            word_dict_a = pickle.load(file)
        return word_dict_a

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
        #global NUM_EPOCH
        data = dataset_iterator.get_next()
        print('[INFO]: Total batch: %d' % total_batch)
        init = tf.global_variables_initializer()

        self.session.run(init)
        self.session.run(dataset_iterator.initializer)
        for epoch in range(num_epoch):
            for i in range(total_batch):
                batch_inputs, batch_labels = self.session.run(data)

                #BATCH_SIZE
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
        #self.personality = personality

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

        #scope_prefix = 'model_{}'.format(self.personality)
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

    def train(self, total_samples, dataset_iterator, validate_iterator,
              num_epoch=5000, display_interval=50):
        #BATCH_SIZE
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

                _, loss = self.session.run([self.optimizer, self.cost],
                                           feed_dict={self.enc_input: input_batch,
                                                      self.dec_input: output_batch,
                                                      self.targets: target_batch})
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

            result = self.session.run(prediction,
                                      feed_dict={self.enc_input: input_batch,
                                                 self.dec_input: output_batch,
                                                 self.targets: target_batch})
            # convert index number to actual token
            for batch in range(result.shape[0]):
                if np.array_equal(result[batch], target_batch[batch]):
                    correct += 1.

        print('[INFO]: eval: correctness = {}'.format(correct / float(total_samples)))

    # Answer the question using the trained seq2seq model
    def answer(self, sentence, wv_model):
        qa_seq = (PrepUtils.preprocess_text(sentence), '_U_')
        input_batch, output_batch, target_batch = PrepUtils.sequence_to_vectors(
            qa_seq, self.input_max_len, wv_model, self.word_dict_a)
        prediction = tf.argmax(self.model, 2)

        result = self.session.run(prediction,
                                  feed_dict={self.enc_input: [input_batch],
                                             self.dec_input: [output_batch],
                                             self.targets: [target_batch]})

        # convert index number to actual token
        decoded = [self.vect_to_answer_dict[i] for i in result[0]]

        # Remove anything after '_E_'
        if "_E_" in decoded:
            end = decoded.index('_E_')
            translated = ' '.join(decoded[:end])
        else:
            translated = ' '.join(decoded[:])

        return translated

    def save_model(self, model_path):
        #model_path = '%s_%s' % (model_path, self.personality)
        path = self.saver.save(self.session, model_path)
        print('[INFO]: Model saved to %s' % path)

    def load_model(self, model_path):
        #model_path = '%s_%s' % (model_path, self.personality)
        self.saver.restore(self.session, model_path)
        print('[INFO]: Model restored from: %s' % model_path)

class ConversationTrigger:
    END = '0'
    CHANGE_PERSONALITY = '1'

    TRIGGERS = {
        END: ['bye', 'goodbye', 'talk to you later', 'i need to go'],
        CHANGE_PERSONALITY: [r'can you be more (.*)(\?)'],
    }

    @staticmethod
    def detect_eoc(input):
        self = ConversationTrigger
        if input.lower() in self.TRIGGERS[self.END]:
            return True
        return False

    @staticmethod
    def detect_change_personality(input):
        self = ConversationTrigger
        for re_templates in self.TRIGGERS[self.CHANGE_PERSONALITY]:
            matched = re.match(re_templates, input.lower(), re.M | re.I)
            if matched:
                return matched.group(1)
        return None


class ChatLogger:
    def __init__(self, save_path):
        self.chats = []
        self.save_path = save_path
        self.user_template = 'User: %s\n'
        self.bot_template = 'Chatbot: %s\n'


    def add(self, user, bot):
        userStr = self.user_template % user
        botStr = self.bot_template % bot
        self.chats.append(userStr)
        self.chats.append(botStr)

    def save(self):
        with open(self.save_path, "w") as file:
            for line in self.chats:
                file.write(line)
            print('[INFO]: Chat log has been saved to:', (self.save_path))


def init():
    print("----------debug in init-------")
    global wv_model, seq2seq_models

    # enable this to load the model saved on Google Drive
    # disable to load model you trained using previous code
    # (model saved on your google drive root/models)
    LOAD_GDRIVE_MODEL = True

    # Please do not modify these
    ####################################################
    MODEL_GDRIVE_ID = '1F-x_ylxEV7rAKRtb7yLUX_fZzZWSw5y5'
    PATH_GDRIVE_ROOT = '/content/gdrive/My Drive'
    FILES_ROOT = '%s/models' % ('..' if LOAD_GDRIVE_MODEL else PATH_GDRIVE_ROOT)

    WORD_DICT_ROOT = '%s/word_dict' % FILES_ROOT
    WV_MODEL_ROOT = '%s/wv_model' % FILES_ROOT
    SEQ_MODEL_ROOT = '%s/seq_model' % FILES_ROOT

    global WORD_DICT_A_PATH
    WORD_DICT_A_PATH = '%s/word_dict_a' % WORD_DICT_ROOT
    WORD_DICT_Q_PATH = '%s/word_dict_q' % WORD_DICT_ROOT
    WV_MODEL_PATH = '%s/word2vect_sg' % WV_MODEL_ROOT
    SEQ_MODEL_PATH = '%s/seq2seq' % SEQ_MODEL_ROOT
    CHAT_LOG_PATH = '%s/models/chat_log.txt' % PATH_GDRIVE_ROOT
    ####################################################

    #WV
    WINDOW_SIZE=1
    VOCAB_SIZE=571
    SAMPLE_SIZE=16
    EMBEDDING_SIZE=50

    #Seq2Seq:hyper-parameters
    BATCH_SIZE=64
    NUM_EPOCH=1000
    LEARNING_RATE=1e-3
    HIDDEN_SIZE=256

    #downloadmodelsfromgoogledrivefirst

    tf.reset_default_graph()

    session=tf.Session()

    #resumepreviouslysavedworddict
    word_dict=SkipGramDatasetReader.resume_word_dict(WORD_DICT_Q_PATH)
    wv_model=SkipGramModel(session=session,word_dict=word_dict,voc_size=VOCAB_SIZE,
    embedding_size=EMBEDDING_SIZE,sample_size=SAMPLE_SIZE)
    wv_model.load_model(WV_MODEL_PATH)

    #recordedfromtraining,maxlengthofquestionforall
    #threedatasetsare12:
    MAX_LEN_Q=12
    global items
    items = {}

    #resumeworddictforanswers(onehotlookup)
    #changetheload_modeltotraintotrainthemodel
    #self,total_samples,dataset_iterator,validate_iterator,
    #num_epoch=5000,display_interval=50
    word_dict_a=Seq2SeqDatasetReader.resume_answer_dict()
    seq2seq_models=Seq2SeqModel(
    session=session,
    word_dict_a=word_dict_a,max_len_q=MAX_LEN_Q,
    embedding_size=wv_model.embedding_size,hidden_size=HIDDEN_SIZE)
    seq2seq_models.load_model(SEQ_MODEL_PATH)

def get_response(user_in):
    print("----------debug in get response-------")
    global seq2seq_models, wv_model
    response = seq2seq_models.answer(user_in, wv_model=wv_model)
    return response

# def fillForm(info):
#     br = mechanize.Browser()
#     br.open("index.html")
#     br.select_form(name="bookingInfoForm")
#     br.form['from'] = 'Enter your Name'
#     br.form['to'] = 'Enter your Title'

def createBrowserScript(items):
    ids = ['fromcity', 'tocity', 'deparure', 'return']
    id_path = os.path.join('.', os.path.dirname(__file__), 'static/js')
    jslines = []
    for k, v in items.items():
        if k in ids:
            jslines.append("document.getElementById('" + k + "').value = '" + v + "';\n")
        else:
            jslines.append("document.getElementById('" + k + "').checked = true;\n")

    file_path = os.path.join(id_path, 'BrowserScript.js')

    outF = open(file_path, 'w')
    outF.writelines(jslines)
    outF.close()
    return outF

'''------------------------------------------------------------------------------------------------------------------------------------'''

# Chatting api
@app.route("/api/response")
def response():
    message = request.args.get('message')
    message = str(message)
    bot_response = get_response(message)
    bot_response = bot_response
    matched_dates =  datefinder.find_dates(message)
    e = extraction.Extractor(text=message)
    e.find_entities()
    matched_places = e.places
    complete = False
    tag = True
    ticket_link = "http://localhost:5000/result"
    for flightDate in matched_dates:
        # Adding a new key value pair
        flightDate = flightDate.strftime("%m-%d-%y")
        if (tag):
            items.update({'deparure': flightDate})
            tag = False
        else:
            items.update({'return': flightDate})
            tag = True
    for place in matched_places:
        # Adding a new key value pair
        if (tag):
            items.update({'fromcity': place})
            tag = False
        else:
            items.update({'tocity': place})
            tag = True
    createBrowserScript(items)
    if len(items) == 4:
        complete = True
    print("huiwen debug:", items)
    info = {
       "input" : message,
       "msg" : bot_response,
       "ticket_msg": ticket_link,
       "indication": complete
       # "dates" : list(matched_dates),
       # "places" : list(matched_places)
    }
    return jsonify(info)

# Load chatting page
@app.route('/')
def index():
    return render_template('chat.html')

#, methods = ['POST', 'GET']
@app.route('/result')
def result():
    return render_template("index.html")

    
# Main entry
if __name__ == '__main__':
    init()
    print("----------debug in main entry-------")
    app.run(debug=False)
