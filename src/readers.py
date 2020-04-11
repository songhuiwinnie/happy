from src.datasets import AirDialogueDataset
from src.utilities import PrepUtility
from src.constants import WORD_DICT_A_PATH
import tensorflow as tf
import numpy as np
import pickle
import os


class SkipGramDatasetReader(AirDialogueDataset):

    def __init__(self, word_dict_path, window_size=1):
        self.word_dict_path = word_dict_path

        # download datasets
        datasets = super().download()

        # ing datasets
        sequences, total = [], []
        for index, row in datasets.iterrows():
            processed_q = PrepUtility.preprocess_text(row[0])
            processed_a = PrepUtility.preprocess_text(row[1])
            sequences.extend([processed_q, processed_a])
            total.extend([processed_q, processed_a])
        self.bow = []
        [self.bow.append(i) for i in total if not self.bow.count(i)]

        # generate word dict for question
        self.word_dict = {w: i for i, w in enumerate(self.bow)}
        with open(self.word_dict_path, 'wb') as file:
            # print("[INFO]: Saving new word dictionary record (one-hot) for question tokens")
            pickle.dump(self.word_dict, file, pickle.HIGHEST_PROTOCOL)
        # print("[INFO]: Total word counts: {}".format(len(self.word_dict)))

        self.skip_grams = []
        for i, seq in enumerate(sequences):
            for j, word in enumerate(seq):
                context = []
                for ws in range(window_size):
                    context.extend(PrepUtility.safe_read(seq, j - ws - 1))
                    context.extend(PrepUtility.safe_read(seq, j + ws + 1))
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
        assert os.path.isfile(path), True
        with open(path, 'rb') as file:
            # print("[INFO]: Resumed from previous word dictionary record (one-hot) for question tokens")
            word_dict = pickle.load(file)
        return word_dict


class Seq2SeqDatasetReader(AirDialogueDataset):

    def __init__(self, word_dict_a_path, wv_model):
        self.wv_model = wv_model
        self.word_dict_a_path = word_dict_a_path

        # section 1: preprocessing: word preprocessing
        datasets = super().download()
        datasets = super().preprocess(datasets)

        # preprocessing: words to vectors and build answer lookup dict
        qa_seqs, unique_answers = datasets

        # append auxiliary tags for begin and end
        unique_answers.extend(["_B_", "_E_", "_U_"])

        # generate answer word dict
        word_dict_path = word_dict_a_path
        self.word_dict_a = {n: i for i, n in enumerate(unique_answers)}
        with open(word_dict_path, 'wb') as file:
            print("[INFO]: Saving new word dictionary record (one-hot) for answers")
            pickle.dump(self.word_dict_a, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO]: Total answers counts: {}".format(len(self.word_dict_a)))

        self.num_seq = len(qa_seqs)
        self.max_len_q = max([len(q) for (q, a) in qa_seqs])
        input_batch, output_batch, target_batch = [], [], []
        for seq in qa_seqs:
            q_in, a_out, a_target = PrepUtility.sequence_to_vectors(seq, self.max_len_q, self.wv_model, self.word_dict_a)
            input_batch.append(q_in)
            output_batch.append(a_out)
            target_batch.append(a_target)
        self.datasets = (input_batch, output_batch, target_batch)
        print("[INFO] Max question length:", self.max_len_q)

    def prepare(self, num_epoch=1000, batch_size=64, shuffle=False):
        dataset = self.datasets
        batch_in = np.array(dataset[0])
        batch_out = np.array(dataset[1])
        batch_target = np.array(dataset[2])
        dataset = tf.data.Dataset.from_tensor_slices((batch_in, batch_out, batch_target)).batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat(num_epoch)
        iterators = dataset.make_initializable_iterator()
        return iterators

    @staticmethod
    def resume_answer_dict():
        assert os.path.isfile(WORD_DICT_A_PATH), True
        with open(WORD_DICT_A_PATH, 'rb') as file:
            print("[INFO]: Resumed from previous word dictionary record (one-hot) for answers")
            word_dict_a = pickle.load(file)
        return word_dict_a
