from pydrive import drive
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import os
import re


class FileUtility:
    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def gdrive_download_file(filename, pk):
        file = drive.CreateFile({'id': pk})
        file.GetContentFile(filename)

    @classmethod
    def gdrive_download_dir(cls, path, pk):
        cls.mkdir(path)
        files = drive.ListFile({'q': "'{0}' in parents".format(pk)}).GetList()
        for f in files:
            file_id, file_path, mime_type = f["id"], '{0}}/{1}'.format(path, f["title"]), f["mimeType"]
            if mime_type == 'application/vnd.google-apps.folder':
                cls.gdrive_download_dir(file_path, file_id)
            else:
                cls.gdrive_download_file(file_path, file_id)


class PrepUtility:
    @staticmethod
    def safe_read(l, idx):
        if idx < 0 or idx > len(l) - 1:
            return []
        return [l[idx]]

    # relax common english contractions
    @staticmethod
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

    @classmethod
    def preprocess_text(cls, text):
        lemmatizer = WordNetLemmatizer()

        # decapitalize
        text = text.lower()
        # replace contractions:
        text = cls.handle_contractions(text)
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
        wv_tag_u = np.zeros((wv_model.embedding_size), dtype=np.float32)
        wv_tag_p = np.ones((wv_model.embedding_size), dtype=np.float32)

        word_vectors = []
        # replace unknown words with pre-defined U tag vector
        for token in tokens:
            vector = wv_model.word2vect(token)
            if vector is None:
                vector = wv_tag_u
            word_vectors.append(vector)

        # pad up to max length with pre-defined P tag vectors
        diff = max_len - len(word_vectors)
        [word_vectors.append(wv_tag_p) for _ in range(diff)]
        return np.array(word_vectors)

    @classmethod
    def get_answer_vector(cls, answer, lookup, convert_index=True):
        # lookup one-hot idx from word_dict
        seq_idx = cls.tokens_to_ids(answer, lookup)
        if convert_index:
            # idx to one-hot vector
            return np.eye(len(lookup), dtype=np.float32)[seq_idx]
        return np.array(seq_idx)

    @classmethod
    def sequence_to_vectors(cls, seq, max_len, wv_model, word_dict_a):
        question, answer = seq
        q = cls.get_question_vector(question, max_len, wv_model)
        a_out = cls.get_answer_vector(['_B_'] + [answer], word_dict_a)
        a_target = cls.get_answer_vector([answer] + ['_E_'], word_dict_a, convert_index=False)
        return q, a_out, a_target
