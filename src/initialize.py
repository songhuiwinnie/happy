from src.readers import SkipGramDatasetReader, Seq2SeqDatasetReader
from src.models import SkipGramModel, Seq2SeqModel
from src.constants import *
import tensorflow as tf


def initialize():

    # download models from google drive first
    tf.reset_default_graph()
    session = tf.Session()

    # resume previously saved word dict
    word_dict = SkipGramDatasetReader.resume_word_dict(WORD_DICT_Q_PATH)
    wv_model = SkipGramModel(session=session, word_dict=word_dict, voc_size=VOCAB_SIZE, embedding_size=EMBEDDING_SIZE,
                             sample_size=SAMPLE_SIZE)
    wv_model.load_model(WV_MODEL_PATH)

    # recorded from training, max length of question for all
    # resume word dict for answers (one hot lookup)
    # change the load_model to train the model
    # self, total_samples, dataset_iterator, validate_iterator,
    # num_epoch=5000, display_interval=50
    word_dict_a = Seq2SeqDatasetReader.resume_answer_dict()
    seq2seq_models = Seq2SeqModel(session=session, word_dict_a=word_dict_a,max_len_q=MAX_LEN_Q,
                                  embedding_size=wv_model.embedding_size, hidden_size=HIDDEN_SIZE)
    seq2seq_models.load_model(SEQ_MODEL_PATH)

    return seq2seq_models, wv_model
