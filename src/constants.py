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

WORD_DICT_A_PATH = '%s/word_dict_a' % WORD_DICT_ROOT
WORD_DICT_Q_PATH = '%s/word_dict_q' % WORD_DICT_ROOT
WV_MODEL_PATH = '%s/word2vect_sg' % WV_MODEL_ROOT
SEQ_MODEL_PATH = '%s/seq2seq' % SEQ_MODEL_ROOT
CHAT_LOG_PATH = '%s/models/chat_log.txt' % PATH_GDRIVE_ROOT
WINDOW_SIZE = 1
VOCAB_SIZE = 571
SAMPLE_SIZE = 16
EMBEDDING_SIZE = 50

# Seq2Seq:hyper-parameters
BATCH_SIZE = 64
NUM_EPOCH = 1000
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256


MAX_LEN_Q = 12