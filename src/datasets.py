from src.utilities import FileUtility, PrepUtility
import pandas as pd
import os


class AirDialogueDataset:

    DATASET_ID_AIRDIALOGUE = '1yGaZzxQ6Nz4wrBd5Eo0wuzoYZEBlhWED'
    DATASET_FILENAME_AIRDIALOGUE = 'airdialogue.tsv'

    @classmethod
    def download(cls):
        if not os.path.exists('./' + cls.DATASET_FILENAME_AIRDIALOGUE):
            FileUtility.gdrive_download_file(cls.DATASET_FILENAME_AIRDIALOGUE, cls.DATASET_ID_AIRDIALOGUE)
        # if not os.path.exists('./' + self.DATASET_FILENAME_COMIC):
        #     FileUtility.gdrive_download_file(self.DATASET_FILENAME_COMIC, self.DATASET_ID_COMIC)
        # if not os.path.exists('./' + self.DATASET_FILENAME_PROFESSIONAL):
        #     FileUtility.gdrive_download_file(self.DATASET_FILENAME_PROFESSIONAL, self.DATASET_ID_PROFESSIONAL)
        return cls.read_dataset()

    @classmethod
    def read_dataset(cls):
        return pd.read_csv(cls.DATASET_FILENAME_AIRDIALOGUE, sep="\t")

    @staticmethod
    def preprocess(datasets):
        qa_sequences = list()
        corpusQ, corpusA = set(), set()
        for index, row in datasets.iterrows():
            question, answer = row[0], row[1]
            q_tokens = PrepUtility.preprocess_text(question)
            corpusQ.update(q_tokens)
            corpusA.add(answer)
            qa_sequences.append((q_tokens, answer))
        datasets = (qa_sequences, list(corpusA))
        return datasets
