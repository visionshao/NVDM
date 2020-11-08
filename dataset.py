from torch.utils.data import Dataset
import numpy as np
import torch


class FeatDataset(Dataset):

    def __init__(self, file_path, vocab_size):
        data, word_count = self.data_set(file_path)
        transformed_docs = self.transform(docs=data, vocab_size=vocab_size)
        self.data = transformed_docs
        self.word_count = word_count

    def __getitem__(self, item):
        return self.data[item], self.word_count[item]

    def __len__(self):
        return len(self.data)

    def data_set(self, file_path):
        """process data input."""
        data = []
        word_count = []
        fin = open(file_path)
        while True:
            line = fin.readline()
            if not line:
                break
            id_freqs = line.split()
            doc = {}
            count = 0
            for id_freq in id_freqs[1:]:
                items = id_freq.split(':')
                # python starts from 0
                doc[int(items[0]) - 1] = int(items[1])
                count += int(items[1])
            if count > 0:
                data.append(doc)
                word_count.append(count)
        fin.close()
        return data, word_count

    def transform(self, docs, vocab_size):
        """transform data to bag-of-words"""
        transformed_docs = []
        for doc in docs:
            bow_doc = np.zeros(vocab_size)
            for word_id, freq in doc.items():
                bow_doc[word_id] = freq
            transformed_docs.append(bow_doc)

        return transformed_docs




