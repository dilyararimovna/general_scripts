import numpy as np
from fasttext_embeddings import text2embeddings

class BatchIteratorFromFasttext(object):

    def __init__(self, data, labels, batch_size, text_size, embedding_size, embedding_model, shuffle=False):
        self._data = data
        self._labels = labels
        self._num_samples = len(data)
        self._batch_size = batch_size
        self._text_size = text_size
        self._embedding_size = embedding_size
        self._embedding_model = embedding_model
        self._shuffle = shuffle
        self._current_permutation = None
        self._epochs_done = 0

        self._cursor = 0
        if self._shuffle:
            self._current_permutation = np.random.permutation(self._num_samples)
        else:
            self._current_permutation = np.arange(self._num_samples)

    def next_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._embedding_size), dtype=np.float)
        if self._cursor + self._batch_size >= self._num_samples:
            text_batch = self._data[self._current_permutation[self._cursor:]]
            batch = text2embeddings(text_batch, self._embedding_model, self._text_size, self._embedding_size)
            batch_labels = self._labels[self._current_permutation[self._cursor:]]

            self._cursor = 0
            self._epochs_done += 1
            print("___%d epochs done___" % self._epochs_done)
        else:
            text_batch = self._data[self._current_permutation[self._cursor:self._cursor+self._batch_size]]

            batch = text2embeddings(text_batch, self._embedding_model, self._text_size, self._embedding_size)
            batch_labels = self._labels[self._current_permutation[self._cursor:self._cursor+self._batch_size]]

            self._cursor += self._batch_size

        return batch, batch_labels
