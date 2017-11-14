import numpy as np
from fasttext_embeddings import text2embeddings

class BatchIteratorFromFasttext(object):

    def __init__(self, data, batch_size, text_size, embedding_size, embedding_model, shuffle=False):
        self._data = data
        self._num_samples = len(data)
        self._batch_size = batch_size
        self._text_size = text_size
        self._embedding_size = embedding_size
        self._embedding_model = embedding_model
        self._shuffle = shuffle
        self._current_permutation = None

        self._cursor = 0
        if self._shuffle:
            self._current_permutation = np.random.permutation(self._num_samples)
        else:
            self._current_permutation = np.arange(self._num_samples)

    def next_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._embedding_size), dtype=np.float)
        if self._cursor + self._batch_size >= self._num_samples:
            self._cursor = self._cursor % self._num_samples

        text_batch = self._current_permutation[self._cursor:self._cursor+self._batch_size]
        self._cursor += self._batch_size

        batch = text2embeddings(text_batch, self._embedding_model, self._text_size, self._embedding_size)
        return batch

