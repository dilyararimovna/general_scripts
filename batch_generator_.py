import numpy as np
from fasttext_embeddings import text2embeddings

class BatchIterator(object):

    def __init__(self, data, labels, batch_size, feature_size, shuffle=False):
        self._data = data
        self._labels = labels
        self._num_samples = len(data)
        self._batch_size = batch_size
        self._feature_size = feature_size
        self._shuffle = shuffle
        self._current_permutation = None
        self._epochs_done = 0

        self._cursor = 0
        if self._shuffle:
            self._current_permutation = np.random.permutation(self._num_samples)
        else:
            self._current_permutation = np.arange(self._num_samples)

    def next_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._feature_size), dtype=np.float)
        if self._cursor + self._batch_size >= self._num_samples:
            batch = self._data[self._current_permutation[self._cursor:]]
            batch_labels = self._labels[self._current_permutation[self._cursor:]]

            self._cursor = 0
            self._epochs_done += 1
            if self._shuffle:
            	self._current_permutation = np.random.permutation(self._num_samples)
            else:
            	self._current_permutation = np.arange(self._num_samples)
            print("___%d epochs done___" % self._epochs_done)
        else:
            batch = self._data[self._current_permutation[self._cursor:self._cursor+self._batch_size]]
            batch_labels = self._labels[self._current_permutation[self._cursor:self._cursor+self._batch_size]]

            self._cursor += self._batch_size

        return batch, batch_labels
