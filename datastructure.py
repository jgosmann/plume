import numpy as np


class EnlargeableArray(object):
    def __init__(self, shape, expected_rows=20):
        self._data = np.empty((expected_rows,) + shape)
        self._num_rows = 0

    data = property(lambda self: self._data[:len(self)])

    def append(self, arr):
        if len(self) >= len(self._data):
            self._data.resize((2 * len(self),) + self._data.shape[1:])

        self._data[self._num_rows] = arr
        self._num_rows += 1

    def __len__(self):
        return self._num_rows
