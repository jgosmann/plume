import numpy as np


class GrowingArray(object):
    def __init__(self, shape, dtype='float', expected_rows=100):
        self.rows = 0
        self._data = np.empty((expected_rows,) + shape, dtype=dtype)

    def get_data(self):
        return self._data[:self.rows]

    data = property(get_data)

    def append(self, item):
        item = np.asarray(item)
        assert self._data.shape[1:] == item.shape, 'Incompatible shape.'
        if self.rows >= len(self._data):
            self._enlarge()
        self._data[self.rows] = item
        self.rows += 1

    def _enlarge(self):
        shape = list(self._data.shape)
        shape[0] *= 2
        new_data = np.empty(shape, self._data.dtype)
        new_data[:self.rows] = self._data[:self.rows]
        self._data = new_data
