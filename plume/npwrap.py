import numpy as np


class GrowingArray(object):
    def __init__(self, shape, dtype='float', expected_rows=100):
        self.rows = 0
        self._data = np.empty((expected_rows,) + shape, dtype=dtype)

    def get_data(self):
        return self._data[:self.rows]

    data = property(get_data)
