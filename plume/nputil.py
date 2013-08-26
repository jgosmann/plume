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

    def extend(self, items):
        items = np.asarray(items)
        assert self._data.shape[1:] == items.shape[1:], 'Incompatible shape.'

        space_left = self._data.shape[0] - self.rows
        if space_left < len(items):
            self._enlarge(len(items) - space_left)
        self._data[self.rows:(self.rows + len(items))] = items
        self.rows += len(items)

    def _enlarge(self, by_at_least=1):
        shape = list(self._data.shape)
        enlarge_by = shape[0]
        while enlarge_by < by_at_least:
            enlarge_by *= 2
        shape[0] += enlarge_by
        new_data = np.empty(shape, self._data.dtype)
        new_data[:self.rows] = self._data[:self.rows]
        self._data = new_data


class Growing2dArray(object):
    def __init__(self, dtype='float', expected_rows=100):
        self._data = np.empty((expected_rows, expected_rows), dtype=dtype)
        self.rows = 0
        pass

    def get_data(self):
        return self._data[:self.rows, :self.rows]

    data = property(get_data)
    cols = property(lambda self: self.rows)

    def enlarge_by(self, by):
        final_size = self.rows + by
        if self._data.shape[0] < final_size:
            new_data = np.empty((final_size, final_size))
            new_data[:self.rows, :self.cols] = self.data
            self._data = new_data
        self.rows = final_size


def meshgrid_nd(*args):
    args = [np.asarray(a) for a in args]
    s = len(args) * (1,)
    return np.broadcast_arrays(*(
        a.reshape(s[:i] + (-1,) + s[i + 1:]) for i, a in enumerate(args)))
