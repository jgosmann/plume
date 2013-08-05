import numpy as np
from numpy.testing import assert_equal

from plume.npwrap import GrowingArray


class TestGrowingArray(object):
    def test_can_initialize_empty_array(self):
        a = GrowingArray((2, 2), dtype='int')
        expected = np.empty((0, 2, 2))
        assert_equal(a.data, expected)

    def test_can_append_data(self):
        a = GrowingArray((2, 2), dtype='int')
        a.append(np.array([[1, 2], [3, 4]]))
        a.append(np.array([[5, 6], [7, 8]]))
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert_equal(a.data, expected)

    def test_enlarges_buffer_as_needed(self):
        a = GrowingArray((2, 2), dtype='int', expected_rows=1)
        a.append(np.array([[1, 2], [3, 4]]))
        a.append(np.array([[5, 6], [7, 8]]))
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert_equal(a.data, expected)

    # check enlarging
    # check appending wrong shape
