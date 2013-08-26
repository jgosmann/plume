from nose.tools import raises
import numpy as np
from numpy.testing import assert_equal

from plume.nputil import GrowingArray, Growing2dArray, meshgrid_nd


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

    def test_append_enlarges_buffer_as_needed(self):
        a = GrowingArray((2, 2), dtype='int', expected_rows=1)
        a.append(np.array([[1, 2], [3, 4]]))
        a.append(np.array([[5, 6], [7, 8]]))
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert_equal(a.data, expected)

    # FIXME use ValueError if possible, but try to keep assertion
    @raises(AssertionError)
    def test_raises_exception_when_appending_wrong_shape(self):
        a = GrowingArray((2, 2), dtype='int')
        a.append(np.array([1, 2]))

    def test_can_extend_data(self):
        a = GrowingArray((2, 2), dtype='int')
        a.extend(np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]]))
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert_equal(a.data, expected)

    def test_extend_enlarges_buffer_as_needed(self):
        a = GrowingArray((2, 2), dtype='int', expected_rows=1)
        a.extend(np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]]))
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert_equal(a.data, expected)

    # FIXME use ValueError if possible, but try to keep assertion
    @raises(AssertionError)
    def test_raises_exception_when_extending_wrong_shape(self):
        a = GrowingArray((2, 2), dtype='int')
        a.extend(np.array([1, 2]))


class TestGrowing2dArray(object):
    def test_can_initialize_empty_array(self):
        a = Growing2dArray(dtype='int')
        expected = np.empty((0, 0))
        assert_equal(a.data, expected)

    def test_can_enlarge(self):
        a = Growing2dArray(dtype='int', expected_rows=1)
        a.enlarge_by(2)
        a.enlarge_by(1)
        assert_equal(a.data.shape, (3, 3))

    def test_enlarging_keeps_existing_data(self):
        test_data = np.array([[1, 2], [3, 4]])
        a = Growing2dArray(dtype='int', expected_rows=1)
        a.enlarge_by(2)
        a.data[:] = test_data
        a.enlarge_by(1)
        assert_equal(a.data[:2, :2], test_data)


class TestMeshgridND(object):
    def test_creates_3d_meshgrid(self):
        a = [0, 0, 1]
        b = [1, 2, 3]
        c = [23, 42]
        expected = [
            np.array([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [0, 0]],
                      [[1, 1], [1, 1], [1, 1]]]),
            np.array([[[1, 1], [2, 2], [3, 3]],
                      [[1, 1], [2, 2], [3, 3]],
                      [[1, 1], [2, 2], [3, 3]]]),
            np.array([[[23, 42], [23, 42], [23, 42]],
                      [[23, 42], [23, 42], [23, 42]],
                      [[23, 42], [23, 42], [23, 42]]])]
        actual = meshgrid_nd(a, b, c)
        assert_equal(expected, actual)
