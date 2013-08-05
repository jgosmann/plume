from hamcrest import assert_that, is_
from nose.tools import raises
from numpy.testing import assert_equal
import numpy as np

from plume.datastructure import EnlargeableArray


class TestEnlargableArray(object):
    def test_allows_to_append_elements(self):
        a = EnlargeableArray((2, 2), 2)
        elem = np.array([[1, 2], [3, 4]])
        a.append(elem)
        a.append(elem)
        assert_equal(a.data, np.array([elem, elem]))

    def test_has_length(self):
        a = EnlargeableArray((2,), 4)
        elem = np.array([1, 2])
        a.append(elem)
        a.append(elem)
        assert_that(len(a), is_(2))

    @raises(IndexError)
    def test_raises_error_if_index_out_of_bounds(self):
        a = EnlargeableArray((2,), 4)
        elem = np.array([1, 2])
        a.append(elem)
        a.data[2]

    def test_enlarges_itself_as_needed(self):
        a = EnlargeableArray((2, 2), 2)
        elem = np.array([[1, 2], [3, 4]])
        for i in xrange(4):
            a.append(elem)
        assert_equal(a.data, np.array(4 * [elem]))
