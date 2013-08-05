import numpy as np
from numpy.testing import assert_equal

from plume.npwrap import GrowingArray


class TestGrowingArray(object):
    def test_can_initialize_empty_array(self):
        a = GrowingArray((2, 2), dtype='int')
        expected = np.empty((0, 2, 2))
        assert_equal(a.data, expected)
