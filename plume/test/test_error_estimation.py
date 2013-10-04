import numpy as np
from numpy.testing import assert_almost_equal

from plume.error_estimation import vegas


class TestVegas(object):
    @staticmethod
    def gauss(*args):
        xs = args[:-1]
        scale = args[-1]
        d = len(xs)
        return (1.0 / scale / np.sqrt(np.pi)) ** d * np.exp(
            -np.sum(((x - 0.5) ** 2 for x in xs), axis=0) / scale / scale)

    def test_can_approx_multidimensional_gaussian_area(self):
        integral, sigma = vegas(
            self.gauss, [0, 0, 0, 0], [1, 1, 1, 1], args=(0.1,))
        assert_almost_equal(integral, 1.0, 1)
        assert_almost_equal(sigma, 0.01, 2)
