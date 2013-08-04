import numpy as np
from numpy.testing import assert_almost_equal

import prediction


class TestOnlineGP(object):
    def setUp(self):
        self.gp = prediction.OnlineGP(prediction.RBFKernel(1.0), noise_var=0.5)

    def test_can_predict(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = [-2, 0, 1, 2, -1]
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = [-0.78511166, 0.37396387]
        pred = self.gp.predict(x_star)
        assert_almost_equal(pred, expected)

    def test_evaluates_mse(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = [-2, 0, 1, 2, -1]
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = [1.04585738, 1.04888027]
        unused, mse = self.gp.predict(x_star, eval_mse=True)
        assert_almost_equal(mse, expected)

    def test_allows_adding_new_datapoints_online(self):
        xs = [-4, -2, -0.5, 0, 2]
        ys = [-2, 0, 1, 2, -1]

        for x, y, in zip(xs, ys):
            self.gp.add_observations(np.array([[x]]), np.array([y]))

        x_star = np.array([[-3, 1]]).T
        expected = [-0.78511166, 0.37396387]
        expected_mse = [1.04585738, 1.04888027]
        pred, mse = self.gp.predict(x_star, eval_mse=True)
        assert_almost_equal(pred, expected)
        assert_almost_equal(mse, expected_mse)
