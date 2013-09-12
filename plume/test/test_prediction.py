from hamcrest import assert_that, equal_to, is_
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import plume.prediction
from plume.prediction import ExponentialKernel, RBFKernel


class TestRBFKernel(object):
    def test_kernel(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [0.00072298179430063214, 6.9693689985354893e-07],
            [0.00289944010460460330, 2.7949898790590032e-06]])
        actual = RBFKernel(lengthscale=0.6, variance=0.75)(x1, x2)
        assert_almost_equal(actual, expected)

    def test_kernel_derivative(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.0, 0.00200828, 0.00401657],
             [5.80780750e-06, 1.93593583e-06, 0.0]],
            [[0.0, 0.0, 0.016108],
             [2.32915823e-05, 0.00000000e+00, 0.00000000e+00]]])
        unused, actual = RBFKernel(lengthscale=0.6, variance=0.75)(
            x1, x2, eval_derivative=True)
        assert_almost_equal(actual, expected)

    def test_diag_symmetric(self):
        x = np.array([[1, 1, 1], [1, 2, 1]])
        expected = 0.75 * np.ones(2)
        actual = RBFKernel(lengthscale=0.6, variance=0.75).diag(x, x)
        assert_almost_equal(actual, expected)

    def test_diag(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.00072298179430063214, 2.7949898790590032e-06])
        actual = RBFKernel(lengthscale=0.6, variance=0.75).diag(x1, x2)
        assert_almost_equal(actual, expected)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([7.22981794e-04, 2.79498988e-06])
        actual = RBFKernel(lengthscale=0.6, variance=0.75).diag(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = RBFKernel(lengthscale=0.6, variance=0.75)
        assert_equal(kernel.get_params(), np.array([0.6, 0.75]))
        assert_equal(kernel.params, np.array([0.6, 0.75]))

    def test_can_set_params_as_array(self):
        kernel = RBFKernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 0.5]))
        assert_that(kernel.lengthscale, is_(equal_to(1.2)))
        assert_that(kernel.variance, is_(equal_to(0.5)))


class TestExponentialKernel(object):
    def test_kernel(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [0.018052663641012889, 0.0038559231230571806],
            [0.026755495010439296, 0.0050534602493140998]])
        actual = ExponentialKernel(lengthscale=0.6, variance=0.75)(x1, x2)
        assert_almost_equal(actual, expected)

    def test_kernel_derivative(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.0, 0.02691132, 0.05382264],
             [0.0121935, 0.0040645, 0.0]],
            [[0.0, 0.0, 0.089184983368130993],
             [0.016844867497713668, 0.0, 0.0]]])
        unused, actual = ExponentialKernel(lengthscale=0.6, variance=0.75)(
            x1, x2, eval_derivative=True)
        assert_almost_equal(actual, expected)

    def test_diag_symmetric(self):
        x = np.array([[1, 1, 1], [1, 2, 1]])
        expected = 0.75 * np.ones(2)
        actual = ExponentialKernel(lengthscale=0.6, variance=0.75).diag(x, x)
        assert_almost_equal(actual, expected)

    def test_diag(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.018052663641012889, 0.0050534602493140998])
        actual = ExponentialKernel(lengthscale=0.6, variance=0.75).diag(x1, x2)
        assert_almost_equal(actual, expected)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.01805266, 0.00505346])
        actual = ExponentialKernel(lengthscale=0.6, variance=0.75).diag(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = ExponentialKernel(lengthscale=0.6, variance=0.75)
        assert_equal(kernel.get_params(), np.array([0.6, 0.75]))
        assert_equal(kernel.params, np.array([0.6, 0.75]))

    def test_can_set_params_as_array(self):
        kernel = ExponentialKernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 0.5]))
        assert_that(kernel.lengthscale, is_(equal_to(1.2)))
        assert_that(kernel.variance, is_(equal_to(0.5)))


class TestOnlineGP(object):
    def setUp(self):
        self.gp = plume.prediction.OnlineGP(
            plume.prediction.RBFKernel(1.0), noise_var=0.5)

    def test_can_predict(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = np.array([[-0.78511166, 0.37396387]]).T
        pred = self.gp.predict(x_star)
        assert_almost_equal(pred, expected)

    def test_evaluates_mse(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = [1.04585738, 1.04888027]
        unused, mse = self.gp.predict(x_star, eval_MSE=True)
        assert_almost_equal(mse, expected)

    def test_allows_adding_new_datapoints_online(self):
        xs = [-4, -2, -0.5, 0, 2]
        ys = [-2, 0, 1, 2, -1]

        for x, y, in zip(xs, ys):
            self.gp.add_observations(np.array([[x]]), np.array([[y]]))

        x_star = np.array([[-3, 1]]).T
        expected = np.array([[-0.78511166, 0.37396387]]).T
        expected_mse = [1.04585738, 1.04888027]
        pred, mse = self.gp.predict(x_star, eval_MSE=True)
        assert_almost_equal(pred, expected)
        assert_almost_equal(mse, expected_mse)

    def test_has_trained_indicator(self):
        assert_that(self.gp.trained, is_(False))
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)
        assert_that(self.gp.trained, is_(True))

    def test_can_calculate_prediction_derivative(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = np.array([[[0.85538797]], [[-1.30833924]]])
        unused, actual = self.gp.predict(x_star, eval_derivatives=True)
        assert_almost_equal(actual, expected)

    def test_can_calculate_mse_derivative(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = np.array([[-0.00352932], [-0.00173095]])
        unused, unused, unused, actual = self.gp.predict(
            x_star, eval_MSE=True, eval_derivatives=True)
        assert_almost_equal(actual, expected)

    def test_can_calculate_neg_log_likelihood(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)
        actual = self.gp.calc_neg_log_likelihood()
        expected = (8.51911832, np.array([0.76088728, -0.49230927]))
        assert_almost_equal(actual[0], expected[0])
        assert_almost_equal(actual[1], expected[1])


class TestLikelihoodGP(object):
    def setUp(self):
        self.gp = plume.prediction.LikelihoodGP(
            plume.prediction.RBFKernel(1.0), noise_var=0.5)
        self.gp.priors[0] = plume.prediction.GaussianLogPrior(0.5, 1.0)
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

    def test_can_predict(self):
        x_test = np.array([[-3, -1, 1, 4]]).T
        expected = np.array(
            [[-0.50845387, 0.46036866, 0.26117605, -0.01261526]]).T
        actual = self.gp.predict(x_test)
        assert_almost_equal(actual, expected)

    def test_can_calculate_neg_log_likelihood(self):
        expected = (9.3495487653878691,
                    np.array([1.50581184e-05, 9.73710450e-06]))
        actual = self.gp.calc_neg_log_likelihood()
        assert_almost_equal(expected[0], actual[0])
        assert_almost_equal(expected[1], actual[1])

    def test_optimizes_kernel_params(self):
        expected = np.array([0.7032695, 1.17616655])
        actual = self.gp.kernel.params
        assert_almost_equal(expected, actual)
