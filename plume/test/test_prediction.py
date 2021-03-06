from hamcrest import assert_that, equal_to, is_
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import plume.prediction
from plume.prediction import AnisotropicExponentialKernel, ExponentialKernel, \
    Matern32Kernel, Matern52Kernel, RBFKernel


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
        actual = self._create_kernel(lengthscale=0.6, variance=0.75)(x1, x2)
        assert_almost_equal(actual, expected)

    def test_kernel_derivative(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.0, 0.01345566, 0.02691132],
             [0.00609675, 0.00203225, -0.0]],
            [[0.0, 0.0, 0.0445924916840655],
             [0.008422433748856834, 0.0, 0.0]]])
        unused, actual = self._create_kernel(lengthscale=0.6, variance=0.75)(
            x1, x2, eval_derivative=True)
        assert_almost_equal(actual, expected)

    def test_diag_symmetric(self):
        x = np.array([[1, 1, 1], [1, 2, 1]])
        expected = 0.75 * np.ones(2)
        actual = self._create_kernel(lengthscale=0.6, variance=0.75).diag(x, x)
        assert_almost_equal(actual, expected)

    def test_diag(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.018052663641012889, 0.0050534602493140998])
        actual = self._create_kernel(lengthscale=0.6, variance=0.75).diag(
            x1, x2)
        assert_almost_equal(actual, expected)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.11213051,  0.03387083],
             [0.14864164,  0.04211217]],
            [[0.02407022,  0.00514123],
             [0.03567399,  0.00673795]]])
        actual = self._create_kernel(
            lengthscale=0.6, variance=0.75).param_derivatives(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = self._create_kernel(lengthscale=0.6, variance=0.75)
        assert_equal(kernel.get_params(), np.array([0.6, 0.75]))
        assert_equal(kernel.params, np.array([0.6, 0.75]))

    def test_can_set_params_as_array(self):
        kernel = self._create_kernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 0.5]))
        assert_that(kernel.lengthscale, is_(equal_to(1.2)))
        assert_that(kernel.variance, is_(equal_to(0.5)))

    def _create_kernel(self, lengthscale, variance):
        return ExponentialKernel(lengthscale=lengthscale, variance=variance)


class TestMatern32Kernel(object):
    def test_kernel(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [0.00879323, 0.00082427],
            [0.01579335, 0.00125588]])
        actual = Matern32Kernel(lengthscale=0.6, variance=0.75)(x1, x2)
        assert_almost_equal(actual, expected)

    def test_kernel_derivative(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.0, 0.00982927, 0.01965854],
             [0.00203448, 0.00067816, 0.0]],
            [[0.0, 0.0, 0.03886061],
             [0.00325013, 0.0, 0.0]]])
        unused, actual = Matern32Kernel(lengthscale=0.6, variance=0.75)(
            x1, x2, eval_derivative=True)
        assert_almost_equal(actual, expected)

    def test_diag_symmetric(self):
        x = np.array([[1, 1, 1], [1, 2, 1]])
        expected = 0.75 * np.ones(2)
        actual = Matern32Kernel(lengthscale=0.6, variance=0.75).diag(x, x)
        assert_almost_equal(actual, expected)

    def test_diag(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.00879323, 0.00125588])
        actual = Matern32Kernel(lengthscale=0.6, variance=0.75).diag(
            x1, x2)
        assert_almost_equal(actual, expected)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.08191057, 0.01130266],
             [0.12953538, 0.01625065]],
            [[0.01172431, 0.00109902],
             [0.0210578, 0.00167451]]])
        actual = Matern32Kernel(
            lengthscale=0.6, variance=0.75).param_derivatives(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = Matern32Kernel(lengthscale=0.6, variance=0.75)
        assert_equal(kernel.get_params(), np.array([0.6, 0.75]))
        assert_equal(kernel.params, np.array([0.6, 0.75]))

    def test_can_set_params_as_array(self):
        kernel = Matern32Kernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 0.5]))
        assert_that(kernel.lengthscale, is_(equal_to(1.2)))
        assert_that(kernel.variance, is_(equal_to(0.5)))


class TestMatern52Kernel(object):
    def test_kernel(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [0.00585567, 0.00033752],
            [0.01172022, 0.0005632]])
        actual = Matern52Kernel(lengthscale=0.6, variance=0.75)(x1, x2)
        assert_almost_equal(actual, expected)

    def test_kernel_derivative(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.0, 0.00778975, 0.0155795],
             [0.00101443, 0.00033814, 0.0]],
            [[0.0, 0.0, 0.03401239],
             [0.00176941, 0.0, 0.0]]])
        unused, actual = Matern52Kernel(lengthscale=0.6, variance=0.75)(
            x1, x2, eval_derivative=True)
        assert_almost_equal(actual, expected)

    def test_diag_symmetric(self):
        x = np.array([[1, 1, 1], [1, 2, 1]])
        expected = 0.75 * np.ones(2)
        actual = Matern52Kernel(lengthscale=0.6, variance=0.75).diag(x, x)
        assert_almost_equal(actual, expected)

    def test_diag(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([0.00585567, 0.0005632])
        actual = Matern52Kernel(lengthscale=0.6, variance=0.75).diag(
            x1, x2)
        assert_almost_equal(actual, expected)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            [[0.25815405, 0.04569556],
             [0.37882633, 0.06316224]],
            [[0.03356713, 0.00411306],
             [0.05560057, 0.00600802]]])
        actual = Matern52Kernel(
            lengthscale=0.6, variance=0.75).param_derivatives(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = Matern32Kernel(lengthscale=0.6, variance=0.75)
        assert_equal(kernel.get_params(), np.array([0.6, 0.75]))
        assert_equal(kernel.params, np.array([0.6, 0.75]))

    def test_can_set_params_as_array(self):
        kernel = Matern32Kernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 0.5]))
        assert_that(kernel.lengthscale, is_(equal_to(1.2)))
        assert_that(kernel.variance, is_(equal_to(0.5)))


class TestAnisotropicExponentialKernel(TestExponentialKernel):
    def _create_kernel(self, lengthscale, variance):
        projection_L = np.diag(np.array(3 * [np.sqrt(lengthscale)]))
        return AnisotropicExponentialKernel(projection_L, variance)

    def test_param_derivatives(self):
        x1 = np.array([[1, 1, 1], [1, 2, 1]])
        x2 = np.array([[1, 2, 3], [4, 2, 1]])
        expected = np.array([
            # lengthscale parameters
            [[0.0, 0.04722522],
             [0.0, 0.06523989]],
            [[0.0, 0.01574174],
             [0.0, 0.0]],
            [[0.03474237, 0.00524725],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.06948473, 0.0],
             [0.0, 0.0]],
            [[0.13896947, 0.0],
             [0.23027464, 0.0]],
            # variance
            [[0.02407022, 0.00514123],
             [0.03567399, 0.00673795]]])
        actual = self._create_kernel(
            lengthscale=0.6, variance=0.75).param_derivatives(x1, x2)
        assert_almost_equal(actual, expected)

    def test_can_get_params_as_array(self):
        kernel = self._create_kernel(lengthscale=0.6, variance=0.75)
        l = np.sqrt(0.6)
        expected = np.array([l, 0.0, l, 0.0, 0.0, l, 0.75])
        assert_equal(kernel.get_params(), expected)
        assert_equal(kernel.params, expected)

    def test_can_set_params_as_array(self):
        kernel = self._create_kernel(lengthscale=0.6, variance=0.75)
        kernel.set_params(np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5]))
        assert_equal(
            kernel.lengthscales, np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7]))
        assert_that(kernel.variance, is_(equal_to(0.5)))


class TestSparseGP(object):
    def setUp(self):
        self.gp = plume.prediction.SparseGP(
            plume.prediction.RBFKernel(1.0), tolerance=0, noise_var=0.5)

    def test_can_predict(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)

        x_star = np.array([[-3, 1]]).T
        expected = np.array([[-0.78511166, 0.37396387]]).T
        pred = self.gp.predict(x_star)
        assert_almost_equal(pred, expected)

    def test_uses_reduced_updates_when_tolerated(self):
        self.gp.tolerance = 0.5
        xs = [-4, -2.1, 0, 2, -2]
        ys = [-2, -0.01, -2, -16, 0]

        for x, y, in zip(xs, ys):
            self.gp.add_observations(np.array([[x]]), np.array([[y]]))

        test_x = np.array([[-3, 1]]).T
        expected = np.array([[-0.70996282], [-6.68149458]])
        actual = self.gp.predict(test_x)

        assert_that(self.gp.num_bv, is_(equal_to(4)))
        assert_almost_equal(expected, actual)

    def test_deletes_basis_vector_if_max_bv_exceeded(self):
        self.gp.max_bv = 4
        xs = [-4, -2, -2.1, 0, 2]
        ys = [-2, 0, -0.01, -2, -16]

        for x, y, in zip(xs, ys):
            self.gp.add_observations(np.array([[x]]), np.array([[y]]))

        test_x = np.array([[-3, 1]]).T
        expected = np.array([[-0.70996282], [-6.68149458]])
        actual = self.gp.predict(test_x)

        assert_that(self.gp.num_bv, is_(equal_to(4)))
        assert_almost_equal(expected, actual)

    def test_training_will_keep_at_most_max_bv_basis_vectors(self):
        self.gp.max_bv = 4
        xs = np.array([[-4, -2.1, 0, 2, -2]]).T
        ys = np.array([[-2, -0.01, -2, -16, 0]]).T
        self.gp.fit(xs, ys)

        test_x = np.array([[-3, 1]]).T
        expected = np.array([[-0.70996282], [-6.68149458]])
        actual = self.gp.predict(test_x)

        assert_that(self.gp.num_bv, is_(equal_to(4)))
        assert_almost_equal(expected, actual)

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
        expected = 8.51911832
        assert_almost_equal(actual, expected)

    def test_can_calculate_neg_log_likelihood_derivative(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)
        actual = self.gp.calc_neg_log_likelihood(eval_derivative=True)
        expected = np.array([0.76088728, -0.49230927])
        assert_almost_equal(actual[1], expected)


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
        expected = 8.51911832
        assert_almost_equal(actual, expected)

    def test_can_calculate_neg_log_likelihood_derivative(self):
        x = np.array([[-4, -2, -0.5, 0, 2]]).T
        y = np.array([[-2, 0, 1, 2, -1]]).T
        self.gp.fit(x, y)
        actual = self.gp.calc_neg_log_likelihood(eval_derivative=True)
        expected = np.array([0.76088728, -0.49230927])
        assert_almost_equal(actual[1], expected)


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
        expected = 9.3495487653878691
        actual = self.gp.calc_neg_log_likelihood()
        assert_almost_equal(expected, actual)

    def test_can_calculate_neg_log_likelihood_derivative(self):
        expected = np.array([1.50581184e-05, 9.73710450e-06])
        actual = self.gp.calc_neg_log_likelihood(eval_derivative=True)
        assert_almost_equal(expected, actual[1])

    def test_optimizes_kernel_params(self):
        expected = np.array([0.7032695, 1.17616655])
        actual = self.gp.kernel.params
        assert_almost_equal(expected, actual)

    def test_add_observations_retrains_as_needed(self):
        x = np.array([[-3.5, -2.7, -1, 0.5, 1.25]]).T
        y = np.array([[0, -0.5, 0.8, 1, 0]]).T
        self.gp.add_observations(x, y)
        expected = np.array([0.9766916, 0.57622054])
        assert_almost_equal(self.gp.kernel.params, expected)
