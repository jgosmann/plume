import logging
import warnings

import numpy as np
from numpy.linalg import cholesky, inv, linalg
from scipy.optimize import fmin_l_bfgs_b

from nputil import GrowingArray, Growing2dArray, meshgrid_nd

logger = logging.getLogger(__name__)


class RBFKernel(object):
    def __init__(self, lengthscale, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def get_params(self):
        return np.array([self.lengthscale, self.variance])

    def set_params(self, values):
        self.lengthscale, self.variance = values

    params = property(get_params, set_params)

    def __call__(self, x1, x2, eval_derivative=False):
        """Returns the Gram matrix for the points given in x1 and x1.

        If eval_derivative=True the derivative in x1 evaluated at the points in
        x1 will be returned.
        """
        d = self._calc_distance(x1, x2)
        res = self.variance * np.exp(-0.5 * d / self.lengthscale ** 2)
        if eval_derivative:
            s = x1[:, None, :] - x2[None, :, :]
            der = -1.0 / (self.lengthscale ** 2) * s * res[:, :, None]
            return res, der
        else:
            return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = -2 * np.einsum('ij,ij->i', x1, x2) + (
            np.sum(np.square(x1), 1) + np.sum(np.square(x2), 1))
        return self.variance * np.exp(-0.5 * d / self.lengthscale ** 2)

    def param_derivatives(self, x1, x2):
        d = self._calc_distance(x1, x2)
        variance_deriv = np.exp(-0.5 * d / self.lengthscale ** 2)
        lengthscale_deriv = 2 * self.variance * d / (self.lengthscale ** 3) * \
            variance_deriv
        return [lengthscale_deriv, variance_deriv]

    def _calc_distance(self, x1, x2):
        return -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])


class ExponentialKernel(object):
    def __init__(self, lengthscale, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def get_params(self):
        return np.array([self.lengthscale, self.variance])

    def set_params(self, values):
        self.lengthscale, self.variance = values

    params = property(get_params, set_params)

    def __call__(self, x1, x2, eval_derivative=False):
        d = self._calc_distance(x1, x2)
        res = self.variance * np.exp(-d / self.lengthscale)
        if eval_derivative:
            s = x1[:, None, :] - x2[None, :, :]
            der = -s / d[:, :, None] / self.lengthscale * res[:, :, None]
            return res, der
        else:
            return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        return self.variance * np.exp(-d / self.lengthscale)

    def param_derivatives(self, x1, x2):
        d = self._calc_distance(x1, x2)
        variance_deriv = np.exp(-d / self.lengthscale)
        lengthscale_deriv = self.variance * d / (self.lengthscale ** 2) * \
            variance_deriv
        return [lengthscale_deriv, variance_deriv]

    def _calc_distance(self, x1, x2):
        return np.sqrt(-2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :]))


class AnisotropicExponentialKernel(object):
    def __init__(self, lengthscale_mat, variance=1.0):
        lengthscale_mat = np.asarray(lengthscale_mat)
        assert lengthscale_mat.shape[0] == lengthscale_mat.shape[1]
        self.num_dim = lengthscale_mat.shape[0]
        self.params = np.concatenate((
            lengthscale_mat[np.tril_indices_from(lengthscale_mat)],
            np.array([variance])))

    def get_params(self):
        return np.concatenate((self.lengthscales, np.array([self.variance])))

    def set_params(self, values):
        self.lengthscales = values[:-1]
        self.variance = values[-1]
        L = np.zeros((self.num_dim, self.num_dim))
        L[np.tril_indices_from(L)] = self.lengthscales
        self.L_inv = inv(L)
        self.projection = np.dot(self.L_inv.T, self.L_inv)

    params = property(get_params, set_params)

    def __call__(self, x1, x2, eval_derivative=False):
        pd = self._calc_projected_distance(x1, x2)
        res = self.variance * np.exp(-pd)
        if eval_derivative:
            sq_proj = np.dot(self.projection, self.projection)
            x1_proj = np.dot(x1, sq_proj)
            x2_proj = np.dot(x2, sq_proj)
            s = x1_proj[:, None, :] - x2_proj[None, :, :]
            der = -s / pd[:, :, None] * res[:, :, None]
            return res, der
        else:
            return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        pd = np.sqrt(np.sum(np.einsum(
            'ij,kj->ki', self.projection, x1 - x2) ** 2, axis=1))
        return self.variance * np.exp(-pd)

    def param_derivatives(self, x1, x2):
        pd = self._calc_projected_distance(x1, x2)
        variance_deriv = np.exp(-pd)

        proj_deriv_component = -np.einsum(
            'ik,jl->ijkl', self.projection, self.L_inv)
        proj_deriv = proj_deriv_component + np.transpose(
            proj_deriv_component, (0, 1, 3, 2))

        d = x1[:, None, :] - x2[None, :, :]
        d_proj = np.einsum(
            'abk,kl,ijlm,abm->ijab', d, self.projection, proj_deriv, d)

        zero_dist_indices = np.abs(pd) < np.finfo(pd.dtype).eps
        pd[zero_dist_indices] = 1.0  # prevent warnings
        lengthscale_deriv = -self.variance * \
            (variance_deriv / pd)[None, None, :, :] * d_proj
        lengthscale_deriv[:, :, zero_dist_indices] = 0.0
        return np.concatenate((
            lengthscale_deriv[np.tril_indices(self.num_dim)],
            variance_deriv[None, :, :]))

    def _calc_projected_distance(self, x1, x2):
        x1_proj = np.einsum('ij,kj->ki', self.projection, x1)
        x2_proj = np.einsum('ij,kj->ki', self.projection, x2)
        return np.sqrt(np.maximum(0, -2 * np.dot(x1_proj, x2_proj.T) + (
            np.sum(np.square(x1_proj), 1)[:, None] +
            np.sum(np.square(x2_proj), 1)[None, :])))


class UniformLogPrior(object):
    def __call__(self, x):
        return 0

    def derivative(self, x):
        return 0


class GaussianLogPrior(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return -0.5 * ((x - self.mean) / self.std) ** 2 - np.log(self.std) - \
            0.5 * np.log(2 * np.pi)

    def derivative(self, x):
        return -(x - self.mean) / self.std


class OnlineGP(object):
    def __init__(self, kernel, noise_var=1.0, expected_samples=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.expected_samples = expected_samples
        self.x_train = None
        self.y_train = None
        self.L_inv = None
        self.trained = False
        self.min_rel_jitter = 1e-6
        self.max_rel_jitter = 1e-1

    def fit(self, x_train, y_train):
        self.x_train = self._create_data_array(np.asarray(x_train))
        self.y_train = self._create_data_array(np.asarray(y_train))
        self.L_inv = Growing2dArray(expected_rows=self.expected_samples)
        self._refit()

    def _refit(self):
        self.L_inv.enlarge_by(len(self.x_train.data) - len(self.L_inv.data))
        self.L_inv.data[:] = inv(self._jitter_cholesky(
            self.kernel(self.x_train.data, self.x_train.data) +
            np.eye(len(self.x_train.data)) * self.noise_var))
        self.K_inv = np.dot(self.L_inv.data.T, self.L_inv.data)
        self.trained = True

    def _create_data_array(self, initial_data):
        growing_array = GrowingArray(
            initial_data.shape[1:], expected_rows=self.expected_samples)
        growing_array.extend(initial_data)
        return growing_array

    def predict(self, x, eval_MSE=False, eval_derivatives=False):
        if eval_derivatives:
            K_new_vs_old, K_new_vs_old_derivative = self.kernel(
                x, self.x_train.data, eval_derivative=True)
        else:
            K_new_vs_old = self.kernel(x, self.x_train.data)

        svs = np.dot(self.K_inv, self.y_train.data)
        pred = np.dot(K_new_vs_old, svs)
        if eval_MSE:
            mse_svs = np.dot(self.K_inv, K_new_vs_old.T)
            mse = np.maximum(
                self.noise_var,
                self.noise_var + self.kernel.diag(x, x) - np.einsum(
                    'ij,ji->i', K_new_vs_old, mse_svs))

        if eval_derivatives:
            pred_derivative = np.einsum(
                'ijk,jl->ilk', K_new_vs_old_derivative, svs)
            if eval_MSE:
                mse_derivative = -2 * np.einsum(
                    'ijk,ji->ik', K_new_vs_old_derivative, mse_svs)
                return pred, pred_derivative, mse, mse_derivative
            else:
                return pred, pred_derivative
        elif eval_MSE:
            return pred, mse
        else:
            return pred

    def add_observations(self, x, y):
        if not self.trained:
            self.fit(x, y)
            return

        K_obs = self.kernel(x, self.x_train.data)
        B = np.dot(K_obs, self.L_inv.data.T)
        CC_T = self.kernel(x, x) + np.eye(len(x)) * self.noise_var - \
            np.dot(B, B.T)
        diag_indices = np.diag_indices_from(CC_T)
        CC_T[diag_indices] = np.maximum(self.noise_var, CC_T[diag_indices])

        self.x_train.extend(x)
        self.y_train.extend(y)

        try:
            C_inv = inv(cholesky(CC_T))
        except linalg.LinAlgError:
            warnings.warn(
                'New submatrix of covariance matrix singular. '
                'Retraining on all data.', NumericalStabilityWarning)
            self._refit()
            return

        l = len(self.L_inv.data)
        self.L_inv.enlarge_by(len(x))
        self.L_inv.data[:l, l:] = 0.0
        self.L_inv.data[l:, :l] = -np.dot(
            np.dot(C_inv, B), self.L_inv.data[:l, :l])
        self.L_inv.data[l:, l:] = C_inv
        self.K_inv = np.dot(self.L_inv.data.T, self.L_inv.data)

    def _jitter_cholesky(self, A):
        try:
            return cholesky(A)
        except linalg.LinAlgError:
            magnitude = np.mean(np.diag(A))
            max_jitter = self.max_rel_jitter * magnitude
            jitter = self.min_rel_jitter * magnitude
            while jitter <= max_jitter:
                try:
                    L = cholesky(A + np.eye(A.shape[0]) * jitter)
                    warnings.warn(
                        'Added jitter of %f.' % jitter,
                        NumericalStabilityWarning)
                    return L
                except linalg.LinAlgError:
                    jitter *= 10.0
        raise linalg.LinAlgError(
            'Singular matrix despite jitter of {}.'.format(jitter))

    def calc_neg_log_likelihood(self, eval_derivative=False):
        svs = np.dot(self.L_inv.data, self.y_train.data)
        log_likelihood = -0.5 * np.dot(svs.T, svs) + \
            np.sum(np.log(np.diag(self.L_inv.data))) - \
            0.5 * len(self.y_train.data) * np.log(2 * np.pi)

        if eval_derivative:
            alpha = np.dot(self.L_inv.data.T, svs)
            grad_weighting = np.dot(alpha, alpha.T) - self.K_inv
            kernel_derivative = np.array([
                0.5 * np.sum(np.einsum('ij,ji->i', grad_weighting, param_deriv))
                for param_deriv in self.kernel.param_derivatives(
                    self.x_train.data, self.x_train.data)])

            return -np.squeeze(log_likelihood), -kernel_derivative
        else:
            return -np.squeeze(log_likelihood)


class LikelihoodGP(object):
    def __init__(self, kernel, noise_var=1.0, expected_samples=100):
        self.priors = [UniformLogPrior() for i in xrange(len(kernel.params))]
        self.bounds = [(None, None)] * len(kernel.params)
        self.kernel = kernel
        self.noise_var = noise_var
        self.expected_samples = expected_samples
        self.gp = OnlineGP(self.kernel, self.noise_var, self.expected_samples)
        self.neg_log_likelihood = -np.inf

    trained = property(lambda self: self.gp.trained)

    def fit(self, x_train, y_train):
        params, unused, unused = fmin_l_bfgs_b(
            self._optimization_fn, self.kernel.params,
            args=(x_train, y_train), bounds=self.bounds)
        self.kernel.params = params
        self.gp.fit(x_train, y_train)
        self.neg_log_likelihood = self._calc_neg_log_likelihood(
            eval_derivative=False)
        logger.info('Log likelihood: {}, params: {}'.format(
            -self.neg_log_likelihood, self.kernel.params))

    def _optimization_fn(self, params, x_train, y_train):
        self.kernel.params = params
        try:
            self.gp.fit(x_train, y_train)
        except linalg.LinAlgError:
            logger.warn(
                'LinAlgError in likelihood optimization', exc_info=True)
            return np.inf, np.zeros_like(params)
        return self._calc_neg_log_likelihood(eval_derivative=True)

    def predict(self, x, eval_MSE=False, eval_derivatives=False):
        return self.gp.predict(x, eval_MSE, eval_derivatives)

    def add_observations(self, x, y):
        if not self.trained:
            self.fit(x, y)
            return

        self.gp.add_observations(x, y)
        new_neg_log_likelihood = self._calc_neg_log_likelihood(
            eval_derivative=False)
        if new_neg_log_likelihood > self.neg_log_likelihood:
            self.fit(self.gp.x_train.data, self.gp.y_train.data)
            self.neg_log_likelihood = self._calc_neg_log_likelihood(
                eval_derivative=False)
        else:
            self.neg_log_likelihood = new_neg_log_likelihood

    def calc_neg_log_likelihood(self, eval_derivative=False):
        if eval_derivative:
            return self._calc_neg_log_likelihood(eval_derivative)
        else:
            return self.neg_log_likelihood

    def _calc_neg_log_likelihood(self, eval_derivative):
        prior_log_likelihood = np.sum([prior(theta) for prior, theta in zip(
            self.priors, self.kernel.params)])
        if eval_derivative:
            gp_neg_log_likelihood, gp_neg_deriv = \
                self.gp.calc_neg_log_likelihood(eval_derivative=True)
            prior_deriv = np.sum([prior.derivative(theta) for prior, theta \
                in zip(self.priors, self.kernel.params)])
            return gp_neg_log_likelihood - prior_log_likelihood, \
                gp_neg_deriv - prior_deriv
        else:
            gp_neg_log_likelihood = self.gp.calc_neg_log_likelihood()
            return gp_neg_log_likelihood - prior_log_likelihood


class NumericalStabilityWarning(RuntimeWarning):
    pass


def predict_on_volume(predictor, area, grid_resolution):
    ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
        area, grid_resolution)]
    x, y, z = meshgrid_nd(*ogrid)

    pred, mse = predictor.predict(
        np.column_stack((x.flat, y.flat, z.flat)), eval_MSE=True)
    np.maximum(0, pred, out=pred)

    assert x.shape == y.shape and y.shape == z.shape
    pred = pred.reshape(x.shape)
    mse = mse.reshape(x.shape)
    return pred, mse, (x, y, z)
