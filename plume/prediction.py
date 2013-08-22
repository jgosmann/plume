import warnings

import GPy as gpy
import numpy as np
from numpy.linalg import cholesky, inv, linalg

from npwrap import GrowingArray, Growing2dArray


class GPyAdapter(object):
    def __init__(self, kernel_str, noise_variance, in_log_space=False):
        self.kernel_str = kernel_str
        self.kernel = eval(kernel_str)
        self.in_log_space = in_log_space
        self.noise_variance = noise_variance
        self.X = None
        self.y = None
        self.trained = False

    def fit(self, X, y):
        self.X = np.asarray(X)
        if hasattr(self, 'in_log_space') and self.in_log_space:
            self.y = np.log(np.asarray(y))
        else:
            self.in_log_space = False
            self.y = np.asarray(y)

        if y.ndim == 1:
            self.y = np.atleast_2d(self.y).T

        self._refit_model()
        self.trained = True

    def add_observations(self, X, y):
        if not self.trained:
            self.fit(X, y)
            return

        self.X = np.append(self.X, X, axis=0)

        if hasattr(self, 'in_log_space') and self.in_log_space:
            y = np.log(np.asarray(y))
        else:
            y = np.asarray(y)
        if y.ndim == 1:
            y = np.atleast_2d(y).T
        self.y = np.append(self.y, y, axis=0)

        self._refit_model()

    def _refit_model(self):
        self.model = gpy.models.GPRegression(self.X, self.y, self.kernel)
        self.model['noise_variance'] = self.noise_variance

    def predict(self, X, eval_MSE=False):
        pred, mse, lcb, ucb = self.model.predict(X)
        if self.in_log_space:
            pred = np.exp(pred)
        if eval_MSE:
            return pred, mse
        else:
            return pred

    def __repr__(self):
        return 'GPyAdapter(%s)' % self.kernel_str


class RBFKernel(object):
    def __init__(self, lengthscale, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2, eval_derivative=False):
        """Returns the Gram matrix for the points given in x1 and x1.

        If eval_derivative=True the derivative in x1 evaluated at the points in
        x1 will be returned.
        """
        d = -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])
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


# FIXME write tests
class ExponentialKernel(object):
    def __init__(self, lengthscale, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1, x2, eval_derivative=False):
        d = np.sqrt(-2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :]))
        res = self.variance * np.exp(-d / self.lengthscale)
        if eval_derivative:
            s = x1[:, None, :] - x2[None, :, :]
            der = -2.0 / d[:, :, None] / self.lengthscale * s * res[:, :, None]
            return res, der
        else:
            return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        return self.variance * np.exp(-d / self.lengthscale)


class OnlineGP(object):
    def __init__(self, kernel, noise_var=1.0, expected_samples=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.expected_samples = expected_samples
        self.x_train = None
        self.y_train = None
        self.L_inv = None
        self.trained = False
        self.min_rel_jitter = 1e-10
        self.max_rel_jitter = 1e-6

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
                    print jitter
                    jitter *= 10.0
        raise linalg.LinAlgError('Singular matrix despite jitter.')


class NumericalStabilityWarning(RuntimeWarning):
    pass


def predict_on_volume(predictor, area, grid_resolution):
    ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
        area, grid_resolution)]
    x, y, z = (np.rollaxis(m, 1) for m in np.meshgrid(*ogrid))

    pred, mse = predictor.predict(
        np.column_stack((x.flat, y.flat, z.flat)), eval_MSE=True)
    np.maximum(0, pred, out=pred)

    assert x.shape == y.shape and y.shape == z.shape
    pred = pred.reshape(x.shape)
    mse = mse.reshape(x.shape)
    return pred, mse, (x, y, z)
