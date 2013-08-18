import GPy as gpy
import numpy as np
from numpy.linalg import cholesky, inv

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
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        d = -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])
        res = self.variance * np.exp(-0.5 * d)
        if eval_derivative:
            s = np.rollaxis(np.atleast_3d(x1) - np.atleast_3d(x2).T, 1, 3)
            der = -self.variance / self.lengthscale * s * \
                np.exp(-0.5 * d)[:, :, None]
            return res, der
        else:
            return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        d = -2 * np.einsum('ij,ij->i', x1, x2) + (
            np.sum(np.square(x1), 1) + np.sum(np.square(x2), 1))
        return self.variance * np.exp(-0.5 * d)


class OnlineGP(object):
    def __init__(self, kernel, noise_var=1.0, expected_samples=100):
        self.kernel = kernel
        self.noise_var = noise_var
        self.expected_samples = expected_samples
        self.x_train = None
        self.y_train = None
        self.K_inv = None
        self.trained = False

    def fit(self, x_train, y_train):
        self.x_train = self._create_data_array(np.asarray(x_train))
        self.y_train = self._create_data_array(np.asarray(y_train))
        self.K_inv = Growing2dArray(expected_rows=self.expected_samples)
        self.K_inv.enlarge_by(len(x_train))
        self.K_inv.data[:] = inv(
            self.kernel(self.x_train.data, self.x_train.data) +
            np.eye(len(self.x_train.data)) * self.noise_var)
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

        svs = np.dot(self.K_inv.data, self.y_train.data)
        pred = np.dot(K_new_vs_old, svs)
        if eval_MSE:
            mse_svs = np.dot(self.K_inv.data, K_new_vs_old.T)
            mse = self.noise_var + self.kernel.diag(x, x) - np.einsum(
                'ij,ji->i', K_new_vs_old, mse_svs)

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

        # The equations used here are stated in
        # Singh, A., Ramos, F., Whyte, H. D., & Kaiser, W. J. (2010).
        # Modeling and decision making in spatio-temporal processes for
        # environmental surveillance, 5490-5497.
        # for example.
        K_obs = self.kernel(x, self.x_train.data)
        projected = np.dot(self.K_inv.data, K_obs.T)

        l = len(self.K_inv.data)
        self.K_inv.enlarge_by(len(x))
        L = inv(cholesky(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var -
            np.dot(K_obs, projected)))
        self.K_inv.data[l:, l:] = np.dot(L.T, L)
        f22_inv = self.K_inv.data[l:, l:]
        f21 = np.dot(projected, f22_inv)
        self.K_inv.data[:l, l:] = -f21
        self.K_inv.data[l:, :l] = -f21.T
        self.K_inv.data[:l, :l] += np.dot(f21, projected.T)

        self.x_train.extend(x)
        self.y_train.extend(y)


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
