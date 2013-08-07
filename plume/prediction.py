import GPy as gpy
import numpy as np
from numpy.linalg import inv

from npwrap import GrowingArray


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

    def __call__(self, x1, x2):
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        d = -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])
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
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        self.x_train = GrowingArray(
            x_train.shape[1:], expected_rows=self.expected_samples)
        self.y_train = GrowingArray(
            y_train.shape[1:], expected_rows=self.expected_samples)
        self.x_train.extend(x_train)
        self.y_train.extend(y_train)
        self.K_inv = inv(
            self.kernel(x_train, x_train) +
            np.eye(len(x_train)) * self.noise_var)
        self.trained = True

    def predict(self, x, eval_MSE=False):
        K_new_vs_old = self.kernel(x, self.x_train.data)
        pred = np.dot(
            K_new_vs_old, np.dot(self.K_inv, self.y_train.data))
        if eval_MSE:
            mse = 1.0 + self.noise_var - np.einsum(
                'ij,jk,ik->i', K_new_vs_old, self.K_inv, K_new_vs_old)
            return pred, mse
        else:
            return pred

    def add_observations(self, x, y):
        if not self.trained:
            self.fit(x, y)
            return

        k_new_vs_old = self.kernel(x, self.x_train.data)
        k_oldinv_new = np.dot(self.K_inv, k_new_vs_old.T)
        f22_inv = inv(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var -
            np.dot(k_new_vs_old, k_oldinv_new))
        f11 = self.K_inv + np.dot(
            k_oldinv_new, np.dot(f22_inv, k_oldinv_new.T))
        f12 = -np.dot(k_oldinv_new, f22_inv)
        l = len(self.K_inv) + len(x)
        self.K_inv = np.empty((l, l))
        self.K_inv[:len(f11), :len(f11)] = f11
        self.K_inv[:len(f11), len(f11):] = f12
        self.K_inv[len(f11):, :len(f11)] = f12.T
        self.K_inv[len(f11):, len(f11):] = f22_inv

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
