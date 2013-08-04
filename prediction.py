import GPy as gpy
import numpy as np
from numpy.linalg import inv


class GPyAdapter(object):
    def __init__(self, kernel_str, in_log_space=False):
        self.kernel_str = kernel_str
        self.kernel = eval(kernel_str)
        self.in_log_space = in_log_space

    def fit(self, X, y):
        X = np.asarray(X)
        if hasattr(self, 'in_log_space') and self.in_log_space:
            y = np.log(np.asarray(y))
        else:
            self.in_log_space = False
            y = np.asarray(y)

        if y.ndim == 1:
            y = np.atleast_2d(y).T

        self.model = gpy.models.GPRegression(X, y, self.kernel)
        self.model['.*_lengthscale'] = 30
        self.model['noise_variance'] = 0.1
        #self.model.constrain_bounded('.*rbf_variance', 0.1, 100)
        #self.model.constrain_bounded('.*rbf_lengthscale', 0.1, 140)
        #self.model.constrain_bounded('.*noise_variance', 0.01, 10)
        #self.model.optimize()
        #print(self.model)

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
        self.sq_lengthscale = lengthscale ** 2
        self.variance = variance

    def __call__(self, x1, x2):
        y = np.empty((len(x1), len(x2)))
        for i, j in np.ndindex(*y.shape):
            d = x1[i] - x2[j]
            y[i, j] = self.variance * np.exp(
                -0.5 * np.dot(d, d) / self.sq_lengthscale)
        return y


class OnlineGP(object):
    def __init__(self, kernel, noise_var=1.0):
        self.kernel = kernel
        self.noise_var = noise_var
        self.x_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.K_inv = inv(
            self.kernel(x_train, x_train) +
            np.eye(len(x_train)) * self.noise_var)

    def predict(self, x, eval_mse=False):
        K_new_vs_old = self.kernel(x, self.x_train)
        pred = np.dot(
            K_new_vs_old, np.dot(self.K_inv, self.y_train))
        if eval_mse:
            mse = self.kernel(x, x) - np.dot(
                K_new_vs_old, np.dot(self.K_inv, K_new_vs_old.T))
            return pred, np.diag(mse) + self.noise_var
        else:
            return pred

    def add_observations(self, x, y):
        if self.K_inv is None:
            self.fit(x, y)
            return

        k_new_vs_old = self.kernel(x, self.x_train)
        k_oldinv_new = np.dot(self.K_inv, k_new_vs_old.T)
        f22_inv = inv(self.kernel(x, x) + np.eye(len(x)) * self.noise_var - \
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

        self.x_train = np.append(self.x_train, x, axis=0)
        self.y_train = np.append(self.y_train, y, axis=0)


def predict_on_volume(predictor, area, grid_resolution):
    ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
        area, grid_resolution)]
    x, y, z = (np.rollaxis(m, 1) for m in np.meshgrid(*ogrid))

    pred, mse = predictor.predict(
        np.column_stack((x.flat, y.flat, z.flat)), eval_MSE=True)
    #np.maximum(0, pred, out=pred)

    assert x.shape == y.shape and y.shape == z.shape
    pred = pred.reshape(x.shape)
    mse = mse.reshape(x.shape)
    return pred, mse, (x, y, z)
