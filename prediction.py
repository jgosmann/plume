import GPy as gpy
import numpy as np


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
