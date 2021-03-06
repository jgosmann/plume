from __future__ import division

import numpy as np
import numpy.random as rnd
from scipy.stats import gaussian_kde


def sample_with_metropolis_hastings(
        client, x0, area, num_samples, proposal_std):
    positions = np.empty((num_samples, 3))
    values = np.empty(num_samples)

    f = 0
    while f <= 0:
        x = _draw_from_proposal_dist(x0, area, proposal_std)
        f = np.squeeze(client.get_samples(x))

    for i in xrange(num_samples):
        x_new = _draw_from_proposal_dist(x, area, proposal_std)
        f_new = np.squeeze(client.get_samples(x_new))
        if f <= 0:
            acceptance_ratio = 1
        else:
            acceptance_ratio = f_new / f
        if rnd.rand() < acceptance_ratio:
            x = x_new
            f = f_new

        positions[i] = x_new
        values[i] = f

    return positions, values


def _draw_from_proposal_dist(x0, area, proposal_std):
    d = len(area)
    x = np.array(3 * [np.inf])
    while np.any(x < area[:, 0]) or np.any(x > area[:, 1]):
        x = x0 + proposal_std * rnd.randn(d)
    return x


def vegas(
        func, a, b, args=(), num_eval=1000, num_iterations=10,
        num_increments=50, num_subincrements=1000):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    num_dim = len(a)

    cum_int_numerator = cum_int_denominator = 0

    incs = np.empty((num_dim, num_increments))
    incs[:, :] = ((b - a) / num_increments)[:, None]

    for iteration in xrange(num_iterations):
        upper_bounds = a[:, None] + np.cumsum(incs, axis=1)

        selection = rnd.randint(num_increments, size=(num_eval, num_dim))
        x = upper_bounds[xrange(num_dim), selection] - \
            incs[xrange(num_dim), selection] * rnd.rand(num_eval, num_dim)
        p = 1.0 / num_increments / incs[xrange(num_dim), selection]
        joint_p = np.prod(p, axis=1)
        y = func(*(tuple(x.T) + args))
        contributions = np.abs(y)[:, None] * p / joint_p[:, None]

        importance_density = np.zeros_like(incs)
        for i in xrange(num_eval):
            importance_density[xrange(num_dim), selection[i]] += \
                contributions[i]

        weighted_y = y / joint_p
        int_estimate = np.mean(weighted_y)
        var_estimate = (np.mean(weighted_y ** 2) - int_estimate ** 2) / (
            num_eval - 1)
        weighted_int_estimate = int_estimate ** 2 / var_estimate
        cum_int_numerator += int_estimate * weighted_int_estimate
        cum_int_denominator += weighted_int_estimate

        unnormalized_importance = importance_density * incs
        importance = unnormalized_importance / np.sum(
            unnormalized_importance, axis=1)[:, None]
        valid = np.logical_and(0 < importance, importance < 1)
        splits = np.ones_like(importance, dtype=int)
        splits[valid] = 1 + np.round(
            num_subincrements * (
                (importance[valid] - 1) / np.log(importance[valid])) ** 2.0)

        new_incs = np.zeros_like(incs)
        incs /= splits
        js = np.zeros(num_dim, dtype=int)
        for i in xrange(num_increments):
            num_subincs_to_join = np.round(
                np.sum(splits, axis=1) / (num_increments - i))
            while np.any(num_subincs_to_join) > 0:
                num_to_take = np.minimum(
                    num_subincs_to_join, splits[xrange(num_dim), js])
                new_incs[:, i] += incs[xrange(num_dim), js] * num_to_take
                num_subincs_to_join -= num_to_take
                splits[xrange(num_dim), js] -= num_to_take
                js += np.logical_and(
                    splits[xrange(num_dim), js] < 1, js < num_increments - 1)
        incs = new_incs

    return cum_int_numerator / cum_int_denominator, \
        cum_int_numerator / cum_int_denominator / np.sqrt(cum_int_denominator)


class ErrorMeasure(object):
    def __init__(self, name, return_value_names):
        self.name = name
        self.return_value_names = return_value_names


class Reward(ErrorMeasure):
    def __init__(self, client):
        super(Reward, self).__init__('reward', ['value'])
        self.client = client

    def __call__(self, gp, test_x, test_y):
        self.locations = self.client.get_locations()
        samples = np.maximum(0, gp.predict(self.locations))
        self.client.set_samples(samples)
        return self.client.get_reward(),


class RMSE(ErrorMeasure):
    def __init__(self):
        super(RMSE, self).__init__('rmse', ['value'])

    def __call__(self, gp, test_x, test_y):
        density = gaussian_kde(test_x.T)
        pred = np.squeeze(np.maximum(0, gp.predict(test_x)))
        return np.sqrt(
            np.mean(np.square((pred - test_y) / density(test_x.T)), axis=0)),


class WRMSE(ErrorMeasure):
    def __init__(self):
        super(WRMSE, self).__init__('wrmse', ['value'])

    def __call__(self, gp, test_x, test_y):
        density = gaussian_kde(test_x.T)
        weighting = test_y / test_y.max()
        pred = np.squeeze(np.maximum(0, gp.predict(test_x)))
        return np.sqrt(np.mean(np.square(
            (pred - test_y) * weighting / density(test_x.T)), axis=0)),


class LogLikelihood(ErrorMeasure):
    def __init__(self):
        super(LogLikelihood, self).__init__('log_likelihood', ['value'])

    def __call__(self, gp, test_x, test_y):
        return -gp.calc_neg_log_likelihood(),


def gen_probe_locations(client, conf):
    area = np.asarray(conf['area'])
    sources = client.get_sources()

    num_uniform_samples = conf['num_uniform_samples']
    num_samples_per_source = conf['num_source_samples'] // len(sources)
    mh_stride = conf['mh_stride']

    uniform_samples = area[:, 0] + rnd.rand(
        num_uniform_samples, 3) * np.squeeze(np.diff(area, axis=1))

    samples = [sample_with_metropolis_hastings(
        client, source, area, num_samples_per_source,
        conf['proposal_std'])[0][::mh_stride] for source in sources]

    samples_gauss = []
    for i in xrange(mh_stride):
        samples_gauss.extend(
            conf['proposal_std'] * rnd.randn(len(s), 3) + s for s in samples)
    samples_gauss = filter(
        lambda x: np.all(x > area[:, 0]) and np.all(x < area[:, 1]),
        samples_gauss)

    return np.concatenate([uniform_samples] + samples + samples_gauss)
