#!/usr/bin/env python

import logging
import os
import sys

import numpy as np
import numpy.random as rnd
import tables

from config import instantiate
from plume import QRSimApplication
from error_estimation import sample_with_metropolis_hastings, vegas
import prediction

logger = logging.getLogger(__name__)


class ErrorMeasure(object):
    def __init__(self, name, return_value_names):
        self.name = name
        self.return_value_names = return_value_names


class Reward(ErrorMeasure):
    def __init__(self, client):
        super(Reward, self).__init__('reward', ['value'])
        self.client = client

    def __call__(self, gp):
        self.locations = self.client.get_locations()
        samples = np.maximum(0, gp.predict(self.locations))
        self.client.set_samples(samples)
        return self.client.get_reward(),


class ISE(ErrorMeasure):
    def __init__(self, client, area):
        super(ISE, self).__init__('ise', ['value', 'sigma'])
        self.client = client
        self.area = np.asarray(area)

    def __call__(self, gp):
        return vegas(
            self.calc_error, self.area[:, 0], self.area[:, 1], args=(gp,),)

    def calc_error(self, x, y, z, gp):
        test_loc = np.vstack(
            (np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z))).T
        pred = np.squeeze(np.maximum(0, gp.predict(test_loc)))
        targets = self.client.get_samples(test_loc)
        return np.square(pred - targets)


class WISE(ErrorMeasure):
    def __init__(self, client, area):
        super(WISE, self).__init__('wise', ['value', 'sigma'])
        self.client = client
        self.area = np.asarray(area)

    def __call__(self, gp):
        return vegas(
            self.calc_error, self.area[:, 0], self.area[:, 1], args=(gp,),)

    def calc_error(self, x, y, z, gp):
        test_loc = np.vstack(
            (np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z))).T
        pred = np.squeeze(np.maximum(0, gp.predict(test_loc)))
        targets = np.asarray(self.client.get_samples(test_loc))
        weighting = targets / targets.max()
        return np.square(pred - targets) * weighting


class LogLikelihood(ErrorMeasure):
    def __init__(self):
        super(LogLikelihood, self).__init__('log_likelihood', ['value'])

    def __call__(self, gp):
        return -gp.calc_neg_log_likelihood(),


class ZeroPredictor(object):
    def predict(self, x):
        return np.zeros(len(x))

    def calc_neg_log_likelihood(self):
        return np.nan


class KernelTester(object):
    def __init__(self, fileh, conf, client):
        self.fileh = fileh
        self.conf = conf
        self.client = client

        self.measures = [
            ISE(client, self.conf['area']), WISE(client, self.conf['area']),
            LogLikelihood(), Reward(client)]

        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)

        fileh.createArray('/', 'lengthscales', conf['lengthscales'])
        fileh.createArray('/', 'variances', conf['variances'])

        for measure in self.measures:
            group = fileh.createGroup('/', measure.name)
            gp_pred_group = fileh.createGroup(group, 'gp_pred')
            zero_pred_group = fileh.createGroup(group, 'zero_pred')

            for name in measure.return_value_names:
                fileh.createArray(
                    zero_pred_group, name, np.zeros(conf['repeats']))
                fileh.createArray(gp_pred_group, name, np.zeros(
                    (len(conf['lengthscales']), len(conf['variances']),
                     conf['repeats'])))

        group = fileh.createGroup('/', 'likelihood_optimization')
        fileh.createArray(group, 'lengthscales', np.zeros(conf['repeats']))
        fileh.createArray(group, 'variances', np.zeros(conf['repeats']))
        for measure in self.measures:
            measure_group = fileh.createGroup(group, measure.name)

            for name in measure.return_value_names:
                fileh.createArray(measure_group, name, np.zeros(
                    (len(conf['lengthscales']), len(conf['variances']),
                     conf['repeats'])))

    def run_and_store_results(self):
        for i in xrange(self.conf['repeats']):
            self._do_trial(i)

    def _do_trial(self, trial):
        self.client.reset_seed(self.conf['seedlist'][trial])
        self.client.reset()

        train_x = self._gen_probe_locations()
        train_y = np.asarray(self.client.get_samples(train_x))

        if self.conf['noise_var'] > 1e-6:
            train_y += np.sqrt(self.conf['noise_var']) * rnd.randn(
                len(train_y))

        for i, j in np.ndindex(
                len(self.conf['lengthscales']), len(self.conf['variances'])):
            lengthscale = self.conf['lengthscales'][i]
            variance = self.conf['variances'][j]

            logger.info('Trial {}, lengthscale={}, variance={}'.format(
                trial, lengthscale, variance))

            kernel = instantiate(
                *self.conf['kernel'], args=(lengthscale, variance))
            gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
            gp.fit(train_x, train_y)

            for measure in self.measures:
                m = measure(gp)
                mz = measure(ZeroPredictor())
                m_group = self.fileh.getNode('/', measure.name)
                gp_pred = self.fileh.getNode(m_group, 'gp_pred')
                zero_pred = self.fileh.getNode(m_group, 'zero_pred')
                for k, name in enumerate(measure.return_value_names):
                    self.fileh.getNode(gp_pred, name)[i, j, trial] = m[k]
                    self.fileh.getNode(zero_pred, name)[trial] = mz[k]

        logger.info('Trial {}, likelihood optimization'.format(trial))
        max_likelihood_idx = np.unravel_index(np.argmax(
            self.fileh.root.log_likelihood.gp_pred.value.read()[:, :, trial]),
            (len(self.conf['lengthscales']), len(self.conf['variances'])))
        lengthscale = self.conf['lengthscales'][max_likelihood_idx[0]]
        variance = self.conf['variances'][max_likelihood_idx[1]]
        kernel = instantiate(
            *self.conf['kernel'], args=(lengthscale, variance))
        gp = prediction.LikelihoodGP(
            kernel, self.conf['noise_var'], self.conf['train_size'])
        gp.fit(train_x, train_y)
        self.fileh.root.likelihood_optimization.lengthscales[trial] = \
            kernel.lengthscale
        self.fileh.root.likelihood_optimization.variances[trial] = \
            kernel.variance
        for measure in self.measures:
            group = self.fileh.getNode(
                '/likelihood_optimization', measure.name)

            m = measure(gp)
            for i, name in enumerate(measure.return_value_names):
                self.fileh.getNode(group, name)[trial] = m[i]

    def _gen_probe_locations(self):
        area = np.asarray(self.conf['area'])
        sources = self.client.get_sources()

        num_uniform_samples = self.conf['uniform_sample_proportion'] * \
            self.conf['train_size']
        num_samples_per_source = int(
            (self.conf['train_size'] - num_uniform_samples) / len(sources))

        samples = [sample_with_metropolis_hastings(
            self.client, source, area, num_samples_per_source,
            self.conf['proposal_std'])[0]
            for source in sources]
        uniform_samples = area[:, 0] + rnd.rand(
            num_uniform_samples, 3) * np.squeeze(np.diff(area, axis=1))

        return np.concatenate([uniform_samples] + samples)


class TryKernelsApplication(QRSimApplication):
    def __init__(self):
        super(TryKernelsApplication, self).__init__()
        self.parser.add_argument(
            '-o', '--output', nargs=1, type=str, default=['kernels'],
            help='Output file name.')

    def _run_application(self, args, conf, client):
        rnd.seed(conf['pyseed'])
        output_filename = os.path.join(args.output_dir[0], args.output[0])
        with tables.openFile(output_filename, 'w') as fileh:
            KernelTester(fileh, conf, client).run_and_store_results()


if __name__ == '__main__':
    if TryKernelsApplication().main():
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_SOFTWARE)
