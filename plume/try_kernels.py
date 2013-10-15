#!/usr/bin/env python

import logging
import os
import sys

import numpy as np
from numpy.linalg import LinAlgError
import numpy.random as rnd
import tables

from config import instantiate
from plume import QRSimApplication
from error_estimation import gen_probe_locations, Reward, RMSE, WRMSE, \
    LogLikelihood
import prediction
from prediction import ZeroPredictor

logger = logging.getLogger(__name__)


class KernelTester(object):
    def __init__(self, fileh, conf, client):
        self.fileh = fileh
        self.conf = conf
        self.client = client

        self.measures = [RMSE(), WRMSE(), LogLikelihood(), Reward(client)]

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
                    conf['repeats']))

    def run_and_store_results(self):
        for i in xrange(self.conf['repeats']):
            self._do_trial(i)

    def _do_trial(self, trial):
        self.client.reset_seed(self.conf['seedlist'][trial])
        self.client.reset()

        x = gen_probe_locations(self.client, self.conf)
        rnd.shuffle(x)
        y = np.asarray(self.client.get_samples(x))
        train_x = x[:self.conf['train_size']]
        train_y = y[:self.conf['train_size']]
        test_x = x[self.conf['train_size']:]
        test_y = y[self.conf['train_size']:]

        if self.conf['noise_var'] > 1e-6:
            train_y += np.sqrt(self.conf['noise_var']) * rnd.randn(
                len(train_y))

        for measure in self.measures:
            mz = measure(ZeroPredictor(), test_x, test_y)
            m_group = self.fileh.getNode('/', measure.name)
            zero_pred = self.fileh.getNode(m_group, 'zero_pred')
            for k, name in enumerate(measure.return_value_names):
                self.fileh.getNode(zero_pred, name)[trial] = mz[k]

        for i, j in np.ndindex(
                len(self.conf['lengthscales']), len(self.conf['variances'])):
            lengthscale = self.conf['lengthscales'][i]
            variance = self.conf['variances'][j]

            logger.info('Trial {}, lengthscale={}, variance={}'.format(
                trial, lengthscale, variance))

            kernel = instantiate(
                *self.conf['kernel'], args=(lengthscale, variance))
            gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
            try:
                gp.fit(train_x, train_y)

                for measure in self.measures:
                    m = measure(gp, test_x, test_y)
                    m_group = self.fileh.getNode('/', measure.name)
                    gp_pred = self.fileh.getNode(m_group, 'gp_pred')
                    for k, name in enumerate(measure.return_value_names):
                        self.fileh.getNode(gp_pred, name)[i, j, trial] = m[k]
            except LinAlgError:
                for measure in self.measures:
                    m_group = self.fileh.getNode('/', measure.name)
                    gp_pred = self.fileh.getNode(m_group, 'gp_pred')
                    for k, name in enumerate(measure.return_value_names):
                        self.fileh.getNode(gp_pred, name)[i, j, trial] = np.nan

        logger.info('Trial {}, likelihood optimization'.format(trial))
        max_likelihood_idx = np.unravel_index(np.argmax(
            self.fileh.root.log_likelihood.gp_pred.value.read()[:, :, trial]),
            (len(self.conf['lengthscales']), len(self.conf['variances'])))
        lengthscale = self.conf['lengthscales'][max_likelihood_idx[0]]
        variance = 1.0
        kernel = instantiate(
            *self.conf['kernel'], args=(lengthscale, variance))
        try:
            gp = prediction.LikelihoodGP(
                kernel, self.conf['noise_var'], self.conf['train_size'])
            gp.bounds = [(0, None), (1, 1)]
            gp.fit(train_x, train_y)
            self.fileh.root.likelihood_optimization.lengthscales[trial] = \
                kernel.lengthscale
            self.fileh.root.likelihood_optimization.variances[trial] = \
                kernel.variance
            for measure in self.measures:
                group = self.fileh.getNode(
                    '/likelihood_optimization', measure.name)

                m = measure(gp, test_x, test_y)
                for i, name in enumerate(measure.return_value_names):
                    self.fileh.getNode(group, name)[trial] = m[i]
        except LinAlgError:
            for measure in self.measures:
                group = self.fileh.getNode(
                    '/likelihood_optimization', measure.name)
                for i, name in enumerate(measure.return_value_names):
                    self.fileh.getNode(group, name)[trial] = np.nan


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
