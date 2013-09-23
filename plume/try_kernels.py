#!/usr/bin/env python

import logging
import os
import sys

import numpy as np
import numpy.random as rnd
import tables

from nputil import meshgrid_nd
from plume import QRSimApplication
import prediction

logger = logging.getLogger(__name__)


class ErrorMeasure(object):
    def __init__(self, name):
        self.name = name


class RMSE(ErrorMeasure):
    def __init__(self):
        super(RMSE, self).__init__('rmse')

    def __call__(self, gp, test_x, test_y):
        pred = np.squeeze(np.maximum(0, gp.predict(test_x)))
        se = (pred - test_y) ** 2
        return np.sqrt(np.mean(se))


class WRMSE(ErrorMeasure):
    def __init__(self):
        super(WRMSE, self).__init__('wrmse')

    def __call__(self, gp, test_x, test_y):
        pred = np.squeeze(np.maximum(0, gp.predict(test_x)))
        se = (pred - test_y) ** 2
        wse = se * test_y / test_y.max()
        return np.sqrt(np.mean(wse))


class LogLikelihood(ErrorMeasure):
    def __init__(self):
        super(LogLikelihood, self).__init__('log_likelihood')

    def __call__(self, gp, test_x, test_y):
        return -gp.calc_neg_log_likelihood()


class KernelTester(object):
    def __init__(self, fileh, conf, client):
        self.fileh = fileh
        self.conf = conf
        self.client = client

        self.measures = [RMSE(), WRMSE(), LogLikelihood()]

        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)

        fileh.createArray('/', 'lengthscales', conf['lengthscales'])
        fileh.createArray('/', 'variances', conf['variances'])
        for measure in self.measures:
            fileh.createArray('/', measure.name, np.zeros(
                (len(conf['lengthscales']), len(conf['variances']),
                 conf['repeats'])))

        group = fileh.createGroup('/', 'likelihood_optimization')
        fileh.createArray(group, 'lengthscales', np.zeros(conf['repeats']))
        fileh.createArray(group, 'variances', np.zeros(conf['repeats']))
        for measure in self.measures:
            fileh.createArray(group, measure.name, np.zeros(conf['repeats']))

    def run_and_store_results(self):
        for i in xrange(self.conf['repeats']):
            self._do_trial(i)

    def _do_trial(self, trial):
        self.client.reset_seed(self.conf['seedlist'][trial])
        self.client.reset()

        locations = self._gen_probe_locations()
        rnd.shuffle(locations)
        ground_truth = np.asarray(self.client.get_samples(locations))
        train_x = locations[:self.conf['train_size']]
        train_y = ground_truth[:self.conf['train_size']]
        test_x = locations[self.conf['train_size']:]
        test_y = ground_truth[self.conf['train_size']:]

        if self.conf['noise_var'] > 1e-6:
            train_y += np.sqrt(self.conf['noise_var']) * rnd.randn(
                len(train_y))

        for i, j in np.ndindex(
                len(self.conf['lengthscales']), len(self.conf['variances'])):
            lengthscale = self.conf['lengthscales'][i]
            variance = self.conf['variances'][j]

            logger.info('Trial {}, lengthscale={}, variance={}'.format(
                trial, lengthscale, variance))

            kernel = self.conf['kernel'](prediction, lengthscale, variance)
            gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
            gp.fit(train_x, train_y)

            for measure in self.measures:
                self.fileh.get_node('/', measure.name)[i, j, trial] = measure(
                    gp, test_x, test_y)

        logger.info('Trial {}, likelihood optimization'.format(trial))
        max_likelihood_idx = np.unravel_index(
            np.argmax(self.fileh.root.log_likelihood.read()[:, :, trial]),
            (len(self.conf['lengthscales']), len(self.conf['variances'])))
        lengthscale = self.conf['lengthscales'][max_likelihood_idx[0]]
        variance = self.conf['variances'][max_likelihood_idx[1]]
        kernel = self.conf['kernel'](prediction, lengthscale, variance)
        gp = prediction.LikelihoodGP(
            kernel, self.conf['noise_var'], self.conf['train_size'])
        gp.fit(train_x, train_y)
        self.fileh.root.likelihood_optimization.lengthscales[trial] = \
            kernel.lengthscale
        self.fileh.root.likelihood_optimization.variances[trial] = \
            kernel.variance
        for measure in self.measures:
            self.fileh.get_node(
                '/likelihood_optimization', measure.name)[trial] = measure(
                gp, test_x, test_y)

    def _gen_probe_locations(self):
        ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
            self.conf['area'], self.conf['resolution'])]
        x, y, z = meshgrid_nd(*ogrid)
        return np.column_stack((x.flat, y.flat, z.flat))


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
