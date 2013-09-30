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


class Reward(ErrorMeasure):
    def __init__(self, client):
        super(Reward, self).__init__('reward')
        self.client = client
        self.locations = self.client.get_locations()

    def __call__(self, gp, test_x, test_y):
        samples = np.maximum(0, gp.predict(self.locations))
        self.client.set_samples(samples)
        return self.client.get_reward()


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

        self.measures = [RMSE(), WRMSE(), LogLikelihood(), Reward(client)]

        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)

    def run_and_store_results(self):
        for i in xrange(self.conf['repeats']):
            self._do_trial(i)

    def _do_trial(self, trial):
        logger.info('Trial {}'.format(trial))

        self.client.reset_seed(self.conf['seedlist'][trial])
        self.client.reset()

        locations = self._gen_probe_locations()
        rnd.shuffle(locations)
        ground_truth = np.asarray(self.client.get_samples(locations))
        train_x = locations[:self.conf['train_size']]
        #train_y = np.asarray(self.client.get_samples(train_x))
        train_y = ground_truth[:self.conf['train_size']]
        test_x = locations[self.conf['train_size']:]
        test_y = ground_truth[self.conf['train_size']:]

        #if self.conf['noise_var'] > 1e-6:
            #train_y += np.sqrt(self.conf['noise_var']) * rnd.randn(
                #len(train_y))

        #ll = []
        #xs = np.linspace(-120, 50, 3*18)
        #for x in xs:
            #kernel = prediction.PlumeKernel([x, 0, -40], 0, 0, 0.33, 0.86)
            ##kernel = prediction.ExponentialKernel(10, 1)
            ##a = kernel.calc_concentration(train_x)
            ##print(np.abs(a - train_y).max())
            #gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
            #gp.fit(train_x, train_y)
            #ll.append(-gp.calc_neg_log_likelihood())
            ##print(-gp.calc_neg_log_likelihood())
        #import matplotlib.pyplot as plt
        #plt.plot(xs, ll)
        #plt.show()

        #kernel = prediction.PlumeKernel([0, 0, -40], 0, 0, 0.33, 0.86)
        #kernel = prediction.StationaryPlumeKernel()
        #kernel = prediction.AnisotropicExponentialKernel(
            #[[480, 0, 0], [0, 10, 0], [0, 0, 10]])
        kernel = prediction.ExponentialKernel(10)
        gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
        gp.fit(train_x, train_y)
        print(-gp.calc_neg_log_likelihood())

        #kernel = prediction.PlumeKernel([0, 0, -40], 0, 3, 0.4, 0.86)
        #gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
        #gp.fit(train_x, train_y)
        #print(-gp.calc_neg_log_likelihood())

        #kernel = prediction.PlumeKernel([0, 0, -40], 0, 3, 0.33, 0.9)
        #gp = prediction.OnlineGP(kernel, self.conf['noise_var'])
        #gp.fit(train_x, train_y)
        #print(-gp.calc_neg_log_likelihood())

        #x0 = train_x[np.argmax(train_y)]
        #print(x0)
        #kernel = prediction.PlumeKernel(x0, 0, 3, 0.33, 0.86)
        #gp = prediction.LikelihoodGP(kernel, self.conf['noise_var'])
        #gp.bounds = [(0.01, 1.0), (0.01, 1.0)]
        #gp.bounds = [(0, None), (0, 0), (10, 10), (0, 0), (0, 0), (10, 10), (1.0, 1.0)]
        #gp.fit(train_x, train_y)
        #print(-gp.calc_neg_log_likelihood())
        #print(kernel.params)

        for measure in self.measures:
            print('{}: {}'.format(measure.name, measure(gp, test_x, test_y)))
            print('  zero pred: {}'.format(
                measure(ZeroPredictor(), test_x, test_y)))

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
