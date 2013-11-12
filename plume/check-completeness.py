#!/usr/bin/env python

import argparse
import itertools
import os.path


def checkfiles_for_paramstr(directory, paramstr, num_trials):
    for i in range(num_trials):
        filename = '{}.{}.h5'.format(paramstr, i)
        if not os.path.exists(os.path.join(directory, filename)):
            print filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs=1, type=str, help='Directory to check')
    args = parser.parse_args()

    behaviors = ['DUCB', 'PDUCB']
    gammas = ['0', '-1e-9', '-1e-8', '-1e-7', '-1e-6', '-1e-5', '-1e-4',
              '-1e-3', '-1e-2']
    kappas = ['0.1', '0.5', '0.75', '1', '1.25', '1.5', '2']
    scaling = {'DUCB': ["'auto'", '1'], 'PDUCB': ["'auto'", '70']}
    num_trials = 20

    for b, k, g in itertools.product(behaviors, kappas, gammas):
        for s in scaling[b]:
            paramstr = '_{b}_scaling={s} kappa={k} gamma={g}'.format(
                b=b, s=s, k=k, g=g)
            checkfiles_for_paramstr(args.dir[0], paramstr, num_trials)

    # Check GO files
    for g in gammas:
        paramstr = '_GO_gamma={g}'.format(g=g)
        checkfiles_for_paramstr(args.dir[0], paramstr, num_trials)
