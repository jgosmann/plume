#!/usr/bin/env python

import argparse
import itertools
import os.path


def missing_trials(directory, paramstr, num_trials):
    for i in range(num_trials):
        filename = '{}.{}.h5'.format(paramstr, i)
        if not os.path.exists(os.path.join(directory, filename)):
            yield i


def checkfiles_for_paramstr(directory, paramstr, num_trials):
    for i in range(num_trials):
        filename = '{}.{}.h5'.format(paramstr, i)
        if not os.path.exists(os.path.join(directory, filename)):
            print filename


def print_tspec_block(paramstr, behavior, kappa, gamma, scaling, trial):
    if behavior == 'DUCB':
        behavior_stmt = 'acq_fn = """(\'behaviors\', \'DUCB\', (), dict(' \
            'kappa=kappa, scaling=scaling, gamma=gamma))"""'
    elif behavior == 'PDUCB':
        behavior_stmt = 'acq_fn = """(\'behaviors\', \'PDUCB\', (), dict(' \
            'kappa=kappa, scaling=scaling, gamma=gamma, epsilon=1e-30))"""'
    elif behavior == 'GO':
        behavior_stmt = 'acq_fn = """(\'behaviors\', \'GO\', (), dict(' \
            'gamma=gamma))"""'
    else:
        raise NotImplementedError()

    print '''[{blockname}]
    {behavior_stmt}
    kappa = {k}
    gamma = {g}
    scaling = {s!r}
    repeat = {r}
'''.format(
        blockname=paramstr[1:] + ' ' + str(trial), behavior_stmt=behavior_stmt,
        k=kappa, g=gamma, s=scaling, r=trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs=1, type=str, help='Directory to check')
    parser.add_argument(
        '--tspec', action='store_true', help='Print tspec block')
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
            if args.tspec:
                for repeat in missing_trials(
                        args.dir[0], paramstr, num_trials):
                    print_tspec_block(paramstr, b, k, g, s, repeat)
            else:
                checkfiles_for_paramstr(args.dir[0], paramstr, num_trials)

    # Check GO files
    for g in gammas:
        paramstr = '_GO_gamma={g}'.format(g=g)
        if args.tspec:
            for repeat in missing_trials(
                    args.dir[0], paramstr, num_trials):
                print_tspec_block(paramstr, b, k, g, s, repeat)
        else:
            checkfiles_for_paramstr(args.dir[0], paramstr, num_trials)
