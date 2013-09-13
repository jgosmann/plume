import argparse
import os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tables


class SkippedDataWarning(RuntimeWarning):
    pass

warnings.simplefilter('always', SkippedDataWarning)


def get_filename(basename, i):
    return '{basename}.{i}.h5'.format(basename=basename, i=i)


def get_data_files(basename):
    i = 0
    print(get_filename(basename, i))
    while os.path.isfile(get_filename(basename, i)):
        yield get_filename(basename, i)
        i += 1


def get_avg_error(basename, error_measure='wrmse'):
    error = None
    i = -1
    for i, filename in enumerate(get_data_files(basename)):
        with tables.openFile(filename, 'r') as data:
            table = getattr(data.root, error_measure)
            size = len(table)
            if error is None:
                error = np.zeros(size)
            if error.size == size:
                error += table.read()
            else:
                warnings.warn(
                    'Skipped trial because of differing trial length',
                    SkippedDataWarning)
    i += 1
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()

    rewards = np.zeros(1000)
    rmse = np.zeros(1000)
    wrmse = np.zeros(1000)

    for i, filename in enumerate(get_data_files(args.filename[0])):
        with tables.openFile(filename, 'r') as data:
            if len(data.root.rewards) < 1000:
                print('{} hast just {}'.format(i, len(data.root.rewards)))
            else:
                rewards += data.root.rewards.read()
                rmse += data.root.rmse.read()
                wrmse += data.root.wrmse.read()

    i += 1
    print(i)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rewards / i)
    ax.plot(-100 * rmse / i, label='rmse')
    ax.plot(-100 * wrmse / i, label='wrmse')
    plt.legend()
    plt.ioff()
    plt.show()
