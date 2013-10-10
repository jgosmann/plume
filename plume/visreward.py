import argparse
import matplotlib.pyplot as plt
import tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()

    with tables.openFile(args.filename[0], 'r') as data:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #ax.plot(data.root.rewards)
        ax.plot(data.root.rmse.read(), label='rmse')
        ax.plot(data.root.wrmse.read(), label='wrmse')
        plt.legend()
        plt.ioff()
        plt.show()
