from __future__ import division
from mayavi import mlab
import argparse
import numpy as np
import tables

from sklearn import gaussian_process


class PlumeVisualizer(object):
    def __init__(self):
        self.figure = mlab.figure(
            bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))

    def finish(self):
        # FIXME read from config or so
        mlab.outline(extent=[-140, 140, -140, 140, -80, 0], figure=self.figure)
        mlab.view(
            azimuth=135, elevation=135, distance=600, roll=-120,
            figure=self.figure)

    def plot_uav_trajectory(self, positions):
        mlab.plot3d(
            *positions.T, tube_radius=1, line_width=0,
            color=(0, 126 / 255, 203 / 255), figure=self.figure)
        mlab.points3d(
            *positions[-1], scale_factor=5, color=(0, 172 / 255, 1),
            figure=self.figure)

    def plot_uav_trajectories(self, trajectories):
        for uav_positions in trajectories:
            self.plot_uav_trajectory(uav_positions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()

    with tables.open_file(args.filename[0], 'r') as data:
        trajectories = data.root.positions.read()

        prediction_visualizer = PlumeVisualizer()
        prediction_visualizer.plot_uav_trajectories(trajectories)

        gp = gaussian_process.GaussianProcess(nugget=0.5)
        gp.fit(
            data.root.positions.read()[0, :, :],
            data.root.plume_measurements.read()[0])

        x, y, z = np.mgrid[-140:140:5, -140:140:5, -80:0:5]
        pred, mse = gp.predict(
            np.column_stack((x.flat, y.flat, z.flat)), eval_MSE=True)
        pred = pred.reshape(x.shape)
        mse = mse.reshape(x.shape)

        mlab.pipeline.volume(
            mlab.pipeline.scalar_field(x, y, z, pred), vmin=0.005)
        prediction_visualizer.finish()

        mse_visualizer = PlumeVisualizer()
        mse_visualizer.plot_uav_trajectories(trajectories)
        mlab.pipeline.volume(
            mlab.pipeline.scalar_field(x, y, z, mse))
        mse_visualizer.finish()

        mlab.sync_camera(prediction_visualizer.figure, mse_visualizer.figure)
        mlab.sync_camera(mse_visualizer.figure, prediction_visualizer.figure)

        mlab.show()
