from qrsim.tcpclient import UAVControls
import numpy as np
from numpy.linalg import norm
import numpy.random as rnd

from sklearn import gaussian_process


class RandomMovement(object):
    def __init__(self, maxv, height):
        self.maxv = maxv
        self.height = height

    def get_controls(self, noisy_states):
        controls = UAVControls(len(noisy_states), 'vel')
        for uav in xrange(len(noisy_states)):
            # random velocity direction scaled by the max allowed velocity
            xy_vel = rnd.rand(2) - 0.5
            xy_vel /= norm(xy_vel)
            controls.U[uav, :2] = 0.5 * self.maxv * xy_vel
            # if the uav is going astray we point it back to the center
            p = np.asarray(noisy_states[uav].position[:2])
            if norm(p) > 100:
                controls.U[uav, :2] = -0.8 * self.maxv * p / norm(p)
            # control height
            controls.U[uav, 2] = max(-self.maxv, min(
                self.maxv,
                -0.25 * self.maxv * (noisy_states[uav].z + self.height)))
            print(noisy_states[uav].z, self.height, controls.U[uav, 2])
        return controls


class ToMaxVariance(object):
    def __init__(self, height, recorder):
        self.height = height
        self.recorder = recorder

    def get_controls(self, noisy_states):
        if len(self.recorder.plume_measurements[0]) < 2:
            b = RandomMovement(3, self.height)
            return b.get_controls(noisy_states)

        controls = UAVControls(len(noisy_states), 'wp')
        for uav in xrange(len(noisy_states)):
            gp = gaussian_process.GaussianProcess(nugget=0.5)
            gp.fit(
                self.recorder.positions[uav, :, :2],
                self.recorder.plume_measurements[uav])
            ep = np.linspace(-40, 40)
            x, y = np.meshgrid(ep, ep)
            xy = np.hstack((
                np.atleast_2d(x.flat).T,
                np.atleast_2d(y.flat).T))  # ,
                #np.repeat([[10]], np.prod(x.shape), 0)))
            unused, mse = gp.predict(xy, eval_MSE=True)
            wp_idx = np.unravel_index(np.argmax(mse), (ep.size, ep.size))

            controls.U[uav, 0] = ep[wp_idx[0]]
            controls.U[uav, 1] = ep[wp_idx[1]]
            controls.U[uav, 2] = -self.height
            controls.U[uav, 3] = noisy_states[uav].phi
        return controls
