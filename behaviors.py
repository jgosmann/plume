from qrsim.tcpclient import UAVControls
import numpy as np
from numpy.linalg import norm
import numpy.random as rnd


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
