from datastructure import EnlargeableArray
from qrsim.tcpclient import UAVControls
import numpy as np
from numpy.linalg import norm
import numpy.random as rnd

from sklearn import gaussian_process


class RandomMovement(object):
    def __init__(self, maxv, height):
        self.maxv = maxv
        self.height = height

    def get_controls(self, noisy_states, plume_measurement):
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
    def __init__(self, height, area, expected_steps=1000):
        self.height = height
        self.area = area
        self.expected_steps = expected_steps
        self.step = 0

    def get_controls(self, noisy_states, plume_measurement):
        if self.step == 0:
            self.positions = EnlargeableArray(
                (len(noisy_states), 3), self.expected_steps)
            self.plume_measurements = EnlargeableArray(
                (len(noisy_states),), self.expected_steps)
            # FIXME Do first step in controller?
            b = RandomMovement(3, self.height)
            return b.get_controls(noisy_states, plume_measurement)

        self.positions.append([s.position for s in noisy_states])
        self.plume_measurements.append(plume_measurement)

        gp = gaussian_process.GaussianProcess(nugget=0.5)
        gp.fit(
            self.positions.data.reshape((-1, 3)),
            self.plume_measurements.flat)

        x, y = np.meshgrid(np.arange(*self.area[0]), np.arange(*self.area[1]))
        z = np.empty_like(x)
        z.fill(self.height)
        unused, mse = gp.predict(
            np.dstack((x, y, z)).reshape((-1, 3)), eval_MSE=True)
        mse = mse.reshape(x.shape)
        wp_idx = np.unravel_index(np.argmax(mse), x.shape)

        controls = UAVControls(len(noisy_states), 'wp')
        for uav in xrange(len(noisy_states)):
            #if noisy_states[uav].z > -8:
                #controls.U[uav, :2] = noisy_states[uav].position[:2]
                #controls.U[uav, 2] = -self.height
                #controls.U[uav, 3] = noisy_states[uav].psi
                #break

            controls.U[uav, 0] = x[wp_idx]
            controls.U[uav, 1] = y[wp_idx]
            controls.U[uav, 2] = -self.height
            # FIXME what is the right angle here?
            controls.U[uav, 3] = noisy_states[uav].psi
            print(controls.U)
        return controls
