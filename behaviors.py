from datastructure import EnlargeableArray
from prediction import predict_on_volume
from qrsim.tcpclient import UAVControls
import numpy as np
from numpy.linalg import norm
import numpy.random as rnd


class VelocityTowardsWaypointController(object):
    def __init__(self, maxv, max_climb):
        self.maxv = maxv
        self.max_climb = max_climb

    def get_controls(self, noisy_states, targets):
        assert len(noisy_states) == len(targets)

        controls = UAVControls(len(noisy_states), 'vel')
        for uav in xrange(len(noisy_states)):
            phi = noisy_states[uav].phi
            theta = noisy_states[uav].theta
            psi = noisy_states[uav].psi

            X = np.matrix(
                [[1, 0, 0],
                 [0, np.cos(phi), -np.sin(phi)],
                 [0, np.sin(phi), np.cos(phi)]])
            Y = np.matrix(
                [[np.cos(theta), 0, np.sin(theta)],
                 [0, 1, 0],
                 [-np.sin(theta), 0, np.cos(theta)]])
            Z = np.matrix(
                [[np.cos(psi), -np.sin(psi), 0],
                 [np.sin(psi), np.cos(psi), 0],
                 [0, 0, 1]])

            world_v = 0.25 * np.array(
                [self.maxv, self.maxv, self.max_climb]) * \
                (targets[uav] - noisy_states[uav].position)
            if norm(world_v[:2]) > self.maxv:
                world_v[:2] *= self.maxv / norm(world_v[:2])
            world_v[2] = np.clip(world_v[2], -self.max_climb, self.max_climb)
            controls.U[uav, :] = np.dot(Z * Y * X, world_v)

        return controls


class RandomMovement(object):
    def __init__(self, maxv, height):
        self.maxv = maxv
        self.height = height

    def get_controls(self, noisy_states, plume_measurement):
        controls = UAVControls(len(noisy_states), 'vel')
        for uav in xrange(len(noisy_states)):
            # random velocity direction scaled by the max allowed velocity
            xy_vel = rnd.rand(2) - 0.5
            if norm(xy_vel) != 0:
                xy_vel /= norm(xy_vel)
            controls.U[uav, :2] = 0.5 * self.maxv * xy_vel
            # if the uav is going astray we point it back to the center
            p = np.asarray(noisy_states[uav].position[:2])
            if norm(p) > 100:
                controls.U[uav, :2] = -0.8 * self.maxv * p / norm(p)
            # control height
            controls.U[uav, 2] = max(-self.maxv, min(
                self.maxv,
                0.25 * self.maxv * (self.height - noisy_states[uav].z)))
        return controls


class ToMaxVariance(object):
    def __init__(
            self, margin, predictor, grid_resolution, area,
            duration_in_steps=1000):
        self.margin = margin
        self.predictor = predictor
        self.grid_resolution = grid_resolution
        self.area = area
        self.expected_steps = duration_in_steps
        self.step = 0
        self._controller = VelocityTowardsWaypointController(3, 3)

    def get_controls(self, noisy_states, plume_measurement):
        if self.step == 0:
            self.positions = EnlargeableArray(
                (len(noisy_states), 3), self.expected_steps)
            self.plume_measurements = EnlargeableArray(
                (len(noisy_states),), self.expected_steps)

        self.positions.append([s.position for s in noisy_states])
        self.plume_measurements.append(plume_measurement)
        self.step += 1

        if self.positions.data.size // 3 < 2:
            b = RandomMovement(3, np.mean(self.get_effective_area()[2]))
            return b.get_controls(noisy_states, plume_measurement)

        # FIXME remove or do only for scikit learn
        #predictor = sklearn.base.clone(self.predictor)
        self.predictor.fit(
            self.positions.data.reshape((-1, 3)),
            self.plume_measurements.data.flatten())
        unused, mse, (x, y, z) = predict_on_volume(
            self.predictor, self.get_effective_area(), self.grid_resolution)
        wp_idx = np.unravel_index(np.argmax(mse), x.shape)

        targets = np.array(
            len(noisy_states) * [[x[wp_idx], y[wp_idx], z[wp_idx]]])
        return self._controller.get_controls(noisy_states, targets)

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])


class DUCB(object):
    def __init__(
            self, margin, predictor, grid_resolution, area, kappa, gamma,
            target_precision, duration_in_steps=1000):
        self.margin = margin
        self.predictor = predictor
        self.grid_resolution = grid_resolution
        self.area = area
        self.kappa = kappa
        self.gamma = gamma
        self.target_precision = target_precision
        self.expected_steps = duration_in_steps
        self.step = 0
        self._controller = VelocityTowardsWaypointController(3, 3)
        self.targets = None

    def __repr__(self):
        return self.__class__.__name__ + '(margin=%(margin)r, ' \
            'predictor=%(predictor)r, grid_resolution=%(grid_resolution)r, ' \
            'area=%(area)r, kappa=%(kappa)r, gamma=%(gamma)r, ' \
            'target_precision=%(target_precision)r)' % self.__dict__

    def get_controls(self, noisy_states, plume_measurement):
        if self.step == 0:
            self.positions = EnlargeableArray(
                (len(noisy_states), 3), self.expected_steps)
            self.plume_measurements = EnlargeableArray(
                (len(noisy_states),), self.expected_steps)

        self.positions.append([s.position for s in noisy_states])
        self.plume_measurements.append(plume_measurement)
        self.step += 1

        if self.positions.data.size // 3 < 2:
            self.targets = np.array([s.position for s in noisy_states])
            controls = UAVControls(len(noisy_states), 'vel')
            controls.U.fill(0.0)
            return controls

        if norm(self.targets - noisy_states[0].position) < \
                self.target_precision:
            # FIXME remove or do only for scikit learn
            #predictor = sklearn.base.clone(self.predictor)
            self.predictor.fit(
                self.positions.data.reshape((-1, 3)),
                self.plume_measurements.data.flatten())
            pred, mse, (x, y, z) = predict_on_volume(
                self.predictor, self.get_effective_area(),
                self.grid_resolution)
            dist = np.apply_along_axis(
                norm, 1, np.column_stack((x.flat, y.flat, z.flat)) -
                self.positions.data[-1]).reshape(x.shape)
            ducb = 0.15e-12 * np.log(pred + 1e-30) + \
                self.kappa * np.sqrt(mse) + self.gamma * dist ** 2

            wp_idx = np.unravel_index(np.argmax(ducb), x.shape)

            self.targets = np.array(
                len(noisy_states) * [[x[wp_idx], y[wp_idx], z[wp_idx]]])

        return self._controller.get_controls(noisy_states, self.targets)

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])
