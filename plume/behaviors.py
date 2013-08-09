import numpy as np
from numpy.linalg import norm
import numpy.random as rnd
from qrsim.tcpclient import UAVControls
from scipy.optimize import fmin_l_bfgs_b

from datastructure import EnlargeableArray
from nputil import meshgrid_nd
from prediction import predict_on_volume


class VelocityTowardsWaypointController(object):
    def __init__(self, maxv, max_climb, area):
        self.maxv = maxv
        self.max_climb = max_climb
        self.max_speeds = np.array([self.maxv, self.maxv, self.max_climb])
        self.area = area

    def get_controls(self, noisy_states, targets):
        assert len(noisy_states) == len(targets)

        controls = UAVControls(len(noisy_states), 'vel')
        for uav in xrange(len(noisy_states)):
            v = 0.05 * np.array([self.maxv, self.maxv, self.max_climb]) * \
                (targets[uav] - noisy_states[uav].position)
            if norm(v[:2]) > self.maxv:
                v[:2] *= self.maxv / norm(v[:2])
            v[2] = np.clip(v[2], -self.max_climb, self.max_climb)

            outside_low = noisy_states[uav].position < self.area[:, 0]
            outside_high = noisy_states[uav].position > self.area[:, 1]
            v[outside_low] = self.max_speeds[outside_low]
            v[outside_high] = -self.max_speeds[outside_high]
            controls.U[uav, :] = v

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
        self._controller = VelocityTowardsWaypointController(3, 3, area)

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


class UCBBased(object):
    def __init__(
            self, margin, predictor, grid_resolution, area, target_precision,
            duration_in_steps=1000):
        self.margin = margin
        self.predictor = predictor
        self.grid_resolution = grid_resolution
        self.area = area
        self.target_precision = target_precision
        self.expected_steps = duration_in_steps
        self.step = 0
        self.last_prediction_update = 0
        self._controller = VelocityTowardsWaypointController(
            3, 3, self.get_effective_area())
        self.targets = None

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
            self.predictor.add_observations(
                self.positions.data[self.last_prediction_update:].reshape(
                    (-1, 3)),
                self.plume_measurements.data[self.last_prediction_update:])
            self.last_prediction_update = self.step

            ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
                self.get_effective_area(), self.grid_resolution)]
            x, y, z = meshgrid_nd(*ogrid)
            ducb, unused = self.calc_neg_ucb(
                np.column_stack((x.flat, y.flat, z.flat)), noisy_states)
            ducb *= -1
            wp_idx = np.unravel_index(np.argmax(ducb), x.shape)
            xs = np.array([x[wp_idx], y[wp_idx], z[wp_idx]])

            x, unused, unused = fmin_l_bfgs_b(
                lambda x, s: self.calc_neg_ucb(x, s), xs,
                args=(noisy_states,), bounds=self.get_effective_area())
            self.targets = np.array(len(noisy_states) * [x])

            pred, mse, (x, y, z) = predict_on_volume(
                self.predictor, self.get_effective_area(),
                self.grid_resolution)
            ducb = self.calc_neg_ucb(np.column_stack((x.flat, y.flat, z.flat)), noisy_states)
            dist = np.apply_along_axis(
                norm, 1, np.column_stack((x.flat, y.flat, z.flat)) -
                self.positions.data[-1]).reshape(x.shape)
            #ducb = np.log(pred + self.epsilon) + self.kappa * np.sqrt(mse) + self.gamma * dist ** 2
            ucb = np.log(np.maximum(0, pred) + self.epsilon) + self.kappa * np.sqrt(mse) + \
                self.gamma * dist ** 2

            wp_idx = np.unravel_index(np.argmax(ducb), x.shape)
            print(-self.calc_ducb(res.x, noisy_states)[0], res.x,
                  -self.calc_ducb(np.array([x[wp_idx], y[wp_idx], z[wp_idx]]), noisy_states)[0],
                  [x[wp_idx], y[wp_idx], z[wp_idx]])
            grad = -self.calc_ducb(res.x, noisy_states)[1]
            print(grad)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(
                ducb[:, :, wp_idx[2]],
                extent=self.get_effective_area()[:2].flatten(), origin='lower')
            plt.colorbar()
            plt.scatter(y[wp_idx], x[wp_idx], color='g')
            plt.scatter(res.x[1], res.x[0], color='r')
            plt.scatter(noisy_states[0].y, noisy_states[0].x)
            plt.subplot(2, 2, 2)
            res_idx = np.argmin(abs(z - res.x[2]))
            plt.imshow(
                ucb[:, :, res_idx],
                extent=self.get_effective_area()[:2].flatten(), origin='lower')
            plt.colorbar()
            plt.scatter(y[wp_idx], x[wp_idx], color='g')
            plt.scatter(res.x[1], res.x[0], color='r')
            plt.scatter(noisy_states[0].y, noisy_states[0].x)
            plt.plot([res.x[1], res.x[1] + grad[1]], [res.x[0], res.x[0] + grad[0]])
            plt.ioff()
            plt.show()

        return self._controller.get_controls(noisy_states, self.targets)

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])

    def calc_neg_ucb(self, x, noisy_states):
        raise NotImplementedError()


class DUCB(UCBBased):
    def __init__(
            self, margin, predictor, grid_resolution, area, kappa, gamma,
            target_precision, duration_in_steps=1000):
        super(DUCB, self).__init__(
            margin, predictor, grid_resolution, area, target_precision,
            duration_in_steps)
        self.kappa = kappa
        self.gamma = gamma

    def __repr__(self):
        return self.__class__.__name__ + '(margin=%(margin)r, ' \
            'predictor=%(predictor)r, grid_resolution=%(grid_resolution)r, ' \
            'area=%(area)r, kappa=%(kappa)r, gamma=%(gamma)r, ' \
            'target_precision=%(target_precision)r)' % self.__dict__

    def calc_neg_ucb(self, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        pred, pred_derivative, mse, mse_derivative = self.predictor.predict(
            x, eval_MSE=True, eval_derivatives=True)
        dist = np.sqrt(-2 * np.dot(x, pos.T) + (
            np.sum(np.square(x), 1)[:, None] +
            np.sum(np.square(pos), 1)[None, :]))
        ucb = pred + self.kappa * mse[:, None] + self.gamma * dist
        ucb_derivative = pred_derivative + self.kappa * mse_derivative + \
            self.gamma * (x - pos) / dist
        return -np.squeeze(ucb), -np.squeeze(ucb_derivative)


class PDUCB(UCBBased):
    def __init__(
            self, margin, predictor, grid_resolution, area, kappa, gamma,
            epsilon, target_precision, duration_in_steps=1000):
        super(PDUCB, self).__init__(
            margin, predictor, grid_resolution, area, target_precision,
            duration_in_steps)
        self.kappa = kappa
        self.gamma = gamma
        self.epsilon = epsilon

    def __repr__(self):
        return self.__class__.__name__ + '(margin=%(margin)r, ' \
            'predictor=%(predictor)r, grid_resolution=%(grid_resolution)r, ' \
            'area=%(area)r, kappa=%(kappa)r, gamma=%(gamma)r, ' \
            'epsilon=%(epsilon)r, ' \
            'target_precision=%(target_precision)r)' % self.__dict__

    def calc_neg_ucb(self, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        pred, pred_derivative, mse, mse_derivative = self.predictor.predict(
            x, eval_MSE=True, eval_derivatives=True)
        sq_dist = np.maximum(0, -2 * np.dot(x, pos.T) + (
            np.sum(np.square(x), 1)[:, None] +
            np.sum(np.square(pos), 1)[None, :]))
        ucb = np.log(np.maximum(0, pred) + self.epsilon) + \
            self.kappa * np.sqrt(mse)[:, None] + self.gamma * sq_dist
        ucb_derivative = pred_derivative / (pred + self.epsilon) + \
            self.kappa * mse_derivative * 0.5 / np.sqrt(mse)[:, None] + \
            self.gamma * 2 * np.sqrt(sq_dist)
        return -np.squeeze(ucb), -np.squeeze(ucb_derivative)


class NDUCB(UCBBased):
    def __init__(
            self, margin, predictor, grid_resolution, area, kappa, gamma,
            epsilon, target_precision, duration_in_steps=1000):
        super(NDUCB, self).__init__(
            margin, predictor, grid_resolution, area, target_precision,
            duration_in_steps)
        self.kappa = kappa
        self.gamma = gamma
        self.epsilon = epsilon

    def __repr__(self):
        return self.__class__.__name__ + '(margin=%(margin)r, ' \
            'predictor=%(predictor)r, grid_resolution=%(grid_resolution)r, ' \
            'area=%(area)r, kappa=%(kappa)r, gamma=%(gamma)r, ' \
            'epsilon=%(epsilon)r, ' \
            'target_precision=%(target_precision)r)' % self.__dict__

    def calc_neg_ucb(self, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        pred, pred_derivative, mse, mse_derivative = self.predictor.predict(
            x, eval_MSE=True, eval_derivatives=True)
        norm_factor = self.predictor.y_train.data.max()
        if norm_factor > 0:
            pred /= norm_factor
            pred_derivative /= norm_factor
        mse_norm_factor = self.predictor.kernel.variance + \
            self.predictor.noise_var
        mse /= mse_norm_factor
        mse_derivative /= norm_factor
        sq_dist = np.maximum(0, -2 * np.dot(x, pos.T) + (
            np.sum(np.square(x), 1)[:, None] +
            np.sum(np.square(pos), 1)[None, :]))
        ucb = np.maximum(0, pred) + \
            self.kappa * np.sqrt(mse)[:, None] + self.gamma * sq_dist
        ucb_derivative = pred_derivative + \
            self.kappa * mse_derivative * 0.5 / np.sqrt(mse)[:, None] + \
            self.gamma * 2 * np.sqrt(sq_dist)
        return -np.squeeze(ucb), -np.squeeze(ucb_derivative)
