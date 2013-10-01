import numpy as np
from numpy.linalg import norm
from qrsim.tcpclient import UAVControls
from scipy.optimize import fmin_l_bfgs_b

from nputil import meshgrid_nd


class VelocityTowardsWaypointController(object):
    def __init__(self, maxv, max_climb, area):
        self.maxv = maxv
        self.max_climb = max_climb
        self.area = area
        self.targets = None

    def get_max_speeds(self):
        return np.array([self.maxv, self.maxv, self.max_climb])

    def set_max_speeds(self, value):
        assert len(value) == 3
        assert value[0] == value[1]
        self.maxv = value[0]
        self.max_climb = value[2]

    max_speeds = property(get_max_speeds, set_max_speeds)

    def get_controls(self, noisy_states):
        controls = UAVControls(len(noisy_states), 'vel')

        if self.targets is None:
            controls.U.fill(0)
        else:
            assert len(noisy_states) == len(self.targets)
            for uav in xrange(len(noisy_states)):
                controls.U[uav, :] = self._get_velocities(
                    noisy_states[uav].position, self.targets[uav])
        return controls

    def _get_velocities(self, current_pos, to):
        v = 0.05 * np.array([self.maxv, self.maxv, self.max_climb]) * \
            (to - current_pos)
        if norm(v[:2]) > self.maxv:
            v[:2] *= self.maxv / norm(v[:2])
        v[2] = np.clip(v[2], -self.max_climb, self.max_climb)

        outside_low = current_pos < self.area[:, 0]
        outside_high = current_pos > self.area[:, 1]
        v[outside_low] = self.max_speeds[outside_low]
        v[outside_high] = -self.max_speeds[outside_high]
        return v


class DifferentiableFn(object):
    def _eval_common_terms(self, eval_fn, eval_derivative, *args, **kwargs):
        pass

    def _eval_fn(self, common_terms, *args, **kwargs):
        raise NotImplementedError()

    def _eval_derivative(self, common_terms, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self._eval_fn(
            self._eval_common_terms(True, False, *args, **kwargs),
            *args, **kwargs)

    def eval_with_derivative(self, *args, **kwargs):
        common_terms = self._eval_common_terms(True, True, *args, **kwargs)
        return self._eval_fn(common_terms, *args, **kwargs), \
            self._eval_derivative(common_terms, *args, **kwargs)


class NegateFn(DifferentiableFn):
    def __init__(self, fn):
        self.fn = fn

    def _eval_fn(self, common_terms, *args, **kwargs):
        return -super(NegateFn, self)._eval_fn(common_terms, *args, **kwargs)

    def _eval_derivative(self, common_terms, *args, **kwargs):
        return -super(NegateFn, self)._eval_derivative(
            common_terms, *args, **kwargs)


class TargetChooser(object):
    def new_targets(self, noisy_states):
        raise NotImplementedError()


class AcquisitionFnTargetChooser(TargetChooser):
    def __init__(self, acquisition_fn, area, margin, grid_resolution):
        self.acquisition_fn = acquisition_fn
        self.area = area
        self.margin = margin
        self.grid_resolution = grid_resolution

    def new_targets(self, noisy_states):
        ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
            self.get_effective_area(), self.grid_resolution)]
        x, y, z = meshgrid_nd(*ogrid)
        acq = self.acquisition_fn(
            np.column_stack((x.flat, y.flat, z.flat)), noisy_states)
        max_idx = np.unravel_index(np.argmax(acq), x.shape)
        x0 = np.array([x[max_idx], y[max_idx], z[max_idx]])

        x, unused, unused = fmin_l_bfgs_b(
            NegateFn(self.acquisition_fn.eval_with_derivative), x0,
            args=(noisy_states,), bounds=self.get_effective_area())

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])


class DUCBBased(DifferentiableFn):
    def __init__(self, predictor):
        self.predictor = predictor

    def _eval_common_terms(self, eval_fn, eval_derivative, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        if eval_derivative:
            pred, pred_derivative, mse, mse_derivative = \
                self.predictor.predict(x, eval_MSE=True, eval_derivatives=True)
        else:
            pred, mse = self.predictor.predict(
                x, eval_MSE=True, eval_derivatives=False)
            pred_derivative = mse_derivative = None
        sq_dist = -2 * np.dot(x, pos.T) + (
            np.sum(np.square(x), 1)[:, None] +
            np.sum(np.square(pos), 1)[None, :])
        return (pred, pred_derivative, mse, mse_derivative, sq_dist)


class DUCB(DUCBBased):
    def __init__(self, predictor, kappa, gamma):
        super(DUCB, self).__init__(predictor)
        self.kappa = kappa
        self.gamma = gamma

    def _eval_fn(self, common_terms, x, noisy_states):
        pred, unused, mse, unused, sq_dist = common_terms
        ucb = pred + self.kappa * mse[:, None] + self.gamma * np.sqrt(sq_dist)
        return np.squeeze(ucb)

    def _eval_derivative(self, common_terms, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        unused, pred_derivative, mse, mse_derivative, dist = common_terms
        ucb_derivative = pred_derivative + self.kappa * mse_derivative + \
            self.gamma * (x - pos) / np.sqrt(dist)
        return np.squeeze(ucb_derivative)


class PDUCB(DUCBBased):
    def __init__(self, predictor, kappa, gamma, epsilon):
        super(DUCBBased, self).__init__(predictor)
        self.kappa = kappa
        self.gamma = gamma
        self.epsilon = epsilon

    def _eval_fn(self, common_terms, x, noisy_states):
        pred, unused, mse, unused, sq_dist = common_terms
        ucb = np.log(np.maximum(0, pred) + self.epsilon) + \
            self.kappa * np.sqrt(mse)[:, None] + self.gamma * sq_dist
        return np.squeeze(ucb)

    def _eval_derivative(self, common_terms, x, noisy_states):
        pred, pred_derivative, mse, mse_derivative, sq_dist = common_terms
        ucb_derivative = pred_derivative / (pred + self.epsilon) + \
            self.kappa * mse_derivative * 0.5 / np.sqrt(mse)[:, None] + \
            self.gamma * 2 * np.sqrt(sq_dist)
        return np.squeeze(ucb_derivative)


class FollowWaypoints(object):
    def __init__(
            self, target_chooser, target_precision, velocity_controller=None):
        self.target_chooser = target_chooser
        self.target_precision = target_precision
        if velocity_controller is None:
            self.velocity_controller = VelocityTowardsWaypointController(
                6, 6, self.target_chooser.get_effective_area())

    def step(self, noisy_states):
        if self.velocity_controller.targets is None:
            update_targets = True
        else:
            dist_to_target = norm(
                self.velocity_controller.targets - noisy_states[0].position)
            update_targets = dist_to_target < self.target_precision
        if update_targets:
            self.velocity_controller.targets = self.target_chooser.new_targets(
                noisy_states)
