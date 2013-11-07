import logging

import numpy as np
from numpy.linalg import norm
from qrsim.tcpclient import UAVControls
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm as normdist

from nputil import GrowingArray, meshgrid_nd

logger = logging.getLogger(__name__)


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
        v = 0.025 * np.array([self.maxv, self.maxv, self.max_climb]) * \
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

    def _eval_common_terms(self, eval_fn, eval_derivative, *args, **kwargs):
        return self.fn._eval_common_terms(
            eval_fn, eval_derivative, *args, **kwargs)

    def _eval_fn(self, common_terms, *args, **kwargs):
        return -self.fn._eval_fn(common_terms, *args, **kwargs)

    def _eval_derivative(self, common_terms, *args, **kwargs):
        return -self.fn._eval_derivative(common_terms, *args, **kwargs)


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
            NegateFn(self.acquisition_fn).eval_with_derivative, x0,
            args=(noisy_states,), bounds=self.get_effective_area(),
            pgtol=1e-10, factr=1e2)

        return [x]

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])


class SurroundArea(TargetChooser):
    def __init__(self, area, margin):
        self.area = area
        self.margin = margin
        self.current_target = -1

    def new_targets(self, noisy_states):
        self.current_target += 1
        if self.current_target == 0:
            self._init_targets(noisy_states)

        if self.current_target >= len(self.targets):
            return None
        return [self.targets[self.current_target]]

    def _init_targets(self, noisy_states):
        ea = self.get_effective_area()
        height = np.mean(ea[2, :])
        distances = np.abs(ea - np.asarray(noisy_states[0].position)[:, None])
        nearest = np.unravel_index(np.argmin(distances[:2, :]), (2, 2))
        start = np.array(noisy_states[0].position[:2] + (height,))
        start[nearest[0]] = ea[nearest[0], nearest[1]]
        corners = np.array([
            [ea[0, 0], ea[1, 0], height],
            [ea[0, 1], ea[1, 0], height],
            [ea[0, 1], ea[1, 1], height],
            [ea[0, 0], ea[1, 1], height]])
        d = np.sum(np.square(corners - start), axis=1)
        nearest_corner = np.argmin(d)
        if d[(nearest_corner + 1) % len(d)] < d[nearest_corner - 1]:
            corners = np.roll(np.flipud(corners), nearest_corner + 1, 0)
        else:
            corners = np.roll(corners, -nearest_corner, 0)
        self.targets = np.vstack(([start], corners, [start]))

    def get_effective_area(self):
        return self.area + np.array([self.margin, -self.margin])


class ChainTargetChoosers(TargetChooser):
    def __init__(self, choosers):
        self.choosers = choosers
        self.current_chooser = 0

    def new_targets(self, noisy_states):
        if self.current_chooser >= len(self.choosers):
            return None

        targets = self.choosers[self.current_chooser].new_targets(noisy_states)
        while targets is None:
            self.current_chooser += 1
            if self.current_chooser >= len(self.choosers):
                return None
            targets = self.choosers[self.current_chooser].new_targets(
                noisy_states)

        return targets

    def get_effective_area(self):
        return self.choosers[self.current_chooser].get_effective_area()


class DUCBBased(DifferentiableFn):
    def __init__(self, predictor):
        self.predictor = predictor

    def _eval_common_terms(self, eval_fn, eval_derivative, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        if eval_derivative:
            pred, pred_derivative, mse, mse_derivative = \
                self.predictor.predict(
                    x, eval_MSE='err', eval_derivatives=True)
            pred_derivative = pred_derivative[:, 0, :]
        else:
            pred, mse = self.predictor.predict(
                x, eval_MSE='err', eval_derivatives=False)
            pred_derivative = mse_derivative = None
        sq_dist = np.maximum(0, -2 * np.dot(x, pos.T) + (
            np.sum(np.square(x), 1)[:, None] +
            np.sum(np.square(pos), 1)[None, :]))
        return (pred, pred_derivative, np.atleast_2d(mse).T, mse_derivative,
                sq_dist)


class DUCB(DUCBBased):
    def __init__(self, predictor, kappa, scaling, gamma):
        super(DUCB, self).__init__(predictor)
        self.kappa = kappa
        self.scaling = scaling
        self.gamma = gamma

    def _eval_fn(self, common_terms, x, noisy_states):
        pred, unused, mse, unused, sq_dist = common_terms
        ucb = pred + self._scaling() * (
            self.kappa * mse + self.gamma * np.sqrt(sq_dist))
        return np.asfortranarray(ucb)

    def _eval_derivative(self, common_terms, x, noisy_states):
        x = np.atleast_2d(x)
        pos = np.atleast_2d(noisy_states[0].position)
        unused, pred_derivative, mse, mse_derivative, sq_dist = common_terms
        ucb_derivative = pred_derivative + self._scaling() * (
            self.kappa * mse_derivative +
            self.gamma * (x - pos) / np.sqrt(np.maximum(sq_dist, 1e-60)))
        return np.asfortranarray(ucb_derivative)

    def _scaling(self):
        if self.scaling == 'auto':
            if hasattr(self.predictor, 'y_bv'):
                return self.predictor.y_bv.max()
            else:
                return self.predictor.y_train.data.max()
        else:
            return self.scaling


class PDUCB(DUCBBased):
    def __init__(self, predictor, kappa, scaling, gamma, epsilon):
        super(PDUCB, self).__init__(predictor)
        self.kappa = kappa
        self.scaling = scaling
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = self.epsilon

    def _eval_fn(self, common_terms, x, noisy_states):
        pred, unused, mse, unused, sq_dist = common_terms
        pred = np.maximum(0, pred)
        epred = np.exp(-pred / self.tau)
        ucb = np.log(pred + self.epsilon) * (1 - epred) + epred * np.log(
            self.epsilon) + self._scaling() * (
            self.kappa * mse + self.gamma * sq_dist)
        return np.asfortranarray(ucb)

    def _eval_derivative(self, common_terms, x, noisy_states):
        pred, pred_derivative, mse, mse_derivative, sq_dist = common_terms

        pred = np.maximum(0, pred)
        epred = np.exp(-pred / self.tau)
        ucb_derivative = pred_derivative * (
            (1.0 - epred) / (pred + self.epsilon) +
            epred / self.tau * (
                np.log(pred + self.epsilon) - np.log(self.epsilon))) + \
            self._scaling() * (
                self.kappa * mse_derivative +
                self.gamma * 2 * (x - noisy_states[0].position))
        return np.asfortranarray(ucb_derivative)

    def _scaling(self):
        if self.scaling == 'auto':
            if hasattr(self.predictor, 'y_bv'):
                max_val = self.predictor.y_bv.max()
            else:
                max_val = self.predictor.y_train.data.max()
            return np.log(max_val + self.epsilon) - np.log(self.epsilon)
        else:
            return self.scaling


class GO(DUCBBased):
    def __init__(self, predictor, gamma):
        super(GO, self).__init__(predictor)
        self.gamma = gamma

    def _eval_fn(self, common_terms, x, noisy_states):
        pred, unused, mse, unused, sq_dist = common_terms
        if hasattr(self.predictor, 'y_bv'):
            eta = self.predictor.y_bv.max()
        else:
            eta = self.predictor.y_train.data.max()

        std = np.sqrt(mse)
        go = eta + (pred - eta) * normdist.cdf((pred - eta) / std) + \
            std * normdist.pdf((pred - eta) / std) + self.gamma * sq_dist
        return np.asfortranarray(go)

    def _eval_derivative(self, common_terms, x, noisy_states):
        pred, pred_der, mse, mse_der, sq_dist = common_terms
        if hasattr(self.predictor, 'y_bv'):
            eta = self.predictor.y_bv.max()
        else:
            eta = self.predictor.y_train.data.max()
        std = np.sqrt(mse)
        std_der = mse_der / 2.0 / std
        a = (pred - eta) / std
        a_der = pred_der / std - std_der / mse * (pred - eta)
        cdf = normdist.cdf(a)
        cdf_der = a_der * 2.0 / np.sqrt(np.pi) * np.exp(-np.square(a))
        pdf = normdist.pdf(a)
        pdf_der = -a_der * a / np.sqrt(2.0 * np.pi) * np.exp(
            -np.square(a) / 2.0)

        return np.asfortranarray(
            pred_der * cdf + pred * cdf_der - eta * cdf_der +
            std_der * pdf + std * pdf_der +
            self.gamma * 2 * (x - noisy_states[0].position))


class FollowWaypoints(object):
    def __init__(
            self, target_chooser, target_precision, velocity_controller=None):
        self.target_chooser = target_chooser
        self.target_precision = target_precision
        if velocity_controller is None:
            self.velocity_controller = VelocityTowardsWaypointController(
                6, 6, self.target_chooser.get_effective_area())
        else:
            self.velocity_controller = velocity_controller
        self.observers = []
        self.num_step = 0

    def step(self, noisy_states):
        self.num_step += 1

        for observer in self.observers:
            observer.step(noisy_states)

        if self.velocity_controller.targets is None:
            update_targets = True
        else:
            dist_to_target = norm(
                self.velocity_controller.targets[0] - noisy_states[0].position)
            update_targets = dist_to_target < self.target_precision
        if self.num_step < 2:
            update_targets = False
        if update_targets:
            for observer in self.observers:
                observer.target_reached()
            nt = np.asarray(self.target_chooser.new_targets(noisy_states))
            if self.velocity_controller.targets is not None:
                change = np.sqrt(np.sum(np.square(
                    self.velocity_controller.targets - nt), axis=1))
                if np.all(change < self.target_precision):
                    raise GotStuckError()
            self.velocity_controller.targets = nt
            logger.info('Updated target {}'.format(
                self.velocity_controller.targets))


class GotStuckError(Exception):
    pass


class BatchPredictionUpdater(object):
    def __init__(self, predictor, plume_recorder):
        self.predictor = predictor
        self.plume_recorder = plume_recorder
        self.last_update = 0
        self.noisy_positions = None

    def step(self, noisy_states):
        if self.noisy_positions is None:
            self.noisy_positions = GrowingArray(
                (len(noisy_states), 3),
                expected_rows=self.plume_recorder.expected_steps)

        self.noisy_positions.append([s.position for s in noisy_states])
        can_do_first_training = self.plume_recorder.num_recorded > 1
        if not self.predictor.trained and can_do_first_training:
            self.update_prediction()

    def target_reached(self):
        if self.predictor.trained:
            self.update_prediction()

    def update_prediction(self):
        self.predictor.add_observations(
            self.noisy_positions.data[self.last_update:, 0, :],
            self.plume_recorder.plume_measurements[0, self.last_update:, None])
        self.last_update = len(self.noisy_positions.data)


class TargetMeasurementPredictionUpdater(object):
    def __init__(self, predictor, plume_recorder):
        self.predictor = predictor
        self.plume_recorder = plume_recorder
        self.noisy_positions = None

    def step(self, noisy_states):
        self.noisy_positions = np.array([s.position for s in noisy_states])

    def target_reached(self):
        self.predictor.add_observations(
            self.noisy_positions,
            self.plume_recorder.plume_measurements[:, -1, None])
