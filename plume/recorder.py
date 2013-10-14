from importlib import import_module
import logging

from qrsim.tcpclient import ctrl_signal_dimensions, UAVControls
import numpy as np
import tables

from error_estimation import gen_probe_locations, RMSE, WRMSE
from nputil import meshgrid_nd
from prediction import ZeroPredictor

logger = logging.getLogger(__name__)


class GeneralRecorder(object):
    def __init__(self, fileh, client, expected_steps=None):
        self.fileh = fileh
        self.client = client
        self.expected_steps = None

    def init(self, area):
        self.area = np.asarray(area)

        self._positions = self.fileh.createEArray(
            self.fileh.root, 'positions', tables.FloatAtom(),
            (self.client.numUAVs, 0, 3), expectedrows=self.expected_steps,
            title='Noiseless positions (numUAVs x timesteps x 3) of the UAVs '
            'over time.')
        self.fileh.createArray(
            self.fileh.root, 'sample_locations', self.client.get_locations(),
            title='Locations where prediction was requested '
            '(num locations x 3)')
        self.fileh.createArray(
            self.fileh.root, 'reference_samples',
            np.asarray(self.client.get_reference_samples()),
            title='Reference samples (num locations)')

        ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
            area, [20, 20, 10])]
        x, y, z = meshgrid_nd(*ogrid)
        locations = np.column_stack((x.flat, y.flat, z.flat))
        self.fileh.createArray(
            self.fileh.root, 'gt_locations', locations,
            title='Locations where ground truth was evaluated '
            '(num locations x 3)')
        self.fileh.createArray(
            self.fileh.root, 'gt_samples',
            np.asarray(self.client.get_samples(locations)),
            title='Ground truth samples (num locations)')

    def record(self):
        self._positions.append(
            [(state.position,) for state in self.client.state])

    positions = property(lambda self: self._positions.read())
    sample_locations = property(
        lambda self: self.fileh.root.sample_locations.read())
    reference_samples = property(
        lambda self: self.fileh.root.reference_samples.read())
    gt_locations = property(
        lambda self: self.fileh.root.gt_locations.read())
    gt_samples = property(
        lambda self: self.fileh.root.gt_samples.read())


def store_obj(fileh, loc, obj):
    loc = fileh.getNode(loc)
    loc._v_attrs.class_name = obj.__class__.__name__
    loc._v_attrs.class_module = obj.__class__.__module__
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__'):
            group = fileh.createGroup(loc, k)
            store_obj(fileh, group, v)
        else:
            fileh.createArray(loc, k, v)


def load_obj(group):
    mod = import_module(group._v_attrs.class_module)
    cls = getattr(mod, group._v_attrs.class_name)
    obj = cls()
    for name, child in group._v_leaves.items():
        setattr(obj, name, child.read())
    for name, child in group._v_groups.items():
        setattr(obj, name, load_obj(child))
    return obj


class TargetsRecorder(object):
    def __init__(self, fileh, behavior, num_uavs, expected_steps=None):
        self.fileh = fileh
        self.behavior = behavior
        self.num_uavs = num_uavs
        self.expected_steps = expected_steps

    def init(self):
        self._targets = self.fileh.createEArray(
            self.fileh.root, 'targets', tables.FloatAtom(),
            (self.num_uavs, 0, 3), expectedrows=self.expected_steps,
            title='Target position for each UAV and timestep.')

    def record(self):
        if self.behavior.targets is None:
            self._targets.append(self.num_uavs * [[3 * [np.nan]]])
        else:
            self._targets.append(np.expand_dims(self.behavior.targets, 1))

    targets = property(lambda self: self._targets.read())


class TaskPlumeRecorder(GeneralRecorder):
    def __init__(self, fileh, client, predictor, expected_steps=None):
        GeneralRecorder.__init__(self, fileh, client, expected_steps)
        self.predictor = predictor

    def init(self, conf):
        area = conf['area']
        GeneralRecorder.init(self, area)
        self._locations = self.client.get_locations()
        self._plume_measurements = self.fileh.createEArray(
            self.fileh.root, 'plume_measurements', tables.FloatAtom(),
            (self.client.numUAVs, 0), expectedrows=self.expected_steps,
            title='Plume measurements (numUAVs x timesteps).')
        self._rewards = self.fileh.createEArray(
            self.fileh.root, 'rewards', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Total reward in each timestep.')
        self._rmse = self.fileh.createEArray(
            self.fileh.root, 'rmse', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Root mean square error in each time step.')
        self._wrmse = self.fileh.createEArray(
            self.fileh.root, 'wrmse', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Weighted root mean square error in each time step.')
        self._kernel_params = self.fileh.createEArray(
            self.fileh.root, 'kernel_params', tables.FloatAtom(),
            (0, self.predictor.kernel.params.size),
            expectedrows=self.expected_steps,
            title='Parameters of the kernel function in each time step.')
        self._log_likelihood = self.fileh.createEArray(
            self.fileh.root, 'log_likelihood', tables.FloatAtom(),
            (0,), expectedrows=self.expected_steps,
            title='Parameters of the kernel function in each time step.')
        self._max_gt_value = self.gt_samples.max()
        self.updates = -1

        self.test_x = gen_probe_locations(self.client, conf)
        self.test_y = np.asarray(self.client.get_samples(self.test_x))
        self._sources = self.fileh.createArray(
            '/', 'sources', self.client.get_sources())

    def record(self):
        GeneralRecorder.record(self)
        self._record_plume_measurement()
        self._record_kernel_params()
        if self.updates < self.predictor.updates:
            self._record_reward()
            self._record_xrmse()
            self._record_log_likelihood()
            self.updates = self.predictor.updates
        else:
            self._rerecord_last_error_values()

    def _rerecord_last_error_values(self):
        self._rewards.append([self._rewards[-1]])
        self._rmse.append([self._rmse[-1]])
        self._wrmse.append([self._wrmse[-1]])
        self._log_likelihood.append([self._log_likelihood[-1]])

    def _record_plume_measurement(self):
        measurement = np.atleast_2d(self.client.get_plume_sensor_outputs()).T
        self._plume_measurements.append(measurement)

    def _record_reward(self):
        if self.predictor.trained:
            samples = np.maximum(0, self.predictor.predict(self._locations))
        else:
            samples = np.zeros(len(self._locations))
        self.client.set_samples(samples)
        reward = self.client.get_reward()
        logger.info('Reward: %f, Sample bounds: (%f, %f)' % (
            reward, samples.min(), samples.max()))
        self._rewards.append([reward])

    def _record_xrmse(self):
        if self.predictor.trained:
            pred = self.predictor
        else:
            pred = ZeroPredictor()

        value, = RMSE()(pred, self.test_x, self.test_y)
        self._rmse.append([value])

        value2, = WRMSE()(pred, self.test_x, self.test_y)
        self._wrmse.append([value2])
        logger.info('Error: {}, {}'.format(value, value2))

    def _record_kernel_params(self):
        self._kernel_params.append([self.predictor.kernel.params])

    def _record_log_likelihood(self):
        if self.predictor.trained:
            self._log_likelihood.append(
                [-self.predictor.calc_neg_log_likelihood()])
        else:
            self._log_likelihood.append([-np.inf])

    plume_measurements = property(lambda self: self._plume_measurements.read())
    rewards = property(lambda self: self._rewards.read())
    rmse = property(lambda self: self._rmse.read())
    wrmse = property(lambda self: self._wrmse.read())
    kernel_params = property(lambda self: self._kernel_params.read())
    log_likelihood = property(lambda self: self._log_likelihood.read())


class ControlsRecorder(object):
    class Controls(tables.IsDescription):
        type = tables.EnumCol(
            ctrl_signal_dimensions.keys(),
            ctrl_signal_dimensions.keys()[0], tables.Int8Atom())
        dt = tables.FloatCol()
        uav = tables.Int8Col()
        U = tables.Float32Col(shape=max(ctrl_signal_dimensions.values()))

    def __init__(self, fileh, client, expected_steps=None):
        self.fileh = fileh
        self.client = client
        self.expected_steps = expected_steps
        self.controls = self.fileh.createTable(
            self.fileh.root, 'controls', self.Controls,
            title='Controls used for each step command.',
            expectedrows=self.expected_steps)

    def step(self, dt, controls):
        row = self.controls.row
        fillup = len(row['U']) - controls.U.shape[1]
        for uav in xrange(controls.num_uavs):
            row['type'] = self.Controls.columns['type'].enum[controls.type]
            row['dt'] = dt
            row['uav'] = uav
            row['U'] = np.concatenate(
                (controls.U[uav, :], np.repeat(0, fillup)))
            row.append()
        self.client.step(dt, controls)

    def step_ctrl(self, dt, ctrl):
        controls = UAVControls(len(ctrl), 'ctrl')
        controls.U[:, :] = ctrl
        self.step(dt, controls)

    def step_vel(self, dt, vels):
        controls = UAVControls(len(vels), 'vel')
        controls.U[:, :] = vels
        self.step(dt, controls)

    def step_wp(self, dt, WPs):
        controls = UAVControls(len(WPs), 'wp')
        controls.U[:, :] = WPs
        self.step(dt, controls)

    def __getattr__(self, name):
        return getattr(self.client, name)
