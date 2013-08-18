import logging

from qrsim.tcpclient import ctrl_signal_dimensions, UAVControls
import numpy as np
import tables

logger = logging.getLogger(__name__)


class GeneralRecorder(object):
    def __init__(self, fileh, client, expected_steps=None):
        self.fileh = fileh
        self.client = client
        self.expected_steps = None

    def init(self, area):
        self._positions = self.fileh.create_earray(
            self.fileh.root, 'positions', tables.FloatAtom(),
            (self.client.numUAVs, 0, 3), expectedrows=self.expected_steps,
            title='Noiseless positions (numUAVs x timesteps x 3) of the UAVs '
            'over time.')
        self.fileh.create_array(
            self.fileh.root, 'sample_locations', self.client.get_locations(),
            title='Locations where prediction was requested '
            '(num locations x 3)')
        self.fileh.create_array(
            self.fileh.root, 'reference_samples',
            np.asarray(self.client.get_reference_samples()),
            title='Reference samples (num locations)')

        ogrid = [np.linspace(*dim, num=res) for dim, res in zip(
            area, [20, 20, 10])]
        x, y, z = (np.rollaxis(m, 1) for m in np.meshgrid(*ogrid))
        locations = np.column_stack((x.flat, y.flat, z.flat))
        self.fileh.create_array(
            self.fileh.root, 'gt_locations', locations,
            title='Locations where ground truth was evaluated '
            '(num locations x 3)')
        self.fileh.create_array(
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


class TargetsRecorder(object):
    def __init__(self, fileh, behavior, num_uavs, expected_steps=None):
        self.fileh = fileh
        self.behavior = behavior
        self.num_uavs = num_uavs
        self.expected_steps = expected_steps

    def init(self):
        self._targets = self.fileh.create_earray(
            self.fileh.root, 'targets', tables.FloatAtom(),
            (self.num_uavs, 0, 3), expectedrows=self.expected_steps,
            title='Target position for each UAV and timestep.')

    def record(self):
        self._targets.append(np.expand_dims(self.behavior.targets, 1))

    targets = property(lambda self: self._targets.read())


class TaskPlumeRecorder(GeneralRecorder):
    def __init__(self, fileh, client, predictor, expected_steps=None):
        GeneralRecorder.__init__(self, fileh, client, expected_steps)
        self.predictor = predictor

    def init(self, area):
        GeneralRecorder.init(self, area)
        self._locations = self.client.get_locations()
        self._plume_measurements = self.fileh.create_earray(
            self.fileh.root, 'plume_measurements', tables.FloatAtom(),
            (self.client.numUAVs, 0), expectedrows=self.expected_steps,
            title='Plume measurements (numUAVs x timesteps).')
        self._rewards = self.fileh.create_earray(
            self.fileh.root, 'rewards', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Total reward in each timestep.')
        self._rmse = self.fileh.create_earray(
            self.fileh.root, 'rmse', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Root mean square error in each time step.')
        self._wrmse = self.fileh.create_earray(
            self.fileh.root, 'wrmse', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Weighted root mean square error in each time step.')
        self._max_gt_value = self.gt_samples.max()

    def record(self):
        GeneralRecorder.record(self)
        self._record_plume_measurement()
        self._record_reward()
        self._record_xrmse()

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
            samples = np.maximum(0, self.predictor.predict(self.gt_locations))
        else:
            samples = np.zeros(len(self.gt_samples))
        se = (samples - self.gt_samples) ** 2
        wse = se * self.gt_samples / self._max_gt_value
        self._rmse.append([np.sqrt(np.sum(se) / len(samples))])
        self._wrmse.append([np.sqrt(np.sum(wse) / len(samples))])

    plume_measurements = property(lambda self: self._plume_measurements.read())
    rewards = property(lambda self: self._rewards.read())
    rmse = property(lambda self: self._rmse.read())
    wrmse = property(lambda self: self._wrmse.read())


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
        self.controls = self.fileh.create_table(
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
