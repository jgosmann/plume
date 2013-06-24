from qrsim.tcpclient import ctrl_signal_dimensions, UAVControls
import numpy as np
import tables


class GeneralRecorder(object):
    def __init__(self, fileh, client, expected_steps=None):
        self.fileh = fileh
        self.client = client
        self.expected_steps = None

    def init(self):
        self._positions = self.fileh.create_earray(
            self.fileh.root, 'positions', tables.FloatAtom(),
            (self.client.numUAVs, 0, 3), expectedrows=self.expected_steps,
            title='Noiseless positions (numUAVs x timesteps x 3) of the UAVs '
            'over time.')

    def record(self):
        self._positions.append(
            [(state.position,) for state in self.client.state])

    positions = property(lambda self: self._positions.read())


class TaskPlumeRecorder(GeneralRecorder):
    def __init__(self, fileh, client, predictor, expected_steps=None):
        GeneralRecorder.__init__(self, fileh, client, expected_steps)
        self.predictor = predictor

    def init(self):
        GeneralRecorder.init(self)
        self._locations = self.client.get_locations()
        self._plume_measurements = self.fileh.create_earray(
            self.fileh.root, 'plume_measurements', tables.FloatAtom(),
            (self.client.numUAVs, 0), expectedrows=self.expected_steps,
            title='Plume measurements (numUAVs x timesteps).')
        self._rewards = self.fileh.create_earray(
            self.fileh.root, 'rewards', tables.FloatAtom(), (0,),
            expectedrows=self.expected_steps,
            title='Total reward in each timestep.')

    def record(self):
        GeneralRecorder.record(self)

        self._plume_measurements.append(np.atleast_2d(
            self.client.get_plume_sensor_outputs()).T)

        reward = -np.inf
        if len(self._plume_measurements) > 1:
            self.predictor.fit(
                self.positions.reshape((len(self._plume_measurements), -1)),
                self.plume_measurements.flat)
            samples = self.predictor.predict(self._locations)
            self.client.set_samples(samples)
            reward = self.client.get_reward()
        self._rewards.append([reward])

    plume_measurements = property(lambda self: self._plume_measurements.read())
    rewards = property(lambda self: self._rewards.read())


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

    def init(self, task, realtime=False):
        self.client.init(task, realtime)
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
