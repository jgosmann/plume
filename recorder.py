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
