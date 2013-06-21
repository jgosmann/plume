#!/usr/bin/env python

from qrsim.tcpclient import TCPClient
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tables

from sklearn import gaussian_process

parser = argparse.ArgumentParser()
parser.add_argument('conf', nargs=1, type=str)
parser.add_argument('output', nargs=1, type=str)
parser.add_argument('ip', nargs='?', type=str, default=['127.0.0.1'])
parser.add_argument('port', nargs='?', type=int, default=[10000])
args = parser.parse_args()

gp = gaussian_process.GaussianProcess(nugget=0.5)

conf = {}
execfile(args.conf[0], conf)


class TaskPlumeClient(TCPClient):
    def get_locations(self):
        return np.array(self.rpc('TASK', 'getLocations')).reshape((3, -1)).T

    def get_samples_per_location(self):
        return self.rpc('TASK', 'getSamplesPerLocation')[0]

    def get_plume_sensor_outputs(self):
        return self.rpc('PLATFORMS', 'getPlumeSensorOutput')

    def set_samples(self, samples):
        self.rpc('TASK', 'setSamples', samples.flat)

    def get_reward(self):
        return self.rpc('TASK', 'reward')[0]


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
        self._positions.append([(state.position,) for state in client.state])

    positions = property(lambda self: self._positions.read())


class TaskPlumeRecorder(GeneralRecorder):
    def __init__(self, fileh, client, predictor):
        GeneralRecorder.__init__(self, fileh, client)
        self.predictor = predictor

    def init(self, duration_in_steps):
        GeneralRecorder.init(self)
        self._locations = client.get_locations()
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
        if len(self.plume_measurements) > 1:
            print(len(self.positions.read()), len(self.plume_measurements))
            self.predictor.fit(
                self.positions.read().reshape(
                    (len(self.plume_measurements), -1)),
                self.plume_measurements.read().flat)
            samples = self.predictor.predict(self.locations)
            self.client.set_samples(samples)
            reward = self.client.get_reward()
        self._rewards.append([reward])

    plume_measurements = property(lambda self: self._plume_measurements.read())
    rewards = property(lambda self: self._rewards.read())


class Controller(object):
    def __init__(self, client, movement_behavior):
        self.client = client
        self.movement_behavior = movement_behavior
        self.recorders = []

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def init(self, taskfile, duration_in_steps):
        self.client.init(taskfile, False)
        for recorder in self.recorders:
            recorder.init(duration_in_steps)

    def reset(self):
        self.client.reset()

    def run(self, num_steps):
        for step in xrange(num_steps):
            # FIXME record controls
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state)
            self.client.step_vel(self.client.timestep, controls.U)
            for recorder in self.recorders:
                recorder.record()


with TaskPlumeClient() as client:
    client.connect_to(args.ip[0], args.port[0])
    duration_in_steps = 200
    movement_behavior = conf['behavior']
    controller = Controller(client, movement_behavior)
    with tables.open_file(args.output[0], 'w') as fileh:
        recorder = TaskPlumeRecorder(fileh, client, gp)
        controller.add_recorder(recorder)
        controller.init(
            'TaskPlumeSingleSourceGaussianDefaultControls', duration_in_steps)
        controller.run(duration_in_steps)

        gp.fit(recorder.positions[0, :, :], recorder.plume_measurements)
        locations = client.get_locations()
        samples = gp.predict(locations)
        client.set_samples(samples)
        rewards = client.get_reward()

        ep = np.linspace(-40, 40)
        x, y = np.meshgrid(ep, ep)
        print(x.shape)
        xy = np.hstack((
            np.atleast_2d(x.flat).T,
            np.atleast_2d(y.flat).T,
            np.repeat([[10]], np.prod(x.shape), 0)))
        print(xy.shape)
        pred = gp.predict(xy).reshape((ep.size, ep.size))
        print(pred.shape)
        plt.imshow(pred)

        plt.figure()
        plt.plot(recorder.rewards[2:])
        plt.show()
