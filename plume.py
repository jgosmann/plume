#!/usr/bin/env python

from numpy.linalg import norm
from qrsim.tcpclient import TCPClient
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from sklearn import gaussian_process

parser = argparse.ArgumentParser()
parser.add_argument('ip', nargs='?', type=str, default=['127.0.0.1'])
parser.add_argument('port', nargs='?', type=int, default=[10000])
args = parser.parse_args()

gp = gaussian_process.GaussianProcess(nugget=0.5)


# FIXME move to py-qrsim-tcpclient and add method taking this object as
# argument and deciding that way which step method to call
class UavControls(object):
    def __init__(self, num_uavs, type):
        dim_for_type = {'ctrl': 5, 'vel': 3, 'wp': 4}
        if not type in dim_for_type:
            raise ValueError('Not a valid UAV control type.')

        self._num_uavs = num_uavs
        self._type = type
        self._U = np.empty((num_uavs, dim_for_type[type]))

    num_uavs = property(lambda self: self._num_uavs)
    type = property(lambda self: self._type)
    U = property(lambda self: self._U)


class RandomMovement(object):
    def __init__(self, maxv, height):
        self.maxv = maxv
        self.height = height

    def get_controls(self, noisy_states):
        controls = UavControls(len(noisy_states), 'vel')
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
    def __init__(self, client):
        self.client = client

    def init(self, duration_in_steps):
        self.positions = np.empty((client.numUAVs, duration_in_steps, 3))

    def record(self, step):
        self.positions[:, step, :] = [state.position for state in client.state]


class TaskPlumeRecorder(GeneralRecorder):
    def __init__(self, client, predictor):
        GeneralRecorder.__init__(self, client)
        self.predictor = predictor

    def init(self, duration_in_steps):
        GeneralRecorder.init(self, duration_in_steps)
        self.locations = client.get_locations()
        self.plume_measurements = np.empty(duration_in_steps)
        self.rewards = np.empty(duration_in_steps)

    def record(self, step):
        GeneralRecorder.record(self, step)
        self.plume_measurements[step] = \
            self.client.get_plume_sensor_outputs()[0]
        if step > 0:
            self.predictor.fit(
                self.positions[0, :(step + 1), :],
                self.plume_measurements[:(step + 1)])
            samples = self.predictor.predict(self.locations)
            self.client.set_samples(samples)
            self.rewards[step] = self.client.get_reward()
        else:
            self.rewards[step] = -np.inf


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
                recorder.record(step)


client = TaskPlumeClient()
client.connect_to(args.ip[0], args.port[0])
try:
    duration_in_steps = 200
    movement_behavior = RandomMovement(3, 10)
    controller = Controller(client, movement_behavior)
    recorder = TaskPlumeRecorder(client, gp)
    controller.add_recorder(recorder)
    controller.init(
        'TaskPlumeSingleSourceGaussianDefaultControls', duration_in_steps)
    controller.run(duration_in_steps)

    gp.fit(recorder.positions[0, :, :], recorder.plume_measurements)
    locations = client.get_locations()
    samples = gp.predict(locations)
    client.set_samples(samples)
    rewards = client.get_reward()
finally:
    client.disconnect()

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
