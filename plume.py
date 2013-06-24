#!/usr/bin/env python

from behaviors import ToMaxVariance
from qrsim.tcpclient import TCPClient
from recorder import TaskPlumeRecorder
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
            recorder.init()

    def reset(self):
        self.client.reset()

    def run(self, num_steps):
        for step in xrange(num_steps):
            # FIXME record controls
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state)
            self.client.step(self.client.timestep, controls)
            for recorder in self.recorders:
                recorder.record()


with TaskPlumeClient() as client:
    client.connect_to(args.ip[0], args.port[0])
    duration_in_steps = 200
    #movement_behavior = conf['behavior']
    with tables.open_file(args.output[0], 'w') as fileh:
        recorder = TaskPlumeRecorder(fileh, client, gp, duration_in_steps)
        movement_behavior = ToMaxVariance(10, recorder)
        controller = Controller(client, movement_behavior)
        controller.add_recorder(recorder)
        controller.init(
            'TaskPlumeSingleSourceGaussianDefaultControls', duration_in_steps)
        controller.run(duration_in_steps)

        gp = gaussian_process.GaussianProcess(nugget=0.5)
        gp.fit(recorder.positions[0, :, :2], recorder.plume_measurements[0])
        #locations = client.get_locations()
        #samples = gp.predict(locations)
        #client.set_samples(samples)
        #rewards = client.get_reward()

        ep = np.linspace(-40, 40)
        x, y = np.meshgrid(ep, ep)
        print(x.shape)
        xy = np.hstack((
            np.atleast_2d(x.flat).T,
            np.atleast_2d(y.flat).T))  # ,
            #np.repeat([[10]], np.prod(x.shape), 0)))
        print(xy.shape)
        pred, mse = gp.predict(xy, eval_MSE=True)
        pred = pred.reshape((ep.size, ep.size))
        mse = mse.reshape((ep.size, ep.size))
        plt.imshow(pred)
        plt.title('pred')

        plt.figure()
        plt.imshow(mse)
        plt.title('mse')

        plt.figure()
        plt.plot(recorder.rewards[2:])
        plt.show()
