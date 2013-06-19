#!/usr/bin/env python

from numpy.linalg import norm
from qrsim.tcpclient import TCPClient
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd


parser = argparse.ArgumentParser()
parser.add_argument('ip', nargs='?', type=str, default=['127.0.0.1'])
parser.add_argument('port', nargs='?', type=int, default=[10000])
args = parser.parse_args()

client = TCPClient()
client.connect_to(args.ip[0], args.port[0])
try:
    client.init('TaskPlumeSingleSourceGaussianDefaultControls', False)

    duration_in_steps = 200
    maxv = 3  # [m/s]
    U = np.zeros((client.numUAVs, 3))
    plume_measurements = np.empty(duration_in_steps)
    rewards = np.empty(duration_in_steps)
    locations = np.array(client.rpc('TASK', 'getLocations')).reshape((3, -1))
    samples_per_location = client.rpc('TASK', 'getSamplesPerLocation')[0]

    for step in xrange(duration_in_steps):
        for uav in xrange(client.numUAVs):
            # random velocity direction scaled by the max allowed velocity
            xy_vel = rnd.rand(2) - 0.5
            xy_vel /= norm(xy_vel)
            U[uav, :2] = 0.5 * maxv * xy_vel
            # if the uav is going astray we point it back to the center
            p = np.asarray(client.noisy_state[uav].position[:2])
            if norm(p) > 100:
                U[uav, :2] = -0.8 * maxv * p / norm(p)
            U[uav, 2] = max(-maxv, min(
                maxv,
                -0.005 * maxv * (
                    client.noisy_state[uav].z + 10.0) / client.timestep))
        client.step_vel(client.timestep, U)
        print(step, client.state[0].z, U[:, 2])
        plume_measurements[step] = client.rpc(
            'PLATFORMS', 'getPlumeSensorOutput')[0]

        samples = rnd.randn(samples_per_location, locations.shape[1])
        client.rpc('TASK', 'setSamples', samples.flat)
        rewards[step] = client.rpc('TASK', 'reward')[0]
finally:
    client.disconnect()

plt.plot(rewards)
plt.show()
