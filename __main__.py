#!/usr/bin/env python

from behaviors import ToMaxVariance
from configloader import load_config
from plume import TaskPlumeClient
from qrsim.tcpclient import UAVControls
from recorder import ControlsRecorder, TaskPlumeRecorder
import argparse
import os.path
import tables

from sklearn import gaussian_process


class Controller(object):
    def __init__(self, client, movement_behavior):
        self.client = client
        self.movement_behavior = movement_behavior
        self.recorders = []

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def init(self, taskfile, duration_in_steps):
        self.client.init(taskfile, False)
        # Ensure that all simulator variables have been set
        self.step_keeping_position()
        for recorder in self.recorders:
            recorder.init()

    def reset(self):
        self.client.reset()

    def run(self, num_steps):
        for step in xrange(num_steps):
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state, client.get_plume_sensor_outputs())
            print(
                self.client.noisy_state[0].position, self.client.state[0].z,
                controls.U)
            self.client.step(self.client.timestep, controls)
            for recorder in self.recorders:
                recorder.record()

    def step_keeping_position(self):
        c = UAVControls(self.client.numUAVs, 'wp')
        c.U[:, :3] = [s.position for s in self.client.state]
        c.U[:, 3] = [s.psi for s in self.client.state]
        self.client.step(self.client.timestep, c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', nargs=1, type=str)
    parser.add_argument('output', nargs=1, type=str)
    parser.add_argument('ip', nargs='?', type=str, default=['127.0.0.1'])
    parser.add_argument('port', nargs='?', type=int, default=[10000])
    args = parser.parse_args()

    conf = load_config(args.conf[0])

    gp = gaussian_process.GaussianProcess(nugget=0.5)

    output_filename = os.path.join(args.output[0], 'plume.h5')

    with tables.open_file(output_filename, 'w') as fileh:
        with TaskPlumeClient() as client:
            client = ControlsRecorder(fileh, client, conf['duration_in_steps'])
            client.connect_to(args.ip[0], args.port[0])
            #movement_behavior = conf['behavior']
            recorder = TaskPlumeRecorder(
                fileh, client, gp, conf['duration_in_steps'])
            movement_behavior = ToMaxVariance(
                -40, conf['area'], conf['duration_in_steps'])

            controller = Controller(client, movement_behavior)
            controller.add_recorder(recorder)
            controller.init(
                'TaskPlumeSingleSourceGaussianDefaultControls',
                conf['duration_in_steps'])
            controller.run(conf['duration_in_steps'])
