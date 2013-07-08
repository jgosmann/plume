#!/usr/bin/env python

from configloader import load_config
from plume import TaskPlumeClient
from qrsim.tcpclient import UAVControls
from recorder import ControlsRecorder, TaskPlumeRecorder
import argparse
import logging
import os.path
import tables

from sklearn import gaussian_process

logger = logging.getLogger(__name__)


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
            logger.info('Step %i', step + 1)
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state, client.get_plume_sensor_outputs())
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

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    conf = load_config(args.conf[0])

    gp = gaussian_process.GaussianProcess(nugget=0.5)

    output_filename = os.path.join(args.output[0], 'plume.h5')

    with tables.open_file(output_filename, 'w') as fileh:
        fileh.root.attrs = conf
        num_steps = conf['global_conf']['duration_in_steps']

        with TaskPlumeClient() as client:
            client = ControlsRecorder(fileh, client, num_steps)
            client.connect_to(args.ip[0], args.port[0])
            recorder = TaskPlumeRecorder(fileh, client, gp, num_steps)

            controller = Controller(client, conf['behavior'])
            controller.add_recorder(recorder)
            controller.init(
                'TaskPlumeSingleSourceGaussianDefaultControls', num_steps)
            controller.run(num_steps)
