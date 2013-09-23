#!/usr/bin/env python

import argparse
import logging
import os.path
import sys

import pexpect
from qrsim.tcpclient import UAVControls
import tables

import behaviors
from config import load_config
from client import TaskPlumeClient
import prediction
from recorder import ControlsRecorder, TargetsRecorder, TaskPlumeRecorder

logger = logging.getLogger(__name__)


class FilterLevelAboveOrEqual(object):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


class Controller(object):
    def __init__(self, client, movement_behavior):
        self.client = client
        self.movement_behavior = movement_behavior
        self.recorders = []

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def init_new_sim(self, seed):
        self.client.reset_seed(seed)
        self.client.reset()
        # Ensure that all simulator variables have been set
        self.step_keeping_position()

    def run(self, num_steps):
        for step in xrange(num_steps):
            logger.info('Step %i', step + 1)
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state,
                self.client.get_plume_sensor_outputs())
            self.client.step(self.client.timestep, controls)
            for recorder in self.recorders:
                recorder.record()

    def step_keeping_position(self):
        c = UAVControls(self.client.numUAVs, 'vel')
        c.U.fill(0.0)
        self.client.step(self.client.timestep, c)


def do_simulation_run(trial, output_filename, conf, client):
    with tables.openFile(output_filename, 'w') as fileh:
        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)
        fileh.createArray(
            '/', 'repeat', [trial], title='Number of repeat run.')

        num_steps = conf['global_conf']['duration_in_steps']
        kernel = conf['kernel'](prediction)
        predictor = conf['predictor'](prediction, kernel)
        if 'bounds' in conf:
            predictor.bounds = conf['bounds']
        if 'priors' in conf:
            for i in range(len(conf['priors'])):
                predictor.priors[i] = conf['priors'][i](prediction)
        behavior = conf['behavior'](behaviors, predictor=predictor)

        client = ControlsRecorder(fileh, client, num_steps)
        controller = Controller(client, behavior)
        controller.init_new_sim(conf['seedlist'][trial])

        recorder = TaskPlumeRecorder(fileh, client, predictor, num_steps)
        recorder.init(conf['global_conf']['area'])
        controller.add_recorder(recorder)

        if hasattr(behavior, 'targets'):
            targets_recorder = TargetsRecorder(
                fileh, behavior, client.numUAVs, num_steps)
            targets_recorder.init()
            controller.add_recorder(targets_recorder)

        controller.run(num_steps)


class QRSimApplication(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '-q', '--quiet', action='store_true',
            help='Reduce output verbosity.')
        self.parser.add_argument(
            '-c', '--config', nargs=1, type=str, help='Configuration to load.')
        self.parser.add_argument(
            '-H', '--host', nargs=1, type=str,
            help='Host running QRSim. If not given it will be tried to launch '
            'an instance locally and connect to that.')
        self.parser.add_argument(
            '-P', '--port', nargs=1, type=int, default=[10000],
            help='Port on which QRSim instance is listening.')
        self.parser.add_argument(
            'output_dir', nargs=1, type=str, help='Output directory.')

    def main(self):
        args = self.parser.parse_args()

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(FilterLevelAboveOrEqual(logging.WARNING))
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        root_logger = logging.getLogger()
        if args.quiet:
            root_logger.setLevel(logging.WARNING)
        else:
            root_logger.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

        conf = load_config(args.config[0])

        with TaskPlumeClient() as client:
            if args.host is not None:
                client.connect_to(args.host[0], args.port[0])
            else:
                qrsim = pexpect.spawn(
                    'matlab -nodesktop -nosplash -r "'
                    "cd(fileparts(which('QRSimTCPServer')));"
                    "QRSimTCPServer(0);"
                    'quit;"',
                    timeout=120)
                qrsim.logfile = sys.stdout
                qrsim.expect(r'Listening on port: (\d+)')
                port = int(qrsim.match.group(1))
                client.connect_to('127.0.0.1', port)
            client.init(conf['task'])

            return self._run_application(args, conf, client)

    def _run_application(self, args, conf, client):
        raise NotImplementedError()


class Plume(QRSimApplication):
    def __init__(self):
        super(Plume, self).__init__()
        self.parser.add_argument(
            '-o', '--output', nargs=1, type=str, default=['plume'],
            help='Output file name without extension (will be add '
            'automatically).')

    def _run_application(self, args, conf, client):
        clean = True
        for i in xrange(conf['repeats']):
            try:
                output_filename = os.path.join(
                    args.output_dir[0], args.output[0] + '.%i.h5' % i)
                do_simulation_run(i, output_filename, conf, client)
            except:
                logger.exception('Repeat failed.')
                clean = False
        return clean


if __name__ == '__main__':
    if Plume().main():
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_SOFTWARE)
