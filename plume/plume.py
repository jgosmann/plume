#!/usr/bin/env python

import argparse
import logging
import os.path
import sys

import numpy as np
import numpy.random as rnd
import pexpect
from qrsim.tcpclient import UAVControls
from scipy.stats import gaussian_kde
import tables

import behaviors
from config import instantiate, load_config
from client import TaskPlumeClient
from recorder import ControlsRecorder, ErrorRecorder, store_obj, \
    TargetsRecorder, TaskPlumeRecorder

logger = logging.getLogger(__name__)


class FilterLevelAboveOrEqual(object):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


class Controller(object):
    def __init__(self, client, controller, movement_behavior):
        self.client = client
        self.controller = controller
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
            for recorder in self.recorders:
                recorder.record()
            self.controller.step(self.client.noisy_state)
            controls = self.movement_behavior.get_controls(
                self.client.noisy_state)
            self.client.step(self.client.timestep, controls)

    def step_keeping_position(self):
        c = UAVControls(self.client.numUAVs, 'vel')
        c.U.fill(0.0)
        self.client.step(self.client.timestep, c)


def do_simulation_run(trial, output_filename, conf, client):
    rnd.seed(conf['pyseedlist'][trial])
    with tables.openFile(output_filename, 'w') as fileh:
        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)
        fileh.createArray(
            '/', 'repeat', [trial], title='Number of repeat run.')

        num_steps = conf['duration_in_steps']
        kernel = instantiate(*conf['kernel'])
        predictor = instantiate(*conf['predictor'], prefix_args=(kernel,))
        if 'bounds' in conf:
            predictor.bounds = conf['bounds']
        if 'priors' in conf:
            for i in range(len(conf['priors'])):
                predictor.priors[i] = instantiate(*conf['priors'][i])

        recorder = TaskPlumeRecorder(fileh, client, predictor, num_steps)
        err_recorder = ErrorRecorder(fileh, client, predictor, num_steps)

        updater = instantiate(
            *conf['updater'], predictor=predictor, plume_recorder=recorder)

        acq_behavior = behaviors.AcquisitionFnTargetChooser(
            instantiate(*conf['acquisition_fn'], predictor=predictor),
            conf['area'], conf['margin'], conf['grid_resolution'])
        if 'noise_search' in conf:
            if conf['noise_search'] == 'wind':
                tc_factory = behaviors.WindBasedPartialSurroundFactory(
                    client, conf['area'], conf['margin'])
            else:
                tc_factory = behaviors.SurroundAreaFactory(
                    conf['area'], conf['margin'])
            surrounder = behaviors.SurroundUntilFound(updater, tc_factory)
            surrounder.observers.append(recorder)
            target_chooser = behaviors.ChainTargetChoosers(
                [surrounder, acq_behavior])
            maxv = 4
        else:
            target_chooser = behaviors.ChainTargetChoosers([
                behaviors.SurroundArea(conf['area'], conf['margin']),
                acq_behavior])
            maxv = 6
        controller = behaviors.FollowWaypoints(
            target_chooser, conf['target_precision'],
            behaviors.VelocityTowardsWaypointController(
                maxv, maxv, target_chooser.get_effective_area()))
        controller.observers.append(updater)

        behavior = controller.velocity_controller

        if conf['full_record']:
            client = ControlsRecorder(fileh, client, num_steps)
        sim_controller = Controller(client, controller, behavior)
        sim_controller.init_new_sim(conf['seedlist'][trial])

        recorder.init(conf)
        err_recorder.init(conf)
        sim_controller.add_recorder(recorder)
        sim_controller.add_recorder(err_recorder)

        if hasattr(behavior, 'targets') and conf['full_record']:
            targets_recorder = TargetsRecorder(
                fileh, behavior, client.numUAVs, num_steps)
            targets_recorder.init()
            sim_controller.add_recorder(targets_recorder)

        try:
            sim_controller.run(num_steps)
        except Exception as err:
            err_tbl = fileh.createVLArray(
                '/', 'exception', tables.ObjectAtom(),
                title='Exception which was raised.')
            err_tbl.append(err)
            raise
        finally:
            try:
                if conf['full_record']:
                    store_obj(fileh, fileh.createGroup('/', 'gp'), predictor)
                else:
                    recorder.prune()
            except:
                pass


def get_correction_factor(trial, conf, client):
    rnd.seed(conf['pyseedlist'][trial])
    with tables.openFile('tmp', 'w') as fileh:
        tbl = fileh.createVLArray(
            '/', 'conf', tables.ObjectAtom(),
            title='Configuration used to generate the stored data.')
        tbl.append(conf)
        fileh.createArray(
            '/', 'repeat', [trial], title='Number of repeat run.')

        num_steps = conf['duration_in_steps']
        kernel = instantiate(*conf['kernel'])
        predictor = instantiate(*conf['predictor'], prefix_args=(kernel,))
        if 'bounds' in conf:
            predictor.bounds = conf['bounds']
        if 'priors' in conf:
            for i in range(len(conf['priors'])):
                predictor.priors[i] = instantiate(*conf['priors'][i])

        recorder = TaskPlumeRecorder(fileh, client, predictor, 1)
        err_recorder = ErrorRecorder(fileh, client, predictor, 1)

        target_chooser = behaviors.ChainTargetChoosers([
            behaviors.SurroundArea(conf['area'], conf['margin']),
            behaviors.AcquisitionFnTargetChooser(
                instantiate(*conf['acquisition_fn'], predictor=predictor),
                conf['area'], conf['margin'], conf['grid_resolution'])])
        controller = behaviors.FollowWaypoints(
            target_chooser, conf['target_precision'])
        updater = instantiate(
            *conf['updater'], predictor=predictor, plume_recorder=recorder)
        controller.observers.append(updater)

        behavior = controller.velocity_controller

        if conf['full_record']:
            client = ControlsRecorder(fileh, client, num_steps)
        sim_controller = Controller(client, controller, behavior)
        sim_controller.init_new_sim(conf['seedlist'][trial])

        recorder.init(conf)
        err_recorder.init(conf)
        volume = np.product(np.diff(conf['area'], axis=1))
        print volume
        test_x = err_recorder.test_x.T
        return np.sqrt(len(test_x) / np.sum(
            1.0 / gaussian_kde(test_x)(test_x) ** 2) / volume)


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
        self.parser.add_argument(
            '-t', '--trial', nargs=1, type=int, required=False,
            help='Only run the given trial.')
        self.parser.add_argument(
            '--error-correction', action='store_true',
            help='Store error correction factors.')

    def _run_application(self, args, conf, client):
        clean = True
        if args.trial is not None:
            trials = args.trial
        else:
            trials = xrange(conf['repeats'])

        err_cor = []

        for i in trials:
            try:
                if args.error_correction:
                    err_cor.append(get_correction_factor(i, conf, client))
                else:
                    output_filename = os.path.join(
                        args.output_dir[0], args.output[0] + '.%i.h5' % i)
                    do_simulation_run(i, output_filename, conf, client)
            except:
                logger.exception('Repeat failed.', exc_info=True)
                clean = False

        if len(err_cor) > 0:
            output_filename = os.path.join(
                args.output_dir[0], args.output[0] + '.errcor.h5')
            with tables.openFile(output_filename, 'w') as fileh:
                fileh.createArray(
                    '/', 'errcor', err_cor, title='Error correction.')
        return clean


if __name__ == '__main__':
    if Plume().main():
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_SOFTWARE)
