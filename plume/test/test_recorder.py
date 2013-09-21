import tempfile

from hamcrest import assert_that, is_, only_contains
from matchers import has_items_in_relative_order
from mock import ANY, call, MagicMock
from numpy.testing import assert_almost_equal, assert_equal
from qrsim.tcpclient import ctrl_signal_dimensions, UAVControls, UAVState
import numpy as np
import tables

from plume.recorder import ControlsRecorder, GeneralRecorder, TaskPlumeRecorder


class TestGeneralRecorder(object):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.fileh = tables.openFile(self.tmp_file.name, 'w')
        self.client = MagicMock()
        self.client.numUAVs = 2
        self.client.state = (UAVState(UAVState.size * [0]),
                             UAVState(UAVState.size * [1]))
        self.client.get_locations.return_value = np.array(
            [[1, 2, 3], [4, 5, 6]])
        self.client.get_reference_samples.return_value = np.array([0.5, 1.0])

        def get_samples(x):
            samples = np.ones(len(x))
            samples[::2] = 0.5
            return samples

        self.client.get_samples.side_effect = get_samples
        self.recorder = self.create_recorder()
        self.recorder.init(3 * [[-10, 10]])

    def tearDown(self):
        self.fileh.close()
        self.tmp_file.close()

    def create_recorder(self):
        return GeneralRecorder(self.fileh, self.client)

    def test_records_positions(self):
        steps = 4
        for i in xrange(steps):
            self.recorder.record()
        expected = np.dstack((np.tile([0], (3, steps)),
                              np.tile([1], (3, steps)))).T
        assert_equal(self.recorder.positions, expected)

    def test_records_sample_locations(self):
        assert_equal(
            self.recorder.sample_locations,
            self.client.get_locations.return_value)

    def test_records_reference_samples(self):
        assert_equal(
            self.recorder.reference_samples,
            self.client.get_reference_samples.return_value)

    def test_records_ground_truth(self):
        assert_that(len(self.client.get_samples.call_args_list), is_(1))
        assert_that(len(self.client.get_samples.call_args_list[0]), is_(2))
        assert_equal(
            self.client.get_samples.call_args_list[0][0][0],
            self.recorder.gt_locations)
        assert_equal(
            self.recorder.gt_samples[::2],
            len(self.recorder.gt_locations) / 2 * [0.5])
        assert_equal(
            self.recorder.gt_samples[1::2],
            len(self.recorder.gt_locations) / 2 * [1.0])


class TestTaskPlumeRecorder(TestGeneralRecorder):
    def setUp(self):
        self.predictor = MagicMock()
        self.predictor.trained = False
        self.predictor.predict.side_effect = lambda x: np.empty(len(x))
        self.predictor.kernel.params = np.array([0, 0])
        self.predictor.calc_neg_log_likelihood.return_value = 0
        TestGeneralRecorder.setUp(self)
        self.client.get_plume_sensor_outputs.return_value = range(
            self.client.numUAVs)
        self.client.get_reward.return_value = 42

    def create_recorder(self):
        return TaskPlumeRecorder(self.fileh, self.client, self.predictor)

    def test_records_plume_measurements(self):
        steps = 4
        for i in xrange(steps):
            self.recorder.record()
        expected = np.tile(np.arange(self.client.numUAVs), (steps, 1)).T
        assert_equal(self.recorder.plume_measurements, expected)

    def test_records_rewards(self):
        steps = 4
        self.predictor.trained = False
        for i in xrange(steps):
            self.recorder.record()
            self.predictor.trained = True
        expected_calls = steps * [call.set_samples(ANY), call.get_reward()]
        assert_that(self.client.mock_calls, has_items_in_relative_order(
            *expected_calls))
        assert_equal(
            self.recorder.rewards,
            steps * [self.client.get_reward.return_value])

    def test_records_rmse(self):
        steps = 4
        self.predictor.trained = True
        self.predictor.predict.side_effect = lambda x: np.ones(len(x))
        for i in xrange(steps):
            self.recorder.record()
        assert_almost_equal(
            self.recorder.rmse, steps * [np.sqrt(0.5 ** 2 / 2)])

    def test_records_wrmse(self):
        steps = 4
        self.predictor.trained = True
        self.predictor.predict.side_effect = lambda x: np.zeros(len(x))
        for i in xrange(steps):
            self.recorder.record()
        assert_almost_equal(
            self.recorder.wrmse, steps * [0.75])

    def test_records_kernel_params(self):
        steps = 4
        self.predictor.trained = True
        self.predictor.kernel.params = [23, 42]
        for i in xrange(steps):
            self.recorder.record()
        assert_equal(self.recorder.kernel_params, steps * [[23, 42]])

    def test_records_log_likelihood(self):
        steps = 4
        self.predictor.trained = True
        self.predictor.calc_neg_log_likelihood.return_value = -1.0
        for i in xrange(steps):
            self.recorder.record()
        assert_almost_equal(self.recorder.log_likelihood, steps * [1.0])


class TestControlsRecorder(object):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.fileh = tables.openFile(self.tmp_file.name, 'w')
        self.client = MagicMock()
        self.client.numUAVs = 2
        self.recorder = ControlsRecorder(self.fileh, self.client)
        self.recorder.init('Task', False)

    def tearDown(self):
        self.fileh.close()
        self.tmp_file.close()

    def test_init_of_client_is_called(self):
        self.client.init.assert_called_once_with('Task', False)

    def test_has_attrs_of_client(self):
        assert_that(self.recorder.numUAVs, is_(self.client.numUAVs))
        assert_that(self.recorder.timestep, is_(self.client.timestep))

    def tests_forwards_unknown_method_calls_to_client(self):
        self.recorder.unknown_method('some arg')
        self.client.unknown_method.assert_called_once_with('some arg')

    def check_control_recording_type_specific_step(self, type):
        dt = 0.1
        steps = 3
        dim = ctrl_signal_dimensions[type]
        for i in xrange(steps):
            getattr(self.recorder, 'step_' + type)(
                dt, [range(dim), range(dim, 2 * dim)])
        self.fileh.flush()

        controls = self.recorder.controls
        assert_that(controls.cols.type, only_contains(
            getattr(ControlsRecorder.Controls.columns['type'].enum, type)))
        assert_that(controls.cols.dt[:], only_contains(0.1))
        assert_equal(
            [x['U'][:dim] for x in controls.where('uav == 0')],
            np.tile(range(dim), (steps, 1)))
        assert_equal(
            [x['U'][:dim] for x in controls.where('uav == 1')],
            np.tile(range(dim, 2 * dim), (steps, 1)))

    def check_control_recording_of_step(self, type):
        input_controls = UAVControls(self.recorder.numUAVs, type)
        for i in xrange(self.recorder.numUAVs):
            input_controls.U[i, :] = i
        dt = 0.1
        steps = 3

        for i in xrange(steps):
            self.recorder.step(dt, input_controls)
        self.fileh.flush()

        dim = ctrl_signal_dimensions[type]
        controls = self.recorder.controls
        assert_that(controls.cols.type, only_contains(
            getattr(ControlsRecorder.Controls.columns['type'].enum, type)))
        assert_that(controls.cols.dt[:], only_contains(0.1))
        for i in xrange(self.recorder.numUAVs):
            assert_equal(
                [x['U'][:dim] for x in controls.where('uav == %i' % i)],
                np.tile(dim * [i], (steps, 1)))

    def check_client_step_called(self, type):
        input_controls = UAVControls(self.recorder.numUAVs, type)
        for i in xrange(self.recorder.numUAVs):
            input_controls.U[i, :] = i
        dt = 0.1

        self.recorder.step(dt, input_controls)
        self.client.step.assert_called_once_with(dt, input_controls)

    def check_client_step_type_specific_called(self, type):
        input_controls = UAVControls(self.recorder.numUAVs, type)
        for i in xrange(self.recorder.numUAVs):
            input_controls.U[i, :] = i
        dt = 0.1

        getattr(self.recorder, 'step_' + type)(dt, input_controls.U)

        class MatchInputControls(object):
            def __init__(self, obj):
                self.obj = obj

            def __eq__(self, other):
                if self.obj.type != other.type:
                    return False
                assert_equal(self.obj.U, other.U)
                return True

        self.client.step.assert_called_once_with(
            dt, MatchInputControls(input_controls))

    def test_records_controls(self):
        for type in ctrl_signal_dimensions.keys():
            yield self.check_control_recording_type_specific_step, type
            yield self.check_control_recording_of_step, type
            yield self.check_client_step_called, type
            yield self.check_client_step_type_specific_called, type
