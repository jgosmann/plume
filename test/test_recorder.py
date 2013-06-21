from hamcrest import assert_that
from matchers import has_items_in_relative_order
from mock import ANY, call, MagicMock
from numpy.testing import assert_equal
from qrsim.tcpclient import UAVState
from recorder import GeneralRecorder, TaskPlumeRecorder
import numpy as np
import tables
import tempfile


class TestGeneralRecorder(object):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.fileh = tables.open_file(self.tmp_file.name, 'w')
        self.client = MagicMock()
        self.client.numUAVs = 2
        self.client.state = (UAVState(UAVState.size * [0]),
                             UAVState(UAVState.size * [1]))
        self.recorder = self.create_recorder()
        self.recorder.init()

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


class TestTaskPlumeRecorder(TestGeneralRecorder):
    def setUp(self):
        self.predictor = MagicMock()
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
        for i in xrange(steps):
            self.recorder.record()
        expected_calls = steps * [call.set_samples(ANY), call.get_reward()]
        assert_that(self.client.mock_calls, has_items_in_relative_order(
            *expected_calls))
        assert_equal(
            self.recorder.rewards,
            [-np.inf] + (steps - 1) * [self.client.get_reward.return_value])
