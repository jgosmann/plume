from qrsim.tcpclient import TCPClient
import numpy as np


class TaskPlumeClient(TCPClient):
    def get_locations(self):
        return np.array(self.rpc('TASK', 'getLocations')).reshape((3, -1)).T

    def get_samples_per_location(self):
        return self.rpc('TASK', 'getSamplesPerLocation')[0]

    def get_plume_sensor_outputs(self):
        return self.rpc('PLATFORMS', 'getPlumeSensorOutput')

    def get_reference_samples(self):
        return self.rpc('TASK', 'getReferenceSamples')

    def set_samples(self, samples):
        self.rpc('TASK', 'setSamples', samples.flat)

    def get_reward(self):
        return self.rpc('TASK', 'reward')[0]

    def get_samples(self, locations):
        return self.rpc('TASK', 'getSamples', locations.flat)

    def get_sources(self):
        return np.array(self.rpc('TASK', 'getSources')).reshape((3, -1)).T
