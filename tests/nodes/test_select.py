import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.select import (ChannelSelector, SpatialFilter, CommonAverageReference, ReferenceChannel)


class TestSelect(unittest.TestCase):

    def test_ChannelSelector(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = ChannelSelector(port, 'index', [1, 2])

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_SpatialFilter(self):

        matrix = {
            'OC2': [4, 0, -1e-2, 0],
            'OC3': [0, -1, 2, 4]
        }
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = SpatialFilter(port, matrix)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_CommonAverageReference(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = CommonAverageReference(port)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_ReferenceChannel(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = ReferenceChannel(port, 'index', 1)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)


if __name__ == '__main__':
    unittest.main()
