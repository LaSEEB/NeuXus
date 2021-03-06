import sys
import os

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.epoching import (TimeBasedEpoching, MarkerBasedSeparation, StimulationBasedEpoching)


class TestEpoching(unittest.TestCase):

    def test_TimeBasedEpoching(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = TimeBasedEpoching(port, 1, 0.5)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_MarkerBasedSeparation(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        marker_port = Port()
        marker_port.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta={})
        node = MarkerBasedSeparation(port, marker_port)
        marker_port.set([[400]], [0.4])

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_StimulationBasedEpoching(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        marker_port = Port()
        marker_port.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta={})
        node = StimulationBasedEpoching(port, marker_port, 400, 0.125, 2)
        marker_port.set([[400]], [0.4])

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)


if __name__ == '__main__':
    unittest.main()
