import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.filter import (NotchFilter, DownSample, ButterFilter)


class TestFilter(unittest.TestCase):

    def test_ButterFilter(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = ButterFilter(port, 4, 10)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_NotchFilter(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = NotchFilter(port, 10, 0.4)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_DownSample(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = DownSample(port, 2)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)
