import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.processing import (HilbertTransform, PsdWelch, Fft)


class TestProcessing(unittest.TestCase):

    def test_HilbertTransform(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = HilbertTransform(port)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_PsdWelch(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = PsdWelch(port)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_Fft(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = Fft(port)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)
