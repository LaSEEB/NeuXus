import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.io import (UdpSend, LslSend)


class TestIo(unittest.TestCase):

    def test_UdpSend(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = UdpSend(port, 'localhost', 400)

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)

    def test_LslSend(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = LslSend(port, 'my_test_signal')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)