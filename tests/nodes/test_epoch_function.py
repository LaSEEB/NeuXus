import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.epoch_function import (Windowing, UnivariateStat)


class TestEpochFunction(unittest.TestCase):

    def test_Windowing(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='epoch',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = Windowing(port, 'blackman')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self, index=[0.2 * i for i in range(250)])

    def test_UnivariateStat(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='epoch',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = UnivariateStat(port, 'mean')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)


if __name__ == '__main__':
    unittest.main()
