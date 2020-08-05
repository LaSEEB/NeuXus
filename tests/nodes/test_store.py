import sys
import os

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.store import ToCsv


class TestStore(unittest.TestCase):

    def test_ToCsv(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = ToCsv(port, 'my_test')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)
        os.remove('my_test.csv')


if __name__ == '__main__':
    unittest.main()
