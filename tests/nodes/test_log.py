import sys
import os

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.log import (Hdf5, Mat)


class TestLog(unittest.TestCase):

    def test_Hdf5(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = Hdf5(port, 'my_test', 'df')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)
        os.remove('my_test.h5')

    def test_Mat(self):
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=COLUMN,
            sampling_frequency=250,
            meta={})
        node = Mat(port, 'my_test')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self)
        node.terminate()
        os.remove('my_test.mat')
