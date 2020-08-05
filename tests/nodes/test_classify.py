import sys

import unittest

sys.path.append('neuxus')
sys.path.append('tests/nodes')

from utils import (INDEX, COLUMN, simulate_loop_and_verify)
from chunks import Port
from nodes.classify import Classify


class TestClassify(unittest.TestCase):

    def test_Classify(self):
        col = ['a', 'b']
        # create a Port and a Node
        port = Port()
        port.set_parameters(
            data_type='signal',
            channels=col,
            sampling_frequency=250,
            meta={})
        node = Classify(port, 'tests/nodes/data/classifier.sav', 'class')

        # simulate NeuXus loops
        simulate_loop_and_verify(port, node, self, column=col)


if __name__ == '__main__':
    unittest.main()
