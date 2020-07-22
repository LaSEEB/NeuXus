import inspect

import unittest
import pandas as pd

import modules.nodes.function as f
from modules.core.port import Port


"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Test module for nodes from function.py
Launch with unittest discover -v
"""


class TestFunction(unittest.TestCase):

    def test_apply_function(self):
        port = Port()
        port.set_parameters(
            channels=[],
            frequency=250,
            meta={})
        node = f.ApplyFunction(port, lambda x: x**2)

        a = pd.DataFrame([[4, 5], [6, 7]], index=[1.2, 1.3], columns=['a', 'b'])
        a_copy = a.copy()
        result = pd.DataFrame([[16, 25], [36, 49]], index=[1.2, 1.3], columns=['a', 'b'])

        port.set_from_df(a)
        node.update()
        self.assertEqual(a.all().all(), a_copy.all().all())
        self.assertEqual(node.output._data[0].all().all(), result.all().all())


if __name__ == '__main__':
    unittest.main()
