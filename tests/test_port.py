import unittest
import pandas as pd

from neuxus.chunks import Port


"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Test module for port.py
Launch with unittest discover -v
"""


class TestPort(unittest.TestCase):

    def test_iter_empty(self):
        port = Port()
        p = True
        for chunk in port:
            p = False
        self.assertTrue(p)

    def test_iter(self):
        port = Port()
        a = pd.DataFrame([[4, 2]], ['A'], ['col1', 'col2'])
        port.set_from_df(a)
        port.set([[4, 5]], ['454'], ['col1', 'col2'])
        verify = []
        for chunk in port:
            verify.append(chunk)
        verify[0]['col1'] = [8]
        self.assertTrue(verify[0] is a)
        self.assertEqual(verify[1].all().all(), pd.DataFrame([[4, 5]], ['454'], ['col1', 'col2']).all().all())

    def test_multiple_iter(self):
        port = Port()
        a = pd.DataFrame([[4, 2]], ['A'], ['col1', 'col2'])
        b = pd.DataFrame([[18, 26]], ['B'], ['col1', 'col2'])
        port.set_from_df(a)
        port.set_from_df(b)
        v1 = []
        v2 = []
        for chunk in port:
            v1.append(chunk)
        for chunk in port:
            v2.append(chunk)
        self.assertEqual(v1[0].all().all(), v2[0].all().all())
        self.assertEqual(v1[1].all().all(), v2[1].all().all())


if __name__ == '__main__':
    unittest.main()
