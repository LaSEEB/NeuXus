import inspect

import unittest
import pandas as pd

import modules.nodes.epoching as f
from modules.core.port import Port


"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Test module for nodes from epoching.py
Launch with unittest discover -v
"""


def generate_fake_df(interval):
    data = [[i, i + 1] for i in range(1, 19, 2)]
    index = [1 + i * interval for i in range(1, 10)]
    return pd.DataFrame(data, index, ['a', 'b'])


class TestEpoching(unittest.TestCase):

    def test_TimeBasedEpoching(self):
        port = Port()
        port.set_parameters(
            channels=[],
            frequency=250,
            meta={})
        node = f.TimeBasedEpoching(port, 5)

        a = pd.DataFrame([[4, 5], [6, 7], [2, 3], [1, 2], [15, 23]] * 2, index=[
                         1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2, 2.1, 2.2], columns=['a', 'b'])
        a_copy = a.copy()
        result = pd.DataFrame([[16, 25], [36, 49]], index=[
                              1.2, 1.3], columns=['a', 'b'])

        port.set_from_df(a)
        node.update()
        print(a)
        self.assertEqual(a.all().all(), a_copy.all().all())
        print(node.output._data)
        # self.assertEqual(node.output._data[0].all().all(), result.all().all())

    def test_StimulationBasedEpoching_simple(self):
        a = generate_fake_df(0.1)
        data_port = Port()
        data_port.set_parameters(
            channels=[],
            frequency=None,
            meta={})

        marker_port = Port()
        marker = pd.DataFrame([['stimulation']], [1.4])
        node = f.StimulationBasedEpoching(
            data_port, marker_port, 'stimulation', 0.02, 0.2)

        data_port.set_from_df(a)
        marker_port.set_from_df(marker)
        node.update()
        self.assertEqual(len(node.output._data), 1)
        self.assertTrue(node.output._data[0].equals(a.loc[
            1.5: 1.605]))

    def test_StimulationBasedEpoching_doublemarker(self):
        a = generate_fake_df(0.1)
        data_port = Port()
        data_port.set_parameters(
            channels=[],
            frequency=None,
            meta={})

        marker_port = Port()
        marker = pd.DataFrame([['stimulation']] * 3, [1.1, 1.5, 45])
        node = f.StimulationBasedEpoching(
            data_port, marker_port, 'stimulation', 0.02, 0.2)

        data_port.set_from_df(a)
        marker_port.set_from_df(marker)
        node.update()
        self.assertEqual(len(node.markers), 1)
        self.assertEqual(len(node.output._data), 2)
        self.assertTrue(node.output._data[0].equals(a.loc[
            1.2: 1.305]))
        self.assertTrue(node.output._data[1].equals(a.loc[
            1.6: 1.705]))

    def test_StimulationBasedEpoching_doublemarker_overlapped(self):
        a = generate_fake_df(0.1)
        data_port = Port()
        data_port.set_parameters(
            channels=[],
            frequency=None,
            meta={})

        marker_port = Port()
        marker = pd.DataFrame([['stimulation']] * 3, [1.1, 1.2, 23])
        node = f.StimulationBasedEpoching(
            data_port, marker_port, 'stimulation', 0.02, 0.2)

        data_port.set_from_df(a)
        marker_port.set_from_df(marker)
        node.update()
        self.assertEqual(len(node.markers), 1)
        self.assertEqual(len(node.output._data), 2)
        self.assertTrue(node.output._data[0].equals(a.loc[
            1.2: 1.305]))
        self.assertTrue(node.output._data[1].equals(a.loc[
            1.3: 1.405]))

    def test_StimulationBasedEpoching_nomarker(self):
        a = generate_fake_df(0.1)
        data_port = Port()
        data_port.set_parameters(
            channels=[],
            frequency=None,
            meta={})

        marker_port = Port()
        marker = pd.DataFrame([])
        node = f.StimulationBasedEpoching(
            data_port, marker_port, 'stimulation', 0.02, 0.2)

        data_port.set_from_df(a)
        marker_port.set_from_df(marker)
        node.update()
        self.assertEqual(len(node.markers), 0)
        self.assertEqual(len(node.output._data), 0)


if __name__ == '__main__':
    unittest.main()
