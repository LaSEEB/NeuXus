import unittest
import numpy

import stimulus.stimulator as stim


"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Test module for stimulaotr.py
Launch with unittest discover -v
"""


class TestConfig(unittest.TestCase):

    def test_open_a_file(self):
        stim.Config('stimulus/test/config1.xml')

    def test_no_file_found(self):
        with self.assertRaises(stim.FileNotFound):
            stim.Config('blabla.pdf')

    def test_invalid_xml(self):
        with self.assertRaises(stim.InvalidXml):
            stim.Config('stimulus/test/config7.xml')

    def test_convert_function(self):
        self.assertIs(type(stim.get_type_function(
            'float32')(456)), numpy.float32)
        self.assertIs(type(stim.get_type_function('string')(456)), str)
        self.assertIs(type(stim.get_type_function('int32')(46)), numpy.int32)
        self.assertIs(type(stim.get_type_function('int16')(456)), numpy.int16)
        self.assertIs(type(stim.get_type_function('int8')(56)), numpy.int8)
        self.assertIs(type(stim.get_type_function('int64')(4567)), numpy.int64)

    def test_global_str(self):
        config = stim.Config('stimulus/test/config1.xml')
        self.assertEqual(config.name, 'Test')
        self.assertEqual(config.number_of_trials, 20.0)
        self.assertEqual(config.type, 'string')
        self.assertEqual(config.type_function, str)
        self.assertEqual(config.author, 'S.Legeay')
        self.assertEqual(config.stream_name, 'MyMarkerStream')
        self.assertEqual(config.classes, ['LEFT ARROW', 'RIGHT ARROW'])
        li = ['ExperimentStart', 'BaselineStart', 'BaselineStop']
        lj = [5.0, 20.0, 5.0]
        for i, step in enumerate(config.init):
            self.assertEqual(step.get_name(), li[i])
            self.assertEqual(step.get_duration(), lj[i])
        li = ['StartOfTrial and CrossOnScreen',
              'LEFT ARROW', 'FeedbackContinuous', 'EndOfTrial']
        lj = [4.0, 1.25, 2.0, 15.2]
        for i, step in enumerate(config.loop):
            self.assertEqual(step.get_name('LEFT ARROW'), li[i])
            if step.get_name('Truc') == 'EndOfTrial':
                self.assertTrue(1 <= step.get_duration()
                                and step.get_duration() <= 3.75)
                self.assertIs(type(step.get_duration()), float)
            else:
                self.assertEqual(step.get_duration(), lj[i])
        li = ['EndOfSession', 'ExperimentStop']
        lj = [5.0, 5.0]
        for i, step in enumerate(config.end):
            self.assertEqual(step.get_name(), li[i])
            self.assertEqual(step.get_duration(), lj[i])

    def test_global_float(self):
        config = stim.Config('stimulus/test/config8.xml')
        self.assertEqual(config.name, 'Test2')
        self.assertEqual(config.number_of_trials, 25.0)
        self.assertEqual(config.type, 'float32')
        self.assertEqual(config.type_function, numpy.float32)
        self.assertEqual(config.author, 'S.Legeay')
        self.assertEqual(config.stream_name, 'MyMarkerStream2')
        self.assertEqual(config.classes, [
                         numpy.float32(40), numpy.float32(41)])
        li = [numpy.float32(1), numpy.float32(2), numpy.float32(3)]
        lj = [5.0, 20.0, 5.0]
        for i, step in enumerate(config.init):
            self.assertEqual(step.get_name(), li[i])
            self.assertEqual(step.get_duration(), lj[i])
        li = [numpy.float32(4),
              numpy.float32(40), numpy.float32(5), numpy.float32(6)]
        lj = [4.0, 1.25, 2.0, 15.2]
        for i, step in enumerate(config.loop):
            self.assertEqual(step.get_name(numpy.float32(40)), li[i])
            if step.get_name('Truc') == numpy.float32(6):
                self.assertTrue(1 <= step.get_duration()
                                and step.get_duration() <= 3.75)
                self.assertIs(type(step.get_duration()), float)
            else:
                self.assertEqual(step.get_duration(), lj[i])
        li = [numpy.float32(8), numpy.float32(9)]
        lj = [5.0, 5.0]
        for i, step in enumerate(config.end):
            self.assertEqual(step.get_name(), li[i])
            self.assertEqual(step.get_duration(), lj[i])

    def test_no_classes(self):
        with self.assertRaisesRegex(stim.ConfigFileNotInAccordance,
                                    'An error occured when loading the config file:\n'
                                    'No classes implemented in config file, please add subsection \'classes\''):
            stim.Config('stimulus/test/config2.xml')

    def test_no_init(self):
        with self.assertRaisesRegex(stim.ConfigFileNotInAccordance,
                                    'An error occured when loading the config file:\n'
                                    'No init implemented in config file, please add subsection \'init\''):
            stim.Config('stimulus/test/config3.xml')

    def test_no_loop(self):
        with self.assertRaisesRegex(stim.ConfigFileNotInAccordance,
                                    'An error occured when loading the config file:\n'
                                    'No loop implemented in config file, please add subsection \'loop\''):
            stim.Config('stimulus/test/config4.xml')

    def test_no_end(self):
        with self.assertRaisesRegex(stim.ConfigFileNotInAccordance,
                                    'An error occured when loading the config file:\n'
                                    'No end implemented in config file, please add subsection \'end\''):
            stim.Config('stimulus/test/config5.xml')

    def test_invalid_value(self):
        with self.assertRaisesRegex(stim.ConfigFileNotInAccordance,
                                    'An error occured when loading the config file:\n'
                                    'm in subsection \'init\' is not a valid duration value'):
            stim.Config('stimulus/test/config6.xml')

    def test_create_a_new_sequence(self):
        scenario = stim.Config('stimulus/test/config1.xml')
        sequence = scenario.create_a_new_sequence()
        self.assertEqual(sequence.count('RIGHT ARROW'),
                         sequence.count('LEFT ARROW'))


class TestMarker(unittest.TestCase):

    def test_get_name(self):
        marker = stim.Marker('Chose', 5)
        self.assertEqual(marker.get_name('Truc'), 'Chose')
        self.assertEqual(marker.get_name(), 'Chose')
        marker2 = stim.MarkerClass('truc', 5)
        self.assertEqual(marker2.get_name('bidule'), 'bidule')

    def test_get_duration(self):
        marker = stim.Marker('Class', 5)
        self.assertEqual(marker.get_duration(), 5.0)
        marker = stim.Marker('Class', 5.2, 2, 5)
        self.assertEqual(marker.get_duration(), 5.2)
        marker = stim.Marker('Class', None, 2, 3)
        for i in range(20):
            self.assertTrue(2 < marker.get_duration()
                            and marker.get_duration() < 3)
        marker = stim.Marker('Class', None, 3, 2)
        for i in range(20):
            self.assertTrue(2 < marker.get_duration()
                            and marker.get_duration() < 3)


if __name__ == '__main__':
    unittest.main()
