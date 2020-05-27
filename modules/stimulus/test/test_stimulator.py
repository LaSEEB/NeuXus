import unittest

import stimulus.stimulator as stim


class TestConfig(unittest.TestCase):

    def test_open_a_file(self):
        stim.Config('stimulus/test/config1.xml')

    def test_no_file_found(self):
        with self.assertRaises(stim.FileNotFound):
            stim.Config('blabla.pdf')

    def test_invalid_xml(self):
        with self.assertRaises(stim.InvalidXml):
            stim.Config('stimulus/test/config7.xml')

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
        self.assertEqual(sequence.count('RIGHT ARROW'), sequence.count('LEFT ARROW'))

class TestMarker(unittest.TestCase):
    """docstring for TestMarker"""

    def test_get_name(self):
        marker = stim.Marker('Class', 5)
        self.assertEqual(marker.get_name('class1'), 'class1')
        self.assertEqual(marker.get_name(), 'Class')
        marker2 = stim.Marker('truc', 5)
        self.assertEqual(marker2.get_name('machin'), 'truc')
    
    def test_get_name(self):
        marker = stim.Marker('Class', 5)
        self.assertEqual(marker.get_duration(), 5.0)
        marker = stim.Marker('Class', 5.2, 2, 5)
        self.assertEqual(marker.get_duration(), 5.2)
        marker = stim.Marker('Class', None, 2, 3)
        for i in range(20):
            self.assertTrue(2 < marker.get_duration() and  marker.get_duration() < 3)
        marker = stim.Marker('Class', None, 3, 2)
        for i in range(20):
            self.assertTrue(2 < marker.get_duration() and  marker.get_duration() < 3)

if __name__ == '__main__':
    unittest.main()
