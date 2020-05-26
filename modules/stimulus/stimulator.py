import time
import random as rd

from pylsl import StreamInfo, StreamOutlet
from xml.dom import minidom

"""
Create a Marker stream of a scenario described in config.xml
"""


class Config(object):
    """docstring for Config"""
    def __init__(self, file):
        super(Config, self).__init__()
        self.file = minidom.parse(file)
        self.number_of_trials = int(self.file.getElementsByTagName('number_of_trials')[0].firstChild.data)

        # get classes from the file
        classes = []
        for class_ in self.file.getElementsByTagName('classes')[0].getElementsByTagName('class'):
            name = class_.getElementsByTagName('name')[0].firstChild.data
            classes.append(name)
        self.classes = classes

        # get init from the file
        init = []
        for step in self.file.getElementsByTagName('init')[0].getElementsByTagName('step'):
            name = step.getElementsByTagName('name')[0].firstChild.data
            duration = step.getElementsByTagName('duration')[0].firstChild.data
            init.append(Marker(name, float(duration)))
        self.init = init

        # get loop from the file
        loop = []
        for step in self.file.getElementsByTagName('loop')[0].getElementsByTagName('step'):
            name = step.getElementsByTagName('name')[0].firstChild.data
            try:
                duration = step.getElementsByTagName('duration')[0].firstChild.data
            except IndexError:
                min_duration = step.getElementsByTagName('min_duration')[0].firstChild.data
                max_duration = step.getElementsByTagName('max_duration')[0].firstChild.data
                loop.append(Marker(name, None, float(min_duration), float(max_duration)))
            else:
                loop.append(Marker(name, float(duration)))
        self.loop = loop

        # get end from the file
        end = []
        for step in self.file.getElementsByTagName('end')[0].getElementsByTagName('step'):
            name = step.getElementsByTagName('name')[0].firstChild.data
            duration = step.getElementsByTagName('duration')[0].firstChild.data
            end.append(Marker(name, float(duration)))
        self.end = end

    def create_a_new_sequence(self):
        sequence = self.classes * self.number_of_trials
        for i in sequence:
            a = rd.randrange(self.number_of_trials * 2)
            b = rd.randrange(self.number_of_trials * 2)
            sequence[a], sequence[b] = sequence[b], sequence[a]
        return sequence


class Marker(object):
    """docstring for Marker"""
    def __init__(self, name, duration, min_duration=None, max_duration=None):
        super(Marker, self).__init__()
        self.name = name
        self.duration = duration
        self.min_duration = min_duration
        self.max_duration = max_duration
        
    def get_duration(self):
        if self.duration:
            return self.duration
        else:
            return rd.uniform(self.min_duration, self.max_duration)

    def is_class(self):
        if self.name == 'Class':
            return True
        return False


if __name__ == "__main__":

    scenario = Config('config.xml')

    # create a new StreamInfo object which shall describe our stream
    info = StreamInfo('MarkerStream', 'Markers', 1, 0, 'string', 'marker1')

    # TO DO
    '''
    # now attach some meta-data
    info.desc().append_child_value("Type", "Graz Motor Imagery")
    config = info.desc().append_child("config")
    config.append_child_value("number_of_trials", str(number_of_trials))
    config.append_child_value("class1", first_class)
    config.append_child_value("class2", second_class)
    config.append_child_value("baseline_duration", str(baseline_duration))
    config.append_child_value("wait_for_beep_duration", str(wait_for_beep_duration))
    config.append_child_value("wait_for_cue_duration", str(wait_for_cue_duration))
    config.append_child_value("display_cue_duration", str(display_cue_duration))
    config.append_child_value("feedback_duration", str(feedback_duration))
    config.append_child_value("end_of_trial_min_duration",
                              str(end_of_trial_min_duration))
    config.append_child_value("end_of_trial_max_duration",
                              str(end_of_trial_max_duration))
    '''
    # next make an outlet
    outlet = StreamOutlet(info)

    sequence = scenario.create_a_new_sequence()

    for marker in scenario.init:
        print(marker.name)
        outlet.push_sample([marker.name])
        time.sleep(marker.get_duration())

    for class_ in sequence:
        for marker in scenario.loop:
            if marker.is_class():
                print(class_)
                outlet.push_sample([class_])
            else:
                print(marker.name)
                outlet.push_sample([marker.name])
            time.sleep(marker.get_duration())

    for marker in scenario.end:
        print(marker.name)
        outlet.push_sample([marker.name])
        time.sleep(marker.get_duration())

    exit()
