import time
import random as rd
import argparse

from pylsl import StreamInfo, StreamOutlet
from xml.dom import minidom
from xml.parsers.expat import ExpatError

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Create a Marker stream of a scenario described in a xml file specified in parser
"""


class ConfigFileNotInAccordance(Exception):
    """Exception raised when the config file is not in accordance with rules"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        message = "An error occured when loading the config file:\n"
        return message + self.message


class FileNotFound(Exception):
    """Exception raised when the specified file does not exist"""

    def __init__(self, file):
        self.file = file

    def __str__(self):
        return f'{self.file} not found'


class InvalidXml(Exception):
    """Exception raised when the config file cannot be opened"""

    def __init__(self, file, err):
        self.file = file
        self.err = err

    def __str__(self):
        return f'{self.file} is not a readable xml, please check the file ({self.err})'


def get_section(file, section):
    """Function used for reading xml file, exttract sections such as init class"""
    try:
        section = file.getElementsByTagName(section)[0]
    except IndexError as err:
        raise ConfigFileNotInAccordance(
            f'No {section} implemented in config file, please add subsection \'{section}\'')
    return section


def get_data(step, data, section):
    """Function used for reading xml file, it extracts data from step"""
    try:
        item = step.getElementsByTagName(data)[0]
    except IndexError:
        raise ConfigFileNotInAccordance(
            f'No {data} implemented in a step of {section}')
    if data in ['duration', 'min_duration', 'max_duration', 'number_of_trials']:
        try:
            return float(item.firstChild.data)
        except ValueError:
            raise ConfigFileNotInAccordance(
                f'{item.firstChild.data} in subsection \'{section}\' is not a valid {data} value')
    return item.firstChild.data


def extract_classes(file):
    """get the different classes from the file"""
    classes = []
    section = get_section(file, 'classes')
    for class_ in section.getElementsByTagName('class'):
        name = get_data(class_, 'name', 'classes')
        classes.append(name)
    return classes


def extract_init(file):
    """get the init of scenario from the file"""
    init = []
    section = get_section(file, 'init')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'init')
        duration = get_data(step, 'duration', 'init')
        init.append(Marker(name, duration))
    return init


def extract_loop(file):
    """get the loop of the scenario from the file"""
    loop = []
    section = get_section(file, 'loop')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'loop')
        if step.getElementsByTagName('duration'):
            duration = get_data(step, 'duration', 'loop')
            loop.append(Marker(name, duration))
        else:
            min_duration = get_data(step, 'min_duration', 'loop')
            max_duration = get_data(step, 'max_duration', 'loop')
            loop.append(Marker(name, None,
                               min_duration, max_duration))
    return loop


def extract_end(file):
    """get the end of the scenario from the file"""
    end = []
    section = get_section(file, 'end')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'end')
        duration = get_data(step, 'duration', 'end')
        end.append(Marker(name, duration))
    return end


class Config(object):
    """object describing the config file"""

    def __init__(self, file):
        super(Config, self).__init__()
        try:
            file = minidom.parse(file)
        except FileNotFoundError:
            raise FileNotFound(file)
        except ExpatError as err:
            raise InvalidXml(file, err)
        self.number_of_trials = get_data(file, 'number_of_trials', 'info')
        self.name = get_data(file, 'name', 'info')
        self.author = get_data(file, 'author', 'info')
        self.classes = extract_classes(file)
        self.init = extract_init(file)
        self.loop = extract_loop(file)
        self.end = extract_end(file)

    def create_a_new_sequence(self):
        """Create a new random sequence of length number_of_trials * nb_of_classes
        with number_of_trials elements by class"""
        sequence = self.classes * int(self.number_of_trials)
        for i in sequence:
            a = rd.randrange(self.number_of_trials * 2)
            b = rd.randrange(self.number_of_trials * 2)
            sequence[a], sequence[b] = sequence[b], sequence[a]
        return sequence


class Marker(object):
    """Object describing a marker"""

    def __init__(self, name, duration, min_duration=None, max_duration=None):
        super(Marker, self).__init__()
        self.name = name
        if duration:
            self.duration = float(duration)
        else:
            self.duration = None
        if min_duration and max_duration:
            self.min_duration = float(min_duration)
            self.max_duration = float(max_duration)
        else:
            self.min_duration = None
            self.max_duration = None

    def get_duration(self):
        if self.duration:
            return self.duration
        else:
            return rd.uniform(self.min_duration, self.max_duration)

    def get_name(self, class_=None):
        if self.name == 'Class' and class_:
            return class_
        return self.name


def main(args):
    try:
        scenario = Config(args.file)
    except (ConfigFileNotInAccordance, FileNotFound, InvalidXml) as err:
        print(err)
        exit()
    sequence = scenario.create_a_new_sequence()

    # create a new StreamInfo object which shall describe our stream
    info = StreamInfo('openvibeMarkers', 'Markers', 1, 0, 'string', 'marker1') # uid it is an optional parameter, to skip

    # now attach some meta-data
    config = info.desc().append_child("config")
    config.append_child_value(
        "number_of_trials", str(scenario.number_of_trials))
    config.append_child_value("type", scenario.name)
    config.append_child_value("author", scenario.author)
    for i, class_ in enumerate(scenario.classes):
        config.append_child_value(f"class{i}", class_)

    # next make an outlet
    outlet = StreamOutlet(info)
    print('Now sending Markers...')

    # send init
    for marker in scenario.init:
        if args.verbose:
            print(marker.get_name())
        outlet.push_sample([marker.get_name()])
        time.sleep(marker.get_duration())

    # send loop
    for class_ in sequence:
        for marker in scenario.loop:
            if args.verbose:
                print(marker.get_name())
            outlet.push_sample([marker.get_name()])
            time.sleep(marker.get_duration())

    # send end
    for marker in scenario.end:
        if args.verbose:
            print(marker.get_name())
        outlet.push_sample([marker.get_name()])
        time.sleep(marker.get_duration())
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Read the config from the specified xml file and launch an LSL marker stream")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="verbose mode")
    parser.add_argument("file", help="path to the config file")
    args = parser.parse_args()

    main(args)

    exit()
