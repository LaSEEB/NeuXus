from time import time
import random as rd
from tkinter import *
from xml.dom import minidom
from xml.parsers.expat import ExpatError
import numpy as np
import multiprocessing as mp
from tkinter.ttk import Progressbar

from neuxus.node import Node

"""
Author: S.LEGEAY, intern at LaSEEB
e-mail: legeay.simon.sup@gmail.com


Create a Marker stream of a scenario described in a xml file specified in parser
Launch: python stimulator.py config.xml
"""

VALID_TYPE = ['float32', 'double64', 'string',
              'int32', 'int16', 'int8', 'int64']


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
    """Function used for reading xml file, extract sections such as init class"""
    try:
        section = file.getElementsByTagName(section)[0]
    except IndexError as err:
        raise ConfigFileNotInAccordance(
            f'No {section} implemented in config file, please add subsection \'{section}\'')
    return section


def get_data(step, data, section, convert=None):
    """Function used for reading xml file, it extracts data from step"""
    try:
        item = step.getElementsByTagName(data)[0]
    except IndexError:
        raise ConfigFileNotInAccordance(
            f'No {data} implemented in a step of {section}')
    if convert:
        try:
            return convert(item.firstChild.data)
        except ValueError:
            raise ConfigFileNotInAccordance(
                f'{item.firstChild.data} in subsection \'{section}\' is not a valid {data} value')
    return item.firstChild.data


def extract_classes(file, convert):
    """get the different classes from the file"""
    classes = []
    section = get_section(file, 'classes')
    for class_ in section.getElementsByTagName('class'):
        name = get_data(class_, 'name', 'classes', convert)
        classes.append(name)
    return classes


def extract_init(file, convert):
    """get the init of scenario from the file"""
    init = []
    section = get_section(file, 'init')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'init', convert)
        duration = get_data(step, 'duration', 'init', float)
        init.append(Marker(name, duration))
    return init


def extract_loop(file, convert):
    """get the loop of the scenario from the file"""
    loop = []
    section = get_section(file, 'loop')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'loop')
        if name == 'Class':
            if step.getElementsByTagName('duration'):
                duration = get_data(step, 'duration', 'loop', float)
                loop.append(MarkerClass(name, duration))
            else:
                min_duration = get_data(step, 'min_duration', 'loop', float)
                max_duration = get_data(step, 'max_duration', 'loop', float)
                loop.append(MarkerClass(name, None,
                                        min_duration, max_duration))
        else:
            name = convert(name)
            if step.getElementsByTagName('duration'):
                duration = get_data(step, 'duration', 'loop', float)
                loop.append(Marker(name, duration))
            else:
                min_duration = get_data(step, 'min_duration', 'loop', float)
                max_duration = get_data(step, 'max_duration', 'loop', float)
                loop.append(Marker(name, None,
                                   min_duration, max_duration))
    return loop


def extract_intersession(file, convert):
    """get the intersession of the scenario from the file"""
    intersession = []
    section = get_section(file, 'intersession')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'intersession', convert)
        duration = get_data(step, 'duration', 'intersession', float)
        intersession.append(Marker(name, duration))
    return intersession


def extract_end(file, convert):
    """get the end of the scenario from the file"""
    end = []
    section = get_section(file, 'end')
    for step in section.getElementsByTagName('step'):
        name = get_data(step, 'name', 'end', convert)
        duration = get_data(step, 'duration', 'end', float)
        end.append(Marker(name, duration))
    return end


def get_type_function(type_):
    """According specified type, return the function that convert to the type specified"""
    if type_ == 'float32':
        return np.float32
    elif type_ == 'string':
        return str
    elif type_ == 'int32':
        return np.int32
    elif type_ == 'int16':
        return np.int16
    elif type_ == 'int8':
        return np.int8
    elif type_ == 'int64':
        return np.int64


def booleen(x):
    if x == 'True' or x == '1':
        return True
    elif x == 'False' or x == '0':
        return False
    else:
        raise ValueError


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
        self.number_of_trials = get_data(file, 'number_of_trials', 'info', int)
        self.type = get_data(file, 'marker_type', 'info')
        if self.type not in VALID_TYPE:
            raise ConfigFileNotInAccordance(f'{self.type} is not an accepted type')
        self.type_function = get_type_function(self.type)
        self.name = get_data(file, 'name', 'info')
        self.author = get_data(file, 'author', 'info')
        self.session = get_data(file, 'session', 'info', int)
        self.random = get_data(file, 'random', 'info', booleen)

        self.classes = extract_classes(file, self.type_function)
        self._init = extract_init(file, self.type_function)
        self._loop = extract_loop(file, self.type_function)
        self._intersession = extract_intersession(file, self.type_function)
        self._end = extract_end(file, self.type_function)

    def create_a_new_scenario(self):
        """Create a scenario of length number_of_trials * nb_of_classes
        with number_of_trials elements by class"""

        scenario = []
        t = 0
        for marker in self._init:
            scenario.append([marker.get_name(), t])
            # Gustavo
            # print('marker: ', marker)
            t += marker.get_duration()

        # send loop
        if self.random:
            sequence = self.classes * int(self.number_of_trials)
            for session in range(self.session):
                for i in sequence:
                    a = rd.randrange(self.number_of_trials * 2)
                    b = rd.randrange(self.number_of_trials * 2)
                    sequence[a], sequence[b] = sequence[b], sequence[a]
                for class_ in sequence:
                    for marker in self._loop:
                        scenario.append([marker.get_name(class_), t])
                        t += marker.get_duration()
                if session != self.session - 1:
                    for marker in self._intersession:
                        scenario.append([marker.get_name(class_), t])
                        t += marker.get_duration()
        else:
            for i in range(len(self.classes) * self.session):
                sequence = [self.classes[i % len(self.classes)]] * self.number_of_trials
                for class_ in sequence:
                    for marker in self._loop:
                        scenario.append([marker.get_name(class_), t])
                        t += marker.get_duration()
                if i != len(self.classes) * self.session - 1:
                    for marker in self._intersession:
                        scenario.append([marker.get_name(class_), t])
                        t += marker.get_duration()

        # send end
        for marker in self._end:
            scenario.append([marker.get_name(), t])
            t += marker.get_duration()
        # /Gustavo:
        print(t//60)
        print(t-(t//60)*60)
        return scenario


class Marker(object):
    """Object describing a marker"""

    def __init__(self, name, duration, min_duration=None, max_duration=None):
        super(Marker, self).__init__()
        self.name = name
        self.duration = duration
        self.min_duration = min_duration
        self.max_duration = max_duration

    def get_duration(self):
        # Gustavo:
        if self.duration is not None:
            return self.duration
        # if self.duration:         # if 0 gives False, but we might want durations = 0!!
        #     return self.duration
        else:
            return rd.uniform(self.min_duration, self.max_duration)

    def get_name(self, class_=None):
        return self.name


class MarkerClass(Marker):
    """"""

    def __init__(self, name, duration, min_duration=None, max_duration=None):
        super().__init__(name, duration, min_duration, max_duration)

    def get_name(self, class_):
        return class_


class CustomCanvas(Canvas):
    """Custom Canvas fitting with graz visualization with function to update the content
    according the windows size, and plot functions"""

    def __init__(self, parent, **kwargs):
        self.progress = Progressbar(parent, length=425, orient=HORIZONTAL, mode='determinate')
        #self.bind("<Configure>", self.on_resize)
        self.parent = parent
        self.progress.pack(side=TOP, expand=NO)

        Canvas.__init__(self, parent, **kwargs)
        self.pack(fill=BOTH, expand=0)
        self._last = ['', '', '', '', '']
        self._i = 0

        #self.height = self.winfo_reqheight()
        #self.width = self.winfo_reqwidth()

    def update(self, marker, end):
        """"""
        self.delete('a')
        self._last = self._last[1:] + [marker]
        self.progress['value'] = self._i * 100 / (end - 1)
        self._i += 1
        self.parent.update_idletasks()
        self.create_text(20, 20, anchor=W, tags='a', text=self._last[0], font=('Helvetica', '12'))
        self.create_text(20, 40, anchor=W, tags='a', text=self._last[1], font=('Helvetica', '12'))
        self.create_text(20, 60, anchor=W, tags='a', text=self._last[2], font=('Helvetica', '12'))
        self.create_text(20, 80, anchor=W, tags='a', text=self._last[3], font=('Helvetica', '12'))
        self.create_text(20, 104, anchor=W, tags='a', text=self._last[4], font=('Helvetica', '18'))

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


class ProcessStim():
    """Create a new window containing the Graz visualization
    Launched in a subprocess"""

    def __init__(self, end):
        self._max = end

    def terminate(self):
        self.root.destroy()

    def call_back(self):
        if self.pipe.poll():
            sample = self.pipe.recv()
            if sample:
                self.myCanvas.update(sample, self._max)
        self.root.after(10, self.call_back)

    def __call__(self, pipe):
        self.pipe = pipe
        # initialise a window.
        self.root = Tk()
        self.root.config()
        self.root.title('Stimulator')
        self.root.geometry("500x150")

        # create a Frame
        myframe = Frame(self.root)
        myframe.pack(fill=BOTH, expand=NO)

        # add a canvas
        self.myCanvas = CustomCanvas(myframe, width=425, height=200,
                                     highlightthickness=0)
        self.myCanvas.pack(fill=BOTH, expand=NO)
        self.root.after(0, self.call_back)
        self.root.mainloop()


class Stimulator(Node):
    """Generate a marker stream from a config file
    Attributes:
      - output (Port): Output port
    Args:
      - file (str): Path to the xml config file

    example: Stimulator('config_ov.xml)

    """

    def __init__(self, file, input_port=None, start_marker=None):
        Node.__init__(self, input_port)
        try:
            config = Config(file)
        except (ConfigFileNotInAccordance, FileNotFound, InvalidXml) as err:
            print(err)
            exit()

        self._scenario = config.create_a_new_scenario()

        # if (start_marker is not None) and (input_port is not None):
        #     self.start_marker = start_marker
        #     self.start = False
        # else:
        #     self.start = True

        self.start_marker = start_marker
        self.start = (start_marker is None) or (input_port is None)

        self.output.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta='')

        # create the plot process
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessStim(len(self._scenario))
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        Node.log_instance(self, {
            'file': file,
            'name': config.name,
            'author': config.author,
            'session': config.session,
            'nb of trials': config.number_of_trials,
            'classes': config.classes,
            'random': config.random,
            'type': config.type})
        self._start_time = None

    def update(self):
        if not self.start:
            for chunk in self.input:
                values = chunk.select_dtypes(include=[np.number]).values
                # self.start = self.start_marker in values
                # if chunk.iloc[:,0].str.contains(self.start_marker).any():
                #     self.start = True
                self.start = chunk.iloc[:,0].str.contains(self.start_marker).any()
                # print('self.start', self.start)

        if self.start:
            if not self._start_time:
                self._start_time = time()
                for markers in self._scenario:
                    markers[1] = markers[1] + self._start_time
            t = time()
            while self._scenario and t >= self._scenario[0][1]:
                self.output.set(self._scenario[0][0], [self._scenario[0][1]], ['marker'])
                try:
                    self.plot_pipe.send(self._scenario[0][0])
                except BrokenPipeError:
                    pass
                self._scenario = self._scenario[1:]

    def terminate(self):
        try:
            self.plot_pipe.send(None,)
        except BrokenPipeError:
            pass


# class Stimulator_when(Node):
#     """Generate a marker stream from a config file
#     Attributes:
#       - output (Port): Output port
#     Args:
#       - file (str): Path to the xml config file
#
#     example: Stimulator('config_ov.xml)
#
#     """
#
#     # def __init__(self, file, input_port):
#         # Node.__init__(self, input_port, False)
#     def __init__(self, file, input_port, start_marker):
#         Node.__init__(self, input_port, True)
#         try:
#             config = Config(file)
#         except (ConfigFileNotInAccordance, FileNotFound, InvalidXml) as err:
#             print(err)
#             exit()
#
#         self._scenario = config.create_a_new_scenario()
#         self.start_marker = start_marker
#         self.start = False
#
#         self.output.set_parameters(
#             data_type='marker',
#             channels=['marker'],
#             sampling_frequency=0,
#             meta='')
#
#         # create the plot process
#         self.plot_pipe, plotter_pipe = mp.Pipe()
#         self.plotter = ProcessStim(len(self._scenario))
#         self.plot_process = mp.Process(
#             target=self.plotter, args=(plotter_pipe,), daemon=True)
#         self.plot_process.start()
#
#         Node.log_instance(self, {
#             'file': file,
#             'name': config.name,
#             'author': config.author,
#             'session': config.session,
#             'nb of trials': config.number_of_trials,
#             'classes': config.classes,
#             'random': config.random,
#             'type': config.type})
#         self._start_time = None
#
#     def update(self):
#
#         if not self.start:
#             # print('self.input: ', self.input)
#             # print('self.input._data: ', self.input._data)
#             for chunk in self.input:
#                 # print('chunk: ', chunk)
#                 # print(chunk[0].str.contains('R128').any())
#                 # print(chunk.iloc[:,0].str.contains('R128').any())
#                 if chunk.iloc[:,0].str.contains(self.start_marker).any():
#                     self.start = True
#
#         if self.start:
#             if not self._start_time:
#                 self._start_time = time()
#                 for markers in self._scenario:
#                     markers[1] = markers[1] + self._start_time
#             t = time()
#             while self._scenario and t >= self._scenario[0][1]:
#                 self.output.set(self._scenario[0][0], [self._scenario[0][1]], ['marker'])
#                 try:
#                     self.plot_pipe.send(self._scenario[0][0])
#                 except BrokenPipeError:
#                     pass
#                 self._scenario = self._scenario[1:]
#
#     def terminate(self):
#         try:
#             self.plot_pipe.send(None,)
#         except BrokenPipeError:
#             pass
