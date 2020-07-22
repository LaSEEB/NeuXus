import sys

import logging
from tkinter import *
import pandas as pd

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

from matplotlib import style
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('../..')

from modules.node import Node

style.use('fivethirtyeight')

"""
Author: Simon Legeay, LaSEEB/CentraleSupélec
e-mail: legeay.simon.sup@gmail.com


Gathers all nodes for signal display and Graz
Includes:
    ProcessPlotter
    Plot(Node)
    Markers
    CustomCanvas
    ProcessGraz
    Graz(Node)
    ProcessSpectralPlotter
    PlotSpectrum(Node)

Plot and Plotspectrum from display.py may make the pipeline slower
because of the use of matplotlib.pyplot. These Nodes may be used to debug
or run on another computer not in the main pipeline
"""


class ProcessPlotter:
    """Class used for plotting signals, launched in a subprocess"""

    def __init__(self, duration, channels, frequency):
        self.trigger = 0
        self.persistent = None
        self.duration = duration
        self.channels = channels
        self.frequency = frequency
        self._x = [i / frequency for i in range(int(duration * frequency) + 1)]
        self.nb = len(self._x) - 1
        self._y = {chan: ['NaN'] * self.nb for chan in channels}
        self.flag = True

    def terminate(self):
        plt.close('all')

    def __call__(self, pipe):

        self.pipe = pipe
        self.fig, self.ax = plt.subplots(len(self.channels), sharex='col')
        if len(self.channels) == 1:
            self.ax = [self.ax]

        def animate(i):
            to_add = pd.DataFrame([], columns=self.channels, index=[])
            while self.pipe.poll():
                command, start, end = self.pipe.recv()
                if command is None:
                    self.terminate()
                to_add = pd.concat([to_add, command])
            try:
                if not self.persistent:
                    self.persistent = pd.DataFrame(
                        [], columns=self.channels, index=[])
            except ValueError:
                pass
            indexes = [i - to_add.index[0] + 1 / self.frequency +
                       self.trigger for i in to_add.index]
            to_add = pd.DataFrame(
                to_add.values, columns=self.channels, index=indexes)
            if indexes:
                trigger = indexes[-1] % self.duration
                a = indexes[-1] // self.duration * self.duration
                if a == 0:
                    self.persistent = pd.concat([self.persistent.iloc[
                                                lambda x: x.index <= self.trigger], to_add, self.persistent.iloc[lambda x: x.index > trigger]])
                    self.trigger = trigger
                else:
                    self.persistent = self.persistent.iloc[
                        lambda x: x.index <= a - self.duration + self.trigger]
                    self.persistent = pd.concat([self.persistent, to_add])
                    self.start = self.persistent.iloc[lambda x: x.index >= a]
                    self.persistent = self.persistent.iloc[
                        lambda x: x.index > a - self.duration + trigger].iloc[lambda x: x.index <= a]
                    index = [i - a for i in self.start.index]
                    self.persistent = pd.concat([pd.DataFrame(
                        self.start.values, index=index, columns=self.channels), self.persistent])
                    self.trigger = trigger
                for index, chan in enumerate(self.channels):
                    self.ax[index].clear()
                    self.ax[index].plot(self.persistent.loc[
                                        :, chan], linewidth=.5, color='blue')
                plt.xlim(0, self.duration)

        ani = animation.FuncAnimation(self.fig, animate, interval=10)
        plt.rc('ytick', labelsize=8)
        plt.rc('xtick', labelsize=8)
        plt.show()


class Plot(Node):
    """Display a signal (epoched or not) in a matplotlib window
    Args:
      - input_port (Port): input signal
      - duration (float): define the length of the observation segment
      - channels (str or list): if 'all', display all channels, else diplay specified
        channels according the way (index or name)
      - way (str): 'name' or 'index', specified the way to choose channels

    example: Plot(port4, duration=5)

    """

    def __init__(self, input_port, duration=10, channels='all', way='index'):
        Node.__init__(self, input_port, None)

        if channels == 'all':
            self._channels = self.input.channels
        else:
            if way == 'index':
                self._channels = [self.input.channels[i - 1] for i in channels]
            elif way == 'name':
                self._channels = channels

        self._duration = float(duration)

        # create the plot process
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(
            self._duration, self._channels, self.input.sampling_frequency)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        # Log new instance
        Node.log_instance(self, {
            'duration': self._duration,
            'channels': self._channels
        })

    def update(self):
        for chunk in self.input:
            end_time = chunk.index[-1]
            try:
                self.plot_pipe.send(
                    (chunk.loc[:, self._channels], end_time - self._duration, end_time))
            except BrokenPipeError:
                pass

    def terminate(self):
        try:
            self.plot_pipe.send((None, None, None))
        except BrokenPipeError:
            pass


MARKERS = {
    'show_cross': 786,
    'show_rigth_arrow': 770,
    'show_left_arrow': 769,
    'hide_arrow': 781,
    'hide_cross': 800,
    'exit_': 1010,
}


class CustomCanvas(Canvas):
    """Custom Canvas fitting with graz visualization with function to update the content
    according the windows size, and plot functions"""

    def __init__(self, parent, **kwargs):
        Canvas.__init__(self, parent, **kwargs)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.on_resize)

        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def show_cross(self):
        """Show cross on the window"""
        x1 = int(self.width / 3)
        x2 = int(self.width * 2 / 3)
        x3 = int(self.width / 2)
        y1 = int(self.height / 2)
        y2 = int(self.height / 3)
        y3 = int(self.height * 2 / 3)
        self.create_line(x1, y1, x2, y1,
                         fill="white", width=1, tags='cross1')
        self.create_line(x3, y2, x3, y3,
                         fill="white", width=1, tags='cross2')

    def show_left_arrow(self):
        """Show red arrow on the window"""
        x3 = int(self.width / 2)
        x4 = int(self.width * 7 / 18)
        y1 = int(self.height / 2)
        w = int(self.width / 25)
        s = f"{w} {w} {int(0.7 * w)}"
        self.create_line(x4, y1, x3, y1,
                         fill="red", width=w, tags='ra',
                         arrow='first', arrowshape=s)

    def show_rigth_arrow(self):
        """Show red arrow on the window"""
        x3 = int(self.width / 2)
        x5 = int(self.width * 11 / 18)
        y1 = int(self.height / 2)
        w = int(self.width / 25)
        s = f"{w} {w} {int(0.7 * w)}"
        self.create_line(x5, y1, x3, y1,
                         fill="red", width=w, tags='la',
                         arrow='first', arrowshape=s)

    def hide_arrow(self):
        """Hide all arrows on the window"""
        self.delete('la')
        self.delete('ra')

    def hide_cross(self):
        """Hide the cross on the window"""
        self.delete('cross1')
        self.delete('cross2')

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


class ProcessGraz():
    """Create a new window containing the Graz visualization
    Launched in a subprocess"""

    def __init__(self):
        pass

    def terminate(self):
        self.root.destroy()

    def call_back(self):
        if self.pipe.poll():
            sample = self.pipe.recv()
            if sample[0] == MARKERS['show_cross']:
                self.myCanvas.show_cross()
            elif sample[0] == MARKERS['show_rigth_arrow']:
                self.myCanvas.show_rigth_arrow()
            elif sample[0] == MARKERS['show_left_arrow']:
                self.myCanvas.show_left_arrow()
            elif sample[0] == MARKERS['hide_arrow']:
                self.myCanvas.hide_arrow()
            elif sample[0] == MARKERS['hide_cross']:
                self.myCanvas.hide_cross()
            elif sample[0] == MARKERS['exit_']:
                self.terminate()
                return
        self.root.after(10, self.call_back)

    def __call__(self, pipe):
        self.pipe = pipe
        # initialise a window.
        self.root = Tk()
        self.root.config(background='black')
        self.root.title('Graz Visualization')
        self.root.geometry("500x350")

        # create a Frame
        myframe = Frame(self.root)
        myframe.pack(fill=BOTH, expand=YES)

        # add a canvas
        self.myCanvas = CustomCanvas(myframe, width=425, height=2,
                                     bg="black", highlightthickness=0)
        self.myCanvas.pack(fill=BOTH, expand=YES)
        self.root.after(0, self.call_back)
        self.root.mainloop()


class Graz(Node):
    """Display the Graz visualization in a new tk window
    Args:
      - input_port (Port): input marker signal

    example: Graz(port4)

    """

    def __init__(self, input_port):
        Node.__init__(self, input_port, None)

        # create the plot process
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessGraz()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        # Log new instance
        Node.log_instance(self, {
        })

    def update(self):
        for chunk in self.input:
            try:
                self.plot_pipe.send(chunk.values[0])
            except BrokenPipeError:
                pass

    def terminate(self):
        try:
            self.plot_pipe.send(None)
        except BrokenPipeError:
            pass


class ProcessSpectralPlotter:
    """Class used for plotting spectral signals, launched in a subprocess"""

    def __init__(self, channels):
        self._channels = channels

    def terminate(self):
        plt.close('all')

    def __call__(self, pipe):

        self.pipe = pipe
        self.fig, self.ax = plt.subplots(len(self._channels), sharex='col')
        if len(self._channels) == 1:
            self.ax = [self.ax]

        def animate(i):
            while self.pipe.poll():
                df = self.pipe.recv()
                if df is None:
                    self.terminate()
                for index, chan in enumerate(self._channels):
                    self.ax[index].clear()
                    self.ax[index].plot(df.loc[[chan], :].transpose(), linewidth=.5)

        ani = animation.FuncAnimation(self.fig, animate, interval=10)
        plt.rc('ytick', labelsize=8)
        plt.rc('xtick', labelsize=8)
        plt.show()


class PlotSpectrum(Node):
    """Display a spectrum signal in a matplotlib window
    Args:
      - input_port (Port): input signal of type 'spectrum'
      - channels (str or list): if 'all', display all channels, else diplay specified
        channels according the way (index or name)
      - way (str): 'name' or 'index', specified the way to choose channels

    example: PlotSpectrum(port4)

    """

    def __init__(self, input_port, channels='all', way='index'):
        Node.__init__(self, input_port, None)

        assert self.input.data_type == 'spectrum'

        if channels == 'all':
            self._channels = self.input.channels
        else:
            if way == 'index':
                self._channels = [self.input.channels[i - 1] for i in channels]
            elif way == 'name':
                self._channels = channels

        # create the plot process
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessSpectralPlotter(self._channels)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        # Log new instance
        Node.log_instance(self, {
            'channels': self._channels
        })

    def update(self):
        for chunk in self.input:
            try:
                self.plot_pipe.send(chunk.loc[self._channels, :])
            except BrokenPipeError:
                pass

    def terminate(self):
        try:
            self.plot_pipe.send(None)
        except BrokenPipeError:
            pass
