import sys
import os

import pandas as pd
from matplotlib import style
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

sys.path.append('../..')

from modules.node import Node

style.use('fivethirtyeight')


class ProcessPlotter:
    """Class used for plotting signals"""

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
