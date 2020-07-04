import sys

import random as rd
from time import time
import numpy as np

sys.path.append('../..')

from modules.node import Node


class Generator(Node):
    """Generate signals for testing purpose
    Attributes:
      - output (Port): Output port
    Args:
      - generator (str): Type of generator used among ['random', 'oscillator', 'simulation']
      - nb_channels (int): Number of channels to send in output
      - sampling_frequency (float): Sampling frequency of the output
      - min_chunk_size (int): Minimum number of rows per output chunk, default is 32

    example: Generator('random', 4, 500)
             Generator('oscillator', 16, 250)
             Generetor('simulation', 32, 500, 16)

    """

    def __init__(self, generator, nb_channels, sampling_frequency, min_chunk_size=32):
        Node.__init__(self, None)

        assert generator in ['random', 'oscillator', 'simulation']
        self._generator = generator
        self._sampling_frequency = sampling_frequency
        self._min_chunk_size = int(min_chunk_size)
        self._channels = [f'Ch{i}' for i in range(1, int(nb_channels) + 1)]
        self._min_period = self._min_chunk_size / self._sampling_frequency

        self.output.set_parameters(
            channels=self._channels,
            frequency=self._sampling_frequency,
            meta='')

        Node.log_instance(self, {
            'generator': self._generator,
            'frequency': self._sampling_frequency,
            'channels': self._channels})

        self._last_t = None

        if self._generator == 'simulation':
            highest_freq = 45
            lowest_freq = 0
            mid_freq = 15
            self._frequency = {}
            self._amplitude = {}
            self._baseline = {}
            for chan in self._channels:
                nb = rd.randint(1, 4)
                self._frequency[chan] = [rd.triangular(lowest_freq, highest_freq, mid_freq) for i in range(nb)]
                self._amplitude[chan] = [rd.triangular(10, 30) for i in range(nb)]
                self._baseline[chan] = rd.gauss(0, 10000)

    def update(self):
        t = time()
        if not self._last_t:
            self._last_t = t
        if t > self._min_period + self._last_t:
            nb_rows = int((t - self._last_t) * self._sampling_frequency)
            timestamps = np.array([self._last_t + i / self._sampling_frequency for i in range(1, nb_rows + 1)])
            self._last_t = timestamps[-1]
            data = []
            for i, chan in enumerate(self._channels):
                if self._generator == 'random':
                    data.append([rd.random() * 20 - 10 for row in range(nb_rows)])
                elif self._generator == 'oscillator':
                    data.append(np.sin(2 * np.pi * (5 + i) * timestamps) * (5 + 5 / (i + 1)))
                elif self._generator == 'simulation':
                    serie = np.zeros(len(timestamps))
                    for i, freq in enumerate(self._frequency[chan]):
                        serie += np.sin(2 * np.pi * freq * timestamps) * self._amplitude[chan][i]
                    serie += np.array([self._baseline[chan] + rd.random() * 20 for row in range(nb_rows)])
                    data.append(serie)
            self.output.set(np.array(data).transpose(), timestamps, self._channels)
