import sys

from scipy import signal
import numpy as np
import logging

sys.path.append('../..')

from modules.node import Node
from modules.registry import *


class ButterFilter(Node):
    """Bandpass filter for continuous signal
    Attributes:
        output: output port
    Args:
        input: get DataFrame and meta from input_ port
        lowcut (float): lowest frequence cut in Hz
        highcut (float): highest frequence cut in Hz
        order (int): order to be applied on the butter filter (recommended < 16)
    """

    def __init__(self, input_port, lowcut, highcut, order=4):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        fs = self.input.frequency
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # calculate a and b, properties of the Butter filter
        self._b, self._a = signal.butter(
            order,
            [low, high],
            analog=False,
            btype='band',
            output='ba')

        # initial condition zi
        len_to_conserve = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((len(self.input.channels), len_to_conserve))

        Node.log_instance(self, {'lowcut': lowcut, 'highcut': highcut, 'order': order})

    def update(self):
        for chunk in self.input:
            # filter
            y, zf = signal.lfilter(
                self._b, self._a, chunk.transpose(), zi=self._zi)
            # zf are the future initial conditions
            self._zi = zf
            # update output port
            self.output.set(np.array(y).transpose(),
                            chunk.index, self.input.channels)
