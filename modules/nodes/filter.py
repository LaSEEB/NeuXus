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
        input_: get DataFrame and meta from input_ port
        output_: output port
    Args:
        lowcut (float): lowest frequence cut
        highcut (float): highest frequence cut
        fs (float): sampling frequence
        order (int): order to be applied on the butter filter (recommended < 16)
    """

    def __init__(self, input_port, lowcut, highcut, order=4):
        Node.__init__(self, input_port)

        logging.info(f'Instanciate a ButterFilter with parameters:'
                     f'\n   input_port {input_port}'
                     f'\n   lowcut {lowcut}'
                     f'\n   highcut {highcut}'
                     f'\n   order {order}')

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        fs = self.input.frequency
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # calculate a and b, properties of the Butter filter
        self.b, self.a = signal.butter(
            order,
            [low, high],
            analog=False,
            btype='band',
            output='ba')

        # initial condition zi
        len_to_conserve = max(len(self.a), len(self.b)) - 1
        self.zi = np.zeros((len(self.input.channels), len_to_conserve))

    def update(self):
        for chunk in self.input:
            # filter
            y, zf = signal.lfilter(
                self.b, self.a, chunk.transpose(), zi=self.zi)
            # zf are the future initial condition
            self.zi = zf
            # update output port
            self.output.set(np.array(y).transpose(),
                            chunk.index, self.input.channels)
