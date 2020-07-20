import sys

from scipy import signal
import numpy as np

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
        order (int): order to be applied on the butter filter (recommended < 16),
        default value is 4

    Example: ButterFilter(Port4, 8, 12, order=5)
    """

    def __init__(self, input_port, lowcut, highcut, order=4):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        fs = self.input.sampling_frequency
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

        Node.log_instance(self, {
            'lowcut': lowcut,
            'highcut': highcut,
            'order': order
        })

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


class NotchFilter(Node):
    """Band-stop filter with a narrow bandwidth (high quality factor). It rejects a narrow
    frequency band and leaves the rest of the spectrum little changed.
    Attributes:
      - output: output port
    Args:
      - input: get DataFrame and meta from input_ port
      - frequency_to_remove(float): frequency to remove from the signal in Hz
      - quality_factor(float): Dimensionless parameter that characterizes notch
        filter -3 dB bandwidth bw relative to its center frequency, Q = frequency_to_remove/bw

    Example: NotchFilter(Port4, 10, 0.8)
    """

    def __init__(self, input_port, frequency_to_remove, quality_factor):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        # calculate a and b, properties of the Butter filter
        self._b, self._a = signal.iirnotch(
            w0=frequency_to_remove,
            Q=quality_factor,
            fs=self.input.sampling_frequency)

        # initial condition zi
        len_to_conserve = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((len(self.input.channels), len_to_conserve))

        Node.log_instance(self, {
            'frequency to remove': frequency_to_remove,
            'quality factor': quality_factor
        })

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


class DownSample(Node):
    """Downsample the signal after applying an anti-aliasing filter
    Attributes:
      - output (Port): output signal port of sampling frequency / factor
    Args:
      - input (Port): input signal port
      - downsampling_factor (int): downsampling factor (recommanded under 13)

    Example: DownSample(port45, 5)
    """

    def __init__(self, input_port, downsampling_factor):
        Node.__init__(self, input_port)

        assert self.input.data_type == 'signal'

        self._downsampling_factor = int(downsampling_factor)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency / self._downsampling_factor,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        Node.log_instance(self, {
            'downsampling factor': self._downsampling_factor
        })
        self.persistent = pd.DataFrame([], [], self.input.channels)

    def update(self):
        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])
            nb_rows = len(self.persistent)
            n = nb_rows // self._downsampling_factor * self._downsampling_factor
            to_compute = self.persistent.iloc[:n, :]
            to_keep = self.persistent.iloc[n:, :]
            try:
                df = signal.decimate(to_compute, self._downsampling_factor, axis=0)
            except ValueError:
                # to_compute does not have enough rows to be downsampled
                pass
            else:
                index = [self.persistent.index[t * self._downsampling_factor] for t in range(len(df))]
                df = pd.DataFrame(df, index, self.input.channels)
                self.output.set(df, index, self.input.channels)
                self.persistent = to_keep
