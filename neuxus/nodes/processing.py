from scipy import signal
import numpy as np
from scipy.fft import fft

from neuxus.node import Node


class HilbertTransform(Node):
    """Compute the analytic signal, using the Hilbert transform.
    Attributes:
        output: output port
    Args:
        input: get DataFrame and meta from input_ port

    Example: hilbert = processing.HilbertTransform(port)
    """

    def __init__(self, input_port):
        Node.__init__(self, input_port)

        assert self.input.data_type in ['epoch', 'signal']

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        Node.log_instance(self, {})

    def update(self):
        for chunk in self.input:
            self.analytic_signal = signal.hilbert(chunk)
            self.amplitude_envelope = np.abs(self.analytic_signal)
            # update output port
            print(self.amplitude_envelope)
            self.output.set(self.amplitude_envelope,
                            chunk.index, self.input.channels)


class PsdWelch(Node):
    """Compute the Power Spectral Density on received chunk
    The number of points used for calculation is the number of points
    received by chunks in input
    Attributes:
      - output (Port): output port of type 'spectrum'
    Args:
      - input (Port): input port of type 'epoch or 'signal'

    Example:
        welch = processing.PsdWelch(port99)
    """

    def __init__(self, input_port):
        Node.__init__(self, input_port)

        assert self.input.data_type in ['epoch', 'signal']

        self.output.set_parameters(
            data_type='spectrum',
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        Node.log_instance(self, {})

    def update(self):
        for chunk in self.input:
            frequency, value = signal.welch(chunk.transpose(), self.input.sampling_frequency, nperseg=len(chunk))
            # update output port
            frequency = [f for f in frequency]
            self.output.set(value, self.input.channels, frequency)


class Fft(Node):
    """Compute the Power Spectral Density on received chunk
    The number of points used for calculation is the number of points
    received by chunks in input
    Attributes:
      - output (Port): output port of type 'spectrum'
    Args:
      - input (Port): input port of type 'epoch or 'signal'

    Example:
        welch = processing.PsdWelch(port99)
    """

    def __init__(self, input_port):
        Node.__init__(self, input_port)

        assert self.input.data_type in ['epoch', 'signal']

        self.output.set_parameters(
            data_type='spectrum',
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        Node.log_instance(self, {})

        self._T = 1 / self.input.sampling_frequency

    def update(self):
        for chunk in self.input:
            N = len(chunk)
            # compute FFT
            value = fft(chunk.transpose())
            # create x
            xf = np.linspace(0.0, 1 / (2 * self._T), N // 2)
            # extract the first value and take abs
            v = [2 / N * np.abs(i[0:N // 2]) for i in value]
            # update output port
            self.output.set(v, self.input.channels, [i for i in xf])
