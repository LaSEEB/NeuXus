import sys

from scipy import signal
import numpy as np

sys.path.append('../..')

from modules.node import Node
from modules.registry import *


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
            self.output.set(self.amplitude_envelope,
                            chunk.index, self.input.channels)
