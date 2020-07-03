import sys

import joblib

sys.path.append('../..')

from modules.node import Node


class Classify(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, model_file):
        Node.__init__(self, input_port)
        self._loaded_model = joblib.load(model_file)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        Node.log_instance(self, {'model': self._loaded_model})

        # TO DO terminate

    def update(self):
        for vector in self.input:
            print(self._loaded_model.predict(vector.values.tolist()))


"""class Train(Node):
    "TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    "

    def __init__(self, input_port, model_file):
        Node.__init__(self, input_port)
        self._loaded_model = joblib.load(model_file)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        Node.log_instance(self, {'model': self._loaded_model})

        # TO DO terminate

    def update(self):
        for vector in self.input:
            print(self._loaded_model.predict(vector.values.tolist()))"""