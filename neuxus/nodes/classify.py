import joblib

from neuxus.node import Node


class Classify(Node):
    """Load a model from a joblib save and classify all input vector
    Attributes:
      - output (Port): signal output port
    Args:
      - model_file (str): path to the model file

    Example: Classify(Port4, 'LDA.sav')

    """

    def __init__(self, input_port, model_file):
        Node.__init__(self, input_port)
        # load model from save
        self._loaded_model = joblib.load(model_file)

        # verify the input signal type

        # set the ouput Port parameters
        self.output.set_parameters(
            data_type='signal',
            channels=['class'],
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta
        )
        # log the new instance
        Node.log_instance(self, {
            'path to model': model_file,
            'model': self._loaded_model
        })

    def update(self):
        for vector in self.input:
            self.output.set(self._loaded_model.predict(vector.values.tolist()), columns=['class'], timestamps=vector.index)


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
            frequency=self.input.sampling_frequency,
            meta=self.input.meta)

        Node.log_instance(self, {'model': self._loaded_model})

        # TO DO terminate

    def update(self):
        for vector in self.input:
            print(self._loaded_model.predict(vector.values.tolist()))"""