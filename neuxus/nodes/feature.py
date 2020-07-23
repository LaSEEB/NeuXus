from neuxus.node import Node


class FeatureAggregator(Node):
    """Each chunk of input will be catenated into one feature vector that can
    be used for classification. It can specified a class as first vector coordinate
    Attributes:
      - output (Port): vector output Port
    Args:
      - input (Port): input signal
      - class_tag (str): class tag to add at first coordinate

    example: FeatureAggregator(port4, 'RIGHT')

    """

    def __init__(self, input_port, class_tag=None):
        Node.__init__(self, input_port)
        self._tag = class_tag
        if self._tag:
            self._channels = ['class'] + self.input.channels
        else:
            self._channels = self.input.channels

        self.output.set_parameters(
            data_type='vector',
            channels=self._channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta
        )

        Node.log_instance(self, {
            'tag': self._tag,
            'coordinates': self._channels
        })
        self._i = 0

    def update(self):
        for chunk in self.input:
            for _, row in chunk.iterrows():
                row = row.values.tolist()
                if self._tag:
                    row = [self._tag] + row
                self.output.set([row], [self._i], self._channels)
                self._i += 1
