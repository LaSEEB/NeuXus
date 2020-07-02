import sys

import pandas as pd

sys.path.append('../..')

from modules.node import Node


class FeatureAggregator(Node):
    """Convert a signal to a CSV file
    Args:
      - file (str): CSV file to write
      - sep (str): Separator between rows, default is ';'
      - decimal (str): Character recognized as decimal separator, default is ','

    example: ToCsv(port4, 'log.csv')

    """

    def __init__(self, input_port, class_tag=None):
        Node.__init__(self, input_port)
        self._tag = class_tag
        if self._tag:
            self._channels = ['class'] + self.input.channels
        else:
            self._channels = self.input.channels
        self.output.set_parameters(
            channels=self._channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        Node.log_instance(self, {'tag': self._tag, 'coordinates': self._channels})
        self._i = 0

    def update(self):
        for chunk in self.input:
            for _, row in chunk.iterrows():
                row = row.values.tolist()
                if self._tag:
                    row = [self._tag] + row
                self.output.set([row], [self._i], self._channels)
                self._i += 1
