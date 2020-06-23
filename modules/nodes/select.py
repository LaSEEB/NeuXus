import sys

import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


class ChannelSelector(Node):
    """Select a subset of signal channels
    Attributes:
     - output (port): output port
    Args:
     - mode ('index' or 'name'): indicate the way to select data
     - selected (list): column to be selected

    example: ChannelSelector(port1, port2, 'index', [2, 4, 5])
    or       ChannelSelector(port1, port2, 'name', ['Channel 2', 'Channel 4'])
    """

    def __init__(self, input_port, mode, selected):
        Node.__init__(self, input_port)

        assert mode in ['index', 'name']
        if mode == 'index':
            channels_name = [self.input.channels[i] for i in selected]
        elif mode == 'name':
            channels_name = selected

        self.output.set_parameters(
            channels=channels_name,
            frequency=self.input.frequency,
            meta=self.input.meta)

        self.mode = mode
        self.selected = selected

    def update(self):
        for chunk in self.input:
            if self.mode == 'name':
                self.output.set_from_df(chunk[self.selected])
            elif self.mode == 'index':
                self.output.set_from_df(chunk.iloc[:, self.selected])


class SpatialFilter(Node):
    """Maps M inputs to N outputs by multiplying the each input vector with a matrix
    usually used after a ChannelSelector
    Attributes:
     - output (port): output GroupOfPorts
    Args:
     - input (port): input port
     - matrix (dict): dictionnary with new channel name as keys and list of coefficients as values,
       list must be of the same length as input.channels

    example: SpatialFilter(input_port, matrix)
    where matrix = {
        'OC2': [4, 0, -1, 0],
        'OC3': [0, -1, 2, 4]
    }
    """

    def __init__(self, input_port, matrix):
        Node.__init__(self, input_port)

        # protected
        self._matrix = matrix
        self._channels = [*self._matrix.keys()]

        # verify that the size of _matrix is correct
        for chan in self._channels:
            assert len(self._matrix[chan]) == len(self.input.channels)

        self.output.set_parameters(
            channels=self._channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

    def update(self):
        for chunk in self.input:
            df = pd.DataFrame([])
            for chan in self._channels:
                flag = True  # use to first create the serie
                for index, coef in enumerate(self._matrix[chan]):
                    # print(f'{index} columns with coef {coef}')
                    if flag:
                        serie = chunk.iloc[:, index] * coef
                        flag = False
                    else:
                        serie += chunk.iloc[:, index] * coef
                df[chan] = serie
            self.output.set_from_df(df)
