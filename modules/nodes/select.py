import sys

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


class ChannelSelector(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        mode ('index' or 'name'): indicate the way to select data
        selected (list): column to be selected

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
