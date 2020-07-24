# description of the module file at the very beginning of the file
"""
Detailed template for creating basic custom nodes
For special and more complex nodes, please refer to the existing nodes

author: Simon Legeay, LaSEEB/CentraleSupélec
mail: simon.legeay.sup@gmail.com

"""


# import all usefull library
# example:
# from scipy import signal
# import numpy as np

from neuxus.node import Node


class MyNewNode(Node):
    """Add global description,
    Attributes:
      - output: output port
    Args:
      - input_port (Port): input port of type data_type
      - arg1 (specify type): description
      - arg2 (specify type): description, default is 4

    Example: MyNewNode(Port4, 8, 12, order=5)
    """

    def __init__(self, input_port, arg1, arg2=4):
        '''Initialize the node before running the pipeline'''

        # create self.input and self.output
        Node.__init__(self, input_port)

        # make sure you get the right input.data_type
        # data_type is either 'epoch', 'signal', 'marker' or 'spectrum'
        assert self.input.data_type in ['epoch', 'signal']

        # update self.output parameters
        self.output.set_parameters(
            # self.input.data_type if the data_type of output is the same as
            # in input or among ['epoch', 'signal', 'marker', 'spectrum']
            data_type=self.input.data_type,
            # self.input.channels if the output channels are the same as
            # in input else list of output channels (['Ch1', 'Ch2'] for example)
            channels=self.input.channels,
            # self.input.sampling_frequency if the output sampling frequency is the same as
            # in input or specify the new output sampling frequency
            sampling_frequency=self.input.sampling_frequency,
            # self.input.meta and/or add every details you want to add in meta
            meta=self.input.meta,
            # if the epoching frequency is unchanged (None or float), specify
            # self.input.epoching_frequency else specify the new epoching frequency
            epoching_frequency=self.input.epoching_frequency
        )

        # initialize parameters that will be useful for calculation:
        # ex:
        self._arg1 = arg1 / 10  # _arg means a protected arg
        self._channels = self.input.channels

        # log all the parameters you think it is useful to log
        Node.log_instance(self, {
            # specify the name of the parameter, and its value
            'my arg1': self._arg1
        })

    def update(self):
        # iter over the input, you might receive several chunks per each global NeuXus iteration
        for chunk in self.input:
            # chunk type depends on self.input.data_type:
            # 'marker': a DataFrame containing one marker
            # 'signal': a DataFrame containing one chunk of the signal
            # 'epoch': a DataFrame containig one epoch
            # 'spectrum': a special DataFrame containing the signal

            # to see more particularly what a chunk looks like add:
            print(chunk)

            # compute all necessary calculation
            # ...

            # to update the output port, use set or set_from_df
            # for example:
            self.output.set(rows=chunk.value, timestamps=chunk.index, columns=self._channels)
            # or
            self.output.set_from_df(chunk)
