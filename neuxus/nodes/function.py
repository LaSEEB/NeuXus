from neuxus.node import Node


class ApplyFunction(Node):
    """Apply function along rows
    Attributes:
      - output: output Port
    Args:
      - function: function to apply, the function can take in input a row or
        a np.array of shape number of input channels as columns and 1 row
        To perform calculation with an unvariateState output it is possible to include
        the .value attribute of this Node (see example)

    Example:
        def f(x):
            return x - 4
        ApplyFunction(port4, f)
        or
        def f(x):
            return x - np.array([3, 2, 5, -1])
        ApplyFunction(port4, f)
        or
        stat = UnivariateStat(port5, 'mean')
        def f(x):
            return x - stat.value
        ApplyFunction(port4, f)

    """

    def __init__(self, input_port, function, *args):
        Node.__init__(self, input_port)

        assert self.input.data_type in ['epoch', 'signal']

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency)

        self.args = args
        self.function = function
        self.y = []

        Node.log_instance(self, {
            'function': self.function
        })

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            self.output.set_from_df(chunk.apply(self.function, args=self.args, axis=1, raw=True))
