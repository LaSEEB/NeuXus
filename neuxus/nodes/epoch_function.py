import pandas as pd
import numpy as np
from scipy import signal, stats

from neuxus.node import Node


class UnivariateStat(Node):
    """Mean, Variance, Median, etc. on the incoming epoched Signal
    perform calculation on each coming epoch and save last value in value
    attribute and update the output
    Attributes:
      - output (Port): signal output port
      - value (float): last value calculated (on the last epoch come)
    Args:
      - input (Port): port of incoming epoched data
      - stat (str):
      - quantile (float between 0 and 1): default is None, used if stat is 'quantile',
        precise the quantile to calculate
      - iqr_quantile (list of two float between 0 and 1): default is [None, None], used
        if stat is 'iqr', precise the limit of range

    Example: UnivariateStat(Port18, 'mean')
             UnivariateStat(Port18, 'quantile', quantile=0.23)
             UnivariateStat(Port18, 'iqr', iqr_quantile=[0.23, 0.78])

    """

    def __init__(self, input_port, stat, quantile=None, iqr_quantile=[None, None], ttest_mean=None):
        Node.__init__(self, input_port)

        assert self.input.data_type == 'epoch'

        assert stat in ['mean', 'min', 'max', 'range', 'std', 'median', 'quantile', 'iqr', 'ttest_1samp']
        self._stat = stat
        self._ttest_mean = ttest_mean

        self.output.set_parameters(
            data_type='signal',
            channels=self.input.channels,
            sampling_frequency=self.input.epoching_frequency,
            meta=self.input.meta
        )

        # value attribute in initialized at 0
        self.value = np.array([0] * len(self.input.channels))

        if self._stat == 'quantile':
            assert 0 <= quantile and quantile <= 1
            self._q = quantile
            Node.log_instance(self, {
                'output frequency': self.input.epoching_frequency,
                'stat': self._stat,
                'quantile': self._q})

        elif self._stat == 'iqr':
            q1 = iqr_quantile[0]
            q2 = iqr_quantile[1]
            assert 0 <= q1 and q1 <= q2 and q2 <= 1
            self._q1 = q1
            self._q2 = q2
            Node.log_instance(self, {
                'output frequency': self.input.epoching_frequency,
                'stat': self._stat,
                'quantile1': self._q1,
                'quantile2': self._q2})
        else:
            Node.log_instance(self, {
                'output frequency': self.input.epoching_frequency,
                'stat': self._stat})

    def update(self):
        for epoch in self.input:
            if self._stat == 'mean':
                stat = epoch.mean()
            elif self._stat == 'min':
                stat = epoch.min()
            elif self._stat == 'max':
                stat = epoch.max()
            elif self._stat == 'std':
                stat = epoch.std()
            elif self._stat == 'range':
                stat = epoch.max() - epoch.min()
            elif self._stat == 'quantile':
                stat = epoch.quantile(q=self._q)
            elif self._stat == 'median':
                stat = epoch.median()
            elif self._stat == 'iqr':
                stat = epoch.quantile(q=self._q2) - epoch.quantile(q=self._q1)
            elif self._stat == 'ttest_1samp':
                tstat, pval = stats.ttest_1samp(epoch, self._ttest_mean.value, axis=0, nan_policy='propagate', alternative='less')
                stat = pd.Series(data=pval, index=epoch.columns)

            self.output.set_from_df(pd.DataFrame(
                stat, columns=[epoch.index[-1]]).transpose())
            self.value = np.array(stat.values)


class Windowing(Node):
    """Apply a windowing function to the input epohed signal
    Attributes:
      - output (Port): epoch output port
    Args:
      - input (Port): port of epoched incoming data
      - window (str): to be choosen between ['blackman', 'hanning', 'hamming', 'triang']

    Example: Windowing(Port48, 'blackman')
    """

    def __init__(self, input_port, window):
        Node.__init__(self, input_port)

        assert self.input.data_type == 'epoch'

        assert window in ['blackman', 'hanning', 'hamming', 'triang']
        self._window = window

        self.output.set_parameters(
            data_type='epoch',
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        self._buffered_windows = {}

        Node.log_instance(self, {'window': self._window})

    def update(self):
        for epoch in self.input:
            nb_rows = len(epoch)
            if nb_rows not in self._buffered_windows:
                print('rcalculed')
                if self._window == 'blackman':
                    self._buffered_windows[nb_rows] = signal.blackman(nb_rows)
                elif self._window == 'hanning':
                    self._buffered_windows[nb_rows] = signal.hanning(nb_rows)
                elif self._window == 'hamming':
                    self._buffered_windows[nb_rows] = signal.hamming(nb_rows)
                elif self._window == 'triang':
                    self._buffered_windows[nb_rows] = signal.triang(nb_rows)
            self.output.set_from_df((epoch.transpose() * self._buffered_windows[nb_rows]).transpose())
