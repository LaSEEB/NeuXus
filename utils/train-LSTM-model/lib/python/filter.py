import numpy as np
import math
import scipy.signal as ssi
import pandas as pd
from scipy import signal


class Downsample:
    def __init__(self, fs, dfs, chans):
        self.ds = round(fs / dfs)
        # [self.ds1, self.ds2] = self.calc_closest_factors(self.ds)

        self.ds = int(self.ds)
        # self.ds1 = int(self.ds1)
        # self.ds2 = int(self.ds2)

        self.persistent = pd.DataFrame([], [], chans)

    def update(self, chunks):
        chunks_output = []
        chunk = chunks[0]
        self.persistent = pd.concat([self.persistent, chunk])
        npnts = len(self.persistent)
        n = int(npnts // self.ds * self.ds)
        to_compute = self.persistent.iloc[:n, :]
        to_keep = self.persistent.iloc[n:, :]
        try:
            # df = ssi.decimate(to_compute, self.ds1, axis=0)
            # df = ssi.decimate(df, self.ds2, axis=0)
            df = ssi.decimate(to_compute, self.ds, axis=0)

        except ValueError:
            # to_compute does not have enough rows to be downsampled
            pass
        else:
            index = [self.persistent.index[t * self.ds] for t in range(len(df))]
            df = pd.DataFrame(df, index, chunk.columns)
            self.persistent = to_keep
            # return df
            chunks_output.append(df)
            return chunks_output


    @staticmethod
    def calc_closest_factors(c: int):
        """Calculate the closest two factors of c.

        Returns:
          [int, int]: The two factors of c that are closest; in other words, the
            closest two integers for which a*b=c. If c is a perfect square, the
            result will be [sqrt(c), sqrt(c)]; if c is a prime number, the result
            will be [1, c]. The first number will always be the smallest, if they
            are not equal.
        """
        if c // 1 != c:
            raise TypeError("c must be an integer.")

        a, b, i = 1, c, 0
        while a < b:
            i += 1
            if c % i == 0:
                a = i
                b = c // a

        return [b, a]


class Filter:
    def __init__(self, order, f, fs, window='hamming', pass_zero=True):  # pass_zero=True => lowpass
        self.coefs = ssi.firwin(order + 1, f / (fs / 2), window=window, pass_zero=pass_zero)
        self.z = np.zeros((1, order))

    def update(self, chunk):
        pf, self.z = ssi.lfilter(self.coefs, 1, chunk.transpose(), zi=self.z)
        return pd.DataFrame(pf.transpose(), index=chunk.index, columns=chunk.columns)
        # return pd.DataFrame(pf, index=timestamps, columns=columns)


class Butter:
    def __init__(self, fs, flim, nchans, type='band', order=4):

        nyq = 0.5 * fs
        # low = lowcut / nyq
        # high = highcut / nyq
        flim_norm = [f/nyq for f in flim]

        # calculate a and b, properties of the Butter filter
        self._b, self._a = signal.butter(
            order,
            flim_norm,
            analog=False,
            btype=type,
            output='ba')

        # initial condition zi
        len_to_conserve = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((nchans, len_to_conserve))

    def update(self, chunks):
        chunks_output = []
        chunk = chunks[0]
        # filter
        y, zf = signal.lfilter(self._b, self._a, chunk.transpose(), zi=self._zi)
        # zf are the future initial conditions
        self._zi = zf
        # update output port
        # return y.transpose()


        # return pd.DataFrame(y.transpose(), index=chunk.index, columns=chunk.columns)
        chunks_output.append(pd.DataFrame(y.transpose(), index=chunk.index, columns=chunk.columns))
        return chunks_output


class Select:
    def __init__(self):
        pass

    @staticmethod
    def update(chunks, chan):
        chunks_output = []
        chunk = chunks[0]
        # return chunk.loc[:, [chan]]
        # return chunk[:][self.chan]
        chunks_output.append(chunk.loc[:, [chan]])
        return chunks_output
