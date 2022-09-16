from scipy import signal
import numpy as np
import pandas as pd
import math
from scipy.special import lpmv
import mne

from neuxus.node import Node


class ButterFilter(Node):
    """Bandpass filter for continuous signal
    Attributes:
        output: output port
    Args:
        input: get DataFrame and meta from input_ port
        lowcut (float): lowest frequence cut in Hz
        highcut (float): highest frequence cut in Hz
        order (int): order to be applied on the butter filter (recommended < 16),
        default value is 4

    Example: ButterFilter(Port4, 8, 12, order=5)
    """

    def __init__(self, input_port, lowcut, highcut, order=4):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        fs = self.input.sampling_frequency
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # calculate a and b, properties of the Butter filter
        self._b, self._a = signal.butter(
            order,
            [low, high],
            analog=False,
            btype='band',
            output='ba')

        # initial condition zi
        len_to_conserve = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((len(self.input.channels), len_to_conserve))

        Node.log_instance(self, {
            'lowcut': lowcut,
            'highcut': highcut,
            'order': order
        })

    def update(self):
        for chunk in self.input:
            # print('chunk = ', chunk)
            # print('self._zi = ', self._zi)
            # print('self._b = ', self._b)
            # print('self._a = ', self._a)
            # print('np.shape(chunk) = ', np.shape(chunk))
            # print('np.shape(self._zi) = ', np.shape(self._zi))
            # print('np.shape(self._b) = ', np.shape(self._b))
            # print('np.shape(self._a) = ', np.shape(self._a))
            # filter
            y, zf = signal.lfilter(self._b, self._a, chunk.transpose(), zi=self._zi)
            # zf are the future initial conditions
            self._zi = zf
            # update output port
            # GUSTAVO:
            # print('y = ', y)
            # print('np.shape(y) = ', np.shape(y))
            # print('zf = ', zf)
            # print('np.shape(zf) = ', np.shape(zf))

            # print('chunk.index = ', chunk.index)
            # print('self.input.channels = ', self.input.channels)
            self.output.set(np.array(y).transpose(),
                            chunk.index, self.input.channels)


class NotchFilter(Node):
    """Band-stop filter with a narrow bandwidth (high quality factor). It rejects a narrow
    frequency band and leaves the rest of the spectrum little changed.
    Attributes:
      - output: output port
    Args:
      - input: get DataFrame and meta from input_ port
      - frequency_to_remove(float): frequency to remove from the signal in Hz
      - quality_factor(float): Dimensionless parameter that characterizes notch
        filter -3 dB bandwidth bw relative to its center frequency, Q = frequency_to_remove/bw

    Example: NotchFilter(Port4, 10, 0.8)
    """

    def __init__(self, input_port, frequency_to_remove, quality_factor):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        # calculate a and b, properties of the Butter filter
        self._b, self._a = signal.iirnotch(
            w0=frequency_to_remove,
            Q=quality_factor,
            fs=self.input.sampling_frequency)

        # initial condition zi
        len_to_conserve = max(len(self._a), len(self._b)) - 1
        self._zi = np.zeros((len(self.input.channels), len_to_conserve))

        Node.log_instance(self, {
            'frequency to remove': frequency_to_remove,
            'quality factor': quality_factor
        })

    def update(self):
        for chunk in self.input:
            # filter
            y, zf = signal.lfilter(
                self._b, self._a, chunk.transpose(), zi=self._zi)
            # zf are the future initial conditions
            self._zi = zf
            # update output port
            self.output.set(np.array(y).transpose(),
                            chunk.index, self.input.channels)


class DownSample(Node):
    """Downsample the signal after applying an anti-aliasing filter
    Attributes:
      - output (Port): output signal port of sampling frequency / factor
    Args:
      - input (Port): input signal port
      - downsampling_factor (int): downsampling factor (recommanded under 13)

    Example: DownSample(port45, 5)
    """

    def __init__(self, input_port, downsampling_factor):
        Node.__init__(self, input_port)

        assert self.input.data_type == 'signal'

        self._downsampling_factor = int(downsampling_factor)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency / self._downsampling_factor,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        Node.log_instance(self, {
            'downsampling factor': self._downsampling_factor
        })
        self.persistent = pd.DataFrame([], [], self.input.channels)

    def update(self):
        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])
            nb_rows = len(self.persistent)
            n = nb_rows // self._downsampling_factor * self._downsampling_factor
            to_compute = self.persistent.iloc[:n, :]
            to_keep = self.persistent.iloc[n:, :]
            try:
                df = signal.decimate(to_compute, self._downsampling_factor, axis=0)
            except ValueError:
                # to_compute does not have enough rows to be downsampled
                pass
            else:
                index = [self.persistent.index[t * self._downsampling_factor] for t in range(len(df))]
                df = pd.DataFrame(df, index, self.input.channels)
                self.output.set(df, index, self.input.channels)
                self.persistent = to_keep

class Laplacian(Node):

    def __init__(self, input_port, loc=None, order=7, m=4, smoothing=10**(-5)):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            data_type=self.input.data_type,
            channels=self.input.channels,
            sampling_frequency=self.input.sampling_frequency,
            meta=self.input.meta,
            epoching_frequency=self.input.epoching_frequency
        )

        if isinstance(loc, mne.io.meas_info.Info):  # If read by Reader from .vhdr, head_size will be 0.085m
            channames = loc.ch_names
            chanlocs = loc['dig']
        else:
            montage = mne.channels.read_custom_montage(loc, head_size=0.095)
            channames = montage.ch_names
            chanlocs = montage.dig

        chanlocs = [chanlocs[c] for chan1 in self.input.channels for c, chan2 in enumerate(channames) if chan1 == chan2]
        if len(self.input.channels) != len(chanlocs):
            raise Exception("Some data channels were not found in the provided locations!")

        # # Check if channel names match channels from output
        # for c, channame in enumerate(self.input.channels):
        #     if channame != channames[c]:
        #         raise Exception("Data channels do not match channels from provided locations")

        # chanlocs = input_port.chanlocs[0]
        X = np.zeros(len(chanlocs))
        Y = np.zeros(len(chanlocs))
        Z = np.zeros(len(chanlocs))
        for c, chanloc in enumerate(chanlocs):
            X[c] = chanloc['r'][0]
            Y[c] = chanloc['r'][1]
            Z[c] = chanloc['r'][2]

        self.G, self.H = Laplacian.generate_laplacian_matrices(X, Y, Z, order, m)
        self.smoothing = smoothing


    @staticmethod
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    @staticmethod
    def generate_laplacian_matrices(X, Y, Z, order, m):
        nchans = len(X)

        # Set 'order' and 'm' in case of more than 100 electrodes
        if nchans > 100:
            order = 40
            m = 3

        # Calculate cosine distance
        _, _, r = Laplacian.cart2sph(X, Y, Z)
        maxr = max(r)
        X = X / maxr
        Y = Y / maxr
        Z = Z / maxr
        cosdist = np.zeros((nchans, nchans))
        for i in range(nchans):
            for j in range(i + 1, nchans):
                cosdist[i, j] = 1 - ((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 + (Z[i] - Z[j]) ** 2) / 2

        cosdist = cosdist + np.transpose(cosdist) + np.eye(nchans)

        # Legendre polynomial
        legpoly = np.zeros((order, nchans, nchans))
        for i in range(order):
            legpoly[i, :, :] = lpmv(0, i + 1, cosdist)

        # Calculate G and H
        orders = np.array(range(1, order + 1))
        twoN1 = 2 * orders + 1
        gdenom = (orders * (orders + 1)) ** m
        hdenom = (orders * (orders + 1)) ** (m - 1)
        G = np.zeros((nchans, nchans))
        H = np.zeros((nchans, nchans))
        for i in range(nchans):
            for j in range(i, nchans):
                g = 0
                h = 0
                for k in range(order):
                    g = g + (twoN1[k] * legpoly[k, i, j]) / gdenom[k]
                    h = h - (twoN1[k] * legpoly[k, i, j]) / hdenom[k]
                G[i, j] = g / (4 * math.pi)
                H[i, j] = -h / (4 * math.pi)

        G = G + np.transpose(G)
        H = H + np.transpose(H)
        G = G - np.eye(nchans) * G[0, 0] / 2
        H = H - np.eye(nchans) * H[0, 0] / 2
        return G, H

    def update(self):
        for chunk in self.input:
            data = np.asarray(chunk).copy()
            nchans = len(data)
            Gs = self.G + np.eye(nchans)*self.smoothing
            GsinvS = np.sum(np.linalg.inv(Gs), axis=0, keepdims=True)
            dataGs = np.transpose(np.linalg.lstsq(Gs, data)[0])
            C = dataGs - (np.sum(dataGs, axis=1, keepdims=True) / np.sum(GsinvS)) @ GsinvS
            lap = np.transpose(C @ np.transpose(self.H))
            self.output.set(lap, chunk.index, columns=self.input.channels)
