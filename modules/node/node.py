from scipy import signal
import pandas as pd
import numpy as np
import uuid
from time import time
from pylsl import (StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, pylsl)


class Send(object):
    """Send to a LSL stream.
    Attributes:
        i (Port): Default data input, expects DataFrame.
    Args:
        name (string): The name of the stream.
        type_ (string): The content type of the stream, .
        format (string): The format type for each channel. Currently, only ``double64`` and ``string`` are supported.
    to add    source (string, None): The unique identifier for the stream. If ``None``, it will be auto-generated.
    """

    _dtypes = {"double64": np.number, "string": np.object}

    def __init__(self, input_, name, frequency, type_="signal", format="double64", uuid_=None):
        self.name = name
        self.type = type_
        self.format = format
        self.frequency = frequency
        self.outlet = None
        self.input = input_
        if not uuid_:
            uuid_ = str(uuid.uuid4())
        self.uuid = uuid_
        self.connect()

    def connect(self):
        '''Create an outlet for streaming data'''
        if not self.outlet:

            # metadata
            info = StreamInfo(
                self.name,
                self.type,
                len(self.input.channels),
                self.frequency,
                self.format,
                self.uuid
            )
            channels = info.desc().append_child("channels")
            print('output' + str(self.input.channels))
            for label in self.input.channels:
                channels.append_child("channel")\
                    .append_child_value("label", str(label))\
                    .append_child_value("unit", "unknown")\
                    .append_child_value("type", "signal")

            # create the outlet
            self.outlet = StreamOutlet(info)

    def update(self):
        '''Send data found in input port'''
        for chunk in self.input:
            values = chunk.select_dtypes(
                include=[self._dtypes[self.format]]).values
            stamps = chunk.index.values.astype(np.float64)
            for row, stamp in zip(values, stamps):
                self.outlet.push_sample(row, stamp)


class Receive(object):
    """Receive from a LSL stream.
    Attributes:
        output_: provides DataFrame and meta
    Args:
        sync (string, None): The method used to synchronize timestamps. Use ``local`` if you receive the stream from another application on the same computer. Use ``network`` if you receive from another computer.
        max_samples (int): The maximum number of samples to return per call.
    """

    def __init__(self, output_, prop, value, sync="local", max_samples=1024 * 4, timeout=10.0):
        self.inlet = None
        self.labels = None
        self._prop = prop
        self._value = value
        self.sync = sync
        self.max_samples = max_samples
        self.offset = time() - pylsl.local_clock()
        self._timeout = timeout

        self.output = output_
        self.connect()

    def connect(self):
        if not self.inlet:
            # resolve streams
            streams = resolve_byprop(self._prop, self._value, timeout=self._timeout)
            if not streams:
                print('No streams found')
                raise Exception
            print('Stream acquired')
            # Stream acquired
            self.inlet = StreamInlet(streams[0])
            info = self.inlet.info()
            self.meta = {
                "name": info.name(),
                "type": info.type(),
                "frequency": info.nominal_srate(),
                "info": str(info.as_xml()).replace("\n", "").replace("\t", ""),
            }

            channels = []
            if not info.desc().child("channels").empty():
                channel = info.desc().child("channels").child("channel")
                for _ in range(info.channel_count()):
                    channel_name = channel.child_value("label")
                    channels.append(channel_name)
                    channel = channel.next_sibling()
            self.channels = channels
            print('Available channels:')
            print(*channels, sep='\n')
            print()
            print(self.meta)

            self.output.set_channels(channels)
            self.output.set_meta(self.meta)
            self.output.set_frequency(info.nominal_srate())

    def update(self):
        if self.inlet:
            values, stamps = self.inlet.pull_chunk(max_samples=self.max_samples)
            if stamps:
                stamps = np.array(stamps)
                if self.sync == "local":
                    stamps += self.offset
                elif self.sync == "network":
                    stamps = stamps + self.inlet.time_correction() + self.offset
                # stamps = pd.to_datetime(stamps, format=None)
            if len(stamps) > 0:
                self.output.set(values, stamps, self.channels)
        else:
            return


class ButterFilter(object):
    """Bandpass filter for continuous signal
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output port
    Args:
        lowcut (float): lowest frequence cut
        highcut (float): highest frequence cut
        fs (float): sampling frequence
        order (int): order to be applied on the butter filter (recommended < 16)
    """

    def __init__(self, input_, output_, lowcut, highcut, order=4):
        super().__init__()
        self.input = input_
        self.output = output_

        self.output.set_frequency(self.input.frequency)
        self.output.set_channels(self.input.channels)

        fs = self.input.frequency
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # calcuate a and b, properties of the Butter filter
        self.b, self.a = signal.butter(
            order,
            [low, high],
            analog=False,
            btype='band',
            output='ba')

        # initial condition zi
        len_to_conserve = max(len(self.a), len(self.b)) - 1
        self.zi = np.zeros((len(self.input.channels), len_to_conserve))

    def update(self):
        for chunk in self.input:
            # filter
            y, zf = signal.lfilter(self.b, self.a, chunk.transpose(), zi=self.zi)
            # zf are the future initial condition
            self.zi = zf
            # update output port
            self.output.set(np.array(y).transpose(), chunk.index, self.input.channels)


class Epoching(object):
    """Cut a continuous signal in epoch of same duration
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_, output_, duration):
        super().__init__()
        self.input = input_
        self.output = output_
        self.duration = duration

        # self.output.set_frequency(1 / duration)
        # self.output.set_channels(self.input.channels)

        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.trigger = None

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            # trigger points to the oldest data in persistence
            if not self.trigger:
                self.trigger = float(chunk.index.values[1])

            # if the new chunk complete an epoch:
            if float(chunk.index[-1]) >= self.trigger + self.duration:
                # number of epoch that can be extracted
                iter_ = int((float(chunk.index[-1]) - self.trigger) / self.duration)
                dfcon = pd.concat([self.persistent, chunk])

                # TO DO treat the case of a working frequency slower than epoching (ie i > 1)
                for i in range(iter_):
                    epoch = dfcon[lambda x: x.index < self.trigger + self.duration]
                    y = dfcon.iloc[lambda x: x.index >= self.trigger + self.duration]
                    self.trigger = self.trigger + self.duration

                    self.output.set_from_df(epoch)
                self.persistent = y
            else:
                self.persistent = pd.concat([self.persistent, chunk])


class Averaging(object):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_, output_):
        super().__init__()
        self.input = input_
        self.output = output_

        # TO DO terminate

    def update(self):
        for epoch in self.input:
            self.output.set_from_df(pd.DataFrame(epoch.mean(), columns=[epoch.index[-1]]).transpose())


class ApplyFunction(object):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_, output_, function, args=()):
        self.input = input_
        self.output = output_

        self.output.set_frequency(self.input.frequency)
        self.output.set_channels(self.input.channels)

        self.function = function
        self.args = args

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            self.output.set_from_df(chunk.apply(self.function, args=self.args))


class ChannelSelector(object):
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

    def __init__(self, input_, output_, mode, selected):
        assert mode in ['index', 'name']
        self.input = input_
        self.output = output_

        self.output.set_frequency(self.input.frequency)
        self.output.set_channels(selected)

        self.mode = mode
        self.selected = selected

    def update(self):
        for chunk in self.input:
            if self.mode == 'name':
                self.output.set_from_df(chunk[self.selected])
            elif self.mode == 'index':
                self.output.set_from_df(chunk.iloc[:, self.selected])
