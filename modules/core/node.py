import sys

from scipy import signal
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import uuid
from time import time
from pylsl import (StreamInfo, StreamOutlet,
                   StreamInlet, resolve_byprop, pylsl)

sys.path.append('.')
sys.path.append('../..')

from modules.core.port import Port
from modules.core.keepref import KeepRefsFromParent


class Node(KeepRefsFromParent, ABC):
    """Abstract class for Node objects with both input and output port"""

    def __init__(self, input_port):
        super(Node, self).__init__()
        self.input = input_port
        self.output = Port()

    @abstractmethod
    def update(self):
        for chunk_or_epoch in self.input:
            pass


class LslSend(Node):
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

    def __init__(self, input_port, name, type_="signal", format="double64", uuid_=None):
        Node.__init__(self, input_port)
        self.name = name
        self.type = type_
        self.format = format
        self.outlet = None
        self.frequency = self.input.frequency
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


class LslReceive(Node):
    """Receive from a LSL stream.
    Attributes:
        output_: provides DataFrame and meta
    Args:
        sync (string, None): The method used to synchronize timestamps. Use ``local`` if you receive the stream from another application on the same computer. Use ``network`` if you receive from another computer.
        max_samples (int): The maximum number of samples to return per call.
    """

    def __init__(self, prop, value, sync="local", max_samples=1024 * 4, timeout=10.0):
        Node.__init__(self, None)
        self.inlet = None
        self.labels = None
        self._prop = prop
        self._value = value
        self.sync = sync
        self.max_samples = max_samples
        self.offset = time() - pylsl.local_clock()
        self._timeout = timeout

        self.connect()

    def connect(self):
        if not self.inlet:
            # resolve streams
            streams = resolve_byprop(
                self._prop, self._value, timeout=self._timeout)
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

            self.output.set_parameters(
                channels=channels,
                frequency=info.nominal_srate(),
                meta=self.meta)

    def update(self):
        if self.inlet:
            values, stamps = self.inlet.pull_chunk(
                max_samples=self.max_samples)
            if stamps:
                stamps = np.array(stamps)
                if self.sync == "local":
                    stamps += self.offset
                elif self.sync == "network":
                    stamps = stamps + self.inlet.time_correction() + self.offset
                # stamps = pd.to_datetime(stamps, format=None)
            if len(stamps) > 0:
                if len(self.channels) > 0:
                    self.output.set(values, stamps, self.channels)
                else:
                    self.output.set(values, stamps)
        else:
            return


class ButterFilter(Node):
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

    def __init__(self, input_port, lowcut, highcut, order=4):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

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
            y, zf = signal.lfilter(
                self.b, self.a, chunk.transpose(), zi=self.zi)
            # zf are the future initial condition
            self.zi = zf
            # update output port
            self.output.set(np.array(y).transpose(),
                            chunk.index, self.input.channels)


class TimeBasedEpoching(Node):
    """Cut a continuous signal in epoch of same duration
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, frequency):
        Node.__init__(self, input_port)

        self.duration = 1 / frequency

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=frequency,
            meta=self.input.meta)

        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.trigger = None

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            # trigger points to the oldest data in persistence
            if not self.trigger:
                self.trigger = float(chunk.index.values[0])

            # if the new chunk complete an epoch:
            if float(chunk.index[-1]) >= self.trigger + self.duration:
                # number of epoch that can be extracted
                iter_ = int(
                    (float(chunk.index[-1]) - self.trigger) / self.duration)
                dfcon = pd.concat([self.persistent, chunk])

                # TO DO treat the case of a working frequency slower than
                # epoching (ie i > 1)
                for i in range(iter_):
                    epoch = dfcon[lambda x: x.index < self.trigger + self.duration]
                    y = dfcon.iloc[lambda x: x.index >= self.trigger + self.duration]
                    self.trigger = self.trigger + self.duration

                    self.output.set_from_df(epoch)
                self.persistent = y
            else:
                self.persistent = pd.concat([self.persistent, chunk])


class MarkerBasedSeparation(Node):
    """Cut a continuous signal in epoch of various duration coming from Markers
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, marker_input_port):
        Node.__init__(self, input_port)

        self.marker_input = marker_input_port

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        # persitent is data of a non-complete epoch
        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.markers_received = pd.DataFrame([], [], self.marker_input.channels)
        # current_name is the name to add for current epoch
        self.current_name = 'first epoch'

        # TO DO terminate

    def get_end_time(self):
        """Test if a new marker arrived and return its timestamp as end of last epoch
        else return None
        """
        if len(self.markers_received) > 0:
            # end_time of current epoch
            end_time = float(self.markers_received.index.values[0])
            name = self.markers_received.loc[end_time, self.marker_input.channels].values[0]
        else:
            end_time = None
            name = None
        return (end_time, name)

    def update(self):
        # concatenate all markers received
        for marker in self.marker_input:
            self.markers_received = pd.concat([self.markers_received, marker])

        # initialize the end_time of current epoch and name of the next epoch
        end_time, next_name = self.get_end_time()

        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])
            while end_time and float(chunk.index[-1]) > end_time:  # while the new chunk complete epochs do:
                # get epoch
                epoch = self.persistent[lambda x: x.index < end_time]
                # the rest is stored in persistence
                self.persistent = self.persistent.iloc[lambda x: x.index >= end_time]

                # update the output
                self.output.set_from_df(epoch, self.current_name)
                print(epoch.meta)
                print(epoch)

                # update the new current epoch name
                self.current_name = next_name

                # drop marker from markers_recived and update end_time and next_name
                self.markers_received = self.markers_received.drop([end_time])
                end_time, next_name = self.get_end_time()


class StimulationBasedEpoching(Node):
    """Cut a continuous signal in epoch of various duration coming from Markers
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, marker_input_port, stimulation, offset, duration):
        Node.__init__(self, input_port)

        self.marker_input = marker_input_port

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        self.stimulation = stimulation
        self.duration = duration
        self.offset = offset

        # persitent is data of a non-complete epoch
        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.markers_received = pd.DataFrame([], [], self.marker_input.channels)
        # current_name is the name to add for current epoch
        self.start_time = None
        self.end_time = None

        # TO DO terminate

    def get_timing_from_marker(self):
        for marker_df in self.marker_input:
            for marker_index in marker_df.index:
                name = marker_df.loc[marker_index].values[0]
                if name == self.stimulation:
                    start_time = marker_index + self.offset
                    end_time = start_time + self.duration
                    return (float(start_time), float(end_time))
        return (None, None)

    def update(self):
        if not self.start_time:
            self.start_time, self.end_time = self.get_timing_from_marker()

        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])
            if self.start_time and float(chunk.index[-1]) > self.end_time:
                # get epoch
                epoch1 = self.persistent.iloc[lambda x: x.index >= self.start_time]
                epoch = epoch1.iloc[lambda x: x.index < self.end_time]
                # the rest is stored in persistence
                self.persistent = self.persistent.iloc[lambda x: x.index >= self.end_time]
                # update the output
                self.output.set_from_df(epoch)

                self.start_time, self.end_time = None, None


class Averaging(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        # TO DO terminate

    def update(self):
        for epoch in self.input:
            self.output.set_from_df(pd.DataFrame(
                epoch.mean(), columns=[epoch.index[-1]]).transpose())


class ApplyFunction(Node):
    """TO DO
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output GroupOfPorts
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_port, function, args=()):
        Node.__init__(self, input_port)

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=self.input.frequency,
            meta=self.input.meta)

        self.function = function
        self.args = args

        # TO DO terminate

    def update(self):
        for chunk in self.input:
            self.output.set_from_df(chunk.apply(self.function, args=self.args))


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
