import sys

import numpy as np
import uuid
from time import time
from pylsl import (StreamInfo, StreamOutlet,
                   StreamInlet, resolve_byprop, pylsl)

sys.path.append('../..')

from modules.node import Node


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
            if not channels:
                channels = [f'{i + 1}' for i in range(info.channel_count())]
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
                elif self.sync == 'special':
                    stamps = stamps
                # stamps = pd.to_datetime(stamps, format=None)
            if len(stamps) > 0:
                if len(self.channels) > 0:
                    self.output.set(values, stamps, self.channels)
                else:
                    self.output.set(values, stamps)
        else:
            return
