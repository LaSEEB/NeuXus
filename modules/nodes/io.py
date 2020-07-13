import sys

import numpy as np
from struct import unpack
from socket import (AF_INET, SOCK_STREAM, socket)
import logging
import uuid
import pandas as pd
from time import (time, sleep)
from pylsl import (StreamInfo, StreamOutlet,
                   StreamInlet, resolve_byprop, pylsl)

sys.path.append('../..')

from modules.node import Node
from modules.chunks import Port


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

    _dtypes = {"double64": np.number, "string": np.object, 'float32': np.number}

    def __init__(self, input_port, name, type="signal", format="double64", uuid_=None):
        Node.__init__(self, input_port, False)
        self._name = name
        self._type = type
        self._format = format
        self.outlet = None
        self._frequency = self.input.sampling_frequency
        if not uuid_:
            uuid_ = str(uuid.uuid4())
        self.uuid = uuid_
        self.connect()

        Node.log_instance(self, {
            'name': self._name,
            'frequency': self._frequency,
            'channels': self.input.channels
        })

    def connect(self):
        '''Create an outlet for streaming data'''
        if not self.outlet:

            # metadata
            info = StreamInfo(
                self._name,
                self._type,
                len(self.input.channels),
                self._frequency,
                self._format,
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
                include=[self._dtypes[self._format]]).values
            stamps = chunk.index.values.astype(np.float64)
            for row, stamp in zip(values, stamps):
                self.outlet.push_sample(row, stamp)


class LslReceive(Node):
    """Receive from a LSL stream.
    Attributes:
      - input: input port
    Args:
      - prop (str): property used to resolve the stream (for example 'name')
      - value (str): value associated to prop for resolving stream
      - data_type (str): type of output value among ['epoch', 'signal', 'vector', 'marker']
      - sync (string, None): The method used to synchronize timestamps. Use ``local`` if
        you receive the stream from another application on the same computer.
        Use ``network`` if you receive from another computer.
      - max_samples (int): The maximum number of samples to return per call. Default is 4096
      - timeout (float): time for the software to wait the stream
    """

    def __init__(self, prop, value, data_type, sync="local", max_samples=1024 * 4, timeout=10.0):
        Node.__init__(self, None)
        assert data_type in ['epoch', 'signal', 'vector', 'marker']
        self._data_type = data_type
        self.inlet = None
        self._prop = prop
        self._value = value
        self.sync = sync
        self.max_samples = max_samples
        self.offset = time() - pylsl.local_clock()
        self._timeout = timeout

        self.connect()

        Node.log_instance(self, {
            'channels': self.channels,
            'sampling frequency': self._frequency})

    def connect(self):
        if not self.inlet:
            # resolve streams
            logging.info(f'Resolving streams with {self._prop} {self._value}')
            streams = resolve_byprop(
                self._prop, self._value, timeout=self._timeout)
            if not streams:
                logging.info('No stream found')
                raise Exception
            logging.info(f'{len(streams)} stream(s) acquired')
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
                channels = [f'Ch{i + 1}' for i in range(info.channel_count())]
            self.channels = channels
            self._frequency = info.nominal_srate()

            self.output.set_parameters(
                data_type=self._data_type,
                channels=channels,
                sampling_frequency=self._frequency,
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


class RdaReceive(Node):
    """Receive a signal from RDA stream
    Attributes:
      - output(Port): output port
    Args:
      - rdaport(int): rdaport to connect with, default is 51254
      - offset(float): offset (in second) to apply to incoming data timestamps
      - host(str): RDA host, default is 'localhost', it could be the IP adress of the host
      - timeout(float): timeout for getting RDA stream

    Example:
        RdaReceive()
        RdaReceive(rdaport=52136, offset=.125)
    """

    def __init__(self, rdaport=51254, offset=.0, host="localhost", timeout=10.0):
        Node.__init__(self, None)
        self._buf_size_max = 2**15
        self._rdaport = rdaport
        self._offset = offset
        self._timeout = timeout
        self._host = host

        self._connect()

        self.output.set_parameters(
            data_type='signal',
            channels=self._channels,
            sampling_frequency=self._frequency,
            meta='')

        self.marker_output = Port()

        self.marker_output.set_parameters(
            data_type='marker',
            channels=['Markers'],
            sampling_frequency=0,
            meta='')

        Node.log_instance(self, {
            'marquers output': self.marker_output.id,
            'channels': self._channels,
            'sampling frequency': self._frequency
        })

        self._persistent = b''
        self._last_block = -1
        self._time = None

    def _connect(self):
        # Create a tcpip socket
        self._my_socket = socket(AF_INET, SOCK_STREAM)
        # RECView: 51254, Recorder: 51244, use 51234 to connect with 16Bit Port
        starttime = time()
        flag = True
        while flag:  # wait for the socket connection
            try:
                self._my_socket.connect((self._host, self._rdaport))
            except ConnectionRefusedError:
                if time() - starttime > self._timeout:
                    print('No RDA stream found')
                    raise Exception
            else:
                flag = False

        while True:  # wait for the starting message
            # Get message header as raw array of chars
            rawhdr = self._receive_data(24)

            # Split array into usefull information id1 to id4 are constants
            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)

            # Get data part of message, which is of variable size
            rawdata = self._receive_data(msgsize - 24)

            if msgtype == 1:
                # Start message, extract eeg properties
                (channel_count, sampling_interval, resolutions,
                 channel_names) = self._get_properties(rawdata)
                self._frequency = 1000000 / sampling_interval
                if not channel_names:
                    channel_names = [f'Ch{i}' for i in range(1, channel_count + 1)]
                self._channels = channel_names
                return
            if time() - starttime > self._timeout:
                print('Timeout')
                raise Exception

    def update(self):
        # receive all data from socket
        raw = self._my_socket.recv(self._buf_size_max)
        # concatenate with last value in persitence
        raw = self._persistent + raw
        # initialize loop
        data_to_send = []
        timestamps = []
        flag = True
        while flag:
            if len(raw) >= 24:
                # get info from 24 fisrt bytes
                info = raw[:24]
                (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', info)
                if len(raw) >= msgsize:  # test if we already get the full message in buffer
                    # get data of current message
                    rawdata = raw[24:msgsize]
                    # store the rest in raw
                    raw = raw[msgsize:]
                    if msgtype == 4:  # if the message contains data do:
                        (block, points, data, markers) = self._extract_data(rawdata)
                        # test overflow of block (ie if we do not receive a block)
                        if self._last_block != -1 and block > self._last_block + 1:
                            logging.warn(
                                "Overflow when getting data from RDA, clock is reset ")
                            self._time = None
                        # update last_block
                        self._last_block = block
                        # concatenate data
                        data_to_send += data
                        if not self._time:  # set the local clock
                            self._time = time() - points / self._frequency
                        timestamps += [self._time - self._offset + i / self._frequency for i in range(points)]
                        for marker in markers:
                            self.marker_output.set([marker['message'][1]] * marker['points'], [timestamps[marker['position'] + i] for i in range(int(marker['points']))])
                        # self._time points to the timestamp of first row from next block
                        self._time = timestamps[-1] + self._offset + 1 / self._frequency
                else:
                    # add to persitence and stop iterations
                    self._persistent = raw
                    flag = False
            else:
                # add to persitence and stop iterations
                self._persistent = raw
                flag = False
        if len(timestamps) > 0:
            # send data in output
            self.output.set(data_to_send, timestamps, self._channels)

    def _receive_data(self, requestedSize):
        """Helper function for receiving a whole message"""
        returnStream = b''
        while len(returnStream) < requestedSize:
            databytes = self._my_socket.recv(requestedSize - len(returnStream))
            if databytes == '':
                print("connection broken")
            returnStream += databytes
        return returnStream

    def _split_string(self, raw):
        """Helper function for splitting a raw array of
        zero terminated strings (C) into an array of python strings"""
        s = [i.decode("utf-8") for i in raw.split(b'\x00')]
        s.remove('')
        return s

    def _get_properties(self, rawdata):
        """Function for extracting eeg properties from a raw data array
        read from tcpip socket"""

        # Extract numerical data
        (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

        # Extract resolutions
        resolutions = []
        for c in range(channelCount):
            index = 12 + c * 8
            restuple = unpack('<d', rawdata[index:index + 8])
            resolutions.append(restuple[0])

        # Extract channel names
        channelNames = self._split_string(rawdata[12 + 8 * channelCount:])

        return (channelCount, samplingInterval, resolutions, channelNames)

    def _extract_data(self, rawdata):
        """function for extracting data from message body"""
        # Extract numerical data
        (block, points, markerCount) = unpack('<LLL', rawdata[:12])

        # Extract eeg data as array of floats
        data = []
        for point in range(points):
            row = []
            for chan in range(len(self._channels)):
                index = 12 + 4 * len(self._channels) * point + 4 * chan
                value = unpack('<f', rawdata[index:index + 4])
                row.append(value[0])
            data.append(row)

        # Extract markers
        markers = []
        index = 12 + 4 * points * len(self._channels)
        for m in range(markerCount):
            markersize = unpack('<L', rawdata[index:index+4])
            (position, points2, channel) = unpack('<LLl', rawdata[index+4:index+16])
            typedesc = self._split_string(rawdata[index+16:index+markersize[0]])
            print(typedesc)
            markers.append({'position': position, 'points': points2, 'message': typedesc})
            index = index + markersize[0]
        return (block, points, data, markers)
