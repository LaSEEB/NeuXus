from scipy import signal
import pandas as pd
import numpy as np
# import uuid
from pylsl import (
    StreamInfo,
    StreamOutlet,
    StreamInlet,
    resolve_stream,
    pylsl,
)
from time import (time, sleep)

from port import Port

import matplotlib.pyplot as plt


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

    def __init__(self, input_, name, frequency, type_="signal", format="double64"):
        self.name = name
        self.type = type_
        self.format = format
        self.frequency = frequency
        self.outlet = None
        self.input = input_
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
                'myuuidtodo'
            )
            channels = info.desc().append_child("channels")
            for label in self.input.channels:
                channels.append_child("channel")\
                    .append_child_value("name", str(label))\
                    .append_child_value("unit", "unknown")\
                    .append_child_value("type", "signal")

            # create the outlet
            self.outlet = StreamOutlet(info)

    def update(self):
        '''Send data found in input port'''
        if not self.input.is_empty():
            values = self.input.data.select_dtypes(
                include=[self._dtypes[self.format]]).values
            stamps = self.input.data.index.values.astype(np.float64)
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

    def __init__(self, output_, sync="local", max_samples=1024):
        self.inlet = None
        self.labels = None
        self.sync = sync
        self.max_samples = max_samples
        self.offset = time() - pylsl.local_clock()

        self.output = output_
        self.connect()

    def connect(self):
        if not self.inlet:
            # resolve streams
            streams = resolve_stream('type', 'signal')
            if not streams:
                return
            # Stream acquired
            self.inlet = StreamInlet(streams[0])
            info = self.inlet.info()
            self.meta = {
                "name": info.name(),
                "type": info.type(),
                "frequency": info.nominal_srate(),
                "info": str(info.as_xml()).replace("\n", "").replace("\t", ""),
            }

            self.output.set_meta(self.meta)

            channels = []
            if not info.desc().child("channels").empty():
                channel = info.desc().child("channels").child("channel")
                for _ in range(info.channel_count()):
                    channel_name = channel.child_value("label")
                    channels.append(channel_name)
                    channel = channel.next_sibling()
            self.channels = channels

            self.output.set_channels(channels)

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

    def __init__(self, input_, output_, lowcut, highcut, fs, order=5):
        super().__init__()
        self.input = input_
        self.output = output_

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
        data = self.input.data
        # filter
        y, zf = signal.lfilter(self.b, self.a, data.transpose(), zi=self.zi)
        # zf are the future initial condition
        self.zi = zf
        # update output port
        self.output.set(np.array(y).transpose(), data.index, self.input.channels)


class Epoching(object):
    """Cut a continuous signal in epoch of same duration
    Attributes:
        input_: get DataFrame and meta from input_ port
        output_: output port
    Args:
        duration: duration of epochs
    """

    def __init__(self, input_, output_, duration):
        super().__init__()
        self.input = input_
        self.output = output_
        self.duration = duration

        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.trigger = None

        # TO DO terminate

    def update(self):
        # trigger points to the oldest data in persistence
        if not self.trigger:
            self.trigger = float(self.input.data.index.values[0])
        df = self.input.data

        # if the new chunk complete an epoch:
        if float(df.index[-1]) >= self.trigger + self.duration:
            # number of epoch that can be extracted
            iter_ = int((float(df.index[-1]) - self.trigger) / self.duration)
            dfcon = pd.concat([self.persistent, self.input.data])

            # TO DO treat the case of a working frequency slower than epoching (ie i > 1)
            for i in range(iter_):
                to_send = dfcon[lambda x: x.index < self.trigger + self.duration]
                y = dfcon.iloc[lambda x: x.index >= self.trigger + self.duration]
                self.trigger = self.trigger + self.duration

            self.output.set_from_df(to_send)
            self.persistent = y
        else:
            self.persistent = pd.concat([self.persistent, self.input.data])
            print(len(self.persistent))


if __name__ == '__main__':

    # for observation via plt
    observe_plt = False

    # initialize the pipeline
    port1 = Port()
    lsl_reception = Receive(port1)
    port2 = Port()
    port2.set_channels(port1.channels)
    port3 = Port()
    port3.set_channels(port1.channels)
    butter_filter = ButterFilter(port1, port2, 8, 40, 512)
    epoch = Epoching(port2, port3, 1)
    lsl_send = Send(port3, 'mySignalEpoched', 512)
    lsl_send2 = Send(port2, 'mySignalFiltered', 512)

    # for dev
    data = pd.DataFrame([])
    data1 = pd.DataFrame([])

    # count iteration
    it = 0

    # working frequency for the loop
    frequency = 5
    t = 1 / frequency

    # run the pipeline
    while True:
        calc_starttime = time()

        # clear port
        port1.clear()
        port2.clear()
        port3.clear()

        lsl_reception.update()
        if port1.ready():
            butter_filter.update()
        if port2.ready():
            epoch.update()
            lsl_send2.update()
        if port3.ready():
            lsl_send.update()

        calc_endtime = time()
        calc_time = calc_endtime - calc_starttime

        print(f'{ int(calc_time / t * 1000) / 10} % for {port1.length} treated rows ({port1.length * len(port1.channels)} data)')

        it += 1

        # for dev
        data1 = pd.concat([data1, port1.data])
        data = pd.concat([data, port2.data])
        if observe_plt and it == 150:
            plt.plot(data.iloc[:, 0:1].values)
            plt.plot(data1.iloc[:, 0:1].values)
            plt.show()

        try:
            sleep(t - calc_time)
        except Exception as err:
    
            print(err)
    # TO DO terminate
