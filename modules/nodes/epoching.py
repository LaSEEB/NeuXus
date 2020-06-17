import sys

import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


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
