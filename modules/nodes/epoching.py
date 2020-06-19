import sys

import pandas as pd

sys.path.append('.')
sys.path.append('../..')

from modules.core.node import Node


class TimeBasedEpoching(Node):
    """Generates signal 'slices' or 'blocks' having a specified duration and interval
    Attributes:
        input_: get DataFrame and meta from input_ port
        output: output Port used to share data to other nodes
    Args:
        input_port: Input signal received in block
        duration: Length of epoched signal
        interval: Time interval between two consecutive epochs for signal
    """

    def __init__(self, input_port, duration, interval):
        Node.__init__(self, input_port)

        self.duration = duration
        self.interval = interval

        self.output.set_parameters(
            channels=self.input.channels,
            frequency=1 / interval,
            meta=self.input.meta)

        self.persistent = pd.DataFrame([], [], self.input.channels)
        self.trigger = None
        self.markers = []

        # TO DO terminate

    def update_timing(self, min_time, max_time):
        if not self.trigger:
            self.trigger = min_time
        while self.trigger < max_time:
            self.markers.append((self.trigger, self.trigger + self.duration))
            self.trigger += self.interval

    def update(self):
        # update persistence
        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])

            # update markers received
            self.update_timing(float(chunk.index.values[0]), float(chunk.index.values[-1]))

        if len(self.persistent) > 0:
            for marker in self.markers.copy():
                start_time, end_time = marker
                if float(self.persistent.index[-1]) > end_time:
                    # get epoch
                    self.persistent = self.persistent.iloc[lambda x: x.index >= start_time]
                    epoch = self.persistent.iloc[lambda x: x.index < end_time]
                    # update the output
                    self.output.set_from_df(epoch)
                    self.markers.remove(marker)

        if len(self.markers) == 0:
            self.persistent = pd.DataFrame([], [], self.input.channels)


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

        self.markers = []
        # current_name is the name to add for current epoch
        self.start_time = None
        self.end_time = None

        # TO DO terminate

    def get_timing_from_marker(self):
        markers = []
        for marker_df in self.marker_input:
            for marker_index in marker_df.index:
                name = marker_df.loc[marker_index].values[0]
                if name == self.stimulation:
                    start_time = marker_index + self.offset
                    end_time = start_time + self.duration
                    markers.append((float(start_time), float(end_time)))
        return markers

    def update(self):
        # update persistence
        for chunk in self.input:
            self.persistent = pd.concat([self.persistent, chunk])

        # update markers received
        self.markers = self.markers + self.get_timing_from_marker()

        if len(self.persistent) > 0:
            for marker in self.markers.copy():
                start_time, end_time = marker
                if float(self.persistent.index[-1]) > end_time:
                    # get epoch
                    self.persistent = self.persistent.iloc[lambda x: x.index >= start_time]
                    epoch = self.persistent.iloc[lambda x: x.index < end_time]
                    # update the output
                    self.output.set_from_df(epoch)
                    self.markers.remove(marker)

        if len(self.markers) == 0:
            self.persistent = pd.DataFrame([], [], self.input.channels)
