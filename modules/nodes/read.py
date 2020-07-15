import sys

import mne
from time import time
import logging

mne.set_log_level('WARNING')

from mne.io import (read_raw_gdf, read_raw_eeglab)
from mne import (find_events, events_from_annotations)

sys.path.append('../..')

from modules.node import Node
from modules.chunks import Port


class Reader(Node):
    """Read a file and stream data in real-time
    Attributes:
      - output (Port): Output port
      - marker_output (Port): output marker port
    Args:
      - file (str): path to the file
      - min_chunk_size (int > 0): default is 4, minimum of rows to send per chunk

    example: Reader('../my_record.gdf')

    """

    def __init__(self, file, min_chunk_size=4):
        Node.__init__(self, None)

        self._raw = read_raw_gdf(file)
        self._sampling_frequency = self._raw.info['sfreq']
        self._channels = self._raw.info.ch_names
        try:
            # to test
            events = find_events(self._raw)
        except ValueError:
            events = events_from_annotations(self._raw)
        print(self._raw.info)

        nb_to_event = {events[1][key]: key for key in events[1]}
        self._events = []
        for h in events[0]:
            self._events.append((h[0] / 1000, int(nb_to_event[h[2]])))
        #self._event = read_raw_eeglab(event_file)

        self.marker_output = Port()

        self.marker_output.set_parameters(
            data_type='marker',
            channels=['marker'],
            sampling_frequency=0,
            meta=''
        )

        self.output.set_parameters(
            data_type='signal',
            channels=self._channels,
            sampling_frequency=self._sampling_frequency,
            meta='')

        Node.log_instance(self, {
            'marquers output': self.marker_output.id,
            'sampling frequency': self._sampling_frequency,
            'channels': self._channels,
            'min chunk size': min_chunk_size,
            'from file': file
        })

        self._last_t = None
        self._min_period = min_chunk_size / self._sampling_frequency
        self._start_time = None
        self._end_record = self._raw.times[-1]
        self._flag = True

    def update(self):
        t = time()
        if not self._last_t:
            self._last_t = t
        if not self._start_time:
            self._start_time = t
        if t > self._min_period + self._last_t and self._last_t - self._start_time < self._end_record:
            df = self._raw.to_data_frame(start=int((self._last_t - self._start_time) * self._sampling_frequency), stop=int((t - self._start_time) * self._sampling_frequency))
            df['time'] = df['time'] / 1000
            df = df.set_index('time')
            df.columns = self._channels
            self.output.set_from_df(df)
            while self._events and t - self._start_time > self._events[0][0]:
                self.marker_output.set([self._events[0][1]], [self._events[0][0]])
                self._events = self._events[1:]
            self._last_t = t
        elif t - self._start_time > self._end_record:
            if self._flag:
                logging.info('End of record, press esc')
                self._flag = False
