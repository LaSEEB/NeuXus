import os

import mne
from time import time
import logging
import pyxdf
import pandas as pd

mne.set_log_level('WARNING')

from mne.io import (read_raw_gdf, read_raw_eeglab, read_raw_brainvision)
from mne import (find_events, events_from_annotations)

from neuxus.node import Node
from neuxus.chunks import Port


class Reader(Node):
    """Read a file and stream data in real-time and markes, can replay EEGLAB set files (.set) (the .ftd
    file must be in the same directory, Genearal Data Format (.gdf), Extensible Data Format (.xdf),
    Brain vision format in .vhdr (.eeg and .vmrk files must be in the same directory)
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

        filename, self._file_extension = os.path.splitext(file)
        self._events = []
        if self._file_extension in ['.gdf', '.set', '.vhdr']:
            if self._file_extension == '.gdf':
                self._raw = read_raw_gdf(file)
            elif self._file_extension == '.set':
                self._raw = read_raw_eeglab(file)
            elif self._file_extension == '.vhdr':
                self._raw = read_raw_brainvision(file)
            self._sampling_frequency = self._raw.info['sfreq']
            self._channels = self._raw.info.ch_names
            try:
                # to test
                events = find_events(self._raw)
                logging.debug('Get from find_events')
            except ValueError:
                events = events_from_annotations(self._raw)
                logging.debug('Get from events_from_annotations')
            nb_to_event = {events[1][key]: key for key in events[1]}
            for h in events[0]:
                try:
                    value = float(nb_to_event[h[2]])
                except ValueError:
                    value = nb_to_event[h[2]]
                # self._events.append((h[0] / 1000, value))
                self._events.append((h[0] / self._sampling_frequency, value))
            self._end_record = self._raw.times[-1]
            self._start_record = self._raw.times[0]
        elif self._file_extension == '.xdf':
            streams, header = pyxdf.load_xdf(file, synchronize_clocks=False, verbose=False)
            logging.info(f'Found {len(streams)} streams in xdf file')
            for ix, stream in enumerate(streams):
                sampling_frequency = float(stream['info']['nominal_srate'][0])
                if sampling_frequency == 0:
                    logging.debug(f'Get marker from stream {ix}')
                    for timestamp, event in zip(stream['time_stamps'], stream['time_series']):
                        self._events.append((timestamp, float(event)))
                else:
                    logging.debug(f'Get data from stream {ix}')
                    self._sampling_frequency = sampling_frequency
                    nb_chan = int(stream['info']['channel_count'][0])
                    self._channels = [(stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0]) for i in range(nb_chan)]
                    self._df = pd.DataFrame(stream['time_series'], stream['time_stamps'], self._channels)
                    self._start_record = stream['time_stamps'][0]
                    self._end_record = stream['time_stamps'][-1]

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
        self._flag = True

    def update(self):
        t = time()
        if not self._start_time:
            self._start_time = t
        t = t - self._start_time
        if self._last_t == None:
            self._last_t = t
        if t > self._min_period + self._last_t and self._last_t < self._end_record:
            start_index = int(self._last_t * self._sampling_frequency)
            end_index = int(t * self._sampling_frequency)
            if self._file_extension in ['.gdf', '.set', '.vhdr']:
                df = self._raw.to_data_frame(start=start_index, stop=end_index)
                df['time'] = df['time'] / 1000
                df = df.set_index('time')
                df.columns = self._channels
            elif self._file_extension == '.xdf':
                df = self._df.iloc[start_index:end_index, :]
            self.output.set_from_df(df)
            while self._events and t + self._start_record > self._events[0][0]:
                self.marker_output.set([self._events[0][1]], [self._events[0][0]])
                self._events = self._events[1:]
            self._last_t = t
        elif t + self._start_record > self._end_record:
            if self._flag:
                logging.info('End of record, press esc')
                self._flag = False
