import pandas as pd


class Port:
    """Class for creating link between nodes, it shares the data between them
    A port is called by iteration ie:
    for chunk in my_port:
        my_chunk_in_dataframe = chunk
        ...
    A port can either contain a chunk of a continuous signal (it means that there is only one iteration)
    or epochs (several iterations)
    To add data use set_from_df(my_df) or set(data, stamps, columns)"""

    def __init__(self, is_epoched=False):
        self.clear()

    def clear(self):
        """Clear all data from _data"""
        self._data = []

    def set_parameters(self, channels, frequency, meta={}):
        """Set channels, samplingfrequency and meta data"""
        self.channels = channels
        self.frequency = frequency
        self.meta = meta

    def set(self, rows, timestamps, columns=None):
        """Set from raw data"""
        if columns:
            self._data.append(pd.DataFrame(rows, index=timestamps, columns=columns))
        else:
            self._data.append(pd.DataFrame(rows, index=timestamps))

    def set_from_df(self, df, name=None):
        """Set from a DataFrame object"""
        if name:
            df.meta = str(name)
        self._data.append(df)

    def __iter__(self):
        """Define iteration"""
        self._index = 0
        return self

    def __next__(self):
        """Define iteration"""
        if self._index == len(self._data):
            raise StopIteration
        self._index += 1
        return self._data[self._index - 1]
