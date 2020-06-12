import pandas as pd


class Port:
    def __init__(self, is_epoched=False):
        self.clear()

    def clear(self):
        self.data = []
        self.meta = {}
        self.index = 0
        # self.length = 0

    def ready(self):
        return len(self.data) > 0

    def set_meta(self, meta={}):
        self.meta = meta

    def set_frequency(self, freq):
        self.frequency = freq

    def set_channels(self, channels):
        self.channels = channels

    def set(self, rows, timestamps, names):
        # self.length = len(timestamps)
        self.data.append(pd.DataFrame(rows, index=timestamps, columns=names))

    def set_from_df(self, df):
        # self.length = len(df)
        self.data.append(df)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self.data):
            raise StopIteration
        self.index += 1
        return self.data[self.index - 1]
