import pandas as pd


class Port:
    def __init__(self, is_epoched=False):
        self.clear()

    def clear(self):
        self.data = None
        self.meta = {}
        self.length = 0

    def ready(self):
        return self.data is not None and len(self.data) > 0

    def set_meta(self, meta={}):
        self.meta = meta

    def set_frequency(self, freq):
        self.frequency = freq

    def set_channels(self, channels):
        self.channels = channels

    def set(self, rows, timestamps, names):
        self.length = len(timestamps)
        self.data = pd.DataFrame(rows, index=timestamps, columns=names)

    def set_from_df(self, df):
        self.length = len(df)
        self.data = df

    def is_empty(self):
        try:
            return self.data.empty
        except AttributeError:
            if not self.data:
                return True

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]



class GroupOfPorts(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.ports = None
        self.length = 0

    def ready(self):
        return self.ports and len(self.ports) > 0

    def set(self, rows, timestamps, names):
        self.length += 1
        if not self.ports:
            self.ports = []
        new_port = Port(is_epoched=True)
        new_port.set(rows, timestamps, names)
        self.ports.append(new_port)

    def set_from_df(self, df):
        self.length += 1
        if not self.ports:
            self.ports = []
        new_port = Port(is_epoched=True)
        new_port.set_from_df(df)
        self.ports.append(new_port)
