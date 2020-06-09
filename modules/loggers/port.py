import pandas as pd


class Port:
    def __init__(self):
        self.clear()

    def clear(self):
        self.data = None
        self.meta = {}
        self.length = 0

    def ready(self):
        return self.data is not None and len(self.data) > 0

    def set_meta(self, meta={}):
        self.meta = meta

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
