import os

import pandas as pd

from neuxus.node import Node


class Hdf5(Node):
    """Log input signal in a HDF5 file
    If the specified file already exists, Hdf5 will override it
    If to Hdf5 Nodes point to the same file (with different keys),
    the two streams will be cleanly stored in this file,
    to read an hdf5 file in Python, use pd.read_hdf()
    Args:
      - file (str): name of the file to write
      - key (str): key to access this particular table

    example: Hdf5(port4, '../my_file', 'marker')

    """

    def __init__(self, input_port, file, key):
        Node.__init__(self, input_port, None)
        filename, file_extension = os.path.splitext(file)
        self._file = filename + '.h5'

        self._key = key
        chan = self.input.channels
        pd.DataFrame([] * len(chan), [], chan).to_hdf(self._file, key=self._key, mode='w', format='table')

        Node.log_instance(self, {'file': self._file})

    def update(self):
        for chunk in self.input:
            chunk.to_hdf(self._file, append=True, mode='a', key=self._key)
