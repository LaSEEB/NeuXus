import os

import pandas as pd
from random import choice
from scipy.io import savemat
import string

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


class Mat(Node):
    """Log input signal in a mat file, easy to read with Matlab
    If the specified file already exists, Mat will override it
    2 Mat Nodes cannot point to the same mat file !
    Args:
      - file (str): name of the file to write
      - matlab_format (str): key to access this particular table

    example: Mat(port4, '../my_file')

    """

    def __init__(self, input_port, file, min_itemsize=None):
        Node.__init__(self, input_port, None)
        filename, file_extension = os.path.splitext(file)
        self._file = filename + '.mat'

        letters = string.ascii_lowercase
        # self._key = ''.join(choice(letters) for i in range(3))

        # GUSTAVO TEST;
        i = 0
        while os.path.exists(f'temp{i:04d}.h5'):
            i += 1
        self._key = f'temp{i:04d}'
        self._save_file = self._key + '.h5'
        self._min_itemsize = min_itemsize

        # self._save_file = 'temporary_file_' + self._key + '.h5'

        chan = self.input.channels
        pd.DataFrame([] * len(chan), [], chan).to_hdf(self._save_file, key=self._key, mode='w', format='table', min_itemsize=self._min_itemsize)

        Node.log_instance(self, {'file': self._file})

    def update(self):
        for chunk in self.input:
            chunk.copy().to_hdf(self._save_file, append=True, mode='a', key=self._key, min_itemsize=self._min_itemsize)

    def terminate(self):
        df = pd.read_hdf(self._save_file, key=self._key)
        savemat(self._file, {'timestamps': [i for i in df.index], 'values': df.values, 'channels': self.input.channels})
        os.remove(self._save_file)
