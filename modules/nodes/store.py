import sys
import os

sys.path.append('../..')

from modules.node import Node


class ToCsv(Node):
    """Convert a signal to a CSV file
    Args:
      - file (str): CSV file to write
      - sep (str): Separator between rows, default is ';'
      - decimal (str): Character recognized as decimal separator, default is ','

    example: ToCsv(port4, 'log.csv')

    """

    def __init__(self, input_port, file, sep=';', decimal=','):
        Node.__init__(self, input_port, None)
        filename, file_extension = os.path.splitext(file)
        self._file = filename + '.csv'
        self._sep = sep
        self._decimal = decimal
        self._first_iter = True

        Node.log_instance(self, {'file': self._file})

    def update(self):
        for chunk in self.input:
            if self._first_iter:
                chunk.to_csv(self._file, mode='w', sep=self._sep, decimal=self._decimal)
                self._first_iter = False
            else:
                chunk.to_csv(self._file, mode='a+', header=False, sep=self._sep, decimal=self._decimal)
