import sys

from abc import ABC, abstractmethod
import logging

sys.path.append('..')

from modules.chunks import Port
from modules.keepref import KeepRefsFromParent
from modules.registry import *


class Node(KeepRefsFromParent, ABC):
    """Abstract class for Node objects with both input and output port"""

    _counter = {}

    def __init__(self, input_port, add_output=True):
        super(Node, self).__init__()
        self.input = input_port
        if add_output:
            self.output = Port()
        else:
            self.output = None
        # get name of subclass (for example ChannelSelector)
        subclass = self.__class__.__name__
        # add instance to counter
        if subclass not in Node._counter:
            Node._counter[subclass] = 1
        else:
            Node._counter[subclass] += 1
        self._id = f'{subclass}{Node._counter[subclass]}'

        self._nb_iter = 0

    def log_instance(self, param):
        to_log = f'Instantiate {self._id} with attributes:'
        if self.input:
            to_log += f'\n   input {self.input.id}'
        if self.output:
            to_log += f'\n   output {self.output.id}'
        for key in param.keys():
            to_log += f'\n   {key} {param[key]}'
        logging.info(to_log)

    def update_to_log(self):
        if self.input:
            for chunk_or_epoch in self.input:
                if self._nb_iter < NB_ITER:
                    logging.debug(f'Input chunk of {self._id}\n{get_chunk_first_value(chunk_or_epoch)}')
                    self._nb_iter += 1

    @abstractmethod
    def update(self):
        for chunk_or_epoch in self.input:
            pass

    def terminate(self):
        pass

    def set_queue(self, q):
        self._q = q

    def second_init(self):
        pass
