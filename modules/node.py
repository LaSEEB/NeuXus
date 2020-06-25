import sys

from abc import ABC, abstractmethod
import logging

sys.path.append('..')

from modules.chunks import IterChunks
from modules.keepref import KeepRefsFromParent
from modules.registry import *


class Node(KeepRefsFromParent, ABC):
    """Abstract class for Node objects with both input and output port"""

    def __init__(self, input_port):
        super(Node, self).__init__()
        self.input = input_port
        self.output = IterChunks()
        self._nb_iter = 0

        if self.input and self.input.is_epoched:
            self.output.set_epoched(
                epoching_frequency=self.input.epoching_frequency)

    def update_to_log(self):
        if self.input:
            for chunk_or_epoch in self.input:
                if self._nb_iter < NB_ITER:
                    logging.debug(f'Input chunk of {self.__class__.__name__}\n{get_chunk_first_value(chunk_or_epoch)}')
                    self._nb_iter += 1

    @abstractmethod
    def update(self):
        for chunk_or_epoch in self.input:
            pass
