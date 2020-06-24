import sys

from abc import ABC, abstractmethod

sys.path.append('..')

from modules.chunks import IterChunks
from modules.keepref import KeepRefsFromParent


class Node(KeepRefsFromParent, ABC):
    """Abstract class for Node objects with both input and output port"""

    def __init__(self, input_port):
        super(Node, self).__init__()
        self.input = input_port
        self.output = IterChunks()

        if self.input and self.input.is_epoched:
            self.output.set_epoched(
                epoching_frequency=self.input.epoching_frequency)

    @abstractmethod
    def update(self):
        for chunk_or_epoch in self.input:
            pass
