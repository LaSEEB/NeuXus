from collections import defaultdict
import weakref
from abc import ABC


class KeepRefs(ABC):
    """Abstract class for keeping the reference a class
    Used for IterChunk, when creating an IterChunk, it adds
    the new instance to the list of IterChunk
    """
    __refs__ = defaultdict(list)

    def __init__(self):
        super(KeepRefs, self).__init__()
        self.__refs__[self.__class__].append(weakref.ref(self))

    @classmethod
    def get_instances(cls):
        """Function to get instances of all the instance of the herited
        class"""
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst


class KeepRefsFromParent(ABC):
    """Abstract class for keeping the reference of Parent
    Used for Node, when creating a subclass of Node, it adds
    the new instance to the list of Nodes
    """
    __refs__ = defaultdict(list)

    def __init__(self):
        super(KeepRefsFromParent, self).__init__()
        self.__refs__[self.__class__.__bases__[0]].append(weakref.ref(self))

    @classmethod
    def get_instances(cls):
        """Function to get instances of all the instance that have the
        current class as parent"""
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst
