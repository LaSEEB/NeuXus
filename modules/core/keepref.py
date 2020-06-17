from collections import defaultdict
import weakref
from abc import ABC


class KeepRefs(ABC):
    __refs__ = defaultdict(list)

    def __init__(self):
        super(KeepRefs, self).__init__()
        self.__refs__[self.__class__].append(weakref.ref(self))

    @classmethod
    def get_instances(cls):
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst


class KeepRefsFromParent(ABC):
    __refs__ = defaultdict(list)

    def __init__(self):
        super(KeepRefsFromParent, self).__init__()
        self.__refs__[self.__class__.__bases__[0]].append(weakref.ref(self))

    @classmethod
    def get_instances(cls):
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst
