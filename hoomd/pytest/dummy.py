# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.trigger import Trigger
from hoomd.operation import Operation


class DummySimulation:

    def __init__(self):
        self.state = DummyState()
        self.operations = DummyOperations()
        self._cpp_sys = DummySystem()
        self._system_communicator = None


class DummySystem:

    def __init__(self):
        self.dummy_list = []


class DummyState:

    def __init__(self):
        pass

    @property
    def particle_types(self):
        return ["A", "B", "C"]


class DummyOperations:
    pass


class DummyCppObj:

    def __init__(self):
        self._dict = dict()

    def setTypeParam(self, type_, value):  # noqa: N802 - this mimics C++ naming
        self._dict[type_] = value

    def getTypeParam(self, type_):  # noqa: N802
        return self._dict[type_]

    @property
    def param1(self):
        return self._param1

    @param1.setter
    def param1(self, value):
        self._param1 = value

    @property
    def param2(self):
        return self._param2

    @param2.setter
    def param2(self, value):
        self._param2 = value

    def notifyDetach(self):  # noqa: N802
        pass

    def __getstate__(self):
        raise RuntimeError("Mimic lack of pickling for C++ objects.")


class DummyOperation(Operation):
    """Requires that user manually add param_dict and typeparam_dict items.

    This is for testing purposes.
    """
    _current_obj_number = 0

    def __init__(self):
        """Increment object counter to enable equality comparison."""
        self.id = self._current_obj_number
        self.__class__._current_obj_number += 1

    def _attach_hook(self):
        self._cpp_obj = DummyCppObj()

    def __eq__(self, other):
        return self.id == other.id


class DummyTrigger(Trigger):

    def __call__(self, ts):
        return True
