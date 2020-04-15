from abc import ABC, abstractmethod
from hoomd.parameterdicts import ParameterDict


class _CustomAction(ABC):
    flags = []
    log_quantities = {}

    def __init__(self):
        pass

    def attach(self, simulation):
        self._state = simulation.state

    def detach(self):
        if hasattr(self, '_state'):
            del self._state

    @abstractmethod
    def act(self, timestep):
        pass


class _InternalCustomAction(_CustomAction):
    _reserved_attrs_with_dft = {'_param_dict': ParameterDict,
                                '_typeparam_dict': dict}

    def __getattr__(self, attr):
        if attr in self._reserved_attrs_with_dft.keys():
            setattr(self, attr, self._reserved_attrs_with_dft[attr]())
            return self.__dict__[attr]
        elif attr in self._param_dict.keys():
            return self._getattr_param(attr)
        elif attr in self._typeparam_dict.keys():
            return self._getattr_typeparam(attr)
        else:
            raise AttributeError("Object {} has no attribute {}"
                                 "".format(self, attr))

    def __setattr__(self, attr, value):
        if attr in self._reserved_attrs_with_dft.keys():
            super().__setattr__(attr, value)
        elif attr in self._param_dict.keys():
            self._param_dict[attr] = value
        elif attr in self._typeparam_dict.keys():
            self._setattr_typeparam(attr, value)
        else:
            super().__setattr__(attr, value)

    def _setattr_typeparam(self, attr, value):
        try:
            for k, v in value.items():
                self._typeparam_dict[attr][k] = v
        except TypeError:
            raise ValueError("To set {}, you must use a dictionary "
                             "with types as keys.".format(attr))
