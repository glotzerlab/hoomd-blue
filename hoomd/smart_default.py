from abc import ABC, abstractmethod
from itertools import repeat
from inspect import isclass
from hoomd.util import is_iterable, is_mapping
from hoomd.typeconverter import RequiredArg


class SmartDefault(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, value):
        pass

    @abstractmethod
    def to_base(self):
        pass


class SmartDefaultSequence(SmartDefault):
    def __init__(self, sequence, default):
        if is_iterable(default):
            raise RuntimeError("Currently only a single explicit default for a "
                               "list is supported.")
        self.default = [toDefault(item, default) for item in sequence]

    def __call__(self, sequence):
        if sequence is None:
            return []
        else:
            new_sequence = []
            for v, d in zip(sequence, self):
                if isinstance(d, SmartDefault):
                    new_sequence.append(d(v))
                else:
                    new_sequence.append(v)
            return new_sequence

    def __iter__(self):
        if len(self.converter) == 1:
            yield from repeat(self.default[0])
        else:
            yield from self.default

    def to_base(self):
        return [fromDefault(item) for item in self.default]


class SmartDefaultMapping(SmartDefault):
    def __init__(self, mapping, defaults):
        if is_mapping(defaults):
            self.default = {key: toDefault(value, defaults.get(key))
                            for key, value in mapping.items()}
        else:
            self.default = {key: toDefault(value, defaults)
                            for key, value in mapping.items()}

    def __call__(self, mapping):
        if mapping is None:
            mapping = dict()
        else:
            new_mapping = dict()
            for key, sdft in self.default.items():
                if key in mapping:
                    if isinstance(sdft, SmartDefault):
                        new_mapping[key] = sdft(mapping[key])
                    else:
                        new_mapping[key] = mapping[key]
                else:
                    if isinstance(sdft, SmartDefault):
                        new_mapping[key] = sdft(None)
                    else:
                        new_mapping[key] = sdft
            return new_mapping

    def keys(self):
        yield from self.default.keys()

    def __getitem__(self, key):
        return self.default[key]

    def __setitem__(self, key, value):
        self.default[key] = value

    def __contains__(self, value):
        return value in self.default

    def to_base(self):
        return {key: fromDefault(value) for key, value in self.default.items()}


def toDefault(value, explicit_defaults=None):
    if is_iterable(value) and not isinstance(value, tuple):
        return SmartDefaultSequence(value, explicit_defaults)
    elif is_mapping(value):
        return SmartDefaultMapping(value, explicit_defaults)
    elif isclass(value) or callable(value):
        return RequiredArg if explicit_defaults is None else explicit_defaults
    else:
        return value if explicit_defaults is None else explicit_defaults


def fromDefault(value):
    if isinstance(value, SmartDefault):
        return value.to_base()
    else:
        return value
