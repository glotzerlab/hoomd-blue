# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implements intelligent and nested defaults for TypeParameters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from itertools import repeat, cycle
from inspect import isclass
from hoomd.util import _is_iterable
from hoomd.data.typeconverter import RequiredArg


class _NoDefault:
    pass


class _SmartDefault(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, value):
        pass

    @abstractmethod
    def to_base(self):
        pass


class _SmartDefaultSequence(_SmartDefault):

    def __init__(self, sequence, default):
        if _is_iterable(default):
            dft_iter = cycle(default)
        else:
            dft_iter = repeat(default)
        self.default = [
            _to_default(item, dft) for item, dft in zip(sequence, dft_iter)
        ]

    def __call__(self, sequence):
        if sequence is None:
            return self.to_base()
        else:
            new_sequence = []
            if len(self.default) == 1:
                for v, d in zip(sequence, self):
                    if isinstance(d, _SmartDefault):
                        new_sequence.append(d(v))
                    else:
                        new_sequence.append(v)
            else:
                given_length = len(sequence)
                for i, d in enumerate(self):
                    if i < given_length:
                        if isinstance(d, _SmartDefault):
                            new_sequence.append(d(sequence[i]))
                        else:
                            new_sequence.append(sequence[i])
                    else:
                        if isinstance(d, _SmartDefault):
                            new_sequence.append(d.to_base())
                        else:
                            new_sequence.append(d)
            return new_sequence

    def __iter__(self):
        if len(self.default) == 1:
            yield from repeat(self.default[0])
        else:
            yield from self.default

    def to_base(self):
        return [_from_default(item) for item in self.default]


class _SmartDefaultFixedLengthSequence(_SmartDefault):

    def __init__(self, sequence, default):
        if _is_iterable(default):
            dft_iter = cycle(default)
        else:
            dft_iter = repeat(default)
        self.default = tuple(
            [_to_default(item, dft) for item, dft in zip(sequence, dft_iter)])

    def __call__(self, sequence):
        if sequence is None:
            return self.to_base()
        else:
            new_sequence = []
            given_length = len(sequence)
            for i, d in enumerate(self):
                if i < given_length:
                    if isinstance(d, _SmartDefault):
                        new_sequence.append(d(sequence[i]))
                    else:
                        new_sequence.append(sequence[i])
                else:
                    if isinstance(d, _SmartDefault):
                        new_sequence.append(d.to_base())
                    else:
                        new_sequence.append(d)
            return tuple(new_sequence)

    def __iter__(self):
        yield from self.default

    def to_base(self):
        return tuple([_from_default(v) for v in self])


class _SmartDefaultMapping(_SmartDefault):

    def __init__(self, mapping, defaults):
        if defaults is _NoDefault:
            self.default = {
                key: _to_default(value, _NoDefault)
                for key, value in mapping.items()
            }
        else:
            self.default = {
                key: _to_default(value, defaults.get(key, _NoDefault))
                for key, value in mapping.items()
            }

    def __call__(self, mapping):
        if mapping is None:
            return self.to_base()
        else:
            new_mapping = dict()
            for key, sdft in self.default.items():
                if key in mapping:
                    if isinstance(sdft, _SmartDefault):
                        new_mapping[key] = sdft(mapping[key])
                    else:
                        new_mapping[key] = mapping[key]
                else:
                    if isinstance(sdft, _SmartDefault):
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
        v = {key: _from_default(value) for key, value in self.default.items()}
        return v


def _to_default(value, defaults=_NoDefault):
    if isinstance(value, tuple):
        if defaults is _NoDefault or _is_iterable(defaults):
            return _SmartDefaultFixedLengthSequence(value, defaults)
        else:
            return defaults
    if _is_iterable(value):
        if defaults is _NoDefault or _is_iterable(defaults):
            return _SmartDefaultSequence(value, defaults)
        else:
            return defaults
    elif isinstance(value, Mapping):
        if defaults is _NoDefault or isinstance(defaults, Mapping):
            return _SmartDefaultMapping(value, defaults)
        else:
            return defaults
    elif isclass(value) or callable(value):
        return RequiredArg if defaults is _NoDefault else defaults
    else:
        return value if defaults is _NoDefault else defaults


def _from_default(value):
    if isinstance(value, _SmartDefault):
        return value.to_base()
    else:
        return value


def _to_base_defaults(value, _defaults=_NoDefault):
    if isinstance(value, Mapping):
        new_default = dict()
        # if _defaults exists use those values over value
        if _defaults is not _NoDefault:
            if isinstance(_defaults, Mapping):
                for key, dft in value.items():
                    sub_explicit_default = _defaults.get(key, _NoDefault)
                    new_default[key] = _to_base_defaults(
                        dft, sub_explicit_default)
            else:
                return None
        else:
            for key, value in value.items():
                new_default[key] = _to_base_defaults(value)
        return new_default
    elif isclass(value) or callable(value):
        return RequiredArg if _defaults is _NoDefault else _defaults
    else:
        return value if _defaults is _NoDefault else _defaults
