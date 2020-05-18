from numpy import array, ndarray
from itertools import repeat
from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import isclass
from hoomd.util import is_iterable, RequiredArg


class TypeConversionError(ValueError):
    pass


def preprocess_list(value):
    if is_iterable:
        return value
    else:
        raise ValueError("Expected an iterable (excluding str and dict).")


class _HelpValidate:
    def __init__(self, preprocess=None, postprocess=None, allow_none=False):
        def identity(value):
            return value

        self._preprocess = identity if preprocess is None else preprocess
        self._postprocess = identity if postprocess is None else postprocess
        self._allow_none = allow_none

    def __call__(self, value):
        if value is None:
            if not self._allow_none:
                raise ValueError("None is not allowed.")
            else:
                return None
        return self._postprocess(self._validate(self._preprocess(value)))


class Either(_HelpValidate):
    def __init__(self, specs, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.specs = specs

    def _validate(self, value):
        for spec in self.specs:
            try:
                return spec(value)
            except Exception:
                continue
        raise ValueError("value {} not converible using {}".format(
            value, [str(spec) for spec in self.specs]))

    def __str__(self):
        return "Either({})".format([str(spec) for spec in self.specs])


class OnlyType(_HelpValidate):
    def __init__(self, type_, strict=False,
                 preprocess=None, postprocess=None, allow_none=False):
        super().__init__(preprocess, postprocess, allow_none)
        self.type = type_
        self.strict = strict

    def _validate(self, value):
        if isinstance(value, self.type):
            return value
        elif self.strict:
            raise ValueError("value {} not instance of type {}."
                             "".format(value, self.type))
        else:
            try:
                return self.type(value)
            except Exception:
                raise ValueError("value {} not convertible into type {}."
                                 "".format(value, self.type))


class OnlyFrom(_HelpValidate):
    def __init__(self, options,preprocess=None, postprocess=None,
                 allow_none=False):
        super().__init__(preprocess, postprocess, allow_none)
        self.options = set(options)

    def _validate(self, value):
        if value in self:
            return value
        else:
            raise ValueError("Value {} not in options: {}".format(value,
                                                                  self.options))

    def __contains__(self, value):
        return value in self.options

    def __str__(self):
        return "OnlyFrom[{}]".format(self.options)


class TypeConverter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, value):
        pass


class TypeConverterValue(TypeConverter):
    _conversion_func_dict = {
        list: OnlyType(list, preprocess=preprocess_list),
        ndarray: OnlyType(ndarray, preprocess=array),
        str: OnlyType(str, strict=True)}

    def __init__(self, value):
        # if constructor with special default setting logic
        if value in self._conversion_func_dict.keys():
            self.converter = self._conversion_func_dict[value]
        # if type with special value setting logic
        elif type(value) in self._conversion_func_dict.keys():
            self.converter = self._conversion_func_dict[type(value)]
        # if object constructor
        elif isclass(value):
            self.converter = OnlyType(value)
        # if callable
        elif callable(value):
            self.converter = value
        # if other object
        else:
            self.converter = OnlyType(type(value))

    def __call__(self, value):
        try:
            return self.converter(value)
        except (TypeError, ValueError, TypeConversionError) as err:
            if value is RequiredArg:
                raise TypeConversionError("Value is a required argument")
            raise TypeConversionError(
                "Value {} of type {} cannot be converted using {}. Raised "
                "error: {}".format(
                    value, type(value), str(self.converter), str(err)))


class TypeConverterSequence(TypeConverter):
    def __init__(self, sequence):
        self.converter = [toTypeConverter(item) for item in sequence]

    def __call__(self, sequence):
        if not is_iterable(sequence):
            raise TypeConversionError(
                "Expected a list like object. Received {} of type {}."
                "".format(sequence, type(sequence)))
        else:
            new_sequence = []
            try:
                for i, (v, c) in enumerate(zip(sequence, self)):
                    new_sequence.append(c(v))
            except (TypeConversionError) as err:
                raise TypeConversionError(
                    "In list item number {}: {}"
                    "".format(i, str(err)))
            return new_sequence

    def __iter__(self):
        if len(self.converter) == 1:
            yield from repeat(self.converter[0])
        else:
            yield from self.converter


class TypeConverterFixedLengthSequence(TypeConverter):
    def __init__(self, sequence):
        self.converter = tuple([toTypeConverter(item) for item in sequence])

    def __call__(self, sequence):
        if not is_iterable(sequence):
            raise TypeConversionError(
                "Expected a tuple like object. Received {} of type {}."
                "".format(sequence, type(sequence)))
        elif len(sequence) != len(self.converter):
            raise TypeConversionError(
                "Expected exactly {} items. Received {}.".format(
                    len(self.converter), len(sequence)))
        else:
            new_sequence = []
            try:
                for i, (v, c) in enumerate(zip(sequence, self)):
                    new_sequence.append(c(v))
            except (TypeConversionError) as err:
                raise TypeConversionError(
                    "In tuple item number {}: {}"
                    "".format(i, str(err)))
            return tuple(new_sequence)

    def __iter__(self):
        yield from self.converter


class TypeConverterMapping(TypeConverter):
    def __init__(self, mapping):
        self.converter = {key: toTypeConverter(value)
                          for key, value in mapping.items()}

    def __call__(self, mapping):
        if not isinstance(mapping, Mapping):
            raise TypeConversionError(
                "Expected a dict like value. Recieved {} of type {}."
                "".format(mapping, type(mapping)))

        new_mapping = dict()
        try:
            for key, value in mapping.items():
                if key in self:
                    new_mapping[key] = self.converter[key](value)
                else:
                    new_mapping[key] = value
        except (TypeConversionError) as err:
            raise TypeConversionError("In key {}: {}"
                                      "".format(str(key), str(err)))
        return new_mapping

    def keys(self):
        yield from self.converter.keys()

    def __getitem__(self, key):
        return self.converter[key]

    def __setitem__(self, key, value):
        self.converter[key] = value

    def __contains__(self, value):
        return value in self.converter


def toTypeConverter(value):
    if isinstance(value, tuple):
        return TypeConverterFixedLengthSequence(value)
    if is_iterable(value):
        return TypeConverterSequence(value)
    elif isinstance(value, Mapping):
        return TypeConverterMapping(value)
    else:
        return TypeConverterValue(value)
