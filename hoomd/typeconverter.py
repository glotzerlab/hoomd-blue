from numpy import array, ndarray
from copy import deepcopy
from itertools import repeat
from abc import ABC, abstractmethod
from inspect import isclass
from hoomd.util import is_iterable, is_mapping


class TypeConversionError(ValueError):
    pass


class RequiredArg:
    pass


def to_list(value):
    if isinstance(value, str) or isinstance(value, dict):
        raise ValueError
    else:
        return list(value)


def to_array(value):
    if isinstance(value, ndarray):
        return TypeConverter(array)


def is_string(value):
    if not isinstance(value, str):
        raise ValueError("Not a string.")
    else:
        return value


class _HelpValidate:
    def __init__(self, preprocess=None, postprocess=None):
        def identity(value):
            return value

        self._preprocess = identity if preprocess is None else preprocess
        self._postprocess = identity if postprocess is None else postprocess

    def __call__(self, value):
        return self._postprocess(self._validate(self._preprocess(value)))


class _HelpValidateWithNone(_HelpValidate):
    def __call__(self, value):
        if value is None:
            return value
        else:
            return super().__call__(value)


class OnlyType(_HelpValidate):
    def __init__(self, type_, strict=False, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
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


class OnlyTypeValidNone(_HelpValidateWithNone):
    def __init__(self, type_, strict=False, preprocess=None, postprocess=None):
        self.type = type_
        self.strict = strict
        super().__init__(preprocess, postprocess)

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
    def __init__(self, options, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
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


class MultipleOnlyFrom(OnlyFrom):
    def _validate(self, value):
        if all(v in self for v in value):
            return value
        else:
            raise ValueError("Value {} all not in options: {}".format(
                value, self.options))

    def __str__(self):
        return "MultipleOnlyFrom{}".format(self.options)


class TypeConverter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, value):
        pass


class TypeConverterValue(TypeConverter):
    _conversion_func_dict = {list: to_list, ndarray: to_array, str: is_string}

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
        except (TypeError, ValueError) as err:
            raise TypeConversionError(
                "Value {} of type {} cannot be converted using {}. The "
                "conversion raised this error: {}".format(
                    value, type(value), self.converter, str(err)))


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


class TypeConverterMapping(TypeConverter):
    def __init__(self, mapping):
        self.converter = {key: toTypeConverter(value)
                          for key, value in mapping.items()}

    def __call__(self, mapping):
        if not is_mapping(mapping):
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
    if is_iterable(value) and not isinstance(value, tuple):
        return TypeConverterSequence(value)
    elif is_mapping(value):
        return TypeConverterMapping(value)
    else:
        return TypeConverterValue(value)


def to_defaults(value, explicit_defaults=None):
    if isinstance(value, dict):
        new_default = dict()
        # if explicit_defaults exists use those values over value
        if explicit_defaults is not None:
            for key, dft in value.items():
                sub_explicit_default = explicit_defaults.get(key)
                new_default[key] = to_defaults(dft, sub_explicit_default)
        else:
            for key, value in value.items():
                new_default[key] = to_defaults(value)
        return new_default
    elif isclass(value) or callable(value) or \
            (is_iterable(value) and not isinstance(value, tuple)):
        return RequiredArg if explicit_defaults is None else explicit_defaults
    else:
        return value if explicit_defaults is None else explicit_defaults


def from_spec(spec, explicit_defaults):
    return (toTypeConverter(spec), to_defaults(spec, explicit_defaults))
