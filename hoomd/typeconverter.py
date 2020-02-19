from numpy import array, ndarray
from hoomd.util import is_constructor


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


class OnlyType(_HelpValidate):
    def __init__(self, type_, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.type = type_

    def _validate(self, value):
        if isinstance(value, self.type):
            return value
        else:
            try:
                return self.type(value)
            except Exception:
                raise ValueError("value {} not convertible into type {}."
                                 "".format(value, self.type))


class OnlyFrom(_HelpValidate):
    def __init__(self, options, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.options = options

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
        if all([v in self for v in value]):
            return value
        else:
            raise ValueError("Value {} all not in options: {}".format(
                value, self.options))

    def __str__(self):
        return "MultipleOnlyFrom[{}]".format(self.options)


class TypeConverter:
    _conversion_func_dict = {list: to_list, ndarray: to_array, str: is_string}

    def __init__(self, converter):
        self.converter = converter

    def keys(self):
        if isinstance(self.converter, dict):
            yield from self.converter.keys()
        else:
            raise RuntimeError("TypeConverter {} does not have keys. "
                               "TypeConverter.keys() only works for objects "
                               "that convert dictionaries.")

    def _raise_error(self, value, conversion_func, prev_err_msg):
        err = "Value {} of type {} cannot be converted using {}. "
        err += "The conversion function raised this error " + prev_err_msg
        err = err.format(value, type(value), conversion_func)
        raise TypeConversionError(err)

    def _raise_from_previous_error(self, err, key):
        if len(err.args) > 1:
            raise TypeConversionError(err.args[0], [key] + err.args[1])
        else:
            raise TypeConversionError(err.args[0], [key])

    def convert(self, value):
        if isinstance(self.converter, dict):
            new_value = dict()
            try:
                for key, v in value.items():
                    temp_value = self._convert_value(v, key)
                    if temp_value == dict():
                        continue
                    else:
                        new_value[key] = temp_value
                return new_value
            except AttributeError:
                raise TypeConversionError("Expected a dictionary like value. "
                                          "Received {}.".format(value))
        else:
            return self._convert_value(value)

    def _convert_value(self, value, key=None):
        if key is None:
            try:
                return self.converter(value)
            except (TypeError, ValueError) as err:
                self._raise_error(value, self.converter, str(err))
        else:
            try:
                return self.converter[key](value)
            except TypeConversionError as err:
                self._raise_from_previous_error(err, key)
            except KeyError:
                return value

    def __getitem__(self, key):
        if isinstance(self.converter, dict):
            return self.converter[key]
        else:
            raise RuntimeError("Cannot call getitem on TypeConverter that is "
                               "not a dictionary.")

    def __setitem__(self, key, value):
        if isinstance(self.converter, dict):
            self.converter[key] = value
        else:
            raise RuntimeError("Cannot call setitem on TypeConverter that is "
                               "not a dictionary.")

    def __call__(self, value):
        return self.convert(value)

    def __str__(self):
        if isinstance(self.converter, dict):
            return '{' + ', '.join(["{}: {}".format(key, value)
                                    for key, value in self.converter.items()]
                                   ) + '}'
        else:
            return str(self.converter)

    @classmethod
    def from_default(cls, default):
        # if default is a dictionary recursively call from_default on values
        if isinstance(default, dict):
            converter = dict()
            for key, value in default.items():
                converter[key] = cls.from_default(value)
            return cls(converter)
        # if constructor with special default setting logic
        elif default in cls._conversion_func_dict.keys():
            return cls(cls._conversion_func_dict[default])
        # if type with special default setting logic
        elif type(default) in cls._conversion_func_dict.keys():
            return cls(cls._conversion_func_dict[type(default)])
        # if object constructor
        elif is_constructor(default):
            return cls(OnlyType(default))
        # if callable
        elif callable(default):
            return cls(default)
        # if other object
        else:
            return cls(OnlyType(type(default)))


def from_type_converter_input_to_default(default, overwrite_default=None):
    if isinstance(default, dict):
        new_default = dict()
        # if overwrite_default exists use those values over default
        if overwrite_default is not None:
            for key, dft in default.items():
                value = overwrite_default.get(key)
                new_default[key] = from_type_converter_input_to_default(dft,
                                                                        value)
        else:
            for key, value in default.items():
                new_default[key] = from_type_converter_input_to_default(value)
        return new_default
    elif is_constructor(default) or callable(default):
        return RequiredArg if overwrite_default is None else overwrite_default
    else:
        return default if overwrite_default is None else overwrite_default
