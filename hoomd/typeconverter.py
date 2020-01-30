from numpy import array, ndarray
from hoomd.util import is_constructor
from inspect import isfunction


class TypeConversionError(Exception):
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


def get_attempt_obj_conversion(type_):

    def attempt_obj_conversion(value):
        if isinstance(value, type_):
            return value
        else:
            try:
                return type_(value)
            except Exception:
                raise ValueError("{} not converted to type {}.".format(value,
                                                                       type_))

    return attempt_obj_conversion


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

    def _raise_error(self, value, expected_type):
        err = "Value {} of type {} cannot be converted using {}."
        err = err.format(value, type(value), expected_type)
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
                    if temp_value == {}:
                        continue
                    else:
                        new_value[key] = temp_value
                return new_value
            except AttributeError:
                self._raise_error(value, dict)
        else:
            return self._convert_value(value)

    def _convert_value(self, value, key=None):
        if key is None:
            try:
                return self.converter(value)
            except (TypeError, ValueError):
                self._raise_error(value, self.converter)
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
            return cls(get_attempt_obj_conversion(default))
        # if function
        elif isfunction(default):
            return cls(default)
        # if other object
        else:
            return cls(get_attempt_obj_conversion(type(default)))


def from_type_converter_input_to_default(default, overwrite_default=None):
    if isinstance(default, dict):
        new_default = dict()
        # if overwrite_default exists use those values over default
        if overwrite_default is not None:
            for key, value in default.items():
                value = overwrite_default.get(key, value)
                new_default[key] = from_type_converter_input_to_default(value)
        else:
            for key, value in default.items():
                new_default[key] = from_type_converter_input_to_default(value)
        return new_default
    elif is_constructor(default) or isfunction(default):
        return RequiredArg if overwrite_default is None else overwrite_default
    else:
        return default if overwrite_default is None else overwrite_default
