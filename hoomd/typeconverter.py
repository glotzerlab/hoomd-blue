from hoomd.util import is_constructor
from inspect import isfunction


class TypeConversionError(Exception):
    pass


class TypeConverter:
    def __init__(self, converter):
        self.converter = converter

    def _raise_error(self, value, expected_type):
        raise TypeConversionError("Value {} of type {} cannot be converted "
                                  "using {}.".format(value, type(value),
                                                     expected_type))

    def convert(self, value):
        if isinstance(self.converter, dict):
            new_value = dict()
            try:
                for key, v in value.items():
                    temp_value = self._convert_value(key, v)
                    if temp_value == {}:
                        continue
                    else:
                        new_value[key] = temp_value
                return new_value
            except AttributeError:
                self._raise_error(value, dict)
        else:
            return self._convert_value(value)

    def _convert_value(self, key, value):
        try:
            return self.converter[key](value)
        except (TypeError, ValueError):
            self._raise_error(value, self.converter[key])
        except KeyError:
            return value

    def __call__(self, value):
        return self.convert(value)

    def __str__(self):
        return '{' + ', '.join(["{}: {}".format(key, value)
                                for key, value in self.converter.items()]
                               ) + '}'


def get_type_converter(default):
    if isinstance(default, dict):
        converter = dict()
        for key, value in default.items():
            converter[key] = get_type_converter(value)
        return TypeConverter(converter)
    elif is_constructor(default) or isfunction(default):
        return default
    else:
        return type(default)


def from_type_converter_input_to_default(default):
    if isinstance(default, dict):
        new_default = dict()
        for key, value in default.items():
            new_default[key] = from_type_converter_input_to_default(value)
        return new_default
    elif is_constructor(default) or isfunction(default):
        return None
    else:
        return default
