# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement type conversion helpers."""

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from inspect import isclass
from hoomd.error import TypeConversionError
from hoomd.util import _is_iterable
from hoomd.variant import Variant, Constant
from hoomd.trigger import Trigger, Periodic
from hoomd.filter import ParticleFilter, CustomFilter
import hoomd


class RequiredArg:
    """Define a parameter as required."""
    pass


def trigger_preprocessing(trigger):
    """Process triggers.

    Convert integers to periodic triggers.
    """
    if isinstance(trigger, Trigger):
        return trigger
    else:
        try:
            return Periodic(period=int(trigger), phase=0)
        except Exception:
            raise ValueError("Expected a hoomd.trigger.trigger_like object.")


def variant_preprocessing(variant):
    """Process variants.

    Convert floats to constant variants.
    """
    if isinstance(variant, Variant):
        return variant
    else:
        try:
            return Constant(float(variant))
        except Exception:
            raise ValueError("Expected a hoomd.variant.variant_like object.")


def box_preprocessing(box):
    """Process boxes.

    Convert values that `Box.from_box` handles.
    """
    if isinstance(box, hoomd.Box):
        return box
    else:
        try:
            return hoomd.Box.from_box(box)
        except Exception:
            raise ValueError(f"{box} is not convertible into a hoomd.Box object"
                             f". using hoomd.Box.from_box")


def box_variant_preprocessing(input):
    """Process box variants.

    Convert boxes and length-6 array-like objects to
    `hoomd.variant.box.Constant`.
    """
    if isinstance(input, hoomd.variant.box.BoxVariant):
        return input
    else:
        try:
            return hoomd.variant.box.Constant(box_preprocessing(input))
        except Exception:
            raise ValueError(f"{input} is not convertible into a "
                             f"hoomd.variant.box.BoxVariant object.")


def positive_real(number):
    """Ensure that a value is positive."""
    try:
        float_number = float(number)
    except Exception as err:
        raise TypeConversionError(
            f"{number} not convertible to float.") from err
    if float_number <= 0:
        raise TypeConversionError("Expected a number greater than zero.")
    return float_number


def nonnegative_real(number):
    """Ensure that a value is not negative."""
    try:
        float_number = float(number)
    except Exception as err:
        raise TypeConversionError(
            f"{number} not convertible to float.") from err
    if float_number < 0:
        raise TypeConversionError("Expected a nonnegative real number.")
    return float_number


def identity(value):
    """Return the given value."""
    return value


class _HelpValidate(ABC):
    """Base class for classes that perform validation on an inputed value.

    Supports arbitrary pre and post processing as well as optionally allowing
    None values. The `_validate` function should raise a `ValueError` or
    `TypeConverterValue` if validation fails, else it should return the
    validated/transformed value.
    """

    def __init__(self, preprocess=None, postprocess=None, allow_none=False):
        self._preprocess = identity if preprocess is None else preprocess
        self._postprocess = identity if postprocess is None else postprocess
        self._allow_none = allow_none

    def __call__(self, value):
        if value is RequiredArg:
            return value
        if value is None:
            if not self._allow_none:
                raise ValueError("None is not allowed.")
            else:
                return None
        try:
            return self._postprocess(self._validate(self._preprocess(value)))
        except Exception as err:
            if isinstance(err, TypeConversionError):
                raise err
            raise TypeConversionError(
                f"Error raised in conversion: {str(err)}") from err

    @abstractmethod
    def _validate(self, value):
        pass


class Any(_HelpValidate):
    """Accept any input."""

    def __init__(self, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)

    def _validate(self, value):
        return value

    def __str__(self):
        """str: String representation of the validator."""
        return "Any()"


class Either(_HelpValidate):
    """Class that has multiple equally valid validation methods for an input.

    For instance if a parameter can either be a length 6 tuple or float then

    Example::

       e = Either(to_type_converter((float,) * 6), to_type_converter(float))

    would allow either value to pass.
    """

    def __init__(self, specs, preprocess=None, postprocess=None):
        super().__init__(preprocess, postprocess)
        self.specs = specs

    def _validate(self, value):
        for spec in self.specs:
            try:
                return spec(value)
            except Exception:
                continue
        raise ValueError(f"value {value} not converible using "
                         f"{[str(spec) for spec in self.specs]}")

    def __str__(self):
        """str: String representation of the validator."""
        return f"Either({[str(spec) for spec in self.specs]})"


class OnlyIf(_HelpValidate):
    """A wrapper around a validation function.

    Not strictly necessary, but keeps the theme of the other classes, and allows
    pre/post-processing and optionally allows None.
    """

    def __init__(self,
                 cond,
                 preprocess=None,
                 postprocess=None,
                 allow_none=False):
        super().__init__(preprocess, postprocess, allow_none)
        self.cond = cond

    def _validate(self, value):
        return self.cond(value)

    def __str__(self):
        """str: String representation of the validator."""
        return f"OnlyIf({str(self.cond)})"


class OnlyTypes(_HelpValidate):
    """Only allow values that are instances of type.

    Developers should consider the `collections.abc` module in using this type.
    In general `OnlyTypes(Sequence)` is more readable than the similar
    `OnlyIf(lambda x: hasattr(x, '__iter__'))`. If a sequence of types is
    provided and ``strict`` is ``False``, conversions will be attempted in the
    order of the ``types`` sequence.
    """

    def __init__(self,
                 *types,
                 disallow_types=None,
                 strict=False,
                 preprocess=None,
                 postprocess=None,
                 allow_none=False):
        super().__init__(preprocess, postprocess, allow_none)
        # Handle if a class is passed rather than an iterable of classes
        self.types = types
        if disallow_types is None:
            self.disallow_types = ()
        else:
            self.disallow_types = disallow_types
        self.strict = strict

    def _validate(self, value):
        if isinstance(value, self.disallow_types):
            raise TypeConversionError(
                f"Value {value} cannot be of type {type(value)}")
        if isinstance(value, self.types):
            return value
        elif self.strict:
            raise ValueError(
                f"Value {value} is not an instance of any of {self.types}.")
        else:
            for type_ in self.types:
                try:
                    return type_(value)
                except Exception:
                    pass
            raise ValueError(
                f"Value {value} is not convertable into any of these types "
                f"{self.types}")

    def __str__(self):
        """str: String representation of the validator."""
        return f"OnlyTypes({str(self.types)})"


class OnlyFrom(_HelpValidate):
    """Validates a value against a given set of options.

    An example that allows integers less than ten `OnlyFrom(range(10))`. Note
    that generator expressions are fine.
    """

    def __init__(self,
                 options,
                 preprocess=None,
                 postprocess=None,
                 allow_none=False):
        super().__init__(preprocess, postprocess, allow_none)
        self.options = set(options)

    def _validate(self, value):
        if value in self:
            return value
        else:
            raise ValueError(f"Value {value} not in options: {self.options}")

    def __contains__(self, value):
        """bool: True when value is in the options."""
        return value in self.options

    def __str__(self):
        """str: String representation of the validator."""
        return "OnlyFrom[{self.options}]"


class SetOnce:
    """Used to make properties read-only after setting."""

    def __init__(self, validation):
        if isclass(validation):
            self._validation = OnlyTypes(validation)
        else:
            self._validation = validation

    def __call__(self, value):
        """Handle setting values."""
        if self._validation is not None:
            val = self._validation(value)
            self._validation = None
            return val
        else:
            raise ValueError("Attribute is read-only.")


class TypeConverter(ABC):
    """Base class for TypeConverter's encodes structure and validation.

    Subclasses represent validating a different data structure. When called they
    are to attempt to validate and transform the inputs as given by the
    specification set up at the initialization.

    Note:
        Subclasses should not be instantiated directly. Instead use
        `to_type_converter`.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, value):
        """Called when values are set."""
        if value is RequiredArg:
            return value
        return self._validate(value)

    @abstractmethod
    def _validate(self, value):
        pass


class NDArrayValidator(_HelpValidate):
    """Validates array and array-like structures.

    Args:
        dtype (numpy.dtype): The type of individual items in the array.
        shape (`tuple` [`int`, ...], optional): The shape of the array. The
            number of dimensions is specified by the length of the tuple and the
            length of a dimension is specified by the value. A value of ``None``
            in an index indicates any length is acceptable. Defaults to
            ``(None,)``.
        order (`str`, optional): The kind of ordering needed for the array.
            Options are ``["C", "F", "K", "A"]``. See `numpy.array`
            documentation for imformation about the orderings. Defaults to
            `"K"`.
        preprocess (callable, optional): An optional function like argument to
            use to preprocess arrays before general validation. Defaults to
            ``None`` which mean on preprocessing.
        preprocess (callable, optional): An optional function like argument to
            use to postprocess arrays after general validation. Defaults to
            ``None`` which means no postprocessing.
        allow_none (`bool`, optional): Whether to allow ``None`` as a valid
            value. Defaults to ``None``.
    The validation will attempt to convert array-like objects to arrays. We will
    change the dtype and ordering if necessary, but do not reshape the given
    arrays since this is non-trivial depending on the shape specification passed
    in.
    """

    def __init__(self,
                 dtype,
                 shape=(None,),
                 order="K",
                 preprocess=None,
                 postprocess=None,
                 allow_none=False):
        """Create a NDArrayValidator object."""
        super().__init__(preprocess, postprocess, allow_none)
        self._dtype = dtype
        self._shape = shape
        self._order = order

    def _validate(self, arr):
        """Validate an array or array-like object."""
        typed_and_ordered = np.array(arr, dtype=self._dtype, order=self._order)
        if len(typed_and_ordered.shape) != len(self._shape):
            raise ValueError(
                f"Expected array of {len(self._shape)} dimensions, but "
                f"recieved array of {len(typed_and_ordered.shape)} dimensions.")

        for i, dim in enumerate(self._shape):
            if dim is not None:
                if typed_and_ordered.shape[i] != dim:
                    raise ValueError(
                        f"In dimension {i}, expected size {dim}, but got size "
                        f"{typed_and_ordered.shape[i]}")
        return typed_and_ordered


class _BaseConverter:
    """Get the base level (i.e. no deeper level exists) validator."""
    _conversion_func_dict = {
        Variant:
            OnlyTypes(Variant, preprocess=variant_preprocessing),
        ParticleFilter:
            OnlyTypes(ParticleFilter, CustomFilter, strict=True),
        str:
            OnlyTypes(str, strict=True),
        Trigger:
            OnlyTypes(Trigger, preprocess=trigger_preprocessing),
        hoomd.Box:
            OnlyTypes(hoomd.Box, preprocess=box_preprocessing),
        hoomd.variant.box.BoxVariant:
            OnlyTypes(hoomd.variant.box.BoxVariant,
                      preprocess=box_variant_preprocessing),
        # arrays default to float of one dimension of arbitrary length and
        # ordering
        np.ndarray:
            NDArrayValidator(float),
    }

    @classmethod
    def to_base_converter(cls, schema):
        # If the schema is a class object
        if isclass(schema):
            # if constructor with special default setting logic
            for special_class in cls._conversion_func_dict:
                if issubclass(schema, special_class):
                    return cls._conversion_func_dict[special_class]
            # constructor with no special logic
            return OnlyTypes(schema)

        # If the schema is a special_class instance
        # if schema is a subtype of a type with special schema setting logic
        for special_class in cls._conversion_func_dict:
            if isinstance(schema, special_class):
                return cls._conversion_func_dict[special_class]

        # if schema is a callable assume that it is the validation function
        if callable(schema):
            return schema
        # if any other object
        else:
            return OnlyTypes(type(schema))


class TypeConverterSequence(TypeConverter):
    """Validation for a generic any length sequence.

    Uses `to_type_converter` for construction the validation. For each item in
    the inputted sequence, a corresponding `TypeConverter` object is
    constructed.

    Args:
         coverter (TypeConverter): Any object compatible with the type converter
            API.

    Specification:
        When validating, the given element was given that element is repeated
        for every element of the inputed sequence. This class is unsuited for
        fix length sequences (`TypeConverterFixedLengthSequence` exists for
        this). An Example,

        Example::

            # All elements should be floats
            TypeConverterSequence(float)
    """

    def __init__(self, converter):
        self.converter = to_type_converter(converter)

    def _validate(self, sequence):
        """Called when the value is set."""
        if not _is_iterable(sequence):
            raise TypeConversionError(
                f"Expected a sequence like instance. Received {sequence} of "
                f"type {type(sequence)}.")
        else:
            new_sequence = []
            try:
                for i, v in enumerate(sequence):
                    new_sequence.append(self.converter(v))
            except (ValueError, TypeError) as err:
                raise TypeConversionError(
                    f"In list item number {i}: {str(err)}") from err
            return new_sequence


class TypeConverterFixedLengthSequence(TypeConverter):
    """Validation for a fixed length sequence (read tuple).

    Uses `to_type_converter` for construction the validation. For each item in
    the inputted sequence, a corresponding `TypeConverter` object is
    constructed.

    Parameters:
        sequence (Sequence[Any]): Any sequence or iterable, anything else passed
            is an error.

    Specification:
        When validating, a sequence of the exact length given on instantiation
        is expected, else an error is raised.

        Example::

            # Three floats
            TypeConverterFixedLengthSequence((float, float, float))

            # a string followed for a float and int
            TypeConverterFixedLengthSequence((string, float, int))
    """

    def __init__(self, sequence):
        self.converter = tuple([to_type_converter(item) for item in sequence])

    def _validate(self, sequence):
        """Called when the value is set."""
        if not _is_iterable(sequence):
            raise TypeConversionError(
                f"Expected a tuple like object. Received {sequence} of type "
                f"{type(sequence)}.")
        elif len(sequence) != len(self.converter):
            raise TypeConversionError(
                f"Expected exactly {len(self.converter)} items. Received "
                f"{len(sequence)}.")
        else:
            new_sequence = []
            try:
                for i, (v, c) in enumerate(zip(sequence, self)):
                    new_sequence.append(c(v))
            except (ValueError, TypeError) as err:
                raise TypeConversionError(
                    f"In tuple item number {i}: {str(err)}") from err
            return tuple(new_sequence)

    def __iter__(self):
        """Iterate over converters in the sequence."""
        yield from self.converter

    def __getitem__(self, index):
        """Return the index-th converter."""
        return self.converter[index]


class TypeConverterMapping(TypeConverter, MutableMapping):
    """Validation for a mapping of string keys to any type values.

    Uses `to_type_converter` for construction the validation. For each value in
    the inputted sequence, a corresponding `TypeConverter` object is
    constructed.

    Parameters:
        mapping (Mapping[str, Any]): Any mapping, anything else passed is an
            error.

    Specification:
        When validating, a subset of keys is expected to be used. No error is
        raised if not all keys are used in the validation. The validation either
        errors or returns a mapping with all the same keys as the inputted
        mapping.

        Example::

            t = TypeConverterMapping({'str': str, 'list_of_floats': [float]})

            # valid
            t({'str': 'hello'})

            # invalid
            t({'new_key': None})
    """

    def __init__(self, mapping):
        self.converter = {
            key: to_type_converter(value) for key, value in mapping.items()
        }

    def _validate(self, mapping):
        """Called when the value is set."""
        if not isinstance(mapping, Mapping):
            raise TypeConversionError(
                f"Expected a dict like value. Recieved {mapping} of type "
                f"{type(mapping)}.")

        new_mapping = {}
        for key, value in mapping.items():
            if key in self:
                try:
                    new_mapping[key] = self.converter[key](value)
                except (ValueError, TypeError) as err:
                    raise TypeConversionError(
                        f"In key {key}: {str(err)}") from err
            else:
                new_mapping[key] = value
        return new_mapping

    def __iter__(self):
        """Iterate over converters in the mapping."""
        yield from self.converter

    def __getitem__(self, key):
        """Get a converter by key."""
        return self.converter[key]

    def __setitem__(self, key, value):
        """Set a converter by key."""
        self.converter[key] = value

    def __delitem__(self, key):
        """Remove a converter by key."""
        del self.converter[key]

    def __len__(self):
        """int: Number of converters."""
        return len(self.converter)


def to_type_converter(value):
    """The function to use for creating a structure of `TypeConverter` objects.

    This is the function to use when defining validation not any of the
    `TypeConverter` subclasses.

    Example::

        # list take a list of tuples of 3 floats each
        validation = to_type_converter(
            {'str': str, 'list': [(float, float, float)]})
    """
    if isinstance(value, tuple):
        return TypeConverterFixedLengthSequence(value)
    if _is_iterable(value):
        if len(value) == 0:
            return TypeConverterSequence(Any())
        return TypeConverterSequence(value[0])
    elif isinstance(value, Mapping):
        return TypeConverterMapping(value)
    else:
        return _BaseConverter.to_base_converter(value)
