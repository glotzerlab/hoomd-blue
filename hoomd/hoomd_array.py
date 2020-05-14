import functools
from collections import Iterable

from numpy import ndarray, array, may_share_memory


class HOOMDArrayError(RuntimeError):
    pass


def WrapClass(methods_wrap_func_list, *args, **kwargs):
    """Factory method for metaclasses that produce methods via a functor.

    Applies the functor to each method given in methods. This occurs before
    class creation. The functor can take any number of arguments, but besides
    for the method name must be the same accross all methods.

    Args:
        methods_wrap_func_list (Sequence[tuple(Sequence[str], Callable]): A
            sequence of method names, functor pairs. For each tuple in the list,
            the provided callable is used to wrap all the methods listed in the
            tuple.
        *args (Any): Required position arguments for the functors.
        **kwargs (Any): Required key word arugments for the functors.
    """
    class _WrapClass(type):
        def __new__(cls, name, bases, class_dict):
            for methods, functor in methods_wrap_func_list:
                for method in methods:
                    class_dict[method] = functor(method, *args, **kwargs)
            return super().__new__(cls, name, bases, class_dict)

    return _WrapClass

"""Various list of NumPy ndarray functions.

We separate them out by the kind of wrapping they need. We have to distinguish
between functions that return a new array, functions that return the same array,
and functions that return a new array with the same underlying data.
"""

# Functions that return a new array and wrapper
def _op_wrap(method):
    func = getattr(ndarray, method)

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        arr = self._coerce_to_ndarray()
        return func(arr, *args, **kwargs)

    return wrapped

_ndarray_ops_ = ([
    # Comparison
    '__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__', '__bool__',
    # Unary
    '__neg__', '__pos__', '__abs__', '__invert__',
    # Arithmetic
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__',
    '__divmod__', '__pow__',
    # Bitwise
    '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',
    # Matrix
    '__matmul__',
    ], _op_wrap)

# Magic methods that never return an array to the same underlying buffer
_magic_wrap = _op_wrap

_ndarray_magic_safe_ = ([
    # Copy
    '__copy__', '__deepcopy__',
    # Pickling
    '__reduce__', '__setstate__',
    # Container based
    '__len__', '__setitem__', '__contains__',
    # Conversion
    '__int__', '__float__', '__complex__'
    ], _magic_wrap)

# Magic methods that may return an array pointing to the same buffer
def _magic_wrap_with_check(method):
    func = getattr(ndarray, method)

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        arr = self._coerce_to_ndarray()
        rtn = func(arr, *args, **kwargs)
        if isinstance(rtn, ndarray) and may_share_memory(rtn, arr):
            return self.__class__(rtn, self._callback)
        else:
            return rtn

    return wrapped

_ndarray_magic_unsafe_ = ([
    # Container based
    '__getitem__',
    ], _magic_wrap_with_check)


# Functions that return an array pointing to the same buffer
def _iop_wrap(method):
    func = getattr(ndarray, method)

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        arr = self._coerce_to_ndarray()
        return self.__class__(func(arr, *args, **kwargs), self._callback)

    return wrapped

_ndarray_iops_ = ([
    # Inplace Arithmetic
    '__iadd__', '__isub__', '__imul__', '__itruediv__', '__ifloordiv__',
    '__imod__', '__ipow__',
    # Inplace Bitwise
    '__ilshift__', '__irshift__', '__iand__', '__ior__', '__ixor__'
    ], _iop_wrap)

# Regular functions that may return an array pointing to the same buffer
_std_func_with_check = _magic_wrap_with_check

_ndarray_std_funcs_ = ([
    # Select subset of array
    'diagonal',
    # Reshapes array
    'reshape', 'transpose', 'swapaxes', 'ravel', 'squeeze',
    ], _std_func_with_check)

# Functions that we disallow use of
def _disallowed_wrap(method):
    def raise_error(*args, **kwargs):
        raise HOOMDArrayError(
            "The {} method is not allowed for {} objects.".format(
                method, self.__class__))

    return raise_error

_ndarray_disallow_funcs_ = ([
    'view', 'resize', 'flat', 'flatiter'
    ], _disallowed_wrap)

# Properties that can return an array pointing to the same buffer
def _wrap_properties_with_check(prop):
    prop = getattr(ndarray, prop)

    @property
    @functools.wraps(prop)
    def wrapped(self):
        arr = self._coerce_to_ndarray()
        rtn = getattr(arr, prop)
        if may_share_memory(rtn, arr):
            return self.__class__(rtn, self._callback)
        else:
            return rtn

    @wrapped.setter
    def wrapped(self, value):
        arr = self._coerce_to_ndarray()
        return setattr(arr, value)

    return wrapped

_ndarray_properties_ = ([
    'T'
    ], _wrap_properties_with_check)

# Properties we disallow access of
def _disallowed_property_wrap(method):

    @property
    def raise_error(self):
        raise HOOMDArrayError(
            "The {} property is not allowed for {} objects.".format(
                method, self.__class__))

    return raise_error

_ndarray_disallow_properties_ = ([
    'data', 'base'
    ], _disallowed_property_wrap)


_wrap_list = [
    _ndarray_ops_,
    _ndarray_magic_safe_,
    _ndarray_magic_unsafe_,
    _ndarray_iops_,
    _ndarray_std_funcs_,
    _ndarray_disallow_funcs_,
    _ndarray_properties_,
    _ndarray_disallow_properties_,
]


def coerce_mock_to_array(val):
    """Helper function for ``__array_{ufunc,function}__``.

    Coerces ``HOOMDArray`` objects into ``numpy.ndarray`` objects.
    """
    if isinstance(val, Iterable) and not isinstance(val, (ndarray, HOOMDArray)):
        return [coerce_mock_to_array(v) for v in val]
    return val if not isinstance(val, HOOMDArray) else val._coerce_to_ndarray()


class HOOMDArray(metaclass=WrapClass(_wrap_list)):
    """A NumPy like interface to internal HOOMD-blue data.

    These objects are returned by HOOMD-blue's zero copy access to system data.
    This class acts like a ``numpy.ndarray`` object through NumPy's provided
    interface
    [https://numpy.org/doc/stable/reference/arrays.classes.html](link).
    For typical use cases, understanding this class is not necessary. Treat it
    as a ``numpy.ndarray``.

    We attempt to escape this class whenever possible. This essentially means
    that whenever a new array is returned we can the `numpy.ndarray`. However,
    any array pointing to the same data will be returned as a `HOOMDArray`. To
    ensure memory safety, a `HOOMDArray` object cannot be accessed outside of
    the context manager in which it was created. To have access outside the
    manager an explicit copy must be made (e.g. ``numpy.array(obj,
    copy=True)``).

    In general this class should be nearly as fast as a standard NumPy array,
    but there is some overhead. This is mitigated by escaping the class when
    possible. If every ounce of performance is necessary,
    ``HOOMDArray._coerce_to_ndarray`` can provide a ``numpy.ndarray`` object
    inside the context manager. *References to a ``HOOMDArray`` object's buffer
    after leaving the context manager is UNSAFE.* It can cause SEGFAULTs and
    cause your program to crash. Use this function only if absolutely necessary.
    """
    def __init__(self, buffer, callback):
        """Create a HOOMDArray.

        Args:
            buffer (hoomd._hoomd.HOOMDHostBuffer): The data buffer for the
                system data.
            callback (Callable): A function when called signifies whether the
                array is in the context manager where it was created.
        """
        self._buffer = buffer
        self._callback = callback

    def __array_function__(self, func, types, args, kwargs):
        """Called when a non-ufunc NumPy method is called.

        It is safe generally to convert `HOOMDArray` objects to `numpy.ndarray`
        objects inside a NumPy function.
        """
        new_inputs = [coerce_mock_to_array(val) for val in args]
        for key, value in kwargs.items():
            if type(value) == tuple:
                kwargs[key] = tuple(
                    [coerce_mock_to_array(val) for val in value])
            else:
                kwargs[key] = coerce_mock_to_array(value)
        arr = func(*new_inputs, **kwargs)
        if isinstance(arr, ndarray):
            if may_share_memory(arr, self._coerce_to_ndarray()):
                return self.__class__(arr, self._callback)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Called when a NumPy ufunc is used.

        It is safe generally to convert `HOOMDArray` objects to `numpy.ndarray`
        objects inside a NumPy function. However, we must ensure that if out is
        specified to be a HOOMDArray, we return the HOOMDArray not a
        `numpy.ndarray` pointing to the internal buffer.
        """
        new_inputs = [coerce_mock_to_array(val) for val in inputs]
        out = kwargs.pop("out", None)
        kwargs = {k: coerce_mock_to_array(v) for k, v in kwargs}
        if out is not None:
            if any(isinstance(o, HOOMDArray) for o in out):
                kwargs['out'] = tuple((coerce_mock_to_array(v) for v in out))
                getattr(ufunc, method)(*new_inputs, **kwargs)
                return self
        return getattr(ufunc, method)(*new_inputs, **kwargs)

    def __getattr__(self, item):
        """Used to mock `numpy.ndarray`'s interface."""
        arr = self._coerce_to_ndarray()
        return getattr(arr, item)

    @property
    def __array_interface__(self):
        """Returns the information for a copy of the underlying data buffer.

        This ensures that calls to functions like `numpy.array` do not cause
        invalid access to the buffer. We must copy because ones we release
        `numpy.ndarray` pointing to the underlying buffer we cannot guarentee
        safety.
        """
        return np.array(self._coerce_to_ndarray(),
                        copy=True).__array_interface__

    def _coerce_to_ndarray(self):
        """Provide a `numpy.ndarray` to the underlying buffer.

        Raises a `HOOMDArrayError` when the provide callback returns False.
        """
        if self._callback():
            return array(self._buffer, copy=False)
        else:
            raise HOOMDArrayError(
                "Cannot access HOOMDArray outside context manager. Use "
                "numpy.array inside context manager instead.")

    @property
    def shape(self):
        return self._coerce_to_ndarray().shape

    @shape.setter
    def shape(self, value):
        raise HOOMDArrayError("Shape cannot be set on a HOOMDArray. Use "
                              "``array.reshape`` instead.")

    def __str__(self):
        cls = self.__class__
        if self._callback():
            return cls.__name__ + "(" + str(self._coerce_to_ndarray()) + ")"
        else:
            return cls.__name__ + "(INVALID)"

    def __repr__(self):
        cls = self.__class__
        if self._callback():
            return cls.__name__ + "(" + str(self._coerce_to_ndarray()) + ")"
        else:
            return cls.__name__ + "(INVALID)"

    def _repr_html_(self):
        cls = self.__class__
        if self._callback():
            return "<emph>" + cls.__name__ + "</emph>" \
                + "(" + str(self._coerce_to_ndarray()) + ")"
        else:
            return "<emph>" + cls.__name__ + "</emph>" \
                + "(<strong>INVALID</strong>)"
