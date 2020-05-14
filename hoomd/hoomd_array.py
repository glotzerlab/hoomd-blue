import functools
from inspect import isclass
from collections import Iterable

from numpy import ndarray, array


class HOOMDArrayError(RuntimeError):
    pass


def WrapClass(methods, functor, *args, **kwargs):
    """Factory method for metaclasses that produce methods via a functor.

    Applies the functor to each method given in methods. This occurs before
    class creation. The functor can take any number of arguments, but besides
    for the method name must be the same accross all methods.

    Args:
        methods (Sequence[str]): A sequence of method names to apply the functor
            to.
        functor (Callable): A callable object that takes a method name and
            the passed ``*args``, ``**kwargs``.
        *args (Any): Required position arguments for ``functor``.
        **kwargs (Any): Required key word arugments for ``functor``.
    """
    class _WrapClass(type):
        def __new__(cls, name, bases, class_dict):
            for method in methods:
                class_dict[method] = functor(method, *args, **kwargs)
            return super().__new__(cls, name, bases, class_dict)

    return _WrapClass


"""
Get all methods that need to be wrapped for a ndarray.

These are essentially all the `__method__` methods, except those we implement or
make sense for an invalid array. By default we include all magic methods so we
are cautious by default.
"""
# don't include normal methods `__getattribute__` handles these
_wrap_ndarray_methods = filter(lambda x: x.startswith('__'), dir(ndarray))
# don't include `__array_...` methods
_wrap_ndarray_methods = filter(lambda x: not x.startswith('__array'),
                               _wrap_ndarray_methods)

_exclude_methods = ['__class__', '__dir__', '__doc__', '__getattribute__',
    '__getattr__', '__init__', '__init_subclass__', '__new__', '__setattr__',
    '__repr__', '__str__', '__subclasshook__']
_wrap_ndarray_methods = filter(lambda x: x not in _exclude_methods,
                               _wrap_ndarray_methods)

_wrap_ndarray_methods = list(_wrap_ndarray_methods)


def _ndarray_wrapper(method):
    """Wraps methods of ``numpy.ndarray`` for use with HOOMDArray.

    Given a method name it calls the corresponding NumPy function with a
    ``ndarray`` view of the underlying `HOOMDArray` buffer. It is designed for
    the dunder (__) methods.
    """
    # This conditional is required since a += b acts like a = a + b. Since we in
    # return a NumPy array in __add__, we must ensure that we do not return
    # access to our internal buffer in __iadd__.
    if method.startswith('__i') and method not in {
            '__index__', '__int__', '__iter__'}:
        def wrapped_method(self, *args, **kwargs):
            arr = self._coerce_to_ndarray()
            getattr(arr, method)(*args, **kwargs)
            return self
    else:
        def wrapped_method(self, *args, **kwargs):
            arr = self._coerce_to_ndarray()
            return getattr(arr, method)(*args, **kwargs)

    return wrapped_method


def coerce_mock_to_array(val):
    """Helper function for ``__array_{ufunc,function}__``.

    Coerces ``HOOMDArray`` objects into ``numpy.ndarray`` objects.
    """
    if isinstance(val, Iterable) and not isinstance(val, (ndarray, MockArray)):
        return [coerce_mock_to_array(v) for v in val]
    return val if not isinstance(val, MockArray) else val._coerce_to_ndarray()


class HOOMDArray(metaclass=WrapClass(_wrap_ndarray_methods, _ndarray_wrapper)):
    """A NumPy like interface to internal HOOMD-blue data.

    These objects are returned by HOOMD-blue's zero copy access to system data.
    This class acts like a ``numpy.ndarray`` object through NumPy's provided
    interface
    [https://numpy.org/doc/stable/reference/arrays.classes.html](link).
    For typical use cases, understanding this class is not necessary. Treat it
    as a ``numpy.ndarray``.

    We attempt to escape this class whenever possible. To ensure memory safety,
    a `HOOMDArray` object cannot be accessed outside of the context manager in
    which it was created. To have access outside the manager an explicit copy
    must be made (e.g. ``numpy.array(obj, copy=True)``).

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
        return func(*new_inputs, **kwargs)

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
            buffer_ = self._buffer
            return ndarray(buffer_.shape, buffer_.dtype, buffer_)
        else:
            raise HOOMDArrayError(
                "Cannot access HOOMDArray outside context manager. Use "
                "numpy.array inside context manager instead.")

    def view(self, dtype=None, cls=None):
        """We disallow views, since the copying for a view is not intuitive."""
        raise HOOMDArrayError(
            "Cannot view HOOMDArray directly. Copy array for use.")

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
