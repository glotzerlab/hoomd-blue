import functools
from inspect import isclass

from numpy import ndarray, generic


class HOOMDArrayError(RuntimeError):
    pass


def _error_outside_manager(cls, method):
    """Higher order function to wrap classmethods of numpy.ndarray.

    This lets us ensure that all calls to dunder (__) functions are valid by
    checking the return value of the callback.
    """
    parent_method = getattr(super(cls, cls), method)
    @functools.wraps(parent_method)
    def error(self, *args, **kwargs):
        if object.__getattribute__(self, '__callback__')():
            return parent_method(self, *args, **kwargs)
        else:
            raise HOOMDArrayError(
                "Cannot access array outside context manager.")

    return error


class _WrapMethods(type):
    """Metaclass for wrapping class methods that error outside context manager.

    Uses the _wrap_methods attribute defined in a class. This attribute is
    not found in the actual class instance, however. The purpose of this class
    is to ensure that :class:`_HOOMDArrayBase` wraps all methods that aren't
    grabbed by a class's `__getattribute` method.
    """
    def __new__(cls, name, bases, class_dict):
        wrap_methods = class_dict.pop("_wrap_methods", [])
        new_cls = super().__new__(cls, name, bases, class_dict)
        for method in wrap_methods:
            setattr(new_cls, method,
                    _error_outside_manager(new_cls, method))
        return new_cls


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
    '__init__', '__init_subclass__', '__new__', '__setattr__', '__repr__',
    '__str__', '__subclasshook__']
_wrap_ndarray_methods = filter(lambda x: x not in _exclude_methods,
                               _wrap_ndarray_methods)


class _HOOMDArrayBase(ndarray, metaclass=_WrapMethods):
    """Internal base class for zero copy NumPy array views of State data.

    We use ``_wrap_methods`` to signify functions that must be wrapped using
    _error_outside_manager. ``_raise_attribute_error`` is to ensure we can view
    invalid ``_HOOMDArrayBase`` instances in notebooks.

    The logic of the class comes in the ``__callback__`` attribute of an
    instance of ``_HOOMDArrayBase``. The attribute is a function that returns a
    Boolean. If True then the array is considered valid, if False invalid. Most
    methods will raise a ``HOOMDArrayError`` if called on an invalid array. The
    ``__callback__`` is designed to tell us whether we are in the context
    manager of for the :class:`LocalSnapshot` object (this is used by the
    :class:`hoomd.State` object.

    A quirk of the class that makes it faster is that ``__callback__`` should
    only every be queried using
    ``object.__getattribute__(self, '__callback__)``.
    """
    _wrap_methods = list(_wrap_ndarray_methods)
    _raise_attribute_error = {
        '_ipython_canary_method_should_not_exist_',
        '_repr_javascript_',
        '_ipython_display_',
        '_repr_mimebundle_',
        '_repr_svg_',
        '_repr_html_',
        '_repr_markdown_',
        '_repr_png_',
        '_repr_jpeg_',
        '_repr_latex_',
        '_repr_json_'}

    def __new__(cls, *args, **kwargs):
        try:
            callback = kwargs.pop('callback')
        except KeyError:
            raise ValueError("callback is a required key word argument.")
        if not callable(callback):
            raise ValueError("callback must be callable.")
            
        arr = super(_HOOMDArrayBase, cls).__new__(cls, *args, **kwargs)
        arr.__callback__ = callback
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            pass
        elif isinstance(obj, _HOOMDArrayBase):
            self.__callback__ = obj.__callback__
        else:
            self.__callback__ = lambda: True
            
    def __getattribute__(self, item):
        try:
            callback = object.__getattribute__(self, '__callback__')()
        except AttributeError:
            callback = True
        if callback:
            return super().__getattribute__(item)
        else:
            if item == '__class__':
                return super().__getattribute__(item)
            elif item in self.__class__._raise_attribute_error:
                raise AttributeError
            else:
                raise HOOMDArrayError(
                    "Cannot access array outside context manager.")

    def __repr__(self):
        if object.__getattribute__(self, '__callback__')():
            return super().__repr__()
        else:
            return "HOOMDArray(INVALID)"

    def __str__(self):
        if object.__getattribute__(self, '__callback__')():
            return super().__str__()
        else:
            return "HOOMDArray(INVALID)"

    @classmethod
    def _from_buffer(cls, buffer, callback):
        """Expects buffer to have a shape and dtype property."""
        shape = buffer.shape
        dtype = buffer.dtype
        return cls(shape, dtype, buffer, callback=callback)


def _get_hoomd_array_subclass():
    class HOOMDArray(_HOOMDArrayBase):
        """Adds logic to enable us to strip numpy.ndarray subclass if possible.

        Keeps a set of pointers that have been used for this class currently.
        Classes are instantiated when necessary by the :class:`hoomd.State` or
        other objects. The class is expected to be destroyed when it is done.
        """
        _existing_pointers = set()

        def __new__(cls, *args, **kwargs):
            arr = super(HOOMDArray, cls).__new__(cls, *args, **kwargs)
            cls._existing_pointers.add(arr.__array_interface__['data'][0])
            return arr

        def __array_wrap__(self, output, context=None):
            """Gets called after a _ufunc_ is called.

            (e.g) in ``a = b + c`` after ``b + c`` is computed. This allows us
            to drop the subclass of ``numpy.ndarray`` when possible.
            """
            pointer = output.__array_interface__['data'][0]
            cls = self.__class__
            if isinstance(output, _HOOMDArrayBase):
                other_pointers = output.__class__._existing_pointers
                if pointer not in cls._existing_pointers.union(
                        other_pointers):
                    return output.view(ndarray)
            else:
                if pointer not in cls._existing_pointers:
                    return output.view(ndarray)
                else:
                    return super().__array_wrap__(output, context)

        def view(self, dtype=None, type=None):
            """In general it is not safe to view a HOOMDArray in another class.

            This would circumvent the logic to prevent invalid accesses of
            memory and segfaults.
            """
            if (type is not None 
                    or (isclass(dtype) and isinstance(dtype, generic))):
                if (self.__array_interface__['data'][0]
                        in self.__class__._existing_pointers):
                    raise HOOMDArrayError(
                        "Cannot view HOOMDArray as another array type.")
            return super().view(dtype)

    return HOOMDArray
