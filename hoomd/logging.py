# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Logging infrastructure.

Use the `Logger` class to collect loggable quantities (e.g. kinetic temperature,
pressure, per-particle energy) during the simulation run. Pass the `Logger` to a
backend such as `hoomd.write.GSD` or `hoomd.write.Table` to write the logged
values to a file.

See Also:
    Tutorial: :doc:`tutorial/02-Logging/00-index`

    Tutorial: :doc:`tutorial/04-Custom-Actions-In-Python/00-index`

.. invisible-code-block: python

    if hoomd.version.md_built:
        lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
"""

import copy
from enum import Flag, auto
from itertools import count
from functools import reduce, wraps
import weakref

import hoomd
from hoomd.util import _SafeNamespaceDict
from hoomd.error import DataAccessError
from collections.abc import Sequence


class LoggerCategories(Flag):
    """Enum that marks all accepted logger types.

    The attribute names of `LoggerCategories` are valid strings for the
    category argument of `Logger` constructor and the `log` method.

    Attributes:
        scalar: `float` or `int` object.

        sequence: Sequence (e.g. `list`, `tuple`, `numpy.ndarray`) of numbers
            of the same type.

        string: A single Python `str` object.

        strings: A sequence of Python `str` objects.

        object: Any Python object outside a sequence, string, or scalar.

        angle: Per-angle quantity.

        bond: Per-bond quantity.

        constraint: Per-constraint quantity.

        dihedral: Per-dihedral quantity.

        improper: Per-improper quantity.

        pair: Per-pair quantity.

        particle: Per-particle quantity.

        ALL: A combination of all other categories.

        NONE: Represents no category.
    """
    NONE = 0
    scalar = auto()
    sequence = auto()
    string = auto()
    strings = auto()
    object = auto()
    angle = auto()
    bond = auto()
    constraint = auto()
    dihedral = auto()
    improper = auto()
    pair = auto()
    particle = auto()

    @classmethod
    def any(cls, categories=None):
        """Return a LoggerCategories enum representing any of the categories.

        Args:
            categories (`list` [`str` ] or `list` [`LoggerCategories`]):
                A list of `str` or `LoggerCategories` objects that should be
                represented by the returned `LoggerCategories` object.

        Returns:
            `LoggerCategories`: the `LoggerCategories` object that represents
            any of the given categories.
        """
        categories = cls.__members__.values(
        ) if categories is None else categories

        return reduce(cls._combine_flags, categories, LoggerCategories.NONE)

    @classmethod
    def _combine_flags(cls, flag1, flag2):
        return cls._from_str(flag1) | cls._from_str(flag2)

    @classmethod
    def _from_str(cls, category):
        if isinstance(category, str):
            return cls[category]
        else:
            return category

    @classmethod
    def _get_string_list(cls, category):
        c = [mem.name for mem in cls.__members__.values() if mem in category]
        # Remove NONE from list
        c.pop(0)
        return c


LoggerCategories.ALL = LoggerCategories.any()


# function defined here to ensure that each class of type Loggable will have a
# loggables property
def _loggables(self):
    """dict[str, str]: Name, category mapping of loggable quantities."""
    return {
        name: quantity.category.name
        for name, quantity in self._export_dict.items()
    }


class _LoggableEntry:
    """Stores entries for _Loggable's store of a class's loggable quantities."""

    def __init__(self, category, default):
        self.category = category
        self.default = default


class _NamespaceFilter:
    """Filter for creating the proper namespace for logging object properties.

    Attributes:
        remove_names (set[str], optional): A set of names which to remove for
            the logging namespace whenever encountered.
        base_names (set[str], optional): A set of names which indicate that the
            next encountered name in the string should be skipped. For example,
            if a module hierarchy was structured as ``project.foo.bar.Bar`` and
            ``foo`` directly imports ``Bar``, ``bar`` may not be desirable to
            have in the logging namespace since users interact with it via
            ``foo.Bar``. Currently, this only handles a single level of nesting
            like this.
        non_native_remove (set[str], optional): A set of names which to remove
            for the logging namespace when found in non-native loggables.
        skip_duplicates (bool, optional): Whether or not to remove consecutive
            duplicates from a logging namespace (e.g. ``foo.foo.bar`` ->
            ``foo.bar``), default ``True``. By default we assume that this
            pattern means that the inner module is imported into its parent.
    """

    def __init__(self,
                 remove_names=None,
                 base_names=None,
                 non_native_remove=None,
                 skip_duplicates=True):
        self.remove_names = set() if remove_names is None else remove_names
        if non_native_remove is None:
            self.non_native_remove = set()
        else:
            self.non_native_remove = non_native_remove
        self.base_names = set() if base_names is None else base_names
        self._skip_next = False
        self.skip_duplicates = skip_duplicates
        if skip_duplicates:
            self._last_name = None

    def __call__(self, namespace, native=True):
        """Filter out parts of the namespace.

        Args:
            namespace (tuple[str]): The namespace of the loggable.
            native (bool, optional): Whether the loggable comes internally from
                hoomd or not.

        Yields:
             str: The filtered parts of a namespace.
        """
        for name in namespace:
            # check for duplicates in the namespace and remove them (e.g.
            # `md.pair.pair.LJ` -> `md.pair.LJ`).
            if self.skip_duplicates:
                last_name = self._last_name
                self._last_name = name
                if last_name == name:
                    continue
            if not native:
                if name not in self.non_native_remove:
                    yield name
                continue
            if name in self.remove_names:
                continue
            elif self._skip_next:
                self._skip_next = False
                continue
            elif name in self.base_names:
                self._skip_next = True
            yield name
        # Reset for next call of filter
        self._skip_next = False
        self._last_name = None


class _LoggerQuantity:
    """The information to automatically log to a `Logger`.

    Args:
        name (str): The name of the quantity.
        cls (``class object``): The class that the quantity comes from.
        category (str or LoggerCategories, optional): The type of quantity.
            Valid values are given in the `LoggerCategories`
            documentation.

    Note:
        For users, this class is meant to be used in conjunction with
        `hoomd.custom.Action` for exposing loggable quantities for custom user
        actions.
    """

    namespace_filter = _NamespaceFilter(
        # Names that are imported directly into the hoomd namespace
        remove_names={"hoomd", 'simulation', 'state', 'operations', 'snapshot'},
        # Names that have their submodules' classes directly imported into them
        # (e.g. `hoomd.update.box_resize.BoxResize` gets used as
        # `hoomd.update.BoxResize`)
        base_names={'update', 'tune', 'write'},
        non_native_remove={"__main__"},
        skip_duplicates=True)

    def __init__(self, name, cls, category='scalar', default=True):
        self.name = name
        self.update_cls(cls)
        if isinstance(category, str):
            self.category = LoggerCategories[category]
        elif isinstance(category, LoggerCategories):
            self.category = category
        else:
            raise ValueError("Flag must be a string convertable into "
                             "LoggerCategories or a LoggerCategories object.")
        self.default = bool(default)

    def yield_names(self, user_name=None):
        """Infinitely yield potential namespaces.

        Used to ensure that all namespaces are unique for a `Logger` object.
        `yield_names` increments a number at the end until the caller stops
        asking for another namespace.

        Yields:
            tuple[str]: A potential namespace for the object.
        """
        if user_name is None:
            namespace = self.namespace
        else:
            namespace = self.namespace[:-1] + (user_name,)
        yield namespace + (self.name,)
        for i in count(start=1, step=1):
            yield namespace[:-1] + (namespace[-1] + '_' + str(i), self.name)

    def update_cls(self, cls):
        """Allow updating the class/namespace of the object.

        Since the namespace is determined by the passed class's module and class
        name, if inheriting from `_LoggerQuantity`, the class needs to be
        updated to the subclass.

        Args:
            cls (``class object``): The class to update the namespace with.
        """
        self.namespace = self._generate_namespace(cls)
        return self

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        attrs = ("namespace", "name", "category", "default")
        return all(getattr(self, a) == getattr(other, a) for a in attrs)

    @classmethod
    def _generate_namespace(cls, loggable_cls):
        """Generate the namespace of a class given its module hierarchy."""
        ns = tuple(loggable_cls.__module__.split('.'))
        cls_name = loggable_cls.__name__
        # Only filter namespaces of objects in the hoomd package
        return tuple(cls.namespace_filter(ns, ns[0] == "hoomd")) + (cls_name,)


class Loggable(type):
    """Loggable quantity."""
    _meta_export_dict = dict()

    def __init__(cls, name, bases, dct):
        """Adds marked quantities for logging in new class.

        Also adds a loggables property that returns a mapping of loggable
        quantity names with the string category. We overwrite __init__ instead
        of __new__ since this plays much more nicely with inheritance. This
        allows, for instance, `Loggable` to be subclassed with metaclasses that
        use __new__ without having to hack the subclass's behavior.
        """
        # grab loggable quantities through class inheritance.
        log_dict = Loggable._get_inherited_loggables(cls)

        # Add property to get all available loggable quantities. We ensure that
        # we haven't already added a loggables property first. The empty dict
        # check is for improved speed while the not any checking of subclasses
        # allows for certainty that a previous class of type Loggable (or one
        # of its subclasses) did not already add that property. This is not
        # necessary, but allows us to check that an user or developer didn't
        # accidentally create a loggables method, attribute, or property
        # already. We can speed this up by just removing the check and
        # overwriting the property every time, but lose the ability to error on
        # improper class definitions.
        if log_dict == {} and not any(
                issubclass(type(c), Loggable) for c in cls.__mro__[1:]):
            Loggable._add_property_for_displaying_loggables(cls)

        # grab the current class's loggable quantities
        log_dict.update(Loggable._get_current_cls_loggables(cls))
        cls._export_dict = log_dict
        Loggable._meta_export_dict = dict()

    @staticmethod
    def _add_property_for_displaying_loggables(new_cls):
        if hasattr(new_cls, 'loggables'):
            raise ValueError("classes of type Loggable cannot implement a "
                             "loggables method, property, or attribute.")
        else:
            new_cls.loggables = property(_loggables)

    @classmethod
    def _get_inherited_loggables(cls, new_cls):
        """Get loggable quantities from new class's __mro__."""
        # We reverse the mro list to ensure that if a conflict in names exist we
        # take the one with the most priority in the mro. Also track if any
        # parent classes also have Loggable as a metaclass. This allows us to
        # know if we should error if a loggables method is defined. We also
        # skip the first entry since that is the new_cls itself.
        inherited_loggables = dict()
        for base_cls in reversed(new_cls.__mro__[1:]):
            # The conditional checks if the type of one of the parent classes of
            # new_cls has a metaclass (or type) which is a subclass of Loggable
            # or one of its subclasses.
            if issubclass(type(base_cls), Loggable):
                inherited_loggables.update({
                    name: copy.deepcopy(quantity).update_cls(new_cls)
                    for name, quantity in base_cls._export_dict.items()
                })
        return inherited_loggables

    @classmethod
    def _get_current_cls_loggables(cls, new_cls):
        """Gets the current class's new loggables (not inherited)."""
        current_loggables = {}
        for name, entry in cls._meta_export_dict.items():
            current_loggables[name] = _LoggerQuantity(name, new_cls,
                                                      entry.category,
                                                      entry.default)
            cls._add_loggable_docstring_info(new_cls, name, entry.category,
                                             entry.default)
        return current_loggables

    @classmethod
    def _add_loggable_docstring_info(cls, new_cls, attr, category, default):
        doc = getattr(new_cls, attr).__doc__
        # Don't add documentation to empty docstrings. This means that the
        # quantity is not documented would needs to be fixed, but this prevents
        # the rendering of invalid docs since we need a non-empty docstring.
        if __doc__ == "":
            return
        str_msg = '\n\n{}(`Loggable <hoomd.logging.Logger>`: '
        str_msg += f'category="{str(category)[17:]}"'
        if default:
            str_msg += ')'
        else:
            str_msg += ', default=False)'
        if doc is None:
            getattr(new_cls, attr).__doc__ = str_msg.format('')
        else:
            indent = 0
            lines = doc.split('\n')
            if len(lines) >= 3:
                cnt = 2
                while lines[cnt] == '':
                    cnt += 1
                indent = len(lines[cnt]) - len(lines[cnt].lstrip())
            getattr(new_cls, attr).__doc__ += str_msg.format(' ' * indent)


def log(func=None,
        *,
        is_property=True,
        category='scalar',
        default=True,
        requires_run=False):
    """Creates loggable quantities for classes of type Loggable.

    Use `log` with `hoomd.custom.Action` to expose loggable quantities from a
    custom action.

    Args:
        func (`method`): class method to make loggable. If using non-default
            arguments, func should not be set.
        is_property (`bool`, optional): Whether to make the method a
            property, defaults to True. Keyword only argument.
        category (`str`, optional): The string represention of the type of
            loggable quantity, defaults to 'scalar'. See `LoggerCategories` for
            available types. Keyword only argument.
        default (`bool`, optional): Whether the quantity should be logged
            by default. Defaults to True. Keyword only argument.
        requires_run (`bool`, optional): Whether this property requires the
            simulation to run before being accessible.

    Note:
        The namespace (where the loggable object is stored in the `Logger`
        object's nested dictionary) is determined by the module/script and class
        name the loggable class comes from. In creating subclasses of
        `hoomd.custom.Action`, for instance, if the module the subclass is
        defined in is ``user.custom.action`` and the class name is ``Foo`` then
        the namespace used will be ``('user', 'custom', 'action', 'Foo')``. This
        helps to prevent naming conflicts and automates the logging
        specification for developers and users.

    .. rubric:: Example:

    .. code-block:: python

        class LogExample(metaclass=hoomd.logging.Loggable):
            @log()
            def loggable(self):
                return 1.5

    Note:
        The metaclass specification is not necessary for subclasses of HOOMD
        classes as they already use this metaclass.

    See Also:
        Tutorial: :doc:`tutorial/04-Custom-Actions-In-Python/00-index`
    """

    def helper(func):
        name = func.__name__
        if name in Loggable._meta_export_dict:
            raise KeyError(
                "Multiple loggable quantities named {}.".format(name))
        Loggable._meta_export_dict[name] = _LoggableEntry(
            LoggerCategories[category], default)
        if requires_run:

            def wrap_with_exception(func):

                @wraps(func)
                def wrapped_func(self, *args, **kwargs):
                    if not self._attached:
                        raise DataAccessError(name)
                    return func(self, *args, **kwargs)

                return wrapped_func

            func = wrap_with_exception(func)
        if is_property:
            return property(func)
        else:
            return func

    if func is None:
        return helper
    else:
        return helper(func)


class _InvalidLogEntryType():
    pass


_InvalidLogEntry = _InvalidLogEntryType()


class _LoggerEntry:
    """Stores the information for an entry in a `Logger`.

    The class deals with the logic of converting `tuple` and
    `_LoggerQuantity` objects into an object that can obtain the
    actually log value when called.

    Note:
        This class could perform verification of the logged quantities. It
        currently doesn't for performance reasons; this can be changed to give
        greater security with regards to user specified quantities.
    """

    def __init__(self, obj, attr, category):
        self.obj = obj
        self.attr = attr
        self.category = category

    @classmethod
    def from_logger_quantity(cls, obj, logger_quantity):
        return cls(obj, logger_quantity.name, logger_quantity.category)

    @classmethod
    def from_tuple(cls, entry):
        err_msg = "Expected either (callable, category) or \
                   (obj, method/property, category)."

        if (not isinstance(entry, Sequence) or len(entry) <= 1
                or len(entry) > 3):
            raise ValueError(err_msg)

        # Get the method and category from the passed entry. Also perform some
        # basic validation.
        if len(entry) == 2:
            if not callable(entry[0]):
                raise ValueError(err_msg)
            category = entry[1]
            method = '__call__'
        elif len(entry) == 3:
            if not isinstance(entry[1], str):
                raise ValueError(err_msg)
            method = entry[1]
            if not hasattr(entry[0], method):
                raise ValueError(
                    "Provided method/property must exist in given object.")
            category = entry[2]

        # Ensure category is valid and converted to LoggerCategories enum.
        if isinstance(category, str):
            category = LoggerCategories[category]
        elif not isinstance(category, LoggerCategories):
            raise ValueError(
                "category must be a string or hoomd.logging.LoggerCategories "
                "object.")
        return cls(entry[0], method, category)

    @property
    def obj(self):
        if isinstance(self._obj, weakref.ReferenceType):
            # We could optimize and check if this is None here or in __call__
            # and switch to a function that just returns None or set self._obj
            # to None, but hopefully users are not triggering this too often.
            return self._obj()
        return self._obj

    @obj.setter
    def obj(self, new_obj):
        if not isinstance(new_obj,
                          (hoomd.operation.Operation, hoomd.Simulation)):
            self._obj = new_obj
            return
        try:
            self._obj = weakref.ref(new_obj)
        except TypeError:
            self._obj = new_obj

    def __call__(self):
        obj = self.obj
        if obj is None:
            return _InvalidLogEntry
        try:
            attr = getattr(obj, self.attr)
        except DataAccessError:
            attr = None

        if callable(attr):
            return (attr(), self.category.name)
        else:
            return (attr, self.category.name)

    def __eq__(self, other):
        return (self.obj == other.obj and self.attr == other.attr
                and self.category == other.category)
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ['obj', 'attr', 'category'])

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        # Remove weak reference
        state["_obj"] = self.obj
        return state


class Logger(_SafeNamespaceDict):
    """Logs HOOMD-blue operation data and custom quantities.

    The `Logger` class provides an intermediary between a backend such as
    `hoomd.write.GSD` or `hoomd.write.Table` and loggable objects. The `Logger`
    class makes use of *namespaces* which organize logged quantities. For
    example internally all loggable quantities are ordered by the module and
    class they come from. For instance, the `hoomd.md.pair.LJ` class has a
    namespace ``('md', 'pair', 'LJ')``. This ensures that logged quantities
    remain unambiguous. Use the ``+=`` operator to add all matching loggable
    quantities provided by an object to the `Logger`.

    `Logger` provides two ways to limit what loggable quantities it will accept:
    ``categories`` and ``only_default`` (the constructor arguments). Once
    instantiated, a `Logger` object will not change the values of these two
    properties. ``categories`` determines what types of loggable quantities (see
    `LoggerCategories`) are appropriate for a given `Logger` object. The
    ``only_default`` flag prevents rarely-used quantities from being added to
    the logger when using`Logger.__iadd__` and `Logger.add` when not explicitly
    requested.

    Note:
        The logger provides a way for users to create their own logger back
        ends. See `log` for details on the intermediate representation.
        `LoggerCategories` defines the various categories available to specify
        logged quantities. Custom backends should be a subclass of
        `hoomd.custom.Action` and used with `hoomd.write.CustomWriter`.

    Note:
        When logging multiple instances of the same class `add` provides a means
        of specifying the class level of the namespace (e.g. ``'LJ`` in ``('md',
        'pair', 'LJ')``). The default behavior (without specifying a user name)
        is to just append ``_{num}`` where ``num`` is the smallest positive
        integer which makes the full namespace unique. This appending will also
        occur for user specified names that are reused.

    Args:
        categories (`list` [`str` ], `LoggerCategories`, optional): Either a
            list of string categories (list of categories can be found in
            `LoggerCategories`) or a `LoggerCategories` instance with the
            desired flags set. These are the only types of loggable quantities
            that can be logged by this logger. Defaults to allowing every type
            ``LoggerCategories.ALL``.
        only_default (`bool`, optional): Whether to log only quantities that are
            logged by default. Defaults to ``True``. Non-default quantities
            are typically measures of operation performance rather than
            information about simulation state.

    .. rubric:: Examples:

    There are various ways to create a logger with different available
    loggables. Create a `Logger` with no options to allow all categories.

    .. code-block:: python

        logger = hoomd.logging.Logger()

    Use a list of strings to log a subset of categories:

    .. code-block:: python

        logger = hoomd.logging.Logger(categories=["string", "strings"])
    """

    def __init__(self, categories=None, only_default=True):
        if categories is None:
            self._categories = LoggerCategories.ALL
        if isinstance(categories, LoggerCategories):
            self._categories = categories
        else:
            self._categories = LoggerCategories.any(categories)
        self._only_default = only_default
        super().__init__()

    @property
    def categories(self):
        """`LoggerCategories`: The enum representing the \
        acceptable categories for the `Logger` object."""
        return self._categories

    @property
    def string_categories(self):
        """list[str]: A list of the string names of the allowed \
        categories for logging."""
        return LoggerCategories._get_string_list(self._categories)

    @property
    def only_default(self):
        """`bool`: Whether the logger object should only add default loggable \
        quantities."""
        return self._only_default

    def _filter_quantities(self, quantities, force_quantities=False):
        for quantity in quantities:
            if quantity.category not in self._categories:
                continue
            # Must be before default check to overwrite _only_default
            if not self._only_default or quantity.default or force_quantities:
                yield quantity

    def _get_loggables_by_name(self, obj, quantities):
        if quantities is None:
            yield from self._filter_quantities(obj._export_dict.values())
        else:
            quantities = self._wrap_quantity(quantities)
            bad_keys = [q for q in quantities if q not in obj._export_dict]
            # ensure all keys are valid
            if bad_keys != []:
                raise ValueError(
                    "object {} has not loggable quantities {}.".format(
                        obj, bad_keys))
            yield from self._filter_quantities(
                map(lambda q: obj._export_dict[q], quantities), True)

    def add(self, obj, quantities=None, user_name=None):
        """Add loggables.

        Args:
            obj (object of class of type ``Loggable``): class of type loggable
                to add loggable quantities from.
            quantities (Sequence[str]): list of str names of quantities to log.
            user_name (`str`, optional): A string to replace the class name in
                the loggable quantities namespace. This allows for easier
                differentiation in the output of the `Logger` and any `Writer`
                which outputs its data.

        Returns:
            list[tuple[str]]: A list of namespaces added to the logger.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            logger.add(obj=lj, quantities=['energy'])
        """
        for quantity in self._get_loggables_by_name(obj, quantities):
            self._add_single_quantity(obj, quantity, user_name)

    def remove(self, obj=None, quantities=None, user_name=None):
        """Remove specified quantities from the logger.

        Args:
            obj (object of class of type ``Loggable``, optional): Object to
                remove quantities from. If ``quantities`` is None, ``obj`` must
                be set. If ``obj`` is set and ``quantities`` is None, all logged
                quantities from ``obj`` will be removed from the logger.

            quantities (Sequence[tuple]): a sequence of namespaces to remove
                from the logger. If specified with ``obj`` only remove
                quantities listed that are exposed from ``obj``. If ``obj`` is
                None, then ``quantities`` must be given.

            user_name (str): A user name to specify the final entry in the
                namespace of the object. This must be used in ``user_name`` was
                specified in `add`.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            logger.remove(obj=lj, quantities=['energy'])
        """
        if obj is None and quantities is None:
            raise ValueError(
                "Either obj, quantities, or both must be specified.")

        if obj is None:
            for quantity in self._wrap_quantity(quantities):
                if quantity in self:
                    del self[quantity]
        else:
            for quantity in self._get_loggables_by_name(obj, quantities):
                # Check all currently used namespaces for object's quantities.
                for namespace in quantity.yield_names(user_name):
                    if namespace in self:
                        if self._contains_obj(namespace, obj):
                            del self[namespace]
                    # We deterministically go through namespaces, so once a
                    # namespace is not in the logger, than we can be sure no
                    # further ones will be as well and break.
                    else:
                        break

    def _add_single_quantity(self, obj, quantity, user_name):
        """Add to first available namespace if obj is not logged."""
        for namespace in quantity.yield_names(user_name):
            if namespace in self:
                # Check if the quantity is already logged by the same object
                if self._contains_obj(namespace, obj):
                    return None
            else:
                self[namespace] = _LoggerEntry.from_logger_quantity(
                    obj, quantity)
                return None

    def __setitem__(self, namespace, value):
        """Allows user specified loggable quantities.

        Args:
            namespace (tuple[str,] or str): key or nested key to determine where
                to store logged quantity.

            value (tuple[Callable, str] or tuple[object, str, str]): Either a
                tuple with a callable and the `LoggerCategories`
                object or associated string or a object with a method/property
                name and category. If using a method it should not take
                arguments or have defaults for all arguments.

        .. rubric:: Example:

        .. invisible-code-block: python

            logger = hoomd.logging.Logger()

        .. code-block:: python

            logger[('custom', 'name')] = (lambda: 42, 'scalar')
        """
        if not isinstance(value, _LoggerEntry):
            value = _LoggerEntry.from_tuple(value)
        if value.category not in self.categories:
            raise ValueError(
                "User specified loggable is not of an accepted category.")
        super().__setitem__(namespace, value)

    def __iadd__(self, obj):
        """Add quantities from an object or list of objects.

        Adds all quantities compatible with given categories and default value.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            logger += lj
        """
        if hasattr(obj, '__iter__'):
            for o in obj:
                self.add(o)
        else:
            self.add(obj)
        return self

    def __isub__(self, value):
        """Remove log entries for a list of quantities or objects.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            logger -= lj
        """
        if isinstance(value, str) or isinstance(value, tuple):
            self.remove(quantities=value)
        elif hasattr(value, '__iter__'):
            for v in value:
                self.__isub__(v)
        else:
            self.remove(obj=value)
        return self

    def log(self):
        """Get a nested dictionary of the current values for logged quantities.

        The nested dictionary consist of one level for each element of a
        namespace. The logged value and category for the namespace ``('example',
        'namespace')`` would be accessible in the returned dictionary via
        ``logger.log()['example']['namespace']``.

        Returns:
            dict: A nested dictionary of the current logged quantities. The end
            values are (value, category) pairs which hold the value along
            with its associated `LoggerCategories` category represented as a
            string (to get the `LoggerCategories` enum value use
            ``LoggerCategories[category]``).

        .. rubric:: Example:

        .. code-block:: python

            values_to_log = logger.log()
        """
        # Use namespace dict to correctly nest log values.
        data = _SafeNamespaceDict()
        # We remove all keys where the reference to the object has become
        # invalid.
        remove_keys = []
        for key, entry in self.items():
            log_value = entry()
            if log_value is _InvalidLogEntry:
                remove_keys.append(key)
                continue
            data[key] = log_value
        if len(remove_keys) > 0:
            for key in remove_keys:
                del self[key]
        return data._dict

    def _contains_obj(self, namespace, obj):
        """Evaluates based on identity."""
        return self._unsafe_getitem(namespace).obj is obj

    @staticmethod
    def _wrap_quantity(quantity):
        """Handles wrapping strings and tuples for iterating over namespaces."""
        if isinstance(quantity, (str, tuple)):
            return [quantity]
        else:
            return quantity

    def __eq__(self, other):
        """Check for equality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.categories == other.categories
                and self.only_default == other.only_default
                and self._dict == other._dict)


def modify_namespace(cls, namespace=None):
    """Modify a class's namespace to a manually assigned one.

    Args:
        cls (type or tuple[str]): The class to modify the namespace of or the
            namespace itself. When passing a namespace (a tuple of strings), the
            function can be used as a decorator.
        namespace (`tuple` [`str` ], optional): The namespace to change the
            class's namespace to.

    Warning:
        This will only persist for the current class. All subclasses will have
        the standard namespace assignment.
    """
    if namespace is None:
        namespace = cls
        cls = None

    def modify(cls):
        for entry in cls._export_dict.values():
            entry.namespace = namespace
        return cls

    if cls is None:
        return modify

    return modify(cls)
