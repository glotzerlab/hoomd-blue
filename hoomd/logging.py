from copy import deepcopy
from enum import Flag, auto
from itertools import count
from functools import reduce
from hoomd.util import dict_map, SafeNamespaceDict
from collections.abc import Sequence


class _LoggableEntry:
    def __init__(self, flag, default):
        self.flag = flag
        self.default = default


class TypeFlags(Flag):
    """Enum that marks all accepted logger types.

    This class does not need to be used by users directly. We directly convert
    from strings to the enum wherever necessary in the API. This class is
    documented to show users what types of quantities can be logged, and what
    flags to use for limiting what data is logged, user specified logged
    quantites, and custom actions (`hoomd.custom.Action`).

    Flags:
        scalar: `float` or `int` objects (i.e. numbers)
        sequence: sequence (e.g. `list`, `tuple`, `numpy.ndarray`) of numbers of
            the same type.
        string: a single Python `str` object
        strings: a sequence of Python `str` objects
        object: any Python object outside a sequence, string, or scalar.
        angle: per-angle quantity
        bond: per-bond quantity
        constraint: per-constraint quantity
        dihedral: per-dihedral quantity
        improper: per-improper quantity
        pair: per-pair quantity
        particle: per-particle quantity
        state: internal flag for specifying object's internal state
        ALL: a combination of all other flags
        NONE: represents no flag
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
    state = auto()

    @classmethod
    def any(cls, flags):
        def from_str(flag):
            if isinstance(flag, str):
                return cls[flag]
            else:
                return flag
        return reduce(lambda x, y: from_str(x) | from_str(y), flags)


TypeFlags.ALL = TypeFlags.any(TypeFlags.__members__.values())


def _loggables(self):
    """dict[str, str] Return a name: flag mapping of loggable quantities
    for the class."""
    return {name: str(quantity.flag)
            for name, quantity in self._export_dict.items()}


class Loggable(type):
    _meta_export_dict = dict()

    @classmethod
    def log(cls, func=None, *, is_property=True, flag='scalar', default=True):
        """Creates loggable quantites for classes of type Loggable.

        Args:
            func (method): class method to make loggable. If using non-default
                arguments, func should not be set.
            is_property (bool, optional): Whether to make the method a property,
                defaults to True. Argument position only
            flag (str, optional): The string represention of the type of
                loggable quantity, defaults to 'scalar'. See
                `hoomd.logging.TypeFlags` for available types. Argument
                position only
            default (boo, optional): Whether the quantity should be logged by
                default, defaults to True. This is orthogonal to the loggable
                quantity's type. An example would be performance orientated
                loggable quantities.  Many users may not want to log such
                quantities even when logging other quantities of that type. The
                default flag allows for these to be pass over by
                `hoomd.logging.Logger` objects by default.
        """

        def helper(func):
            name = func.__name__
            if name in cls._meta_export_dict:
                raise KeyError(
                    "Multiple loggable quantities named {}.".format(name))
            cls._meta_export_dict[name] = _LoggableEntry(
                TypeFlags[flag], default)
            if is_property:
                return property(func)
            else:
                return func

        if func is None:
            return helper
        else:
            return helper(func)

    def __init__(cls, name, bases, dct):
        """Adds marked quantites for logging in new class.

        Also adds a loggables property that returns a mapping of loggable
        quantity names with the string flag. We overwrite __init__ instead of
        __new__ since this plays much more nicely with inheritance. This allows,
        for instance, `Loggable` to be subclassed with metaclasses that use
        __new__ without having to hack the subclass's behavior.
        """
        # This grabs the metaclass of the newly created class which will be a
        # subclass of the Loggable metaclass (or the class itself)
        loggable_cls = type(cls)

        # grab loggable quantities through class inheritance.
        log_dict = loggable_cls._get_inherited_loggables(cls)

        # Add property to get all available loggable quantities. We ensure that
        # we haven't already added a loggables property first. The empty dict
        # check is for improved speed while the not any checking of subclasses
        # allows for certainty that a previous class of type Loggable (or one of
        # its subclasses) did not already add that property. This is not
        # necessary, but allows us to check that an user or developer didn't
        # accidentally create a loggables method, attribute, or property
        # already. We can speed this up by just removing the check and
        # overwriting the property every time, but lose the ability to error on
        # improper class definitions.
        if log_dict == {} and not any(issubclass(type(c), Loggable)
                                      for c in cls.__mro__):
            loggable_cls._add_loggable_property(cls)

        # grab the current class's loggable quantities
        log_dict.update(loggable_cls._get_current_cls_loggables(cls))
        cls._export_dict = log_dict
        loggable_cls._meta_export_dict = dict()

    @staticmethod
    def _add_loggable_property(new_cls):
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
        # know if we should errorr if a loggables method is defined. We also
        # skip the first entry since that is the new_cls itself.
        inherited_loggables = dict()
        for base_cls in reversed(new_cls.__mro__[1:]):
            # The conditional checks if the type of one of the parent classes of
            # new_cls has a metaclass (or type) which is a subclass of Loggable
            # or one of its subclasses.
            if issubclass(type(base_cls), Loggable):
                inherited_loggables.update(
                    {name: deepcopy(quantity).update_cls(new_cls)
                     for name, quantity in base_cls._export_dict.items()})
        return inherited_loggables

    @classmethod
    def _get_current_cls_loggables(cls, new_cls):
        """Gets the current class's new loggables (not inherited."""
        return {name: LoggerQuantity(name, new_cls, entry.flag, entry.default)
                for name, entry in cls._meta_export_dict.items()}


class LoggerQuantity:
    """The information to automatically log to a `hoomd.logging.Logger`.

    Args:
        name (str): The name of the quantity.
        cls (class object): The class that the quantity comes from.
        flag (str or TypeFlags, optional): The type of quantity it is.
            Valid values are given in the `hoomd.logging.TypeFlags`
            documentation.

    Note:
        For users, this class is meant to be used in conjunction with
        `hoomd.custom.Action` for exposing loggable quantities for custom user
        actions.
    """

    def __init__(self, name, cls, flag='scalar', default=True):
        self.name = str(name)
        self.update_cls(cls)
        if isinstance(flag, str):
            self.flag = TypeFlags[flag]
        elif isinstance(flag, TypeFlags):
            self.flag = flag
        else:
            raise ValueError("Flag must be a string convertable into "
                             "TypeFlags or a TypeFlags object.")
        self.default = bool(default)

    def yield_names(self):
        """Infinitely yield potential namespaces.

        Used to ensure that all namespaces are unique for a
        `hoomd.logging.Logger` object. We simple increment a number at the end
        until the caller stops asking for another namespace.

        Yields:
            tuple[str]: A potential namespace for the object.
        """
        yield self.namespace + (self.name,)
        for i in count(start=1, step=1):
            yield self.namespace[:-1] + \
                (self.namespace[-1] + '_' + str(i), self.name)

    def update_cls(self, cls):
        """Allow updating the class/namespace of the object.

        Since the namespace is determined by the passed class's module and class
        name, if inheritanting `hoomd.logging.LoggerQuantity`, the class needs
        to be updated to the subclass.

        Args:
            cls (class object): The class to update the namespace with.
        """
        self.namespace = self._generate_namespace(cls)
        return self

    @staticmethod
    def _generate_namespace(cls):
        """Infite iterator of namespaces for a given class.

        If namespace is taken add a number and increment until unique.
        """
        ns = tuple(cls.__module__.split('.') + [cls.__name__])
        if ns[0] == 'hoomd':
            return ns[1:]
        else:
            return ns


class _LoggerEntry:
    """Stores the information for an entry in a `hoomd.logging.Logger`."""

    def __init__(self, obj, attr, flag):
        self.obj = obj
        self.attr = attr
        self.flag = flag

    @classmethod
    def from_logger_quantity(cls, obj, logger_quantity):
        return cls(obj, logger_quantity.name, logger_quantity.flag)

    @classmethod
    def from_tuple(cls, entry):
        err_msg = "Expected either (callable, flag) or \
                   (obj, method/property, flag)."
        if (not isinstance(entry, Sequence) or
                len(entry) <= 1 or
                len(entry) > 3):
            raise ValueError(err_msg)
        print(len(entry))

        # Get the method and flag from the passed entry. Also perform some basic
        # validation.
        if len(entry) == 2:
            if not callable(entry[0]):
                raise ValueError(err_msg)
            flag = entry[1]
            method = '__call__'
        elif len(entry) == 3:
            if not isinstance(entry[1], str):
                raise ValueError(err_msg)
            method = entry[1]
            if not hasattr(entry[0], method):
                raise ValueError(
                    "Provided method/property must exist in given object.")
            flag = entry[2]

        # Ensure flag is valid and converted to TypeFlags enum.
        if isinstance(flag, str):
            flag = TypeFlags[flag]
        elif not isinstance(flag, TypeFlags):
            raise ValueError(
                "flag must be a string or hoomd.logging.TypeFlags object.")
        return cls(entry[0], method, flag)

    def __call__(self):
        attr = getattr(self.obj, self.attr)
        if self.flag is TypeFlags.state:
            return attr
        if callable(attr):
            return (attr(), self.flag)
        else:
            return (attr, self.flag)

    def __eq__(self, other):
        return (self.obj == other.obj and
                self.attr == other.attr and
                self.flag == other.flag)
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['obj', 'attr', 'flag'])


class Logger(SafeNamespaceDict):
    '''Logs HOOMD-blue operation data and custom quantities.

    The `Logger` class provides an intermediatary between a back end such as the
    `hoomd.output.CSV` and many of HOOMD-blue's object (as most objects are
    loggable). The `Logger` class makes use of *namespaces* which denote where a
    logged quantity fits in. For example internally all loggable quantities are
    ordered by the module and class them come from. For instance, the
    `hoomd.md.pair.LJ` class has a namespace `('md', 'pair', 'LJ')`. This
    applies to all loggable internal objects in HOOMD-blue. This ensures that
    logged quantities remain unambigious. To add a loggable object's quantities
    two methods exist `Logger.add` and the ``+=`` operator. Here we show an
    example using the ``+=`` operator.

    Example:
        .. code-block: python

            logger = hoomd.logging.Logger()
            lj = md.pair.lj(nlist)
            # Log all default quantites of the lj object
            logger += lj
            logger = hoomd.logging.Logger(flags=['scalar'])
            # Log all default scalar quantites of the lj object
            logger += lj

    The `Logger` class also supports user specified quantities using namespaces
    as well.

    Example:
        .. code-block: python

            logger = hoomd.logging.Logger()
            # Add quantity to ('custom', 'name') namespace
            logger[('custom', 'name')] = (lambda: 42, 'scalar')
            # Add quantity to ('custom_name',) namespace
            logger[('custom_name',)] = (lambda: 43, 'scalar')

    `Logger` objects support two ways of discriminating what loggable quantities
    they will accept: ``flags`` and ``only_default`` (the constructor
    arguements). Both of these are static meaning that once instantiated a
    `Logger` object will not change the values of these two properties.
    ``flags`` determines what if any types of loggable quantities (see
    `hoomd.logging.TypeFlags`) are appropriate for a given `Logger` object. This
    helps logging back ends determine if a `Logger` object is compatible. The
    ``only_default`` flag is mainly a convenience by allowing quantities not
    commonly logged (but available) to be passed over unless explicityly asked
    for. You can override the ``only_default`` flag by explicitly listing the
    quantities you want in `Logger.add`, but the same is not true with regards
    to ``flags``.
    '''

    def __init__(self, flags=None, only_default=True):
        self._flags = TypeFlags.ALL if flags is None else TypeFlags.any(flags)
        self._only_default = only_default
        super().__init__()

    def _filter_quantities(self, quantities, overwrite_default=False):
        if overwrite_default:
            def filter_func(log_quantity):
                return log_quantity.flag in self._flags
        else:
            def filter_func(log_quantity):
                return log_quantity.default and log_quantity.flag in self._flags
        yield from filter(filter_func, quantities)

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
                map(lambda q: obj._export_dict[q], quantities))

    def add(self, obj, quantities=None):
        """Add loggables from obj to logger. Returns the used namespaces.

        Args:
            obj (loggable class): class of type loggable to add loggable
                quantities from.
            quantities (Sequence[str]): list of str names of quantities to log.

        Returns:
            used_namespaces (list[tuple[str]]): A list of namespaces that were
                added to the logger.
        """
        used_namespaces = []
        for quantity in self._get_loggables_by_name(obj, quantities):
            used_namespaces.append(self._add_single_quantity(obj, quantity))
        return used_namespaces

    def remove(self, obj=None, quantities=None):
        """Remove specified quantities from the logger.

        Args:
            obj (loggable obj, optional): Object to remove quantities from. If
                ``quantities`` is None, ``obj`` must be set. If ``obj`` is set
                and ``quantities`` is None, all logged quanties from ``obj``
                will be removed from the logger.
            quantities (Sequence[tuple]): a sequence of namespaces to remove
                from the logger. If specified with ``obj`` only remove
                quantities listed that are exposed from ``obj``. If ``obj`` is
                None, then ``quantities`` must be given.

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
                for namespace in quantity.yield_names():
                    if namespace in self:
                        if self._contains_obj(namespace, obj):
                            del self[namespace]
                    # We deterministically go through namespaces, so once a
                    # namespace is not in the logger, than we can be sure no
                    # further ones will be as well and break.
                    else:
                        break

    def _add_single_quantity(self, obj, quantity):
        '''If quantity for obj is not logged add to first available namespace.
        '''
        for namespace in quantity.yield_names():
            if namespace in self:
                # Check if the quantity is already logged by the same object
                if self._contains_obj(namespace, obj):
                    return namespace
                else:
                    continue
            else:
                self[namespace] = _LoggerEntry.from_logger_quantity(
                    obj, quantity)
                return namespace

    def __setitem__(self, namespace, value):
        """Allows user specified loggable quantities.

        Args:
            namespace (tuple[str,] or str): key or nested key to determine where
                to store logged quantity.
            value (tuple[callable, str] or tuple[object, str, str]):
                Either a tuple with a callable and the `hoomd.logging.TypeFlags`
                object or associated string or a object with a method/property
                name and flag. If using a method it should not take arguments or
                have defaults for all arguments.
        """
        if isinstance(value, _LoggerEntry):
            super().__setitem__(namespace, value)
        else:
            super().__setitem__(namespace, _LoggerEntry.from_tuple(value))

    def __iadd__(self, obj):
        """Add quantities from object or list of objects to logger.

        Adds all quantities compatible with given flags and default value.

        Examples:
            ..code-block: python

                logger += lj
                logger += [lj, harmonic_bonds]
        """
        if hasattr(obj, '__iter__'):
            for o in obj:
                self.add(o)
        else:
            self.add(obj)
        return self

    def __isub__(self, value):
        """Remove log entries for a list of quantities or objects.

        Examples:
            ..code-block: python

                logger -= ('md', 'pair', 'lj')
                logger -= [('md', 'pair', 'lj', 'energy'),
                           ('md', 'pair', 'lj', 'forces')]
                logger -= lj
                logger -= [lj, harmonic_bonds]
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

        Returns:
            log_dict (dict): A nested dictionary of the current logged
                quantities. The end values are (value, flag) pairs which hold
                the value along with its associated TypeFlags flag.
        """
        return dict_map(self._dict, lambda x: x())

    def _contains_obj(self, namespace, obj):
        '''evaulates based on identity.'''
        return self._unsafe_getitem(namespace).obj is obj

    @staticmethod
    def _wrap_quantity(quantity):
        """Handles wrapping strings and tuples for iterating over namespaces."""
        if isinstance(quantity, (str, tuple)):
            return [quantity]
        else:
            return quantity
