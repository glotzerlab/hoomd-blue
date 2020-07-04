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

    def __new__(cls, name, base, dct):
        """Adds marked quantites for logging in new class.

        Also adds a loggables property that returns a mapping of loggable
        quantity names with the string flag.
        """
        new_cls = super().__new__(cls, name, base, dct)

        log_dict = dict()
        # grab loggable quantities through class inheritance. We reverse the mro
        # list to ensure that if a conflict in names exist we take the one with
        # the most priority in the mro. Also track if any parent classes also
        # have Loggable as a metaclass. This allows us to know if we should
        # errorr if a loggables method is defined.
        add_loggables = True
        for base_cls in reversed(new_cls.__mro__):
            if type(base_cls) == cls:
                add_loggables = False
            if hasattr(base_cls, '_export_dict'):
                log_dict.update(
                    {name: deepcopy(quantity).update_cls(new_cls)
                     for name, quantity in base_cls._export_dict.items()})
        # handle new loggable quantities from current class
        for name, entry in cls._meta_export_dict.items():
            log_dict[name] = LoggerQuantity(
                name, new_cls, entry.flag, entry.default)
        new_cls._export_dict = log_dict
        cls._meta_export_dict = dict()

        # Add property to get all available loggable quantities
        if add_loggables:
            if hasattr(new_cls, 'loggables'):
                raise ValueError("classes of type Loggable cannot implement a "
                                 "loggables method, property, or attribute.")
            else:
                new_cls.loggables = property(_loggables)

def generate_namespace(cls):
    ns = tuple(cls.__module__.split('.') + [cls.__name__])
    if ns[0] == 'hoomd':
        return ns[1:]
    else:
        return ns


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
        self.namespace = generate_namespace(cls)
        return self


class Logger(SafeNamespaceDict):
    '''Logs HOOMD-blue operation data and custom quantities.'''

    def __init__(self, accepted_flags=None, only_default=True):
        if accepted_flags is None:
            accepted_flags = TypeFlags.ALL
        else:
            accepted_flags = TypeFlags.any(accepted_flags)
        self._flags = accepted_flags
        self._only_default = only_default
        super().__init__()

    def _grab_log_quantities_from_names(self, obj, quantities):
        if quantities is None:
            if self._only_default:
                return filter(lambda x: x.default, obj._export_dict.values())
            else:
                return obj._export_dict.values()

        else:
            log_quantities = []
            bad_keys = []
            for quantity in self.wrap_quantity(quantities):
                try:
                    log_quantities.append(obj._export_dict[quantity])
                except KeyError:
                    bad_keys.append(quantity)
            if bad_keys != []:
                raise KeyError("Log quantities {} do not exist for {} obj."
                               "".format(bad_keys, obj))
            return log_quantities

    def add(self, obj, quantities=None):
        used_namespaces = []
        for quantity in self._grab_log_quantities_from_names(obj, quantities):
            if quantity.flag in self._flags:
                used_namespaces.append(self._add_single_quantity(obj,
                                                                 quantity))
        return used_namespaces

    def remove(self, obj=None, quantities=None):
        if obj is None and quantities is None:
            return None

        if obj is None:
            for quantity in self.wrap_quantity(quantities):
                if quantity in self:
                    del self[quantity]
        else:
            for quantity in self._grab_log_quantities_from_names(obj,
                                                                 quantities):
                # Check all currently used namespaces for object's quantities
                for namespace in quantity.yield_names():
                    if namespace in self:
                        if self._contains_obj(namespace, obj):
                            del self[namespace]
                    else:
                        break

    def _add_single_quantity(self, obj, quantity):
        '''If quantity for obj is not logged add to first available namespace.
        '''
        for namespace in quantity.yield_names():
            if namespace in self:
                if self._contains_obj(namespace, obj):
                    return namespace
                else:
                    continue
            else:
                self[namespace] = (obj, quantity.name, quantity.flag)
                return namespace

    def __setitem__(self, namespace, value):
        # raise errors if necessary
        err_msg = "Expected either (callable, flag) or \
                   (obj, method/property, flag)."
        if not isinstance(value, Sequence):
            raise ValueError(err_msg)
        if len(value) == 2:
            if not callable(value[0]):
                raise ValueError(err_msg)
            if isinstance(value[1], str):
                flag = TypeFlags[value[1]]
            elif isinstance(value[1], TypeFlags):
                flag = value[1]
            else:
                raise ValueError(
                    "flag must be a string or hoomd.logging.TypeFlags object.")
            super().__setitem__(namespace, (value[0], '__call__', flag))
        elif len(value) == 3:
            if not isinstance(value[1], str):
                raise ValueError(err_msg)
            if isinstance(value[2], str):
                flag = TypeFlags[value[2]]
            elif isinstance(value[2], TypeFlags):
                flag = value[2]
            else:
                raise ValueError(
                    "flag must be a string or hoomd.logging.TypeFlags object.")
            super().__setitem__(namespace, (*value[:2], flag))
        else:
            raise ValueError(err_msg)

    def __iadd__(self, obj):
        self.add(obj)
        return self

    def __isub__(self, value):
        if isinstance(value, str) or isinstance(value, tuple):
            self.remove(quantities=value)
        elif hasattr(value, '__iter__'):
            for v in value:
                self.__isub__(v)
        else:
            self.remove(obj=value)
        return self

    def log(self):
        return dict_map(self._dict, self._log_conversion)

    def _log_conversion(self, obj_prop_tuple):
        obj, prop, flag = obj_prop_tuple
        value = getattr(obj, prop)
        if hasattr(value, '__call__'):
            value = value()
        if flag == 'dict' or flag == 'state':
            return value
        else:
            return (value, flag)

    def _contains_obj(self, namespace, obj):
        '''evaulates based on identity.'''
        val = self._unsafe_getitem(namespace)
        if isinstance(val, tuple):
            return val[0] is obj
        else:
            return False

    def wrap_quantity(self, quantity):
        if isinstance(quantity, str) or isinstance(quantity, tuple):
            return [quantity]
        else:
            return quantity
