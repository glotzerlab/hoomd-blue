from itertools import count
from copy import deepcopy
from hoomd.util import dict_map, SafeNamespaceDict
from collections.abc import Sequence


class Loggable(type):
    _meta_export_dict = dict()

    @classmethod
    def log(cls, func=None, is_property=True, flag='scalar'):
        def helper(func):
            name = func.__name__
            if name in cls._meta_export_dict:
                raise KeyError(
                    "Multiple loggable quantities named {}.".format(name))
            cls._meta_export_dict[name] = flag
            if is_property:
                return property(func)
            else:
                return func
        if func is None:
            return helper
        else:
            return helper(func)

    def __new__(cls, name, base, dct):
        new_cls = super().__new__(cls, name, base, dct)
        log_dict = dict()
        for name, flag in cls._meta_export_dict.items():
            log_dict[name] = LoggerQuantity(name, new_cls, flag)
        if hasattr(new_cls, '_export_dict'):
            old_dict = deepcopy(new_cls._export_dict)
            for key, value in old_dict.items():
                old_dict[key] = value.update_cls(new_cls)
            old_dict.update(log_dict)
            new_cls._export_dict = old_dict
        else:
            new_cls._export_dict = log_dict
        cls._meta_export_dict = dict()
        return new_cls


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
        flag (str, optional): The type of quantity it is. Valid values include
            scalar, multi (array-like), particle (per particle quantity), bond,
            angle, dihedral, constraint, pair, dict (a mapping of multiple
            logged quantity names with their values), and object.

    Note:
        For users, this class is meant to be used in conjunction with
        `hoomd.custom.Action` for exposing loggable quantities for custom user
        actions.
    """
    def __init__(self, name, cls, flag='scalar'):
        self.name = name
        self.update_cls(cls)
        self.flag = flag

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

    def __init__(self, accepted_flags=None):
        accepted_flags = [] if accepted_flags is None else accepted_flags
        self._flags = accepted_flags
        super().__init__()

    def _grab_log_quantities_from_names(self, obj, quantities):
        if quantities is None:
            return list(obj._export_dict.values())
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
            if self.flag_checks(quantity):
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
        if not all(isinstance(v, str) for v in value[1:]):
            raise ValueError("Method/property and flags must be strings.")

        # Check length for setting with either (obj, prop, flag) or (func, flag)
        elif len(value) == 3:
            super().__setitem__(namespace, value)
        elif len(value) == 2:
            if not callable(value[0]):
                raise ValueError(err_msg)
            else:
                super().__setitem__(namespace, (value[0], '__call__', value[1]))
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

    def flag_checks(self, log_quantity):
        if self._flags == []:
            return True
        else:
            return log_quantity.flag in self._flags

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
