from itertools import count
from copy import deepcopy
from hoomd.util import is_iterable


class Loggable(type):
    _meta_export_dict = dict()

    @classmethod
    def log(cls, func=None, is_property=True, flag='scalar'):
        def helper(func):
            name = func.__name__
            if name in cls._meta_export_dict.keys():
                raise KeyError("Multiple loggable quantities named "
                            "{}.".format(name))
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
        namespace = generate_namespace(new_cls)
        log_dict = dict()
        for name, flag in cls._meta_export_dict.items():
            log_dict[name] = LoggerQuantity(name, namespace, flag)
        if hasattr(new_cls, '_export_dict'):
            old_dict = deepcopy(new_cls._export_dict)
            old_dict.update(log_dict)
            new_cls._export_dict = old_dict
        else:
            new_cls._export_dict = log_dict
        cls._meta_export_dict = dict()
        return new_cls


def dict_map(dict_, func):
    new_dict = dict()
    for key, value in dict_.items():
        if isinstance(value, dict):
            new_dict[key] = dict_map(value, func)
        else:
            new_dict[key] = func(value)
    return new_dict


def generate_namespace(cls):
    return tuple(cls.__module__.split('.') + [cls.__name__])


class LoggerQuantity:
    def __init__(self, name, namespace, flag='scalar'):
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        self.name = name
        if not isinstance(namespace, tuple):
            raise ValueError("Namespace must be an ordered tuple of "
                             "namespaces.")
        self.namespace = namespace
        self.flag = flag

    def yield_names(self):
        yield self.namespace + (self.name,)
        for i in count(start=1, step=1):
            yield self.namespace[:-1] + \
                (self.namespace[-1] + '_' + str(i), self.name)


class NamespaceDict:
    def __init__(self):
        self._dict = dict()

    def key_exists(self, namespace):
        if isinstance(namespace, str):
            namespace = (namespace,)
        elif not is_iterable(namespace) or len(namespace) == 0:
            return False
        current_dict = self._dict
        current_namespace = []
        for name in namespace:
            current_namespace.append(name)
            try:
                if name in current_dict.keys():
                    current_dict = current_dict[name]
                    continue
                else:
                    return False
            except (TypeError, AttributeError):
                return False
        return True

    def keys(self):
        pass

    def _setitem(self, namespace, value):
        if self.key_exists(namespace):
            raise KeyError("Namespace {} is being used. Remove before "
                           "replacing.".format(namespace))
        # Grab parent dictionary creating sub dictionaries as necessary
        parent_dict = self._dict
        current_namespace = []
        for name in namespace[:-1]:
            current_namespace.append(name)
            # If key does not exist create key with empty dictionary
            try:
                parent_dict = parent_dict[name]
            except KeyError:
                parent_dict[name] = dict()
                parent_dict = parent_dict[name]
        # Attempt to set the value
        parent_dict[namespace[-1]] = value

    def __setitem__(self, namespace, value):
        if isinstance(namespace, tuple):
            if isinstance(namespace, str):
                namespace = (namespace,)
            else:
                namespace = tuple(namespace)
        self._setitem(namespace, value)


class Logger(NamespaceDict):
    '''Logs Hoomd Operation data and custom quantities.'''

    def _grab_log_quantities_from_names(self, obj, quantities):
        if quantities is None:
            return list(obj._export_dict.values())
        else:
            log_quantities = []
            bad_keys = []
            for quantity in quantities:
                try:
                    log_quantity.append(obj._export_dict[quantity])
                except KeyError:
                    bad_keys.append(quantity)
            if bad_keys != []:
                raise KeyError("Log quantities {} do not exist for {} obj."
                               "".format(bad_keys, obj))
            return log_quantity

    def add(self, obj, quantities=None):
        for quantity in self._grab_log_quantities_from_names(quantities):
            self._add_single_quantity(quantity, obj)

    def remove(self, obj=None, quantities=None):
        if obj is None and quantities is None:
            return None

        if obj is None:
            for quantity in quantities:
                if self.key_exists(quantity):
                    self._remove_quantity(quantity)
        else:
            for quantity in self._grab_log_quantities_from_names(obj,
                                                                 quantities):
                # Check all currently used namespaces for object's quantities
                for namespace in quantity.yield_names():
                    if self.key_exists(namespace):
                        parent_dict = self._unsafe_getitem(namespace[:-1])
                        try:
                            # If namespace contains object remove
                            if parent_dict[namespace[-1]][0] is obj:
                                del parent_dict[namespace[1]]
                                continue
                        except TypeError:
                            continue
                    else:
                        break

    def _add_single_quanity(self, quantity, obj):
        for namespace in quantity.yield_names():
            if self.key_exists(namespace):
                quantity_val = self._unsafe_getitem(namespace)
                try:
                    if quantity_val[0] is obj:
                        return None
                except TypeError:
                    raise RuntimeError("Logger is in an undefined state. "
                                       "In namespace {} is an improper value."
                                       "".format(namespace))
                continue
            else:
                self._setitem(namespace, (obj, quantity.name))

    def __setitem__(self, namespace, value):
        if not isinstance(value, tuple):
            raise ValueError("Logger expects values of (obj, method/property)")
        super().__setitem__(namespace, value)

    def _unsafe_getitem(self, namespace):
        ret_val = self._dict
        for name in namespace:
            ret_val = ret_val[name]
        return ret_val

    def __iadd__(self, obj):
        self.add(obj)

    def __isub__(self, value):
        if isinstance(value, str):
            self.remove(quantities=[value])
        elif hasattr(value, '__iter__'):
            self.remove(quantities=value)
        else:
            self.remove(obj=value)

    def log(self):
        return dict_map(self._dict, self._log_conversion)

    def _log_conversion(obj_prop_tuple):
        attr = getattr(obj_prop_tuple[0], obj_prop_tuple[1])
        if hasattr(attr, '__call__'):
            return attr()
        else:
            return attr
