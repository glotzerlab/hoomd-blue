from itertools import count
from copy import deepcopy
from hoomd.util import is_iterable, dict_fold, dict_map, SafeNamespaceDict


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
    def __init__(self, name, cls, flag='scalar'):
        self.name = name
        self.update_cls(cls)
        self.flag = flag

    def yield_names(self):
        yield self.namespace + (self.name,)
        for i in count(start=1, step=1):
            yield self.namespace[:-1] + \
                (self.namespace[-1] + '_' + str(i), self.name)

    def update_cls(self, cls):
        self.namespace = generate_namespace(cls)
        return self


class Logger(SafeNamespaceDict):
    '''Logs Hoomd Operation data and custom quantities.'''

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
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("Logger expects values of "
                             "(obj, method/property, flag)")
        super().__setitem__(namespace, value)

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
        if flag == 'dict':
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
