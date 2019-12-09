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


def dict_fold(dict_, func, init_value, use_keys=False):
    final_value = init_value
    for key, value in dict_.items():
        if isinstance(value, dict):
            final_value = dict_fold(value, func, final_value)
        else:
            if use_keys:
                final_value = func(key, final_value)
            else:
                final_value = func(value, final_value)
    return final_value


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


class SafeNamespaceDict:
    def __init__(self):
        self._dict = dict()

    def __len__(self):
        return dict_fold(self._dict, lambda x, incr: incr + 1, 0)

    def key_exists(self, namespace):
        try:
            namespace = self.validate_namespace(namespace)
        except ValueError:
            return False
        current_dict = self._dict
        # traverse through dictionary hierarchy
        for name in namespace:
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
        raise NotImplementedError

    def pop_namespace(self, namespace):
        return (namespace[-1], namespace[:-1])

    def _setitem(self, namespace, value):
        if namespace in self:
            raise KeyError("Namespace {} is being used. Remove before "
                           "replacing.".format(namespace))
        # Grab parent dictionary creating sub dictionaries as necessary
        parent_dict = self._dict
        base_name, parent_namespace = self.pop_namespace(namespace)
        for name in parent_namespace:
            # If key does not exist create key with empty dictionary
            try:
                parent_dict = parent_dict[name]
            except KeyError:
                parent_dict[name] = dict()
                parent_dict = parent_dict[name]
        # Attempt to set the value
        parent_dict[base_name] = value

    def __setitem__(self, namespace, value):
        try:
            namespace = self.validate_namespace(namespace)
        except ValueError:
            raise KeyError("Expected a tuple or string key.")
        self._setitem(namespace, value)

    def _unsafe_getitem(self, namespace):
        ret_val = self._dict
        if isinstance(namespace, str):
            namespace = (namespace,)
        for name in namespace:
            ret_val = ret_val[name]
        return ret_val

    def __delitem__(self, namespace):
        '''Does not check that key exists.'''
        if isinstance(namespace, str):
            namespace = (namespace,)
        parent_dict = self._unsafe_getitem(namespace[:-1])
        del parent_dict[namespace[-1]]

    def __contains__(self, namespace):
        return self.key_exists(namespace)

    def validate_namespace(self, namespace):
        if isinstance(namespace, str):
            namespace = (namespace,)
        if not isinstance(namespace, tuple):
            raise ValueError("Expected a string or tuple namespace.")
        return namespace


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
