from copy import deepcopy
from enum import Flag, auto
from itertools import count
from functools import reduce
from hoomd.util import dict_map, SafeNamespaceDict
from collections.abc import Sequence


class LoggerCategories(Flag):
    """Enum that marks all accepted logger types.

    This class does not need to be used by users directly. We directly convert
    from strings to the enum wherever necessary in the API. This class is
    documented to show users what types of quantities can be logged, and what
    categories to use for limiting what data is logged, user specified logged
    quantities, and custom actions (`hoomd.custom.Action`).

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

        state: internal category for specifying object's internal state

        ALL: a combination of all other categories

        NONE: represents no category
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
    def any(cls, categories=None):
        """Return a LoggerCategories enum representing any of the given categories.

        Args:
            categories (list[str] or list[`LoggerCategories`]): A list of `str` or
            `LoggerCategories` objects that should be represented by the returned
            `LoggerCategories` object.

        Returns:
            `LoggerCategories`: the `LoggerCategories` object that represents any of the given
            categories.
        """
        categories = cls.__members__.values() if categories is None else categories

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
        return [mem.name for mem in cls.__members__.values() if mem in category]


LoggerCategories.ALL = LoggerCategories.any()


# function defined here to ensure that each class of type Loggable will have a
# loggables property
def _loggables(self):
    """dict[str, str]: Return a name, category mapping of loggable quantities."""
    return {name: quantity.category.name
            for name, quantity in self._export_dict.items()}


class _LoggableEntry:
    """Stores entries for _Loggable's store of a class's loggable quantities."""
    def __init__(self, category, default):
        self.category = category
        self.default = default


class _NamespaceFilter:
    """Filter for creating the proper namespace for logging object properties.

    Attributes:
        remove_names (set[str]): A set of names which to remove for the logging
            namespace whenever encountered.
        base_names (set[str]): A set of names which indicate that the next
            encountered name in the string should be skipped. For example, if a
            module hierarchy went like ``project.foo.bar.Bar`` and ``foo``
            directly imports ``Bar``, ``bar`` may not be desirable to have in
            the logging namespace since users interact with it via ``foo.Bar``.
            Currently, this only handles a single level of nesting like this.
        skip_duplicates (bool, optional): Whether or not to remove consecutive
            duplicates from a logging namespace (e.g. ``foo.foo.bar`` ->
            ``foo.bar``), default ``True``. By default we assume that this
            pattern means that the inner module is imported into its parent.
    """

    def __init__(self,
                 remove_names=None,
                 base_names=None,
                 skip_duplicates=True):
        self.remove_names = set() if remove_names is None else remove_names
        self.base_names = set() if base_names is None else base_names
        self._skip_next = False
        self.skip_duplicates = skip_duplicates
        if skip_duplicates:
            self._last_name = None

    def __call__(self, namespace):
        for name in namespace:
            # check for duplicates in the namespace and remove them (e.g.
            # `md.pair.pair.LJ` -> `md.pair.LJ`).
            if self.skip_duplicates:
                last_name = self._last_name
                self._last_name = name
                if last_name == name:
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


class _LoggerQuantity:
    """The information to automatically log to a `hoomd.logging.Logger`.

    Args:
        name (str): The name of the quantity.
        cls (``class object``): The class that the quantity comes from.
        category (str or LoggerCategories, optional): The type of quantity it is.
            Valid values are given in the `hoomd.logging.LoggerCategories`
            documentation.

    Note:
        For users, this class is meant to be used in conjunction with
        `hoomd.custom.Action` for exposing loggable quantities for custom user
        actions.
    """

    namespace_filter = _NamespaceFilter(
        # Names that are imported directly into the hoomd namespace
        remove_names={'simulation', 'state', 'operations', 'snapshot'},
        # Names that have their submodules' classes directly imported into them
        # (e.g. `hoomd.update.box_resize.BoxResize` gets used as
        # `hoomd.update.BoxResize`)
        base_names={'update', 'tune', 'write'},
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

        Used to ensure that all namespaces are unique for a
        `hoomd.logging.Logger` object. We simple increment a number at the end
        until the caller stops asking for another namespace.

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
        name, if inheritanting `hoomd.logging._LoggerQuantity`, the class needs
        to be updated to the subclass.

        Args:
            cls (``class object``): The class to update the namespace with.
        """
        self.namespace = self._generate_namespace(cls)
        return self

    @classmethod
    def _generate_namespace(cls, loggable_cls):
        """Generate the namespace of a class given its module hierarchy."""
        ns = tuple(loggable_cls.__module__.split('.'))
        cls_name = loggable_cls.__name__
        # Only filter namespaces of objects in the hoomd package
        if ns[0] == 'hoomd':
            return tuple(cls.namespace_filter(ns[1:])) + (cls_name,)
        else:
            return ns + (cls_name,)


class Loggable(type):
    _meta_export_dict = dict()

    def __init__(cls, name, bases, dct):
        """Adds marked quantities for logging in new class.

        Also adds a loggables property that returns a mapping of loggable
        quantity names with the string category. We overwrite __init__ instead of
        __new__ since this plays much more nicely with inheritance. This allows,
        for instance, `Loggable` to be subclassed with metaclasses that use
        __new__ without having to hack the subclass's behavior.
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
        if log_dict == {} and not any(issubclass(type(c), Loggable)
                                      for c in cls.__mro__[1:]):
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
                inherited_loggables.update(
                    {name: deepcopy(quantity).update_cls(new_cls)
                     for name, quantity in base_cls._export_dict.items()})
        return inherited_loggables

    @classmethod
    def _get_current_cls_loggables(cls, new_cls):
        """Gets the current class's new loggables (not inherited)."""
        current_loggables = {}
        for name, entry in cls._meta_export_dict.items():
            current_loggables[name] = _LoggerQuantity(
                name, new_cls, entry.category, entry.default)
            cls._add_loggable_docstring_info(
                new_cls, name, entry.category, entry.default)
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


def log(func=None, *, is_property=True, category='scalar', default=True):
    """Creates loggable quantities for classes of type Loggable.

    For users this should be used with `hoomd.custom.Action` for exposing
    loggable quantities from a custom action.

    Args:
        func (`method`): class method to make loggable. If using non-default
            arguments, func should not be set.
        is_property (:obj:`bool`, optional): Whether to make the method a
            property, defaults to True. Argument keyword only
        category (:obj:`str`, optional): The string represention of the type of
            loggable quantity, defaults to 'scalar'. See
            `hoomd.logging.LoggerCategories` for available types. Argument
            keyword only
        default (:obj:`bool`, optional): Whether the quantity should be logged
            by default, defaults to True. This is orthogonal to the loggable
            quantity's type. An example would be performance orientated
            loggable quantities.  Many users may not want to log such
            quantities even when logging other quantities of that type. The
            default category allows for these to be pass over by
            `hoomd.logging.Logger` objects by default. Argument keyword only.

    Note:
        The namespace (where the loggable object is stored in the
        `hoomd.logging.Logger` object's nested dictionary, is determined by
        the module/script and class name the loggable class comes from. In
        creating subclasses of `hoomd.custom.Action`, for instance, if the
        module the subclass is defined in is ``user.custom.action`` and the
        class name is ``Foo`` then the namespace used will be ``('user',
        'custom', 'action', 'Foo')``. This helps to prevent naming conflicts,
        and automate the logging specification for developers and users.
    """

    def helper(func):
        name = func.__name__
        if name in Loggable._meta_export_dict:
            raise KeyError(
                "Multiple loggable quantities named {}.".format(name))
        Loggable._meta_export_dict[name] = _LoggableEntry(
            LoggerCategories[category], default)
        if is_property:
            return property(func)
        else:
            return func

    if func is None:
        return helper
    else:
        return helper(func)


class _LoggerEntry:
    """Stores the information for an entry in a `hoomd.logging.Logger`.

    The class deals with the logic of converting `tuple` and
    `hoomd.logging._LoggerQuantity` objects into an object that can obtain the
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
        if (not isinstance(entry, Sequence)
                or len(entry) <= 1
                or len(entry) > 3):
            raise ValueError(err_msg)

        # Get the method and category from the passed entry. Also perform some basic
        # validation.
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
                "category must be a string or hoomd.logging.LoggerCategories object.")
        return cls(entry[0], method, category)

    def __call__(self):
        attr = getattr(self.obj, self.attr)
        if self.category is LoggerCategories.state:
            return attr
        if callable(attr):
            return (attr(), self.category.name)
        else:
            return (attr, self.category.name)

    def __eq__(self, other):
        return (self.obj == other.obj and
                self.attr == other.attr and
                self.category == other.category)
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in ['obj', 'attr', 'category'])


class Logger(SafeNamespaceDict):
    '''Logs HOOMD-blue operation data and custom quantities.

    The `Logger` class provides an intermediary between a back end such as the
    `hoomd.write.Table` and many of HOOMD-blue's object (as most objects are
    loggable). The `Logger` class makes use of *namespaces* which denote where a
    logged quantity fits in. For example internally all loggable quantities are
    ordered by the module and class them come from. For instance, the
    `hoomd.md.pair.LJ` class has a namespace ``('md', 'pair', 'LJ')``. This
    applies to all loggable internal objects in HOOMD-blue. This ensures that
    logged quantities remain unambigious. To add a loggable object's quantities
    two methods exist `Logger.add` and the ``+=`` operator. Here we show an
    example using the ``+=`` operator.

    Example:
        .. code-block:: python

            logger = hoomd.logging.Logger()
            lj = md.pair.lj(nlist)
            # Log all default quantities of the lj object
            logger += lj
            logger = hoomd.logging.Logger(categories=['scalar'])
            # Log all default scalar quantities of the lj object
            logger += lj

    The `Logger` class also supports user specified quantities using namespaces
    as well.

    Example:
        .. code-block:: python

            logger = hoomd.logging.Logger()
            # Add quantity to ('custom', 'name') namespace
            logger[('custom', 'name')] = (lambda: 42, 'scalar')
            # Add quantity to ('custom_name',) namespace
            logger[('custom_name',)] = (lambda: 43, 'scalar')

    `Logger` objects support two ways of discriminating what loggable quantities
    they will accept: ``categories`` and ``only_default`` (the constructor
    arguments). Both of these are static meaning that once instantiated a
    `Logger` object will not change the values of these two properties.
    ``categories`` determines what if any types of loggable quantities (see
    `hoomd.logging.LoggerCategories`) are appropriate for a given `Logger`
    object. This helps logging back ends determine if a `Logger` object is
    compatible. The ``only_default`` flag is mainly a convenience by allowing
    quantities not commonly logged (but available) to be passed over unless
    explicitly asked for. You can override the ``only_default`` flag by
    explicitly listing the quantities you want in `Logger.add`, but the same is
    not true with regards to ``categories``.

    Note:
        The logger provides a way for users to create their own logger back ends
        if they wish. In making a custom logger back end, understanding the
        intermediate representation is key. To get an introduction see
        `hoomd.logging.Logger.log`. To understand the various categories
        available to specify logged quantities, see
        `hoomd.logging.LoggerCategories`.  To integrate with `hoomd.Operations`
        the back end should be a subclass of `hoomd.custom.Action` and used with
        `hoomd.writer.CustomWriter`.

    Note:
        When logging multiple instances of the same class `Logger.add` provides
        a means of specifying the class level of the namespace (e.g. ``'LJ`` in
        ``('md', 'pair', 'LJ')``). The default behavior (without specifying a
        user name) is to just append ``_{num}`` where ``num`` is the smallest
        positive integer which makes the full namespace unique. This appending
        will also occur for user specified names that are reused.

    Args:
        categories (`list` of `str`, optional): A list of string categories
            (list of categories can be found in `hoomd.logging.LoggerCategories`).
            These are the only types of loggable quantities that can be logged
            by this logger. Defaults to allowing every type.
        only_default (`bool`, optional): Whether to log only quantities that are
            logged by "default", defaults to ``True``. This mostly means that
            performance centric loggable quantities will be passed over when
            logging when false.
    '''

    def __init__(self, categories=None, only_default=True):
        self._categories = LoggerCategories.ALL if categories is None else LoggerCategories.any(categories)
        self._only_default = only_default
        super().__init__()

    @property
    def categories(self):
        """`hoomd.logging.LoggerCategories`: The enum representing the
        acceptable categories for the `Logger` object.
        """
        return self._categories

    @property
    def string_categories(self):
        """`list` of `str`: A list of the string names of the allowed categories
        for logging.
        """
        return LoggerCategories._get_string_list(self._categories)

    @property
    def only_default(self):
        """`bool`: Whether the logger object should only grab default loggable
        quantities.
        """
        return self._only_default

    def _filter_quantities(self, quantities):
        for quantity in quantities:
            if self._only_default and not quantity.default:
                continue
            elif quantity.category in self._categories:
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
                map(lambda q: obj._export_dict[q], quantities))

    def add(self, obj, quantities=None, user_name=None):
        """Add loggables from obj to logger.

        Args:
            obj (object of class of type ``Loggable``): class of type loggable
                to add loggable quantities from.
            quantities (Sequence[str]): list of str names of quantities to log.
            user_name (`str`, optional): A string to replace the class name in
                the loggable quantities namespace. This allows for easier
                differentiation in the output of the `Logger` and any `Writer`
                which outputs its data.

        Returns:
            list[tuple[str]]: A list of namespaces that were
                added to the logger.
        """
        for quantity in self._get_loggables_by_name(obj, quantities):
            self._add_single_quantity(obj, quantity, user_name)

    def remove(self, obj=None, quantities=None, user_name=None):
        """Remove specified quantities from the logger.

        Args:
            obj (object of class of type ``Loggable``, optional):
                Object to remove quantities from. If ``quantities`` is None,
                ``obj`` must be set. If ``obj`` is set and ``quantities`` is
                None, all logged quanties from ``obj`` will be removed from the
                logger.
            quantities (Sequence[tuple]): a sequence of namespaces to remove
                from the logger. If specified with ``obj`` only remove
                quantities listed that are exposed from ``obj``. If ``obj`` is
                None, then ``quantities`` must be given.
            user_name (str): A user name to specify the final entry in the
                namespace of the object. This must be used in ``user_name`` was
                specified in `Logger.add`.
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
        '''If quantity for obj is not logged add to first available namespace.
        '''
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
            value (tuple[Callable, str] or tuple[object, str, str]):
                Either a tuple with a callable and the `hoomd.logging.LoggerCategories`
                object or associated string or a object with a method/property
                name and category. If using a method it should not take
                arguments or have defaults for all arguments.
        """
        if isinstance(value, _LoggerEntry):
            super().__setitem__(namespace, value)
        else:
            super().__setitem__(namespace, _LoggerEntry.from_tuple(value))

    def __iadd__(self, obj):
        """Add quantities from object or list of objects to logger.

        Adds all quantities compatible with given categories and default value.

        Examples:
            .. code-block:: python

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
            .. code-block:: python

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

        The nested dictionary consist of one level for each element of a
        namespace. The logged value and category for the namespace ``('example',
        'namespace')`` would be accessible in the returned dictionary via
        ``logger.log()['example']['namespace']``.

        Returns:
            dict: A nested dictionary of the current logged quantities. The end
                values are (value, category) pairs which hold the value along
                with its associated `hoomd.logging.LoggerCategories` category
                represented as a string (to get the
                `hoomd.logging.LoggerCategories` enum value use
                ``LoggerCategories[category]``.
        """
        return dict_map(self._dict, lambda x: x())

    def _contains_obj(self, namespace, obj):
        '''Evaluates based on identity.'''
        return self._unsafe_getitem(namespace).obj is obj

    @staticmethod
    def _wrap_quantity(quantity):
        """Handles wrapping strings and tuples for iterating over namespaces."""
        if isinstance(quantity, (str, tuple)):
            return [quantity]
        else:
            return quantity
