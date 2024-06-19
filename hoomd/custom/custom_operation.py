# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomOperation."""

from abc import abstractmethod
import functools
import itertools

from hoomd.data.parameterdicts import ParameterDict
from hoomd.custom.custom_action import Action, _AbstractLoggable
from hoomd.operation import TriggeredOperation
from hoomd.trigger import Trigger
from hoomd import _hoomd


class CustomOperation(TriggeredOperation, metaclass=_AbstractLoggable):
    """User defined operation.

    This is the parent class for `hoomd.tune.CustomTuner`,
    `hoomd.update.CustomUpdater`. and `hoomd.write.CustomWriter`.  These
    classes wrap Python objects that inherit from `hoomd.custom.Action`
    so they can be added to the simulation operations.

    This class also implements a "pass-through" system for attributes.
    Attributes and methods from the passed in `action` will be available
    directly in this class. This does not apply to attributes with these names:
    ``trigger``, ``_action``, and ``action``.

    Note:
        Due to the pass through no attribute should exist both in
        `hoomd.custom.CustomOperation` and the `hoomd.custom.Action`.

    Note:
        This object should not be instantiated or subclassed by an user.

    Attributes:
        trigger (hoomd.trigger.Trigger): A trigger to determine when the
            wrapped `hoomd.custom.Action` is run.
    """

    _override_setattr = {'_action', "_export_dict", "_simulation"}

    @abstractmethod
    def _cpp_class_name(self):
        """C++ Class to use for attaching."""
        raise NotImplementedError

    def __init__(self, trigger, action):
        if not isinstance(action, Action):
            raise ValueError("action must be a subclass of "
                             "hoomd.custom_action.custom.Action.")
        self._action = action
        self._export_dict = action._export_dict

        param_dict = ParameterDict(trigger=Trigger)
        param_dict['trigger'] = trigger
        self._param_dict.update(param_dict)

    def __getattr__(self, attr):
        """Pass through attributes/methods of the wrapped object."""
        try:
            return super().__getattr__(attr)
        except AttributeError:
            try:
                return getattr(self._action, attr)
            except AttributeError:
                raise AttributeError("{} object has no attribute {}".format(
                    type(self), attr))

    def _setattr_hook(self, attr, value):
        """This implements the __setattr__ pass through to the Action."""
        if attr not in self.__dict__ and hasattr(self._action, attr):
            setattr(self._action, attr, value)
            return
        object.__setattr__(self, attr, value)

    def _attach_hook(self):
        """Create the C++ custom operation."""
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            self._simulation.state._cpp_sys_def, self.trigger, self._action)
        self._action.attach(self._simulation)

    def _detach_hook(self):
        """Detaching from a `hoomd.Simulation`."""
        self._action.detach()

    def act(self, timestep):
        """Perform the action of the custom action if attached.

        Calls through to the action property of the instance.

        Args:
            timestep (int): The current timestep of the state.
        """
        if self._attached:
            self._action.act(timestep)

    @property
    def action(self):
        """`hoomd.custom.Action` The action the operation wraps."""
        return self._action

    def __setstate__(self, state):
        """Set object state from pickling or deepcopying."""
        self._action = state.pop("_action")
        for attr, value in state.items():
            setattr(self, attr, value)


class _AbstractLoggableWithPassthrough(_AbstractLoggable):
    """Enhances wrapping of an internal action class for custom operations.

    Attributes:
        _internal_cls (type): The action class to wrap.
        _wrap_methods (list[str]): A list of ``_internal_cls`` methods to
            actively wrap. Note all loggables are automatically wrapped.

    Extra Features:
    * Wrap loggable properties/methods to allow for sphinx documentation.
    * Pass through non-wrapped internal class attributes and methods to custom
      wrapping class.

    Note:
        Sphinx can only document wrapped methods/properties.
    """

    def __init__(cls, name, base, dct):  # noqa: N805
        """Wrap extant internal class loggables for documentation."""
        action_cls = dct.get("_internal_class", None)
        if action_cls is None or isinstance(action_cls, property):
            return
        extra_methods = dct.get("_wrap_methods", [])
        for name in itertools.chain(action_cls._export_dict, extra_methods):
            wrapped_method = _AbstractLoggableWithPassthrough._wrap_loggable(
                name, getattr(action_cls, name))
            setattr(cls, name, wrapped_method)
        cls._export_dict = action_cls._export_dict
        _AbstractLoggable.__init__(cls, name, base, dct)

    @staticmethod
    def _wrap_loggable(name, mthd):
        if isinstance(mthd, property):

            @property
            def getter(self):
                return getattr(self._action, name)

            if mthd.fset is not None:

                @getter.setter
                def setter(self, new_value):
                    setattr(self._action, name, new_value)

            getter.__doc__ = mthd.__doc__
            return getter

        @functools.wraps(mthd)
        def func(self, *args, **kwargs):
            return getattr(self._action, name)(*args, **kwargs)

        return func

    def __getattr__(self, attr):
        """Treat class attributes/methods of inner class as from this class."""
        try:
            # This will not work with classmethods that are constructors. We
            # need a trigger for operations, and the action does not contain a
            # trigger. This can be made to work for alternate constructors but
            # would require wrapping the classmethod in question. Since this
            # should only ever matter for internal actions, putting such
            # classmethods in the wrapping operation should be fine.
            return getattr(self._internal_class, attr)
        except AttributeError:
            raise AttributeError("{} object {} has no attribute {}".format(
                type(self), self, attr))


class _InternalCustomOperation(CustomOperation,
                               metaclass=_AbstractLoggableWithPassthrough):
    """Internal class for Python `Action`s. Offers a streamlined ``__init__``.

    Adds a wrapper around an hoomd Python action. This extends the attribute
    getting and setting wrapper of `hoomd.CustomOperation` with a wrapping of
    the ``__init__`` method as well as a error raised if the ``action`` is
    attempted to be accessed directly.
    """

    # These attributes are not accessible or able to be passed through to
    # prevent leaky abstractions and help promote the illusion of a single
    # object for cases of internal custom actions.
    _disallowed_attrs = {'detach', 'attach', 'action', "act"}

    def __getattribute__(self, attr):
        if attr in object.__getattribute__(self, "_disallowed_attrs"):
            raise AttributeError("{} object {} has no attribute {}.".format(
                type(self), self, attr))
        return object.__getattribute__(self, attr)

    def __getattr__(self, attr):
        if attr in self._disallowed_attrs:
            raise AttributeError("{} object {} has no attribute {}.".format(
                type(self), self, attr))
        return super().__getattr__(attr)

    @property
    @abstractmethod
    def _internal_class(self):
        """Internal class to use for the Action of the Operation."""
        pass

    def __init__(self, trigger, *args, **kwargs):
        super().__init__(trigger, self._internal_class(*args, **kwargs))
        # handle pass through logging
        self._export_dict = {
            key: value.update_cls(self.__class__)
            for key, value in self._export_dict.items()
        }
        # Wrap action act method with operation appropriate one.
        wrapping_method = getattr(self, self._operation_func).__func__
        setattr(wrapping_method, "__doc__", self._action.act.__doc__)

    def __dir__(self):
        """Expose all attributes for dynamic querying in notebooks and IDEs."""
        list_ = super().__dir__()
        act = self._action
        action_list = [
            k for k in itertools.chain(act._param_dict, act._typeparam_dict)
        ]
        list_.remove("action")
        list_.remove("act")
        return list_ + action_list
