from abc import ABCMeta, abstractmethod
from enum import IntEnum
from hoomd.logging import Loggable
from hoomd.operation import _HOOMDGetSetAttrBase


class _AbstractLoggable(Loggable, ABCMeta):
    """Allows the use of abstractmethod with log."""
    def __init__(cls, name, base, dct):
        Loggable.__init__(cls, name, base, dct)
        ABCMeta.__init__(cls, name, base, dct)


class Action(metaclass=_AbstractLoggable):
    """Base class for all Python Action's.

    This class is the parent class for all Python `Action` subclasses. This
    class requires all subclasses to implement the :meth:`~.act` method which
    performs the Python object's task whether that be updating the system,
    writing output, or analyzing some property of the system.

    To use subclasses of this class, the object must be passed as an argument
    to a `hoomd.update.CustomUpdater` or `hoomd.analyze.CustomAnalyzer`
    constructor.

    If the pressure, rotational kinetic energy, or external field virial is
    needed for a subclass, the flags attribute of the class needs to be set with
    the appropriate flags from the internal `Action.Flags` enumeration.

    .. code-block:: python

        from hoomd.custom import Action


        class ExampleActionWithFlag(Action):
            flags = [Action.Flags.ROTATIONAL_KINETIC_ENERGY,
                     Action.Flags.PRESSURE_TENSOR,
                     Action.Flags.EXTERNAL_FIELD_VIRIAL]

            def act(self, timestep):
                pass

    For advertising loggable quantities through the wrapping object, the
    decorator `hoomd.logging.log` can be used.

    .. code-block:: python

        from hoomd.python_action import Action
        from hoomd.logging import log


        class ExampleActionWithFlag(Action):

            @log
            def answer(self):
                return 42

            def act(self, timestep):
                pass

    An example of a `Action` that actually performs an action is given
    below. This `Action` computes the center of mass of the system.

    .. code-block:: python

        from hoomd.custom import Action


        class ExampleAction(Action):
            def act(self, timestep):
                self.com = self.snapshot.particles.position.mean(axis=0)


    Attributes:
        flags (list[hoomd.custom.Action.Flags]): List of flags from the
            `hoomd.custom.Action.Flags`. Used to tell the integrator if
            specific quantities are needed for the action.
    """
    class Flags(IntEnum):
        """Flags to indictate the integrator should calcuate certain quantities.

        * PRESSURE_TENSOR = 0
        * ROTATIONAL_KINETIC_ENERGY = 1
        * EXTERNAL_FIELD_VIRIAL = 2
        """
        PRESSURE_TENSOR = 0
        ROTATIONAL_KINETIC_ENERGY = 1
        EXTERNAL_FIELD_VIRIAL = 2

    flags = []
    log_quantities = {}

    def __init__(self):
        pass

    def attach(self, simulation):
        """Attaches the Action to the `hoomd.Simulation`.

        Args:
            simulation (hoomd.Simulation): The simulation to attach the action
                to.
        """
        self._state = simulation.state

    @property
    def is_attached(self):
        return getattr(self, '_state', None) is not None

    def detach(self):
        """Detaches the Action from the `hoomd.Simulation`."""
        if hasattr(self, '_state'):
            del self._state

    @abstractmethod
    def act(self, timestep):
        """Performs whatever action a subclass implements.

        This method can change the state (updater) or compute or store data
        (analyzer).

        Args:
            timestep (int): The current timestep in a simulation.

        Note:
            A `hoomd.State` is not given here. This means that if the default
            `attach` method is overwritten, there is no way to query or change
            the state when called. By default, the state is accessible through
            ``self.state`` after attaching.
        """
        pass


class _InternalAction(Action, _HOOMDGetSetAttrBase):
    """An internal class for Python ``Action``s.

    Gives additional support in using HOOMD constructs like ``ParameterDict``s
    and ``TypeParameters``.

    When wrapped around a subclass of `hoomd._InternalCustomOperation`, the
    operation acts like the action (i.e. we mock the behavior of this object
    with the wrapping object). That means we can use ``op.a = 3`` rather than
    ``op.action.a = 3``. In addition, when creating Python Actions, all logic
    should go in these classes. In general, in creating a subclass of
    `hoomd.custom_operation._InternalCustomOperation` only a ``_internal_class``
    should be specified in the subclass. No other methods or attributes should
    be created.
    """
    def _setattr_param(self, attr, value):
        """Necessary to prevent errors on setting after attaching.

        See hoomd/operation.py _Operation._setattr_param for details.
        """
        self._param_dict[attr] = value
