from abc import ABC, abstractmethod
from hoomd.operation import _HOOMDGetSetAttrBase


class CustomAction(ABC):
    """Base class for all Python Action's.

    This class is the parent class for all Python ``Action`` subclasses. This
    class requires all subclasses to implement the :meth:`~.act` method which
    performs the Python object's task whether that be updating the system,
    writing output, or analyzing some property of the system.

    To use subclasses of this class, the object must be passed as an argument
    to a `hoomd.update.CustomUpdater` or `hoomd.analyze.CustomAnalyzer`
    constructor.

    If the pressure, rotational kinetic energy, or external field virial is
    needed for a subclass, the flags attribute of the class needs to be set with
    the appropriate flags from `hoomd.util.ParticleDataFlags`.

    .. code-block:: python

        from hoomd.python_action import CustomAction
        from hoomd.util import ParticleDataFlags


        class ExampleActionWithFlag(CustomAction):
            flags = [ParticleDataFlags.ROTATIONAL_KINETIC_ENERGY,
                     ParticleDataFlags.PRESSURE_TENSOR,
                     ParticleDataFlags.EXTERNAL_FIELD_VIRIAL]

            def act(self, timestep):
                pass

    For advertising loggable quantities through the wrappping object, the class
    attribute ``log_quantities`` can be used. The dictionary expects string keys
    with the name of the loggable and `hoomd.logger.LoggerQuantity` objects as
    the values.

    .. code-block:: python

        from hoomd.python_action import CustomAction
        from hoomd.logger import LoggerQuantity


        class ExampleActionWithFlag(CustomAction):
            def __init__(self):
                self.log_quantities = {
                    'loggable': LoggerQuantity('scalar_loggable',
                                               self.__class__,
                                               flag='scalar')}

            def loggable(self):
                return 42

            def act(self, timestep):
                pass

    An example of a ``CustomAction`` that actually performs an action is given
    below. This ``CustomAction`` computes the center of mass of the system.

    .. code-block:: python

        from hoomd.python_action import CustomAction


        class ExampleAction(CustomAction):
            def act(self, timestep):
                self.com = self.snapshot.particles.position.mean(axis=0)


    Attributes:
        flags (list[hoomd.util.ParticleDataFlags]): List of flags from the
            `hoomd.util.ParticleDataFlags`. Used to tell the integrator if
            specific quantities are needed for the action.
        log_quantities (dict[str, hoomd.logger.LoggerQuantity]): Dictionary of
            the name of loggable quantites to the `hoomd.logger.LoggerQuantity`
            instance for the class method or property. Allows for subclasses of
            `CustomAction` to specify to a `hoomd.Logger` that is exposes
            loggable quantities.
    """
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


class _InternalCustomAction(CustomAction, _HOOMDGetSetAttrBase):
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
    pass
