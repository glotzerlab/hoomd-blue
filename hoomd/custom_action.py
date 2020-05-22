from abc import ABC, abstractmethod
from hoomd.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDGetSetAttrBase


class CustomAction(ABC):
    """Base class for all Python Action's.

    This class must be the parent class for all Python ``Action``s. This class
    requires all subclasses to implement the :meth:`~.act` method which performs the
    Python object's task whether that be updating the system, writing output, or
    analyzing some property of the system.

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
    with the name of the loggable and `hooomd.logger.LoggerQuantity` objects as
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
            the state when called.
        """
        pass


class _InternalCustomAction(CustomAction, _HOOMDGetSetAttrBase):
    """An internal class for Python ``Action``s.

    Gives additional support in using HOOMD constructs like ``ParameterDict``s
    and ``TypeParameters``.
    """
    pass
