from abc import ABC, abstractmethod
from hoomd.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDGetSetAttrBase


class _CustomAction(ABC):
    """Base class for all Python ``Action``s.

    This class must be the parent class for all Python ``Action``s. This class
    requires all subclasses to implement the act method which performs the
    Python object's task whether that be updating the system, writing output, or
    analyzing some property of the system.

    To use subclasses of this class, the object must be passed as an argument
    for the `hoomd.python_action._CustomOperation` constructor.

    If the pressure, rotational kinetic energy, or external field virial is
    needed for a subclass, the flags attribute of the class needs to be set with
    the appropriate flags from `hoomd.util.ParticleDataFlags`.

    .. code-block:: python

        from hoomd.python_action import _CustomAction
        from hoomd.util import ParticleDataFlags


        class ExampleActionWithFlag(_CustomAction):
            flags = [ParticleDataFlags.ROTATIONAL_KINETIC_ENERGY,
                     ParticleDataFlags.PRESSURE_TENSOR,
                     ParticleDataFlags.EXTERNAL_FIELD_VIRIAL]

            def act(self, timestep):
                pass

    For advertising loggable quantities through the
    `hoomd.python_action._CustomOperation` object, the class attribute
    ``log_quantities`` can be used. The dictionary expects string keys with the
    name of the loggable and `hooomd.logger.LoggerQuantity` objects as the
    values.

    .. code-block:: python

        from hoomd.python_action import _CustomAction
        from hoomd.logger import LoggerQuantity


        class ExampleActionWithFlag(_CustomAction):
            def __init__(self):
                self.log_quantities = {
                    'loggable': LoggerQuantity('scalar_loggable',
                                               self.__class__,
                                               flag='scalar')}

            def loggable(self):
                return 42

            def act(self, timestep):
                pass
    """
    flags = []
    log_quantities = {}

    def __init__(self):
        pass

    def attach(self, simulation):
        self._state = simulation.state

    def detach(self):
        if hasattr(self, '_state'):
            del self._state

    @abstractmethod
    def act(self, timestep):
        pass


class _InternalCustomAction(_CustomAction, _HOOMDGetSetAttrBase):
    """An internal class for Python ``Action``s.

    Gives additional support in using HOOMD constructs like ``ParameterDict``s
    and ``TypeParameters``.
    """
    pass
