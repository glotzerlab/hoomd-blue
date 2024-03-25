# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Action.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    logger = hoomd.logging.Logger()
"""

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
    """Base class for user-defined actions.

    To implement a custom operation in Python, subclass `hoomd.custom.Action`
    and implement the :meth:`~.act` method to perform the desired action. To
    include the action in the simulation run loop, pass an instance of the
    action to `hoomd.update.CustomUpdater`, `hoomd.write.CustomWriter`, or
    `hoomd.tune.CustomTuner`.

    .. rubric:: Examples:

    .. code-block:: python

        class ExampleAction(hoomd.custom.Action):
            def act(self, timestep):
                snapshot = self._state.get_snapshot()
                if snapshot.communicator.rank == 0:
                    self.com = snapshot.particles.position.mean(axis=0)

    To request that HOOMD-blue compute virials, pressure, the rotational kinetic
    energy, or the external field virial, set the flags attribute with the
    appropriate flags from the internal `Action.Flags` enumeration:

    .. code-block:: python

        class ExampleAction(hoomd.custom.Action):
            flags = [Action.Flags.ROTATIONAL_KINETIC_ENERGY,
                     Action.Flags.PRESSURE_TENSOR,
                     Action.Flags.EXTERNAL_FIELD_VIRIAL]

            def act(self, timestep):
                pass

    Use the `hoomd.logging.log` decorator to define loggable properties:

    .. code-block:: python

        class ExampleAction(hoomd.custom.Action):

            @hoomd.logging.log
            def answer(self):
                return 42

            def act(self, timestep):
                pass

        example_action = ExampleAction()
        logger.add(example_action, quantities=['answer'])

    Attributes:
        flags (list[Action.Flags]): List of flags from the
            `Action.Flags`. Used to tell the integrator if
            specific quantities are needed for the action.
    """

    class Flags(IntEnum):
        """Flags to indictate the integrator should calculate quantities.

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

        Stores the simulation state in ``self._state``. Override this in derived
        classes to implement other behaviors.
        """
        self._state = simulation.state

    @property
    def _attached(self):
        return getattr(self, '_state', None) is not None

    def detach(self):
        """Detaches the Action from the `hoomd.Simulation`."""
        self._state = None

    @abstractmethod
    def act(self, timestep):
        """Performs whatever action a subclass implements.

        Args:
            timestep (int): The current timestep in a simulation.

        Note:
            Use ``self._state`` to access the simulation state via
            `hoomd.State` when using the base class `attach`.
        """
        pass


class _InternalAction(Action, _HOOMDGetSetAttrBase):
    """An internal class for Python Actions.

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

        See hoomd/operation.py BaseHOOMDObject._setattr_param for details.
        """
        self._param_dict[attr] = value
