# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical MD integration methods."""

import hoomd
from hoomd.md.alchemy.pair import AlchemicalPairParticle
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data import syncedlist
from hoomd.md.methods import Method
from hoomd.variant import Variant


class Alchemostat(Method):
    """Alchemostat Base Class."""

    def __init__(self, alchemical_particles):
        self.alchemical_particles = self._OwnedAlchemicalParticles(self)
        if alchemical_particles is not None:
            self.alchemical_particles.extend(alchemical_particles)

    def _update(self):
        if self._attached:
            self._cpp_obj.alchemical_particles = \
                    self.alchemical_particles._synced_list

    def _attach(self):
        super()._attach()
        # TODO: handle forces not attached
        # for force in {alpha.force for alpha in self.alchemical_particles}

        # keep a separate local cpp list because static casting
        # won't let us access directly
        self.alchemical_particles._sync(None, [])
        self._update()

        # if we're holding onto a temporary variable, ditch it now
        if hasattr(self, "_time_factor"):
            self.time_factor

    class _OwnedAlchemicalParticles(syncedlist.SyncedList):
        """Owned alchemical particles.

        Accessor/wrapper to specialize a synced list

        Alchemical particles which will be integrated by this integrator
        method.
        """

        def __init__(self, outer):
            self._outer = outer
            super().__init__(AlchemicalPairParticle,
                             syncedlist._PartialGetAttr('_cpp_obj'))

        def __setitem__(self, i, item):
            item._own(self._outer)
            super().__setitem__(i, item)
            self._outer._update()

        def __delitem__(self, i):
            self._outer.alchemical_particles[i]._disown()
            super().__delitem__(i)
            self._outer._update()

        def insert(self, i, item):
            """Insert value to list at index, handling list syncing."""
            item._own(self._outer)
            super().insert(i, item)
            self._outer._update()


class NVT(Alchemostat):
    r"""Alchemical NVT Integration.

    Args:
        kT (`hoomd.variant.Variant` or `float`): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        time_factor (`int`): Time factor for the alchemostat

        alchemical_particles (list): List of alchemical particles

    Examples::

        nvt=hoomd.md.methods.NVT(kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])

    Attributes:
        kT (hoomd.variant.Variant): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        time_factor (int): Time factor for the alchemostat

        alchemical_particles (list): List of alchemical particles

    """

    def __init__(self, kT, time_factor=1, alchemical_particles=[]):

        # store metadata
        param_dict = ParameterDict(kT=Variant, time_factor=int(time_factor))
        param_dict.update(dict(kT=kT))
        # set defaults
        self._param_dict.update(param_dict)
        super().__init__(alchemical_particles)

    def _attach(self):
        cpp_class = hoomd.md._md.TwoStepNVTAlchemy
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_obj = cpp_class(cpp_sys_def, self.time_factor, self.kT)
        self._cpp_obj.setNextAlchemicalTimestep(self._simulation.timestep)
        super()._attach()


class NVE(Alchemostat):
    r"""Alchemical NVE Integration.

    Args:
        time_factor (`int`): Time factor for the alchemostat

        alchemical_particles (list): List of alchemical particles

    Examples::

        nve=hoomd.md.methods.NVE()
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])

    Attributes:
        time_factor (int): Time factor for the alchemostat

        alchemical_particles (list): List of alchemical particles

    """

    def __init__(self, time_factor=1, alchemical_particles=[]):

        # store metadata
        param_dict = ParameterDict(time_factor=int(time_factor))
        # set defaults
        self._param_dict.update(param_dict)
        super().__init__(alchemical_particles)

    def _attach(self):
        cpp_class = hoomd.md._md.TwoStepNVEAlchemy
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_obj = cpp_class(cpp_sys_def, self.time_factor)
        self._cpp_obj.setNextAlchemicalTimestep(self._simulation.timestep)
        super()._attach()
