# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical MD integration methods."""

import hoomd
from hoomd.md.alchemy.pair import AlchemicalDOF
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data import syncedlist
from hoomd.md.methods import Method
from hoomd.variant import Variant


class Alchemostat(Method):
    """Alchemostat Base Class."""

    def __init__(self, alchemical_dof):
        self.alchemical_dof = self._OwnedAlchemicalParticles(self)
        if alchemical_dof is not None:
            self.alchemical_dof.extend(alchemical_dof)

    def _update(self):
        if self._attached:
            self._cpp_obj.alchemical_dof = self.alchemical_dof._synced_list

    def _attach(self):
        super()._attach()
        # TODO: handle forces not attached
        # for force in {alpha.force for alpha in self.alchemical_dof}

        # keep a separate local cpp list because static casting
        # won't let us access directly
        self.alchemical_dof._sync(None, [])
        self._update()

    class _OwnedAlchemicalParticles(syncedlist.SyncedList):
        """Owned alchemical particles.

        Accessor/wrapper to specialize a synced list

        Alchemical degrees of freedom which will be integrated by this
        integration method.
        """

        def __init__(self, outer):
            self._outer = outer
            super().__init__(AlchemicalDOF,
                             syncedlist._PartialGetAttr('_cpp_obj'))

        def __setitem__(self, i, item):
            item._own(self._outer)
            super().__setitem__(i, item)
            self._outer._update()

        def __delitem__(self, i):
            self._outer.alchemical_dof[i]._disown()
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
        alchemical_kT (`hoomd.variant.Variant` or `float`): Temperature set
            point for the alchemostat :math:`[\mathrm{energy}]`.

        alchemical_dof (list): List of alchemical particles

        period (`int`): Time factor for the alchemostat

    Attention:
        `hoomd.md.alchemy.methods.NVT` does not support execution on GPUs.

    Attention:
        `hoomd.md.alchemy.methods.NVT` does not support MPI parallel
        simulations.

    Attention:
        `hoomd.md.alchemy.methods.NVT` objects are not picklable.

    Examples::

        nvt=hoomd.md.methods.NVT(kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])

    Attributes:
        kT (hoomd.variant.Variant): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        alchemical_dof (list): List of alchemical particles

        period (int): Time factor for the alchemostat

    """

    def __init__(self, alchemical_kT, alchemical_dof, period=1):

        # store metadata
        param_dict = ParameterDict(alchemical_kT=Variant, period=int)
        param_dict.update(dict(alchemical_kT=alchemical_kT, period=period))
        # set defaults
        self._param_dict.update(param_dict)
        super().__init__(alchemical_dof)

    def _attach(self):
        cpp_class = hoomd.md._md.TwoStepNVTAlchemy
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_obj = cpp_class(cpp_sys_def, self.period, self.alchemical_kT)
        self._cpp_obj.setNextAlchemicalTimestep(self._simulation.timestep)
        super()._attach()
