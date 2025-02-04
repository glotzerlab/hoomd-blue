# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical MD integration methods."""

import hoomd
from hoomd.md.alchemy.pair import AlchemicalDOF
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data import syncedlist
from hoomd.md.methods import Method
from hoomd.variant import Variant


class Alchemostat(Method):
    """Alchemostat base class.

    Note:
        `Alchemostat` is the base class for all alchemical integration methods.
        Users should use the subclasses and not instantiate `Alchemostat`
        directly.

    .. versionadded:: 3.1.0
    """

    def __init__(self, alchemical_dof):
        self._alchemical_dof = syncedlist.SyncedList(
            AlchemicalDOF, syncedlist._PartialGetAttr("_cpp_obj"))
        if alchemical_dof is not None:
            self._alchemical_dof.extend(alchemical_dof)

    def _attach_hook(self):
        self._alchemical_dof._sync(self._simulation,
                                   self._cpp_obj.alchemical_dof)

    def _detach_hook(self):
        self._alchemical_dof._unsync()

    @property
    def alchemical_dof(self):
        """`list` [`hoomd.md.alchemy.pair.AlchemicalDOF` ]: List of \
                alchemical degrees of freedom."""
        return self._alchemical_dof

    @alchemical_dof.setter
    def alchemical_dof(self, new_dof):
        self._alchemical_dof.clear()
        self._alchemical_dof.extend(new_dof)


class NVT(Alchemostat):
    r"""Alchemical NVT integration.

    Implements molecular dynamics simulations of an extended statistical
    mechanical ensemble that includes alchemical degrees of freedom describing
    particle attributes as thermodynamic variables.

    Args:
        alchemical_kT (hoomd.variant.variant_like): Temperature set
            point for the alchemostat :math:`[\mathrm{energy}]`.

        alchemical_dof (`list` [`hoomd.md.alchemy.pair.AlchemicalDOF`]): List
            of alchemical degrees of freedom.

        period (int): Timesteps between applications of the alchemostat.

    Attention:
        `hoomd.md.alchemy.methods.NVT` does not support execution on GPUs.

    Attention:
        `hoomd.md.alchemy.methods.NVT` does not support MPI parallel
        simulations.

    Attention:
        `hoomd.md.alchemy.methods.NVT` objects are not picklable.

    Danger:
        `NVT` must be the first item in the `hoomd.md.Integrator.methods` list.

    See Also:
        `Zhou et al. 2019 <https://doi.org/10.1080/00268976.2019.1680886>`_.

    .. versionadded:: 3.1.0

    Attributes:
        alchemical_kT (hoomd.variant.Variant): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        alchemical_dof (`list` [`hoomd.md.alchemy.pair.AlchemicalDOF`]): List of
            alchemical degrees of freedom.

        period (int): Timesteps between applications of the alchemostat.
    """

    def __init__(self, alchemical_kT, alchemical_dof, period=1):

        # store metadata
        param_dict = ParameterDict(alchemical_kT=Variant, period=int)
        param_dict.update(dict(alchemical_kT=alchemical_kT, period=period))
        # set defaults
        self._param_dict.update(param_dict)
        super().__init__(alchemical_dof)

    def _attach_hook(self):
        cpp_class = hoomd.md._md.TwoStepNVTAlchemy
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_obj = cpp_class(cpp_sys_def, self.period, self.alchemical_kT)
        self._cpp_obj.setNextAlchemicalTimestep(self._simulation.timestep)
        super()._attach_hook()
