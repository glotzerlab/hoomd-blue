# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Alchemical potentials."""

from hoomd.logging import log, Loggable
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeconverter import SetOnce
from hoomd.md.pair import pair
from hoomd.md.methods import Method
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant

# alternative access? hoomd.alch.md.pair would be nice to somehow link here
# without making overly complicated
# I just like alch(MD/MC) as alchemy specification aliases

# TODO: remove this variable when the log names are finalized
_api = ['value', 'mass', 'mu', 'avalue', 'amomentum', 'net_aforce', 'aforces']


class _AlchemicalPairPotential(Loggable):

    def __new__(cls, name, superclasses, attributedict):
        attributedict[
            '_cpp_class_name'] = 'Alchemical' + superclasses[0]._cpp_class_name
        superclasses += (_AlchemicalMethods,)
        return super().__new__(cls, name, superclasses, attributedict)

    def __init__(cls, name, superclasses, attributedict):
        cls._reserved_default_attrs['_alchemical_parameters'] = list
        cls._accepted_modes = ('none', 'shift')
        super().__init__(name, superclasses, attributedict)


# Perhaps define a true base alch particle to base this off of just adding the
# MD components mirroring the cpp side
class AlchemicalMDParticle(_HOOMDBaseObject):
    __slots__ = ['force', 'name', 'typepair', '_cpp_obj', '_mass']

    def __new__(cls,
                force: _AlchemicalPairPotential,
                name: str = '',
                typepair: tuple = None,
                mass: float = 1.0):
        typepair = tuple(sorted(typepair))
        if (typepair, name) in force.alchemical_particles:
            return force.alchemical_particles[typepair, name]
        return super().__new__(cls)

    def __init__(self,
                 force: _AlchemicalPairPotential,
                 name: str = '',
                 typepair: tuple = None,
                 mass: float = 1.0):
        self.force = force
        self.name = name
        self.typepair = typepair
        self.force._add_dependent(self)
        self._mass = mass
        if self.force._attached:
            self._attach()

    def _attach(self):
        self._cpp_obj = self.force._cpp_obj.getAlchemicalPairParticle(
            *map(self.force._simulation.state.particle_types.index,
                 self.typepair),
            self.force._alchemical_parameters.index(self.name))
        self.mass = self._mass
        self._mass = self.mass

    # Need to enable and disable via synced list of alchemostat
    def _add(self, simulation):
        self._enable()
        super()._add(simulation)

    @log(default=False)
    def mass(self):
        if self._attached:
            return self._cpp_obj.mass
        else:
            return self._mass

    @mass.setter
    def mass(self, M):
        if self._attached:
            self._cpp_obj.mass = M
        else:
            self._mass = M

    @log
    def value(self):
        return self.force.params[self.typepair][self.name] * (
            self._cpp_obj.alpha if self._attached else 1.)

    @log(default=False, requires_run=True)
    def avalue(self):
        return self._cpp_obj.alpha

    @log(default=False, requires_run=True)
    def amomentum(self):
        return self._cpp_obj.alpha

    @log(default=False, requires_run=True)
    def mu(self):
        return self._cpp_obj.mu

    @log(default=False, requires_run=True, category='particle')
    def aforces(self):
        return self._cpp_obj.forces

    @log(requires_run=True)
    def net_aforce(self):
        return self._cpp_obj.net_force()

    def _enable(self):
        self.force._cpp_obj.enableAlchemicalPairParticle(self._cpp_obj)


class _AlchemicalMethods(_HOOMDBaseObject):

    def __init__(self):
        self.alchemical_particles = self.AlchemicalParticleAccess(self)

    class AlchemicalParticleAccess(TypeParameterDict):

        def __init__(self, outer):
            self.outer = outer
            super().__init__(SetOnce(AlchemicalMDParticle), len_keys=2)

        def _validate_and_split_alchem(self, key):
            if isinstance(key, tuple) and len(key) == self._len_keys:
                for param, typepair in zip(key, reversed(key)):
                    if set(self.outer._alchemical_parameters).issuperset(
                            self._validate_and_split_len_one(param)) and set(
                                self.outer._simulation.state.particle_types
                            ).issuperset(
                                self._validate_and_split_len_one(typepair)):
                        break
                else:
                    raise KeyError(
                        key,
                        'Not a valid combination of particle types and alchemical parameter keys.'
                    )
                return tuple(self._yield_keys(typepair)), tuple(
                    self._validate_and_split_len_one(param))

        def __getitem__(self, key):
            typepair, param = self._validate_and_split_alchem(key)
            vals = dict()
            for t in typepair:
                for p in param:
                    k = (t, p)
                    if not k in self._dict:
                        self._dict[k] = AlchemicalMDParticle(self.outer, p, t)
                    vals[k] = self._dict[k]
            if len(vals) > 1:
                return vals
            else:
                return vals[k]

        def __setitem__(self, key, val):
            raise NotImplementedError

        def __contains__(self, key):
            key = self._validate_and_split_alchem(key)
            return key in self._dict.keys()

        def keys(self):
            return self._dict.keys()

    def _attach(self):
        super()._attach()
        for v in self.alchemical_particles._dict.values():
            v._attach()


class LJGauss(pair.LJGauss, metaclass=_AlchemicalPairPotential):
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)


class NVT(Method):
    r"""Alchemical NVT Integration.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which
            to apply this method.

        kT (`hoomd.variant.Variant` or `float`): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        tau (`float`): Time factor for the alchemostat

    Examples::

        nvt=hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])

    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        kT (hoomd.variant.Variant): Temperature set point
            for the alchemostat :math:`[\mathrm{energy}]`.

        tau (float): Time factor for the alchemostat

    """

    def __init__(self, filter, kT, time_factor):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   time_factor=float(tau))
        param_dict.update(
            dict(kT=kT,
                 filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        cpp_class = hoomd.md._md.TwoStepNVTAlchemy
        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_obj = cpp_class(cpp_sys_def, group, self.time_factor, self.kT)
        super()._attach()


# # TODO: the rest of this should likely be moved to a new namespace
# from collections.abc import ABC, abstractmethod
# from hoomd.util import SyncedList
# from hoomd.md.integrate import BaseIntegrator

# class Alchemostat(ABC, _HOOMDBaseObject):

#     # synced list? or operation style?
#     @property
#     def alchemical_particles(self):
#         return self._alchemical_particles

#     @alchemical_particles.setter
#     def alchemical_particles(self, alchemical_particles):
#         # This condition is necessary to allow for += and -= operators to work
#         # correctly with alchemostat.alchemical_particles (+=/-=).
#         if alchemical_particles is self._alchemical_particles:
#             return
#         else:
#             # Handle error cases first
#             if alchemical_particles._added or alchemical_particles._simulation is not None:
#                 raise RuntimeError(
#                     "Cannot add `hoomd.Alchemicalalchemical_particles` object that belongs to "
#                     "another `hoomd.Simulation` object.")

#     @property
#     def time_factor(self):
#         if self.attached:
#             return self._cpp_obj.alchemicalTimeFactor
#         else:
#             return storedparams
