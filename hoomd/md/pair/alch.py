# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Alchemical potentials."""
import hoomd
from hoomd.logging import log, Loggable
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeconverter import SetOnce
from hoomd.data import syncedlist
from hoomd.md.pair import pair
from hoomd.md.methods import Method
from hoomd.variant import Variant

# alternative access? hoomd.alch.md.pair would be nice to somehow link here
# without making overly complicated
# I just like alch(MD/MC) as alchemy specification aliases

# TODO: remove this variable when the log names are finalized
_api = {}
_api[0.1] = [
    'value', 'mass', 'mu', 'avalue', 'amomentum', 'net_aforce', 'aforces'
]


class _AlchemicalPairPotential(Loggable):
    """A metaclass to make alchemical modifications to the pair potential."""

    def __new__(cls, name, superclasses, attributedict):
        new_cpp_name = [
            'PotentialPair', 'Alchemical', superclasses[0]._cpp_class_name[13:]
        ]
        if attributedict.get('noramlized', False):
            new_cpp_name.insert(2, 'Normalized')
            attributedict['_particle_type'] = AlchemicalNormalizedPairParticle
        else:
            attributedict['normalized'] = False
            attributedict['_particle_type'] = AlchemicalPairParticle
        attributedict['_cpp_class_name'] = ''.join(new_cpp_name)

        superclasses += (_AlchemicalMethods,)
        return super().__new__(cls, name, superclasses, attributedict)

    def __init__(self, name, superclasses, attributedict):
        self._reserved_default_attrs['_alchemical_parameters'] = list
        self._accepted_modes = ('none', 'shift')
        super().__init__(name, superclasses, attributedict)


# Perhaps define a true base alch particle to base this off of just adding the
# MD components mirroring the cpp side
class AlchemicalPairParticle(_HOOMDBaseObject):
    """Alchemical pair particle associated with a specific force."""

    __slots__ = ['force', 'name', 'typepair', '_cpp_obj', '_mass']

    def __new__(cls,
                force: _AlchemicalPairPotential,
                name: str = '',
                typepair: tuple = None,
                mass: float = 1.0):
        """This is not a public method @flake8."""
        typepair = tuple(sorted(typepair))
        # if an instenace already exists, return that one
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
        self._mass = mass
        if self.force._attached:
            self._attach()
        # store metadata
        param_dict = ParameterDict(force=force, typepair=tuple)
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        self._cpp_obj = self.force._cpp_obj.getAlchemicalPairParticle(
            *map(self.force._simulation.state.particle_types.index,
                 self.typepair),
            self.force._alchemical_parameters.index(self.name))
        self._enable()
        self.mass = self._mass
        self._mass = self.mass

    # Need to enable and disable via synced list of alchemostat
    def _add(self, simulation):
        super()._add(simulation)

    def _detach(self):
        if self._attached:
            self.force.params[self.typepair][self.name] = self.value
            self._disable()
            super()._detach()

    @log(default=False)
    def mass(self):
        """Alchemical mass."""
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
        """Current value of the alchemical parameter."""
        return self.force.params[self.typepair][self.name] * (
            self._cpp_obj.alpha if self._attached else 1.)

    @log(default=False, requires_run=True)
    def avalue(self):
        """Dimensionless alchemical alpha space value."""
        return self._cpp_obj.alpha

    @log(default=False, requires_run=True)
    def amomentum(self):
        """Momentum in alchemical alpha space."""
        return self._cpp_obj.alpha

    @log(default=False, requires_run=True)
    def mu(self):
        """Alchemical potential."""
        return self._cpp_obj.mu

    @log(default=False, requires_run=True, category='particle')
    def aforces(self):
        """Per particle forces in alchemical alpha space."""
        return self._cpp_obj.forces

    @log(requires_run=True)
    def net_aforce(self):
        """Net force in alchemical alpha space."""
        return self._cpp_obj.net_force()

    def _enable(self):
        self.force._cpp_obj.enableAlchemicalPairParticle(self._cpp_obj)

    def _disable(self):
        self.force._cpp_obj.disableAlchemicalPairParticle(self._cpp_obj)


class AlchemicalNormalizedPairParticle(AlchemicalPairParticle):
    """Alchemical normalized pair particle."""

    @log(default=False, requires_run=True)
    def norm_value(self):
        """Normalization Value."""
        return self._cpp_obj.norm_value

    @log(default=False, requires_run=True, category='particle')
    def aforces(self):
        """Per particle forces in alchemical alpha space."""
        return self._cpp_obj.forces * self._cpp_obj.norm_value


class _AlchemicalMethods(_HOOMDBaseObject):

    def __init__(self):
        self.alchemical_particles = self.AlchemicalParticleAccess(self)

    class AlchemicalParticleAccess(TypeParameterDict):

        def __init__(self, outer):
            self.outer = outer
            super().__init__(SetOnce(outer._particle_type), len_keys=2)

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
                        key, """Not a valid combination of particle types and
                        alchemical parameter keys.""")
                return tuple(self._yield_keys(typepair)), tuple(
                    self._validate_and_split_len_one(param))

        def __getitem__(self, key):
            typepair, param = self._validate_and_split_alchem(key)
            vals = dict()
            for t in typepair:
                for p in param:
                    k = (t, p)
                    if k not in self._dict:
                        self._dict[k] = self.outer._particle_type(
                            self.outer, p, t)
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
    """Alchemical Lennard Jones Gauss pair potential."""
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)


class NLJGauss(pair.LJGauss, metaclass=_AlchemicalPairPotential):
    """Alchemical Lennard Jones Gauss pair potential."""
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']
    noramlized = True

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)


class Alchemostat(Method):
    """Alchemostat Base Class."""

    def __init__(self, alchemical_particles):
        if alchemical_particles is None:
            alchemical_particles = []
        self._alchemical_particles = syncedlist.SyncedList(
            AlchemicalPairParticle,
            syncedlist._PartialGetAttr('_cpp_obj'),
            iterable=alchemical_particles)

    def _attach(self):
        self.alchemical_particles._sync(self._simulation,
                                        self._cpp_obj.alchemical_particles)
        super()._attach()

    @property
    def alchemical_particles(self):
        """Owned alchemical particles.

        Alchemical particles which will be integrated by this integrator
        method.
        """
        return self._alchemical_particles

    @alchemical_particles.setter
    def alchemical_particles(self, value):
        self._alchemical_particles.clear()
        self._alchemical_particles = value


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
