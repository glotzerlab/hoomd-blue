# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical pair potentials."""

from hoomd.logging import log, Loggable
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.md.pair import LJGauss as BaseLJGauss

from hoomd.md.alchemy._alchemical_methods import _AlchemicalMethods


class _AlchemicalPairPotential(Loggable):
    """A metaclass to make alchemical modifications to the pair potential."""

    def __new__(cls, name, superclasses, attributedict):
        new_cpp_name = [
            'PotentialPair', 'Alchemical', superclasses[0]._cpp_class_name[13:]
        ]
        if attributedict.get('normalized', False):
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
        assert self.force._attached
        self._cpp_obj = self.force._cpp_obj.getAlchemicalPairParticle(
            *map(self.force._simulation.state.particle_types.index,
                 self.typepair),
            self.force._alchemical_parameters.index(self.name))
        self.mass = self._mass
        self._mass = self.mass
        if self._owned:
            self._enable()

    # Need to enable and disable via synced list of alchemostat
    def _add(self, simulation):
        super()._add(simulation)

    @property
    def _owned(self):
        return hasattr(self, '_owner')

    def _own(self, alchemostat):
        if self._owned:
            raise RuntimeError(
                "Attempting to iterate an alchemical particle twice")
        self._owner = alchemostat
        if self._attached:
            self._enable()

    def _disown(self):
        self._disable()
        delattr(self, '_owner')

    def _detach(self):
        if self._attached:
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
    def alpha(self):
        """Dimensionless alchemical alpha space value."""
        return self._cpp_obj.alpha

    @log(default=False, requires_run=True)
    def alchemical_momentum(self):
        """Momentum in alchemical alpha space."""
        return self._cpp_obj.momentum

    @log(default=False, requires_run=True)
    def mu(self):
        """Alchemical potential."""
        return self._cpp_obj.mu

    @log(default=False, requires_run=True, category='particle')
    def alchemical_forces(self):
        """Per particle forces in alchemical alpha space."""
        return self._cpp_obj.forces

    @log(requires_run=True)
    def net_alchemical_force(self):
        """Net force in alchemical alpha space."""
        return self._cpp_obj.net_force

    def _enable(self):
        assert self._attached
        self.force._cpp_obj.enableAlchemicalPairParticle(self._cpp_obj)

    def _disable(self):
        assert self._attached
        self.force.params[self.typepair][self.name] = self.value
        self.force._cpp_obj.disableAlchemicalPairParticle(self._cpp_obj)


class AlchemicalNormalizedPairParticle(AlchemicalPairParticle):
    """Alchemical normalized pair particle."""

    @log(default=False, requires_run=True)
    def norm_value(self):
        """Normalization Value."""
        return self._cpp_obj.norm_value

    @norm_value.setter
    def norm_value(self, value):
        self._cpp_obj.norm_value = value

    @log(default=False, requires_run=True, category='particle')
    def aforces(self):
        """Per particle forces in alchemical alpha space."""
        return self._cpp_obj.forces * self._cpp_obj.norm_value


class LJGauss(BaseLJGauss, metaclass=_AlchemicalPairPotential):
    """Alchemical Lennard Jones Gauss pair potential."""
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)


class NLJGauss(BaseLJGauss, metaclass=_AlchemicalPairPotential):
    """Alchemical Lennard Jones Gauss pair potential."""
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']
    normalized = True

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)
