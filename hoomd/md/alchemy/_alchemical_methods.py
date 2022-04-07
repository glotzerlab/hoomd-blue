# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeconverter import SetOnce
from hoomd.operation import _HOOMDBaseObject


class _AlchemicalMethods(_HOOMDBaseObject):

    def __init__(self):
        self.alchemical_particles = self.AlchemicalParticleAccess(self)

    class AlchemicalParticleAccess(TypeParameterDict):

        def __init__(self, outer):
            self.outer = outer
            super().__init__(SetOnce(outer._particle_type), len_keys=2)

        # FIXME: breaks delayed instantiation by validating types
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
