# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical pair forces."""

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
            attributedict['_dof_type'] = AlchemicalNormalizedDOF
        else:
            attributedict['normalized'] = False
            attributedict['_dof_type'] = AlchemicalDOF
        attributedict['_cpp_class_name'] = ''.join(new_cpp_name)

        superclasses += (_AlchemicalMethods,)
        return super().__new__(cls, name, superclasses, attributedict)

    def __init__(self, name, superclasses, attributedict):
        self._reserved_default_attrs['_alchemical_parameters'] = list
        self._accepted_modes = ('none', 'shift')
        super().__init__(name, superclasses, attributedict)


class AlchemicalDOF(_HOOMDBaseObject):
    """Alchemical degree of freedom associated with a specific pair force.

    Args:
        force (``_AlchemicalPairPotential``): Pair force containing the
        alchemical degree of freedom.
        name (str): The name of the pair force.
        typepair (tuple[str]): The particle types upon which the pair force
            acts.
        alpha (float): The value of the alchemical parameter.
        mass (float): The mass of the alchemical degree of freedom.
        mu (float): The alchemical potential.

    `AlchemicalDOF` defines alchemical degrees of freedom that are to be
    numerically integrated via an alchemical MD integration method.

    Tip:
        Use the ``create_alchemical_dof`` method of any of the alchemical pair
        forces to construct an `AlchemicalDOF` instance.

    """

    def __new__(cls,
                force: _AlchemicalPairPotential,
                name: str = '',
                typepair: tuple = None,
                mass: float = 1.0):
        """Cache existing instances of AlchemicalDOF.

        Args:
            force (``_AlchemicalPairPotential``): Pair force containing the
            alchemical degree of freedom.
            name (str): The name of the pair force.
            typepair (tuple[str]): The particle types upon which the pair force
                acts.
            mass (float): The mass of the alchemical degree of freedom.

        """
        typepair = tuple(sorted(typepair))
        # if an instenace already exists, return that one
        if (typepair, name) in force._alchemical_dof:
            return force._alchemical_dof[typepair, name]
        return super().__new__(cls)

    def __init__(self,
                 force: _AlchemicalPairPotential,
                 name: str = '',
                 typepair: tuple = None,
                 alpha: float = 1.0,
                 mass: float = 1.0,
                 mu: float = 0.0):
        self.force = force
        self.name = name
        self.typepair = typepair
        if self.force._attached:
            self._attach()
        # store metadata
        param_dict = ParameterDict(mass=float,
                                   mu=float,
                                   alpha=float,
                                   alchemical_momentum=float)
        param_dict['mass'] = mass
        param_dict['mu'] = mu
        param_dict['alpha'] = alpha
        param_dict['alchemical_momentum'] = 0.0

        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        assert self.force._attached
        self._cpp_obj = self.force._cpp_obj.getAlchemicalPairParticle(
            *map(self.force._simulation.state.particle_types.index,
                 self.typepair),
            self.force._alchemical_parameters.index(self.name))
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
                "Attempting to iterate an alchemical degree of freedom twice")
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

    @log
    def value(self):
        """Current value of the alchemical degree of freedom."""
        return self.force.params[self.typepair][self.name] * (
            self._cpp_obj.alpha if self._attached else 1.)

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


class AlchemicalNormalizedDOF(AlchemicalDOF):
    """Alchemical normalized degree of freedom."""

    def __init__(self,
                 force: _AlchemicalPairPotential,
                 name: str = '',
                 typepair: tuple = None,
                 alpha: float = 1.0,
                 norm_value: float = 0.0,
                 mass: float = 1.0,
                 mu: float = 0.0):
        super().__init__(force, name, typepair, alpha, mass, mu)
        self._param_dict.update(dict(norm_value=norm_value))

    @log(default=False, requires_run=True, category='particle')
    def alchemical_forces(self):
        """Per particle forces in alchemical alpha space."""
        return self._cpp_obj.forces * self._cpp_obj.norm_value


class LJGauss(BaseLJGauss, metaclass=_AlchemicalPairPotential):
    r"""Alchemical Lennard Jones Gauss pair force.

    Args:
        nlist (`hoomd.md.nlist.NeighborList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJGauss` computes the Lennard-Jones Gauss force on all particles in the
    simulation state, see `hoomd.md.pair.LJGauss` for more details.

    Attention:
        `hoomd.md.alchemy.pair.LJGauss` does not support execution on GPUs.

    Attention:
        `hoomd.md.alchemy.pair.LJGauss` does not support MPI parallel
        simulations.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma2`` (`float`, **required**) -
          Gaussian variance :math:`\sigma^2` :math:`[\mathrm{length}]^2`
        * ``r0`` (`float`, **required**) -
          Gaussian center :math:`r_0` :math:`[\mathrm{length}]`

    """
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)

    def create_alchemical_dof(self, typepair, parameter):
        """Create an alchemical degree of freedom on a potential parameter.

        Args:
            typepair (tuple[str]): The pair of particle types for which to
                create the alchemical degree of freedom
            parameter (str): The name of the parameter to make an alchemical
                degree of freedom

        Returns:
            AlchemicalDOF: The alchemical degree of freedom that can be\
                    integrated via the :doc:`alchemical integration methods\
                    </module-md-alchemy-methods>`.
        """
        return self._alchemical_dof[typepair, parameter]


class NLJGauss(BaseLJGauss, metaclass=_AlchemicalPairPotential):
    """Alchemical normalized Lennard Jones Gauss pair force.

    See `LJGauss` for more information.

    Attention:
        `hoomd.md.alchemy.pair.NLJGauss` does not support execution on GPUs.

    Attention:
        `hoomd.md.alchemy.pair.NLJGauss` does not support MPI parallel
        simulations.

    Attention:
        `hoomd.md.alchemy.pair.NLJGauss` is only valid for systems that contain
        a single particle type with a single pair force.

    """
    _alchemical_parameters = ['epsilon', 'sigma2', 'r0']
    normalized = True

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalMethods.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)

    def create_alchemical_dof(self, typepair, parameter):
        """Create an alchemical degree of freedom on a potential parameter.

        Args:
            typepair (tuple[str]): The pair of particle types for which to
                create the alchemical degree of freedom
            parameter (str): The name of the parameter to make an alchemical
                degree of freedom

        Returns:
            AlchemicalDOF: The alchemical degree of freedom that can be\
                    integrated via the :doc:`alchemical integration methods\
                    </module-md-alchemy-methods>`.

        """
        return self._alchemical_dof[typepair, parameter]
