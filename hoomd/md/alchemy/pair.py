# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical pair forces."""
from collections.abc import Mapping

import hoomd.data
from hoomd.logging import log
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.md.pair import LJGauss as BaseLJGauss


def _modify_pair_cls_to_alchemical(cls):
    new_cpp_name = [
        'PotentialPair', 'Alchemical', cls.__mro__[0]._cpp_class_name[13:]
    ]
    if getattr(cls, 'normalized', False):
        new_cpp_name.insert(2, 'Normalized')
        cls._dof_cls = AlchemicalNormalizedDOF
    else:
        cls.normalized = False
        cls._dof_cls = AlchemicalDOF
    cls._cpp_class_name = ''.join(new_cpp_name)
    cls._reserved_default_attrs['_alchemical_parameters'] = list
    cls._accepted_modes = ('none', 'shift')
    return cls


class AlchemicalPairDOFStore(Mapping):
    """A read-only mapping of alchemical particles accessed by type."""

    def __init__(self, name, pair_instance, dof_cls):
        """Create an `AlchemicalPairDOFStore` object.

        Warning:
            Should not be instantiated by users.
        """
        self._name = name
        self._dof_cls = dof_cls
        self._pair_instance = pair_instance
        self._indexer = hoomd.data.parameterdicts._SmartTypeIndexer(2)
        self._data = {}

    def __getitem__(self, key):
        """Get the alchemical particle for the given type pair."""
        items = {}
        for k in self._indexer(key):
            if k not in self._data:
                self._data[k] = self._dof_cls(self._pair_instance, self._name,
                                              k)
            items[k] = self._data[k]
        if len(items) == 1:
            return items.popitem()[1]
        return items

    def __iter__(self):
        """Iterate over keys."""
        yield from self._data

    def __contains__(self, key):
        """Return whether the key is in the mapping."""
        keys = list(self._indexer(key))
        if len(keys) == 1:
            return keys[0] in self._data
        return [k in self._data for k in keys]

    def __len__(self):
        """Get the length of the mapping."""
        return len(self._data)

    def _attach(self, types):
        self._indexer.valid_types = types
        for key in self:
            if not self._indexer.are_valid_types(key):
                raise RuntimeError(
                    f"Alchemical DOF ({self._name}) for non-existent type pair "
                    f"{key} was accessed.")

    def _detach(self):
        self._indexer.valid_types = None


class _AlchemicalPairForce(_HOOMDBaseObject):
    _alchemical_dofs = []
    _dof_cls = None

    def __init__(self):
        self._set_alchemical_parameters()

    def _set_alchemical_parameters(self):
        self._alchemical_params = {}
        for dof in self._alchemical_dofs:
            self._alchemical_params[dof] = AlchemicalPairDOFStore(
                name=dof, pair_instance=self, dof_cls=self._dof_cls)

    def _setattr_hook(self, attr, value):
        if attr in self._alchemical_dofs:
            raise RuntimeError(f"{attr} is not settable.")
        super()._setattr_hook(attr, value)

    def _getattr_hook(self, attr):
        if attr in self._alchemical_dofs:
            return self._alchemical_params[attr]
        return super()._getattr_hook(attr)


class AlchemicalDOF(_HOOMDBaseObject):
    """Alchemical degree of freedom :math:`\\alpha_i` associated with a\
    specific pair force.

    `AlchemicalDOF` represents an alchemical degree of freedom to be
    numerically integrated via an `alchemical integration method
    <hoomd.md.alchemy.methods>`.

    Note:
        To access an alchemical particle for a particular degree of freedom
        query the corresponding attribute in the class instance with alchemical
        degrees of freedom.

    Attributes:
        mass (float): The mass of the alchemical degree of freedom.
        mu (float): The value of the alchemical potential.
        alpha (float): The value of the dimensionless alchemical degree of
            freedom :math:`\\alpha_i`.

        alchemical_momentum (float): The momentum of the alchemical parameter.

    """

    def __init__(self,
                 force: _AlchemicalPairForce,
                 name: str = '',
                 typepair: tuple = None,
                 alpha: float = 1.0,
                 mass: float = 1.0,
                 mu: float = 0.0):
        """Cache existing instances of AlchemicalDOF.

        Args:
            force (``_AlchemicalPairForce``): Pair force containing the
                alchemical degree of freedom.
            name (str): The name of the pair force.
            typepair (tuple[str]): The particle types upon which the pair force
                acts.
            alpha (float): The value of the alchemical parameter.
            mass (float): The mass of the alchemical degree of freedom.
            mu (float): The alchemical potential.

        """
        self._force = force
        self.name = name
        self.typepair = typepair
        if self._force._attached:
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
        if not self._force._attached:
            raise RuntimeError("Call Simulation.run(0) before attaching "
                               "alchemical degrees of freedom.")
        self._cpp_obj = self._force._cpp_obj.getAlchemicalPairParticle(
            self.typepair, self.name)
        self._force._cpp_obj.enableAlchemicalPairParticle(self._cpp_obj)

    def _add(self, simulation):
        super()._add(simulation)

    def _detach(self):
        if self._attached:
            self._force.params[self.typepair][self.name] = self.value
            self._force._cpp_obj.disableAlchemicalPairParticle(self._cpp_obj)
            super()._detach()

    @log(requires_run=True)
    def value(self):
        """Current value of alpha multiplied by its corresponding parameter."""
        return self._force.params[self.typepair][self.name] * (
            self._cpp_obj.alpha)

    @log(default=False, requires_run=True, category='particle')
    def alchemical_forces(self):
        r"""Per particle forces in alchemical alpha space.

        .. math::

            F_{\mathrm{alchemical},i} = -\frac{\mathrm{d}U_i}{\mathrm{d}\alpha}

        """
        return self._cpp_obj._forces

    @log(requires_run=True)
    def net_alchemical_force(self):
        """Net force in alchemical alpha space.

        .. math::
            F_{\\mathrm{alchemical}} = \\sum_{i=0}^{N_{\\mathrm{paricles}}-1}
            F_{\\mathrm{alchemical},i}

        """
        return self._cpp_obj.net_force


# hiding this class from the sphinx docs until we figure out how the
# normalization scheme works
class AlchemicalNormalizedDOF(AlchemicalDOF):
    """Alchemical normalized degree of freedom."""

    def __init__(self,
                 force: _AlchemicalPairForce,
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
        return self._cpp_obj._forces * self._cpp_obj.norm_value


@_modify_pair_cls_to_alchemical
class LJGauss(BaseLJGauss, _AlchemicalPairForce):
    r"""Alchemical Lennard Jones Gauss pair force.

    Args:
        nlist (`hoomd.md.nlist.NeighborList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJGauss` computes the Lennard-Jones Gauss force on all particles in the
    simulation state, with additional alchemical degrees of freedom:

    .. math::
        U(r) = 1\ [\mathrm{energy}] \cdot \left[
                 \left ( \frac{1\ [\mathrm{length}]}{r} \right)^{12} -
                 \left ( \frac{2\ [\mathrm{length}]}{r} \right)^{6} \right] -
            \alpha_{1}\epsilon
            \exp \left[ - \frac{\left(r - \alpha_{2}r_{0}\right)^{2}}{2
            (\alpha_{3}\sigma)^{2}} \right],

    where :math:`\alpha_i` are the alchemical degrees of freedom.

    Note:
        :math:`\alpha_i` not accessed are set to 1.

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
    _alchemical_dofs = ['epsilon', 'sigma2', 'r0']

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalPairForce.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)


# hiding this class from the sphinx docs until we figure out how the
# normalization scheme works
@_modify_pair_cls_to_alchemical
class _NLJGauss(BaseLJGauss, _AlchemicalPairForce):
    """Alchemical normalized Lennard Jones Gauss pair force.

    Attention:
        `hoomd.md.alchemy.pair._NLJGauss` does not support execution on GPUs.

    Attention:
        `hoomd.md.alchemy.pair._NLJGauss` does not support MPI parallel
        simulations.

    Attention:
        `hoomd.md.alchemy.pair._NLJGauss` is only valid for systems that contain
        a single particle type with a single pair force.

    """
    _alchemical_dofs = ['epsilon', 'sigma2', 'r0']
    normalized = True

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        _AlchemicalPairForce.__init__(self)
        super().__init__(nlist, default_r_cut, default_r_on, mode)
