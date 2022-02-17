# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MD integration methods with manifold constraints."""

from hoomd.md import _md
import hoomd
from hoomd.md.manifold import Manifold
from hoomd.md.methods.methods import Method
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant


class MethodRATTLE(Method):
    """Base class RATTLE integration method.

    Provides common methods for all integration methods which implement the
    RATTLE algorithm to constrain particles to a manifold surface.

    Warning:
        The particles should be initialised close to the implicit surface of
        the manifold. Even though the particles are mapped to the set surface
        automatically, the mapping can lead to small inter-particle distances
        and, hence, large forces between particles!

    See Also:
    * `Paquay and Kusters 2016 <https://doi.org/10.1016/j.bpj.2016.02.017>`__

    Note:
        Users should use the subclasses and not instantiate `MethodRATTLE`
        directly.
    """

    def __init__(self, manifold_constraint, tolerance):

        param_dict = ParameterDict(manifold_constraint=OnlyTypes(
            Manifold, allow_none=False),
                                   tolerance=float(tolerance))
        param_dict['manifold_constraint'] = manifold_constraint
        # set defaults
        self._param_dict.update(param_dict)

    def _add(self, sim):
        self.manifold_constraint._add(sim)
        super()._add(sim)

    def _attach_constraint(self, sim):
        if not self.manifold_constraint._attached:
            self.manifold_constraint._attach()

    def _setattr_param(self, attr, value):
        if attr == "manifold_constraint":
            raise AttributeError(
                "Cannot set manifold_constraint after construction.")
        super()._setattr_param(attr, value)


class NVE(MethodRATTLE):
    r"""NVE Integration via Velocity-Verlet with RATTLE constraint.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
         apply this method.

        manifold_constraint (:py:mod:`hoomd.md.manifold.Manifold`): Manifold
            constraint.

        limit (None or `float`): Enforce that no particle moves more than a
            distance of a limit in a single time step. Defaults to None

        tolerance (`float`): Defines the tolerated error particles are
            allowed to deviate from the manifold in terms of the implicit
            function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

    `NVE` performs constant volume, constant energy simulations as described
    in `hoomd.md.methods.NVE`. In addition the particles are constrained to a
    manifold by using the RATTLE algorithm.

    Examples::

        sphere = hoomd.md.manifold.Sphere(r=10)
        nve_rattle = hoomd.md.methods.rattle.NVE(
            filter=hoomd.filter.All(),maifold=sphere)
        integrator = hoomd.md.Integrator(
            dt=0.005, methods=[nve_rattle], forces=[lj])


    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        limit (None or float): Enforce that no particle moves more than a
            distance of a limit in a single time step. Defaults to None

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

    """

    def __init__(self,
                 filter,
                 manifold_constraint,
                 limit=None,
                 tolerance=0.000001):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            limit=OnlyTypes(float, allow_none=True),
            zero_force=OnlyTypes(bool, allow_none=False),
        )
        param_dict.update(dict(filter=filter, limit=limit, zero_force=False))

        # set defaults
        self._param_dict.update(param_dict)

        super().__init__(manifold_constraint, tolerance)

    def _attach(self):

        self._attach_constraint(self._simulation)

        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = getattr(
                _md, 'TwoStepRATTLENVE'
                + self.manifold_constraint.__class__.__name__)
        else:
            my_class = getattr(
                _md, 'TwoStepRATTLENVE'
                + self.manifold_constraint.__class__.__name__ + 'GPU')

        self._cpp_obj = my_class(self._simulation.state._cpp_sys_def,
                                 self._simulation.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj, False,
                                 self.tolerance)

        # Attach param_dict and typeparam_dict
        super()._attach()


class Langevin(MethodRATTLE):
    r"""Langevin dynamics with RATTLE constraint.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles to
            apply this method to.

        kT (`hoomd.variant.Variant` or `float`): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (:py:mod:`hoomd.md.manifold.Manifold`): Manifold
            constraint.

        alpha (`float`): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot
            \mathrm{length}^{-1} \cdot \mathrm{time}^{-1}]`.
            Defaults to None.

        tally_reservoir_energy (`bool`): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities.
            Defaults to False :math:`[\mathrm{energy}]`.

        tolerance (`float`): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

    .. rubric:: Translational degrees of freedom

    `Langevin` uses the same integrator as `hoomd.md.methods.Langevin`, which
    follows the Langevin equations of motion with the additional force term
    :math:`- \lambda \vec{F}_\mathrm{M}`. The force :math:`\vec{F}_\mathrm{M}`
    keeps the particles on the manifold constraint, where the Lagrange
    multiplier :math:`\lambda` is calculated via the RATTLE algorithm. For
    more details about Langevin dynamics see `hoomd.md.methods.Langevin`.

    Use `Brownian` if your system is not underdamped.

    Examples::

        sphere = hoomd.md.manifold.Sphere(r=10)
        langevin_rattle = hoomd.md.methods.rattle.Langevin(
            filter=hoomd.filter.All(), kT=0.2, manifold_constraint=sphere,
            seed=1, alpha=1.0)

    Examples of using ``gamma`` or ``gamma_r`` on drag coefficient::

        sphere = hoomd.md.manifold.Sphere(r=10)
        langevin_rattle = hoomd.md.methods.rattle.Langevin(
        filter=hoomd.filter.All(), kT=0.2,
        manifold_constraint = sphere, seed=1, alpha=1.0)

    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`. Defaults to None.

        tolerance (float): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

    """

    def __init__(self,
                 filter,
                 kT,
                 manifold_constraint,
                 alpha=None,
                 tally_reservoir_energy=False,
                 tolerance=0.000001):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            alpha=OnlyTypes(float, allow_none=True),
            tally_reservoir_energy=bool(tally_reservoir_energy),
        )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))

        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _add(self, simulation):
        """Add the operation to a simulation.

        Langevin uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):

        sim = self._simulation
        self._attach_constraint(sim)

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = getattr(
                _md, 'TwoStepRATTLELangevin'
                + self.manifold_constraint.__class__.__name__)
        else:
            my_class = getattr(
                _md, 'TwoStepRATTLELangevin'
                + self.manifold_constraint.__class__.__name__ + 'GPU')

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj, self.kT,
                                 self.tolerance)

        # Attach param_dict and typeparam_dict
        super()._attach()


class Brownian(MethodRATTLE):
    r"""Brownian dynamics with RATTLE constraint.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles to
            apply this method to.

        kT (`hoomd.variant.Variant` or `float`): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (:py:mod:`hoomd.md.manifold.Manifold`): Manifold
            constraint.

        alpha (`float`): When set, use :math:`\alpha d_i` for the
            drag coefficient where :math:`d_i` is particle diameter.
            Defaults to None
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.

        tolerance (`float`): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

    `Brownian` uses the same integrator as `hoomd.md.methods.Brownian`, which
    follows the overdamped Langevin equations of motion with the additional
    force term :math:`- \lambda \vec{F}_\mathrm{M}`. The force
    :math:`\vec{F}_\mathrm{M}` keeps the particles on the manifold constraint,
    where the Lagrange multiplier :math:`\lambda` is calculated via the RATTLE
    algorithm. For more details about Brownian dynamics see
    `hoomd.md.methods.Brownian`.

    Examples of using ``manifold_constraint``::

        sphere = hoomd.md.manifold.Sphere(r=10)
        brownian_rattle = hoomd.md.methods.rattle.Brownian(
        filter=hoomd.filter.All(), kT=0.2, manifold_constraint=sphere,
        seed=1, alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[brownian_rattle],
        forces=[lj])

    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`. Defaults to None.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``, [`float`, `float`, `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{force} \cdot \mathrm{length} \cdot
            \mathrm{radian}^{-1} \cdot \mathrm{time}^{-1}]`.
    """

    def __init__(self,
                 filter,
                 kT,
                 manifold_constraint,
                 tolerance=0.000001,
                 alpha=None):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            alpha=OnlyTypes(float, allow_none=True),
        )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _add(self, simulation):
        """Add the operation to a simulation.

        Brownian uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):

        sim = self._simulation
        self._attach_constraint(sim)

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = getattr(
                _md,
                'TwoStepRATTLEBD' + self.manifold_constraint.__class__.__name__)
        else:
            my_class = getattr(
                _md, 'TwoStepRATTLEBD'
                + self.manifold_constraint.__class__.__name__ + 'GPU')

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj, self.kT,
                                 False, False, self.tolerance)

        # Attach param_dict and typeparam_dict
        super()._attach()


class OverdampedViscous(MethodRATTLE):
    r"""Overdamped viscous dynamics with RATTLE constraint.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles to
            apply this method to.

        manifold_constraint (:py:mod:`hoomd.md.manifold.Manifold`): Manifold
            constraint.

        alpha (`float`): When set, use :math:`\alpha d_i` for the
            drag coefficient where :math:`d_i` is particle diameter.
            Defaults to None
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.

        tolerance (`float`): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

    `OverdampedViscous` uses the same integrator as
    `hoomd.md.methods.OverdampedViscous`, with the additional force term
    :math:`- \lambda \vec{F}_\mathrm{M}`. The force
    :math:`\vec{F}_\mathrm{M}` keeps the particles on the manifold constraint,
    where the Lagrange multiplier :math:`\lambda` is calculated via the RATTLE
    algorithm. For more details about overdamped viscous dynamics see
    `hoomd.md.methods.OverdampedViscous`.

    Examples of using ``manifold_constraint``::

        sphere = hoomd.md.manifold.Sphere(r=10)
        odv_rattle = hoomd.md.methods.rattle.OverdampedViscous(
            filter=hoomd.filter.All(), manifold_constraint=sphere, seed=1,
            alpha=1.0)
        integrator = hoomd.md.Integrator(
            dt=0.001, methods=[odv_rattle], forces=[lj])


    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles to
            apply this method to.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`. Defaults to None.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``, [`float`, `float`, `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{force} \cdot \mathrm{length} \cdot
            \mathrm{radian}^{-1} \cdot \mathrm{time}^{-1}]`.
    """

    def __init__(self,
                 filter,
                 manifold_constraint,
                 tolerance=0.000001,
                 alpha=None):
        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            alpha=OnlyTypes(float, allow_none=True),
        )
        param_dict.update(dict(alpha=alpha, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _add(self, simulation):
        """Add the operation to a simulation.

        OverdampedViscous uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):

        sim = self._simulation
        self._attach_constraint(sim)

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = getattr(
                _md,
                'TwoStepRATTLEBD' + self.manifold_constraint.__class__.__name__)
        else:
            my_class = getattr(
                _md, 'TwoStepRATTLEBD'
                + self.manifold_constraint.__class__.__name__ + 'GPU')

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj,
                                 hoomd.variant.Constant(0.0), True, True,
                                 self.tolerance)

        # Attach param_dict and typeparam_dict
        super()._attach()
