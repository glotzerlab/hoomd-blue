# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MD integration methods with manifold constraints.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.001)
    logger = hoomd.logging.Logger()

    # Rename pytest's tmp_path fixture for clarity in the documentation.
    path = tmp_path

"""

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

    def _attach_constraint(self, sim):
        self.manifold_constraint._attach(sim)

    def _setattr_param(self, attr, value):
        if attr == "manifold_constraint":
            raise AttributeError(
                "Cannot set manifold_constraint after construction.")
        super()._setattr_param(attr, value)


class NVE(MethodRATTLE):
    r"""NVE Integration via Velocity-Verlet with RATTLE constraint.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold
            constraint.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function. The
            units of tolerance match that of the selected manifold's implicit
            function. Defaults to 1e-6

    `NVE` performs constant volume, constant energy simulations as described
    in `hoomd.md.methods.ConstantVolume` without any thermostat. In addition
    the particles are constrained to a manifold by using the RATTLE algorithm.

    .. rubric:: Example

    .. code-block:: python

        sphere = hoomd.md.manifold.Sphere(r=5)
        nve_rattle = hoomd.md.methods.rattle.NVE(
            filter=hoomd.filter.All(),
            manifold_constraint=sphere)
        simulation.operations.integrator.methods = [nve_rattle]


    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function. The
            units of tolerance match that of the selected manifold's implicit
            function. Defaults to 1e-6

    """

    def __init__(self, filter, manifold_constraint, tolerance=0.000001):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            zero_force=OnlyTypes(bool, allow_none=False),
        )
        param_dict.update(dict(filter=filter, zero_force=False))

        # set defaults
        self._param_dict.update(param_dict)

        super().__init__(manifold_constraint, tolerance)

    def _attach_hook(self):
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
                                 self.manifold_constraint._cpp_obj,
                                 self.tolerance)


class DisplacementCapped(NVE):
    r"""Newtonian dynamics with a cap on the maximum displacement per time step.

    Integration is via a maximum displacement capped Velocity-Verlet with
    RATTLE constraint. This class is useful to relax a simulation on a manifold.

    Warning:
        This method does not conserve energy or momentum.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.
        maximum_displacement (hoomd.variant.variant_like): The maximum
            displacement allowed for a particular timestep
            :math:`[\mathrm{length}]`.
        manifold_constraint (hoomd.md.manifold.Manifold): Manifold
            constraint.
        tolerance (`float`, optional): Defines the tolerated error particles are
            allowed to deviate from the manifold in terms of the implicit
            function. The units of tolerance match that of the selected
            manifold's implicit function. Defaults to 1e-6

    `DisplacementCapped` performs constant volume simulations as described in
    `hoomd.md.methods.DisplacementCapped`. In addition the particles are
    constrained to a manifold by using the RATTLE algorithm.

    .. rubric:: Example

    .. code-block:: python

        sphere = hoomd.md.manifold.Sphere(r=5)
        relax_rattle = hoomd.md.methods.rattle.DisplacementCapped(
            filter=hoomd.filter.All(),
            maximum_displacement=0.01,
            manifold_constraint=sphere)
        simulation.operations.integrator.methods = [relax_rattle]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.
        maximum_displacement (hoomd.variant.variant_like): The maximum
            displacement allowed for a particular timestep
            :math:`[\mathrm{length}]`.
        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.
        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function. The
            units of tolerance match that of the selected manifold's implicit
            function. Defaults to 1e-6

    """

    def __init__(self,
                 filter: hoomd.filter.filter_like,
                 maximum_displacement: hoomd.variant.variant_like,
                 manifold_constraint: "hoomd.md.manifold.Manifold",
                 tolerance: float = 1e-6):

        # store metadata
        super().__init__(filter, manifold_constraint, tolerance)
        param_dict = ParameterDict(maximum_displacement=hoomd.variant.Variant)
        param_dict["maximum_displacement"] = maximum_displacement

        # set defaults
        self._param_dict.update(param_dict)


class Langevin(MethodRATTLE):
    r"""Langevin dynamics with RATTLE constraint.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold
            constraint.

        tally_reservoir_energy (bool): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities.
            Defaults to False :math:`[\mathrm{energy}]`.

        tolerance (float): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{time}^{-1}]`.

    .. rubric:: Translational degrees of freedom

    `Langevin` uses the same integrator as `hoomd.md.methods.Langevin`, which
    follows the Langevin equations of motion with the additional force term
    :math:`- \lambda \vec{F}_\mathrm{M}`. The force :math:`\vec{F}_\mathrm{M}`
    keeps the particles on the manifold constraint, where the Lagrange
    multiplier :math:`\lambda` is calculated via the RATTLE algorithm. For
    more details about Langevin dynamics see `hoomd.md.methods.Langevin`.

    Use `Brownian` if your system is not underdamped.

    .. rubric:: Example

    .. code-block:: python

        sphere = hoomd.md.manifold.Sphere(r=5)
        langevin_rattle = hoomd.md.methods.rattle.Langevin(
            filter=hoomd.filter.All(),
            kT=1.5,
            manifold_constraint=sphere,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0))
        simulation.operations.integrator.methods = [langevin_rattle]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        tolerance (float): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{time}^{-1}]`.
    """

    def __init__(
            self,
            filter,
            kT,
            manifold_constraint,
            tally_reservoir_energy=False,
            tolerance=0.000001,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0),
    ):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            tally_reservoir_energy=bool(tally_reservoir_energy),
        )
        param_dict.update(dict(kT=kT, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        gamma_r.default = default_gamma_r

        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _attach_hook(self):
        sim = self._simulation
        # Langevin uses RNGs. Warn the user if they did not set the seed.
        sim._warn_if_seed_unset()
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


class Brownian(MethodRATTLE):
    r"""Brownian dynamics with RATTLE constraint.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold
            constraint.

        tolerance (float): Defines the tolerated error particles are allowed
            to deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{time}^{-1}]`.

    `Brownian` uses the same integrator as `hoomd.md.methods.Brownian`, which
    follows the overdamped Langevin equations of motion with the additional
    force term :math:`- \lambda \vec{F}_\mathrm{M}`. The force
    :math:`\vec{F}_\mathrm{M}` keeps the particles on the manifold constraint,
    where the Lagrange multiplier :math:`\lambda` is calculated via the RATTLE
    algorithm. For more details about Brownian dynamics see
    `hoomd.md.methods.Brownian`.

    .. rubric:: Example

    .. code-block:: python

        sphere = hoomd.md.manifold.Sphere(r=5)
        brownian_rattle = hoomd.md.methods.rattle.Brownian(
            filter=hoomd.filter.All(),
            kT=1.5,
            manifold_constraint=sphere,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0))
        simulation.operations.integrator.methods = [brownian_rattle]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function.
            The units of tolerance match that of the selected manifold's
            implicit function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{time}^{-1}]`.
    """

    def __init__(self,
                 filter,
                 kT,
                 manifold_constraint,
                 tolerance=1e-6,
                 default_gamma=1.0,
                 default_gamma_r=(1.0, 1.0, 1.0)):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
        )
        param_dict.update(dict(kT=kT, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        gamma_r.default = default_gamma_r

        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _attach_hook(self):
        sim = self._simulation
        # Brownian uses RNGs. Warn the user if they did not set the seed.
        sim._warn_if_seed_unset()
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


class OverdampedViscous(MethodRATTLE):
    r"""Overdamped viscous dynamics with RATTLE constraint.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function. The
            units of tolerance match that of the selected manifold's implicit
            function. Defaults to 1e-6

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{time}^{-1}]`.

    `OverdampedViscous` uses the same integrator as
    `hoomd.md.methods.OverdampedViscous`, with the additional force term
    :math:`- \lambda \vec{F}_\mathrm{M}`. The force :math:`\vec{F}_\mathrm{M}`
    keeps the particles on the manifold constraint, where the Lagrange
    multiplier :math:`\lambda` is calculated via the RATTLE algorithm. For more
    details about overdamped viscous dynamics see
    `hoomd.md.methods.OverdampedViscous`.

    .. rubric:: Example

    .. code-block:: python

        sphere = hoomd.md.manifold.Sphere(r=5)
        odv_rattle = hoomd.md.methods.rattle.OverdampedViscous(
            filter=hoomd.filter.All(),
            manifold_constraint=sphere,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0))
        simulation.operations.integrator.methods = [odv_rattle]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint
            which is used by and as a trigger for the RATTLE algorithm of this
            method.

        tolerance (float): Defines the tolerated error particles are allowed to
            deviate from the manifold in terms of the implicit function. The
            units of tolerance match that of the selected manifold's implicit
            function. Defaults to 1e-6

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{time}^{-1}]`.
    """

    def __init__(self,
                 filter,
                 manifold_constraint,
                 tolerance=1e-6,
                 default_gamma=1.0,
                 default_gamma_r=(1.0, 1.0, 1.0)):
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,)
        param_dict.update(dict(filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        gamma_r.default = default_gamma_r

        self._extend_typeparam([gamma, gamma_r])

        super().__init__(manifold_constraint, tolerance)

    def _attach_hook(self):
        sim = self._simulation
        # OverdampedViscous uses RNGs. Warn the user if they did not set the
        # seed.
        sim._warn_if_seed_unset()
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
