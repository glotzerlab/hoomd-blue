# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Energy minimizer."""

import hoomd

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data import syncedlist
from hoomd.data.typeconverter import OnlyTypes, positive_real
from hoomd.logging import log
from hoomd.md import _md
from hoomd.md.integrate import _DynamicIntegrator


class FIRE(_DynamicIntegrator):
    """Energy Minimizer (FIRE).

    Args:
        dt (float):
            This is the maximum step size the minimizer is permitted to use
            :math:`[\\mathrm{time}]`. Consider the stability of the system when
            setting.
        integrate_rotational_dof (bool): When True, integrate rotational degrees
            of freedom.
        forces (Sequence[hoomd.md.force.Force]):
            Sequence of forces applied to the particles in the system. All the
            forces are summed together. The default value of ``None``
            initializes an empty list.
        methods (Sequence[hoomd.md.methods.Method]):
            Sequence of integration methods. Each integration method can be
            applied to only a specific subset of particles. The intersection of
            the subsets must be null. The default value of ``None`` initializes
            an empty list.
        constraints (Sequence[hoomd.md.constrain.Constraint]):
            Sequence of constraint forces applied to the particles in the
            system. The default value of ``None`` initializes an empty list.
            Rigid body objects (i.e. `hoomd.md.constrain.Rigid`) are not
            allowed in the list.
        rigid (hoomd.md.constrain.Rigid):
            A rigid bodies object defining the rigid bodies in the simulation.
        min_steps_adapt (int):
            Number of steps energy change is negative before allowing
            :math:`\\alpha` and :math:`\\delta t` to adapt.
        finc_dt (float):
            Factor to increase :math:`\\delta t` by
            :math:`[\\mathrm{dimensionless}]`.
        fdec_dt (float):
            Factor to decrease :math:`\\delta t` by
            :math:`[\\mathrm{dimensionless}]`.
        alpha_start (float):
            Initial (and maximum) :math:`\\alpha [\\mathrm{dimensionless}]`.
        fdec_alpha (float):
            Factor to decrease :math:`\\alpha t` by
            :math:`[\\mathrm{dimensionless}]`.
        force_tol (float):
            Force convergence criteria
            :math:`[\\mathrm{force} / \\mathrm{mass}]`.
        angmom_tol (float):
            Angular momentum convergence criteria
            :math:`[\\mathrm{energy} * \\mathrm{time}]`.
        energy_tol (float):
            Energy convergence criteria :math:`[\\mathrm{energy}]`.
        min_steps_conv (int):
            A minimum number of attempts before convergence criteria are
            considered.

    `FIRE` is a `hoomd.md.Integrator` that uses the Fast Inertial Relaxation
    Engine (FIRE) algorithm to minimize the potential energy for a group of
    particles while keeping all other particles fixed. This method is published
    in `Bitzek, et. al., PRL, 2006
    <http://dx.doi.org/10.1103/PhysRevLett.97.170201>`_. HOOMD-blue's
    implementation extends the original formulation to include rotational
    degrees of freedom.

    At each time step, :math:`\\delta t`, the algorithm uses the supplied
    integration methods to generate :math:`\\vec{r}`, :math:`\\vec{v}`, and
    :math:`\\vec{F}`, and then adjusts :math:`\\vec{v}` according
    to

    .. math::

        \\vec{v} = (1-\\alpha)\\vec{v} + \\alpha \\hat{F}|\\vec{v}|

    where :math:`\\alpha` and :math:`\\delta t` are dynamically adaptive
    quantities.  While a current search has been lowering the energy of system
    for more than :math:`N_{min}` steps, :math:`\\alpha` is decreased by
    :math:`\\alpha \\rightarrow \\alpha \\cdot \\mathrm{fdec}_{\\alpha}` and
    :math:`\\delta t` is increased by :math:`\\delta t \\rightarrow
    \\max(\\delta t \\cdot \\mathrm{finc}_{dt}, \\ \\delta t_{max})`. If the
    energy of the system increases (or stays the same), the velocity of the
    particles is set to 0, :math:`\\alpha \\rightarrow \\ \\alpha_{start}` and
    :math:`\\delta t \\rightarrow \\delta t \\cdot \\mathrm{fdec}_{\\alpha}`.
    The method converges when the force per particle is below `force_tol`, the
    angular momentum is below `angmom_tol` and the change in potential energy
    from one step to the next is below `energy_tol`:

    .. math::

        \\frac{\\sum |F|}{N*\\sqrt{N_{dof}}} < \\mathrm{\\text{force_tol}}
        \\;\\;, \\;\\ \\Delta \\frac{\\sum|E|}{N} <
        \\mathrm{\\text{energy_tol}} \\;\\;, and \\;\\ \\frac{\\sum|L|}{N} <
        \\mathrm{\\text{angmom_tol}}

    where :math:`N_{\\mathrm{dof}}` is the number of degrees of freedom the
    minimization is acting over. Any of the criterion can be effectively
    disabled by setting the tolerance to a large number.

    If the minimization acts on a subset of all the particles in the
    system, the other particles will be kept frozen but will still interact
    with the particles being moved.

    Examples::

        fire = md.minimize.FIRE(dt=0.05,
                                force_tol=1e-2,
                                angmom_tol=1e-2,
                                energy_tol=1e-7)
        fire.methods.append(md.methods.ConstantVolume(hoomd.filter.All()))
        sim.operations.integrator = fire
        while not(fire.converged):
           sim.run(100)

        fire = md.minimize.FIRE(dt=0.05)
        fire.methods.append(md.methods.ConstantPressure(
            hoomd.filter.All(), S=1, tauS=1, couple='none'))
        sim.operations.integrator = fire
        while not(fire.converged):
           sim.run(100)

    Note:
        To use `FIRE`, set it as the simulation's integrator in place of the
        typical `hoomd.md.Integrator`.

    Note:
        The algorithm requires an integration method to update the particle
        position and velocities. This should either be either
        `hoomd.md.methods.ConstantVolume` (to minimize energy) or
        `hoomd.md.methods.ConstantPressure` (to minimize energy and relax
        the box), without setting any thermostat. The quantity minimized is in
        any case the potential energy (not the enthalpy or any other quantity).

    Note:
        In practice, the default parameters prevents the simulation from making
        too aggressive a first step, but also from quitting before having found
        a good search direction. Adjust the parameters as needed for your
        simulations.

    Attributes:
        dt (float):
            This is the maximum step size the minimizer is permitted to use
            :math:`[\\mathrm{time}]`. Consider the stability of the system when
            setting.
        force_tol (float):
            Force convergence criteria
            :math:`[\\mathrm{force} / \\mathrm{mass}]`.
        angmom_tol (float):
            Angular momentum convergence criteria
            :math:`[\\mathrm{energy} * \\mathrm{time}]`.
        energy_tol (float):
            Energy convergence criteria :math:`[\\mathrm{energy}]`.
        integrate_rotational_dof (bool): When True, integrate rotational degrees
            of freedom.
        forces (Sequence[hoomd.md.force.Force]):
            Sequence of forces applied to the particles in the system. All the
            forces are summed together. The default value of ``None``
            initializes an empty list.
        methods (Sequence[hoomd.md.methods.Method]):
            Sequence of integration methods. Each integration method can be
            applied to only a specific subset of particles. The intersection of
            the subsets must be null. The default value of ``None`` initializes
            an empty list.
        constraints (Sequence[hoomd.md.constrain.Constraint]):
            Sequence of constraint forces applied to the particles in the
            system. The default value of ``None`` initializes an empty list.
            Rigid body objects (i.e. `hoomd.md.constrain.Rigid`) are not
            allowed in the list.
        rigid (hoomd.md.constrain.Rigid):
            A rigid bodies object defining the rigid bodies in the simulation.
        min_steps_adapt (int):
            Number of steps energy change is negative before allowing
            :math:`\\alpha` and :math:`\\delta t` to adapt.
        finc_dt (float):
            Factor to increase :math:`\\delta t` by
            :math:`[\\mathrm{dimensionless}]`.
        fdec_dt (float):
            Factor to decrease :math:`\\delta t` by
            :math:`[\\mathrm{dimensionless}]`.
        alpha_start (float):
            Initial (and maximum) :math:`\\alpha [\\mathrm{dimensionless}]`.
        fdec_alpha (float):
            Factor to decrease :math:`\\alpha t` by
            :math:`[\\mathrm{dimensionless}]`.
        min_steps_conv (int):
            A minimum number of attempts before convergence criteria are
            considered.

    """
    _cpp_class_name = "FIREEnergyMinimizer"

    def __init__(self,
                 dt,
                 force_tol,
                 angmom_tol,
                 energy_tol,
                 integrate_rotational_dof=False,
                 forces=None,
                 constraints=None,
                 methods=None,
                 rigid=None,
                 min_steps_adapt=5,
                 finc_dt=1.1,
                 fdec_dt=0.5,
                 alpha_start=0.1,
                 fdec_alpha=0.99,
                 min_steps_conv=10):

        super().__init__(forces, constraints, methods, rigid)

        pdict = ParameterDict(
            dt=float(dt),
            integrate_rotational_dof=bool(integrate_rotational_dof),
            min_steps_adapt=OnlyTypes(int, preprocess=positive_real),
            finc_dt=float(finc_dt),
            fdec_dt=float(fdec_dt),
            alpha_start=float(alpha_start),
            fdec_alpha=float(fdec_alpha),
            force_tol=float(force_tol),
            angmom_tol=float(angmom_tol),
            energy_tol=float(energy_tol),
            min_steps_conv=OnlyTypes(int, preprocess=positive_real),
            _defaults={
                'min_steps_adapt': 5,
                'min_steps_conv': 10
            })

        self._param_dict.update(pdict)

        # set these values explicitly so they can be validated
        self.min_steps_adapt = min_steps_adapt
        self.min_steps_conv = min_steps_conv

        # have to remove methods from old syncedlist so new syncedlist doesn't
        # think members are attached to multiple syncedlists
        self._methods.clear()

        methods_list = syncedlist.SyncedList(
            OnlyTypes((hoomd.md.methods.ConstantVolume,
                       hoomd.md.methods.ConstantPressure,
                       hoomd.md.methods.rattle.NVE)),
            syncedlist._PartialGetAttr("_cpp_obj"),
            iterable=methods)
        self._methods = methods_list

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def, self.dt)
        super()._attach_hook()

    @log(requires_run=True)
    def energy(self):
        """float: Get the energy after the last iteration of the minimizer."""
        return self._cpp_obj.energy

    @log(default=False)
    def converged(self):
        """bool: True when the minimizer has converged, else False."""
        if not self._attached:
            return False

        return self._cpp_obj.converged

    def reset(self):
        """Reset the minimizer to its initial state."""
        return self._cpp_obj.reset()
