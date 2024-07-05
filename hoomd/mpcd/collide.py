# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD collision methods

An MPCD collision method is required to update the particle velocities over time.
It is meant to be used in conjunction with an :py:class:`~hoomd.mpcd.integrator`
and streaming method (see :py:mod:`~hoomd.mpcd.stream`). Particles are binned into
cells based on their positions, and all particles in a cell undergo a stochastic
collision that updates their velocities while conserving linear momentum. Collision
rules can optionally be extended to also enforce angular-momentum conservation.
The stochastic collision lead to a build up of hydrodynamic interactions, and the
choice of collision rule and solvent properties determine the transport coefficients.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd
import numpy as np


class _collision_method():
    """ Base collision method

    Args:
        seed (int): Seed to the collision method random number generator (must be positive)
        period (int): Number of integration steps between collisions

    This class is not intended to be initialized directly by the user. Instead,
    initialize a specific collision method directly. It is included in the documentation
    to supply signatures for common methods.

    """

    def __init__(self, seed, period):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            raise RuntimeError(
                'mpcd.collide: system must be initialized before collision method\n'
            )

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.collide: an MPCD system must be initialized before the collision method\n'
            )
            raise RuntimeError('MPCD system not initialized')

        # check for multiple collision rule initializations
        if hoomd.context.current.mpcd._collide is not None:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.collide: only one collision method can be created.\n')
            raise RuntimeError('Multiple initialization of collision method')

        self.period = period
        self.seed = seed
        self.group = None
        self.shift = True
        self.enabled = True
        self._cpp = None

        self.enable()

    def embed(self, group):
        """ Embed a particle group into the MPCD collision

        Args:
            group (``hoomd.group``): Group of particles to embed

        The *group* is embedded into the MPCD collision step and cell properties.
        During collisions, the embedded particles are included in determining
        per-cell quantities, and the collisions are applied to the embedded
        particles.

        No integrator is generated for *group*. Usually, you will need to create
        a separate method to integrate the embedded particles. The recommended
        (and most common) integrator to use is :py:class:`~hoomd.md.methods.NVE`.
        It is generally **not** a good idea to use a thermostatting integrator for
        the embedded particles, since the MPCD particles themselves already act
        as a heat bath that will thermalize the embedded particles.

        Examples::

            polymer = hoomd.group.type('P')
            md.integrate.nve(group=polymer)
            method.embed(polymer)

        """

        self.group = group
        self._cpp.setEmbeddedGroup(group.cpp_group)

    def enable(self):
        """ Enable the collision method

        Examples::

            method.enable()

        Enabling the collision method adds it to the current MPCD system definition.
        Only one collision method can be attached to the system at any time.
        If another method is already set, ``disable`` must be called
        first before switching.

        """

        self.enabled = True
        hoomd.context.current.mpcd._collide = self

    def disable(self):
        """ Disable the collision method

        Examples::

            method.disable()

        Disabling the collision method removes it from the current MPCD system definition.
        Only one collision method can be attached to the system at any time, so
        use this method to remove the current collision method before adding another.

        """

        self.enabled = False
        hoomd.context.current.mpcd._collide = None

    def set_period(self, period):
        """ Set the collision period.

        Args:
            period (int): New collision period.

        The MPCD collision period can only be changed to a new value on a
        simulation timestep that is a multiple of both the previous *period*
        and the new *period*. An error will be raised if it is not.

        Examples::

            # The initial period is 5.
            # The period can be updated to 2 on step 10.
            hoomd.run_upto(10)
            method.set_period(period=2)

            # The period can be updated to 4 on step 12.
            hoomd.run_upto(12)
            hoomd.set_period(period=4)

        """

        cur_tstep = hoomd.context.current.system.getCurrentTimeStep()
        if cur_tstep % self.period != 0 or cur_tstep % period != 0:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.collide: collision period can only be changed on multiple of current and new period.\n'
            )
            raise RuntimeError(
                'collision period can only be changed on multiple of current and new period'
            )

        self._cpp.setPeriod(cur_tstep, period)
        self.period = period


class at(_collision_method):
    r""" Andersen thermostat method

    Args:
        seed (int): Seed to the collision method random number generator (must be positive)
        period (int): Number of integration steps between collisions
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature set
            point for the thermostat (in energy units).
        group (``hoomd.group``): Group of particles to embed in collisions

    This class implements the Andersen thermostat collision rule for MPCD, as described
    by `Allahyarov and Gompper <https://doi.org/10.1103/PhysRevE.66.036702>`_.
    Every *period* steps, the particles are binned into cells. The size of the cell
    can be selected as a property of the MPCD system (see :py:meth:`.data.system.set_params`).
    New particle velocities are then randomly drawn from a Gaussian distribution
    (using *seed*) relative to the center-of-mass velocity for the cell. The random
    velocities are given zero-mean so that the cell momentum is conserved. This
    collision rule naturally imparts the canonical (NVT) ensemble consistent
    with *kT*. The properties of the AT fluid are tuned using *period*, *kT*, the
    underlying size of the MPCD cell list, and the particle density.

    Note:
        The *period* must be chosen as a multiple of the MPCD
        :py:mod:`~hoomd.mpcd.stream` period. Other values will result in an
        error when ```hoomd.run``` is called.

    When the total mean-free path of the MPCD particles is small, the underlying
    MPCD cell list must be randomly shifted in order to ensure Galilean
    invariance. Because the performance penalty from grid shifting is small,
    shifting is enabled by default in all simulations. Disable it using
    :py:meth:`set_params()` if you are sure that you do not want to use it.

    HOOMD particles in *group* can be embedded into the collision step (see
    ``embed``). A separate integration method (:py:mod:`~hoomd.md.methods`)
    must be specified in order to integrate the positions of particles in *group*.
    The recommended integrator is :py:class:`~hoomd.md.methods.NVE`.

    Examples::

        collide.at(seed=42, period=1, kT=1.0)
        collide.at(seed=77, period=50, kT=1.5, group=hoomd.group.all())

    """

    def __init__(self, seed, period, kT, group=None):

        _collision_method.__init__(self, seed, period)
        self.kT = hoomd.variant._setup_variant_input(kT)

        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            collide_class = _mpcd.ATCollisionMethod
            thermo_class = _mpcd.CellThermoCompute
        else:
            collide_class = _mpcd.ATCollisionMethodGPU
            thermo_class = _mpcd.CellThermoComputeGPU

        # create an auxiliary thermo compute and disable logging on it
        if hoomd.context.current.mpcd._at_thermo is None:
            rand_thermo = thermo_class(hoomd.context.current.mpcd.data)
            hoomd.context.current.system.addCompute(rand_thermo,
                                                    "mpcd_at_thermo")
            hoomd.context.current.mpcd._at_thermo = rand_thermo

        self._cpp = collide_class(
            hoomd.context.current.mpcd.data,
            hoomd.context.current.system.getCurrentTimeStep(), self.period, 0,
            self.seed, hoomd.context.current.mpcd._thermo,
            hoomd.context.current.mpcd._at_thermo, self.kT.cpp_variant)

        if group is not None:
            self.embed(group)

    def set_params(self, shift=None, kT=None):
        """ Set parameters for the SRD collision method

        Args:
            shift (bool): If True, perform a random shift of the underlying cell list.
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature
                set point for the thermostat (in energy units).

        Examples::

            srd.set_params(shift=False)
            srd.set_params(shift=True, kT=1.0)
            srd.set_params(kT=hoomd.data.variant.linear_interp([[0,1.0],[100,5.0]]))

        """

        if shift is not None:
            self.shift = shift
            self._cpp.enableGridShifting(shift)
        if kT is not None:
            self.kT = hoomd.variant._setup_variant_input(kT)
            self._cpp.setTemperature(self.kT.cpp_variant)


class srd(_collision_method):
    r""" Stochastic rotation dynamics method

    Args:
        seed (int): Seed to the collision method random number generator (must be positive)
        period (int): Number of integration steps between collisions
        angle (float): SRD rotation angle (degrees)
        kT (:py:mod:`hoomd.variant` or :py:obj:`float` or bool): Temperature set
            point for the thermostat (in energy units). If False (default), no
            thermostat is applied and an NVE simulation is run.
        group (``hoomd.group``): Group of particles to embed in collisions

    This class implements the classic stochastic rotation dynamics collision
    rule for MPCD as first proposed by `Malevanets and Kapral <http://dx.doi.org/10.1063/1.478857>`_.
    Every *period* steps, the particles are binned into cells. The size of the cell
    can be selected as a property of the MPCD system (see :py:meth:`.data.system.set_params`).
    The particle velocities are then rotated by *angle* around an
    axis randomly drawn from the unit sphere. The rotation is done relative to
    the average velocity, so this rotation rule conserves momentum and energy
    within each cell, and so also globally. The properties of the SRD fluid
    are tuned using *period*, *angle*, *kT*, the underlying size of the MPCD
    cell list, and the particle density.

    Note:
        The *period* must be chosen as a multiple of the MPCD
        :py:mod:`~hoomd.mpcd.stream` period. Other values will
        result in an error when ```hoomd.run``` is called.

    When the total mean-free path of the MPCD particles is small, the underlying
    MPCD cell list must be randomly shifted in order to ensure Galilean
    invariance. Because the performance penalty from grid shifting is small,
    shifting is enabled by default in all simulations. Disable it using
    :py:meth:`set_params()` if you are sure that you do not want to use it.

    HOOMD particles in *group* can be embedded into the collision step (see
    ``embed()``). A separate integration method (:py:mod:`~hoomd.md.methods`)
    must be specified in order to integrate the positions of particles in *group*.
    The recommended integrator is :py:class:`~hoomd.md.methods.NVE`.

    The SRD method naturally imparts the NVE ensemble to the system comprising
    the MPCD particles and *group*. Accordingly, the system must be properly
    initialized to the correct temperature. (SRD has an H theorem, and so
    particles exchange momentum to reach an equilibrium temperature.) A thermostat
    can be applied in conjunction with the SRD method through the *kT* parameter.
    SRD employs a `Maxwell-Boltzmann thermostat <https://doi.org/10.1016/j.jcp.2009.09.024>`_
    on the cell level, which generates the (correct) isothermal ensemble. The
    temperature is defined relative to the cell-average velocity, and so can be
    used to dissipate heat in nonequilibrium simulations. Under this thermostat, the
    SRD algorithm still conserves momentum, but energy is of course no longer conserved.

    Examples::

        collide.srd(seed=42, period=1, angle=130.)
        collide.srd(seed=77, period=50, angle=130., group=hoomd.group.all())
        collide.srd(seed=1991, period=10, angle=90., kT=1.5)

    """

    def __init__(self, seed, period, angle, kT=False, group=None):

        _collision_method.__init__(self, seed, period)

        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            collide_class = _mpcd.SRDCollisionMethod
        else:
            collide_class = _mpcd.SRDCollisionMethodGPU
        self._cpp = collide_class(
            hoomd.context.current.mpcd.data,
            hoomd.context.current.system.getCurrentTimeStep(), self.period, 0,
            self.seed, hoomd.context.current.mpcd._thermo)

        self.set_params(angle=angle, kT=kT)
        if group is not None:
            self.embed(group)

    def set_params(self, angle=None, shift=None, kT=None):
        """ Set parameters for the SRD collision method

        Args:
            angle (float): SRD rotation angle (degrees)
            shift (bool): If True, perform a random shift of the underlying cell list
            kT (:py:mod:`hoomd.variant` or :py:obj:`float` or bool): Temperature
                set point for the thermostat (in energy units). If False, any
                set thermostat is removed and an NVE simulation is run.

        Examples::

            srd.set_params(angle=90.)
            srd.set_params(shift=False)
            srd.set_params(angle=130., shift=True, kT=1.0)
            srd.set_params(kT=hoomd.data.variant.linear_interp([[0,1.0],[100,5.0]]))
            srd.set_params(kT=False)

        """

        if angle is not None:
            self.angle = angle
            self._cpp.setRotationAngle(angle * np.pi / 180.)
        if shift is not None:
            self.shift = shift
            self._cpp.enableGridShifting(shift)
        if kT is not None:
            if kT is False:
                self._cpp.unsetTemperature()
                self.kT = kT
            else:
                self.kT = hoomd.variant._setup_variant_input(kT)
                self._cpp.setTemperature(self.kT.cpp_variant)
