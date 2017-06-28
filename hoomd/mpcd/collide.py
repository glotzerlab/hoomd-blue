# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD collision methods

MPCD collision rules.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd
import numpy as np

class _collision_method(hoomd.meta._metadata):
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
            hoomd.context.msg.error("mpcd.collide: system must be initialized before collision method\n")
            raise RuntimeError('System not initialized')

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.msg.error('mpcd.collide: an MPCD system must be initialized before the collision rule\n')
            raise RuntimeError('MPCD system not initialized')

        # check for multiple collision rule initializations
        if hoomd.context.current.mpcd._collide is not None:
            hoomd.context.msg.error('mpcd.collide: only one collision method can be set, use disable() first.\n')
            raise RuntimeError('Multiple initialization of collision method')

        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = ['period','seed','group','shift','enabled']

        self.period = period
        self.seed = seed
        self.group = None
        self.shift = True
        self.enabled = True
        self._cpp = None

        hoomd.util.quiet_status()
        self.enable()
        hoomd.util.unquiet_status()

    def embed(self, group):
        """ Embed a particle group into the MPCD collision

        Args:
            group (:py:mod:`hoomd.group`): Group of particles to embed

        The *group* is embedded into the MPCD collision step and cell properties.
        During collisions, the embedded particles are included in determining
        per-cell quantities, and the collisions are applied to the embedded
        particles.

        No integrator is generated for *group*. Usually, you will need to create
        a separate method to integrate the embedded particles. The recommended
        (and most common) integrator to use is :py:class:`~hoomd.md.integrate.nve`.
        It is generally **not** a good idea to use thermostatting integrator for
        the embedded particles, since the MPCD particles themselves already act
        as a heat bath that will thermalize the embedded particles.

        Note:
            The group momentum is included in any net properties reported to the
            logger. Be aware of this when computing the energy of the system.

        Examples::

            polymer = hoomd.group.type('P')
            md.integrate.nve(group=polymer)
            method.embed(polymer)

        """
        hoomd.util.print_status_line()

        self.group = group
        self._cpp.setEmbeddedGroup(group.cpp_group)

    def enable(self):
        """ Enable the collision method

        Examples::

            method.enable()

        Enabling the collision method adds it to the current MPCD system definition.
        Only one collision method can be attached to the system at any time.
        If another method is already set, :py:meth:`disable()` must be called
        first before switching.

        """
        hoomd.util.print_status_line()

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
        hoomd.util.print_status_line()

        self.enabled = False
        hoomd.context.current.mpcd._collide = None

class srd(_collision_method):
    """ Stochastic rotation dynamics method

    Args:
        seed (int): Seed to the collision method random number generator (must be positive)
        period (int): Number of integration steps between collisions
        angle (float): SRD rotation angle (degrees)
        group (:py:mod:`hoomd.group`): Group of particles to embed in collisions
        kT (:py:mod:`hoomd.variant` or :py:obj:`float` or bool): Temperature set
            point for the thermostat (in energy units). If False (default), no
            thermostat is applied and an NVE simulation is run.

    This class implements the classic stochastic rotation dynamics collision
    rule for MPCD. Every *period* steps, the particles are binned into cells
    (see XX). The particle velocities are then rotated by *angle* around an
    axis randomly drawn from the unit sphere. The rotation is done relative to
    the average velocity, so this rotation rule conserves momentum and energy
    within each cell, and so also globally.

    The properties of the SRD fluid are tuned using *period*, *angle*, the
    underlying size of the MPCD cell list (see XX), and the particle density.
    These parameters combine in a nontrivial way to set specific fluid properties.
    See XX for a discussion of how to choose these values.

    Note:
        The *period* must be chosen as a multiple of the MPCD
        :py:class:`~hoomd.mpcd.integrate.integrator` period. Other values will
        result in an error when :py:meth:`hoomd.run()` is called.

    When the total mean-free path of the MPCD particles is small, the underlying
    MPCD cell list must be randomly shifted in order to ensure Galilean
    invariance. Because the performance penalty from grid shifting is small,
    shifting is enabled by default in all simulations. Disable it using
    :py:meth:`set_params()` if you are sure that you do not want to use it.

    HOOMD particles in *group* can be embedded into the collision step. See
    :py:meth:`embed()`. A separate integration method (:py:mod:`~hoomd.md.integrate`)
    must be specified in order to integrate the positions of particles in *group*.
    The recommended integrator is :py:class:`~hoomd.md.integrate.nve`.

    The SRD method naturally imparts the NVE ensemble to the system comprising
    the MPCD particles and *group*. Accordingly, the system must be properly
    initialized to the correct temperature. (SRD has an H theorem, and so
    particles exchange momentum to reach an equilibrium temperature.) A thermostat
    can be applied in conjunction with the SRD method through the *kT* parameter.
    SRD employs a Maxwell-Boltzmann thermostat on the cell level, which generates
    the (correct) isothermal ensemble. The temperature is defined relative to the
    cell-average velocity, and so can be used to dissipate heat in nonequilibrium
    simulations. Under this thermostat, the SRD algorithm still conserves momentum,
    but energy is of course no longer conserved.

    Note:
        Setting *kT* will automatically enable the thermostat, while omitting
        it will perform an NVE simulation. Use :py:meth:`set_thermostat()` to
        enable / disable the thermostat or change the temperature setpoint
        during a simulation.

    Examples::

        collide.srd(seed=42, period=1, angle=130.)
        collide.srd(seed=77, period=50, angle=130., group=hoomd.group.all())
        collide.srd(seed=1991, period=10, angle=90., kT=1.5)

    """
    def __init__(self, seed, period, angle, group=None, kT=False):
        hoomd.util.print_status_line()

        _collision_method.__init__(self, seed, period)
        self.metadata_fields += ['angle','kT']

        if not hoomd.context.exec_conf.isCUDAEnabled():
            collide_class = _mpcd.SRDCollisionMethod
        else:
            collide_class = _mpcd.SRDCollisionMethodGPU
        self._cpp = collide_class(hoomd.context.current.mpcd.data,
                                  hoomd.context.current.system.getCurrentTimeStep(),
                                  self.period,
                                  -1,
                                  self.seed,
                                  hoomd.context.current.mpcd._thermo)

        hoomd.util.quiet_status()
        self.set_params(angle=angle, kT=kT)
        if group is not None:
            self.embed(group)
        hoomd.util.unquiet_status()

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
        hoomd.util.print_status_line()

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
