# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD streaming methods

An MPCD streaming method is required to update the particle positions over time.
It is meant to be used in conjunction with an :py:class:`~hoomd.mpcd.integrator`
and collision method (see :py:mod:`~hoomd.mpcd.collide`). Particle positions are
propagated ballistically according to Newton's equations using a velocity-Verlet
scheme for a time :math:`\Delta t`:

.. math::

    \mathbf{v}(t + \Delta t/2) &= \mathbf{v}(t) + (\mathbf{f}/m)(\Delta t / 2)

    \mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t+\Delta t/2) \Delta t

    \mathbf{v}(t + \Delta t) &= \mathbf{v}(t + \Delta t/2) + (\mathbf{f}/m)(\Delta t / 2)

where **r** and **v** are the particle position and velocity, respectively, and **f**
is the external force acting on the particles of mass *m*. For a list of forces
that can be applied, see :py:mod:`.mpcd.force`.

Since one of the main strengths of the MPCD algorithm is that it can be coupled to
complex boundaries, the streaming geometry can be configured. MPCD solvent particles
will be reflected from boundary surfaces using specular reflections (bounce-back)
rules consistent with either "slip" or "no-slip" hydrodynamic boundary conditions.
(The external force is only applied to the particles at the beginning and the end
of this process.) To help fully enforce the boundary conditions, "virtual" MPCD
particles can be inserted near the boundary walls.

Although a streaming geometry is enforced on the MPCD solvent particles, there are
a few important caveats:

    1. Embedded particles are not coupled to the boundary walls. They must be confined
       by an appropriate method, e.g., an external potential, an explicit particle wall,
       or a bounce-back method (see :py:mod:`.mpcd.integrate`).
    2. The confined geometry exists inside a fully periodic simulation box. Hence, the
       box must be padded large enough that the MPCD cells do not interact through the
       periodic boundary. Usually, this means adding at least one extra layer of cells
       in the confined dimensions. Your periodic simulation box will be validated by
       the confined geometry.
    3. It is an error for MPCD particles to lie "outside" the confined geometry. You
       must initialize your system carefully using the snapshot interface to ensure
       all particles are "inside" the geometry. An error will be raised otherwise.

"""

import hoomd
from hoomd import _hoomd

from . import _mpcd


class _streaming_method:
    """Base streaming method

    Args:
        period (int): Number of integration steps between streaming step

    This class is not intended to be initialized directly by the user. Instead,
    initialize a specific streaming method directly. It is included in the documentation
    to supply signatures for common methods.

    """

    def __init__(self, period):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            raise RuntimeError(
                "mpcd.stream: system must be initialized before streaming method\n"
            )

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.current.device.cpp_msg.error(
                "mpcd.stream: an MPCD system must be initialized before the streaming method\n"
            )
            raise RuntimeError("MPCD system not initialized")

        # check for multiple collision rule initializations
        if hoomd.context.current.mpcd._stream is not None:
            hoomd.context.current.device.cpp_msg.error(
                "mpcd.stream: only one streaming method can be created.\n")
            raise RuntimeError("Multiple initialization of streaming method")

        self.period = period
        self.enabled = True
        self.force = None
        self._cpp = None
        self._filler = None

        # attach the streaming method to the system
        self.enable()

    def enable(self):
        """Enable the streaming method

        Examples::

            method.enable()

        Enabling the streaming method adds it to the current MPCD system definition.
        Only one streaming method can be attached to the system at any time.
        If another method is already set, ``disable`` must be called
        first before switching. Streaming will occur when the timestep is the next
        multiple of *period*.

        """

        self.enabled = True
        hoomd.context.current.mpcd._stream = self

    def disable(self):
        """Disable the streaming method

        Examples::

            method.disable()

        Disabling the streaming method removes it from the current MPCD system definition.
        Only one streaming method can be attached to the system at any time, so
        use this method to remove the current streaming method before adding another.

        """

        self.enabled = False
        hoomd.context.current.mpcd._stream = None

    def set_period(self, period):
        """Set the streaming period.

        Args:
            period (int): New streaming period.

        The MPCD streaming period can only be changed to a new value on a
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
                "mpcd.stream: streaming period can only be changed on multiple of current and new period.\n"
            )
            raise RuntimeError(
                "Streaming period can only be changed on multiple of current and new period"
            )

        self._cpp.setPeriod(cur_tstep, period)
        self.period = period

    def set_force(self, force):
        """Set the external force field for streaming.

        Args:
            force (:py:mod:`.mpcd.force`): External force field to apply to MPCD particles.

        Setting an external *force* will generate a flow of the MPCD particles subject to the
        boundary conditions of the streaming geometry. Note that the force field should be
        chosen in a way that makes sense for the geometry (e.g., so that the box is not
        continually accelerating).

        Warning:
            The *force* applies only to the MPCD particles. If you have embedded
            particles, you should usually additionally specify a force from :py:mod:`.md.force`
            for that particle group.

        Examples::

            f = mpcd.force.constant(field=(1.0,0.0,0.0))
            streamer.set_force(f)

        """
        self.force = force
        self._cpp.setField(self.force._cpp)

    def remove_force(self):
        """Remove the external force field for streaming.

        Warning:
            This only removes the force on the MPCD particles. If you have embedded
            particles, you must separately disable any corresponding external force.

        Example::

            streamer.remove_force()

        """
        self.force = None
        self._cpp.removeField()

    def _process_boundary(self, bc):
        """Process boundary condition string into enum

        Args:
            bc (str): Boundary condition, either "no_slip" or "slip"

        Returns:
            A valid boundary condition enum.

        The enum interface is still fairly clunky for the user since the boundary
        condition is buried too deep in the package structure. This is a convenience
        method for interpreting.

        """
        if bc == "no_slip":
            return _mpcd.boundary.no_slip
        elif bc == "slip":
            return _mpcd.boundary.slip
        else:
            hoomd.context.current.device.cpp_msg.error(
                "mpcd.stream: boundary condition " + bc + " not recognized.\n")
            raise ValueError("Unrecognized streaming boundary condition")


class bulk(_streaming_method):
    """Bulk fluid streaming geometry.

    Args:
        period (int): Number of integration steps between collisions.

    :py:class:`bulk` performs the streaming step for MPCD particles in a fully
    periodic geometry (2D or 3D). This geometry is appropriate for modeling
    bulk fluids. The streaming time :math:`\\Delta t` is equal to *period* steps
    of the :py:class:`~hoomd.mpcd.integrator`. For a pure MPCD fluid,
    typically *period* should be 1. When particles are embedded in the MPCD fluid
    through the collision step, *period* should be equal to the MPCD collision
    *period* for best performance. The MPCD particle positions will be updated
    every time the simulation timestep is a multiple of *period*. This is
    equivalent to setting a *phase* of 0 using the terminology of other
    periodic :py:mod:`~hoomd.update` methods.

    Example for pure MPCD fluid::

        mpcd.integrator(dt=0.1)
        mpcd.collide.srd(seed=42, period=1, angle=130.)
        mpcd.stream.bulk(period=1)

    Example for embedded particles::

        mpcd.integrator(dt=0.01)
        mpcd.collide.srd(seed=42, period=10, angle=130., group=hoomd.group.all())
        mpcd.stream.bulk(period=10)

    """

    def __init__(self, period=1):
        _streaming_method.__init__(self, period)

        # create the base streaming class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            stream_class = _mpcd.ConfinedStreamingMethodBulk
        else:
            stream_class = _mpcd.ConfinedStreamingMethodGPUBulk
        self._cpp = stream_class(
            hoomd.context.current.mpcd.data,
            hoomd.context.current.system.getCurrentTimeStep(),
            self.period,
            0,
            _mpcd.BulkGeometry(),
        )


class slit(_streaming_method):
    r"""Parallel plate (slit) streaming geometry.

    Args:
        H (float): channel half-width
        V (float): wall speed (default: 0)
        boundary (str): boundary condition at wall ("slip" or "no_slip"")
        period (int): Number of integration steps between collisions

    The slit geometry represents a fluid confined between two infinite parallel
    plates. The slit is centered around the origin, and the walls are placed
    at :math:`z=-H` and :math:`z=+H`, so the total channel width is *2H*.
    The walls may be put into motion, moving with speeds :math:`-V` and
    :math:`+V` in the *x* direction, respectively. If combined with a
    no-slip boundary condition, this motion can be used to generate simple
    shear flow.

    The "inside" of the :py:class:`slit` is the space where :math:`|z| < H`.

    Examples::

        stream.slit(period=10, H=30.)
        stream.slit(period=1, H=25., V=0.1)

    .. versionadded:: 2.6

    """

    def __init__(self, H, V=0.0, boundary="no_slip", period=1):
        _streaming_method.__init__(self, period)

        self.H = H
        self.V = V
        self.boundary = boundary

        bc = self._process_boundary(boundary)

        # create the base streaming class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            stream_class = _mpcd.ConfinedStreamingMethodSlit
        else:
            stream_class = _mpcd.ConfinedStreamingMethodGPUSlit
        self._cpp = stream_class(
            hoomd.context.current.mpcd.data,
            hoomd.context.current.system.getCurrentTimeStep(),
            self.period,
            0,
            _mpcd.SlitGeometry(H, V, bc),
        )

    def set_filler(self, density, kT, seed, type="A"):
        r"""Add virtual particles to slit channel.

        Args:
            density (float): Density of virtual particles.
            kT (float): Temperature of virtual particles.
            seed (int): Seed to pseudo-random number generator for virtual particles.
            type (str): Type of the MPCD particles to fill with.

        The virtual particle filler draws particles within the volume *outside* the
        slit walls that could be overlapped by any cell that is partially *inside*
        the slit channel (between the parallel plates). The particles are drawn from
        the velocity distribution consistent with *kT* and with the given *density*.
        The mean of the distribution is zero in *y* and *z*, but is equal to the wall
        speed in *x*. Typically, the virtual particle density and temperature are set
        to the same conditions as the solvent.

        The virtual particles will act as a weak thermostat on the fluid, and so energy
        is no longer conserved. Momentum will also be sunk into the walls.

        Example::

            slit.set_filler(density=5.0, kT=1.0, seed=42)

        .. versionadded:: 2.6

        """

        type_id = hoomd.context.current.mpcd.particles.getTypeByName(type)
        T = hoomd.variant._setup_variant_input(kT)

        if self._filler is None:
            if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
                fill_class = _mpcd.SlitGeometryFiller
            else:
                fill_class = _mpcd.SlitGeometryFillerGPU
            self._filler = fill_class(
                hoomd.context.current.mpcd.data,
                density,
                type_id,
                T.cpp_variant,
                seed,
                self._cpp.geometry,
            )
        else:
            self._filler.setDensity(density)
            self._filler.setType(type_id)
            self._filler.setTemperature(T.cpp_variant)
            self._filler.setSeed(seed)

    def remove_filler(self):
        """Remove the virtual particle filler.

        Example::

            slit.remove_filler()

        .. versionadded:: 2.6

        """

        self._filler = None

    def set_params(self, H=None, V=None, boundary=None):
        """Set parameters for the slit geometry.

        Args:
            H (float): channel half-width
            V (float): wall speed (default: 0)
            boundary (str): boundary condition at wall ("slip" or "no_slip"")

        Changing any of these parameters will require the geometry to be
        constructed and validated, so do not change these too often.

        Examples::

            slit.set_params(H=15.0)
            slit.set_params(V=0.2, boundary="no_slip")

        .. versionadded:: 2.6

        """

        if H is not None:
            self.H = H

        if V is not None:
            self.V = V

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self._cpp.geometry = _mpcd.SlitGeometry(self.H, self.V, bc)
        if self._filler is not None:
            self._filler.setGeometry(self._cpp.geometry)


class slit_pore(_streaming_method):
    r"""Parallel plate (slit) pore streaming geometry.

    Args:
        H (float): channel half-width
        L (float): pore half-length
        boundary (str): boundary condition at wall ("slip" or "no_slip"")
        period (int): Number of integration steps between collisions

    The slit pore geometry represents a fluid partially confined between two
    parallel plates that have finite length in *x*. The slit pore is centered
    around the origin, and the walls are placed at :math:`z=-H` and
    :math:`z=+H`, so the total channel width is *2H*. They extend from
    :math:`x=-L` to :math:`x=+L` (total length *2L*), where additional solid walls
    with normals in *x* prevent penetration into the regions above / below the
    plates. The plates are infinite in *y*. Outside the pore, the simulation box
    has full periodic boundaries; it is not confined by any walls. This model
    hence mimics a narrow pore in, e.g., a membrane.

    .. image:: mpcd_slit_pore.png

    The "inside" of the :py:class:`slit_pore` is the space where
    :math:`|z| < H` for :math:`|x| < L`, and the entire space where
    :math:`|x| \ge L`.

    Examples::

        stream.slit_pore(period=10, H=30., L=10.)
        stream.slit_pore(period=1, H=25., L=25.)

    .. versionadded:: 2.7

    """

    def __init__(self, H, L, boundary="no_slip", period=1):
        _streaming_method.__init__(self, period)

        self.H = H
        self.L = L
        self.boundary = boundary

        bc = self._process_boundary(boundary)

        # create the base streaming class
        if not hoomd.context.current.device.mode == "gpu":
            stream_class = _mpcd.ConfinedStreamingMethodSlitPore
        else:
            stream_class = _mpcd.ConfinedStreamingMethodGPUSlitPore
        self._cpp = stream_class(
            hoomd.context.current.mpcd.data,
            hoomd.context.current.system.getCurrentTimeStep(),
            self.period,
            0,
            _mpcd.SlitPoreGeometry(H, L, bc),
        )

    def set_filler(self, density, kT, seed, type="A"):
        r"""Add virtual particles to slit pore.

        Args:
            density (float): Density of virtual particles.
            kT (float): Temperature of virtual particles.
            seed (int): Seed to pseudo-random number generator for virtual particles.
            type (str): Type of the MPCD particles to fill with.

        The virtual particle filler draws particles within the volume *outside* the
        slit pore boundaries that could be overlapped by any cell that is partially *inside*
        the slit pore. The particles are drawn from the velocity distribution consistent
        with *kT* and with the given *density*. The mean of the distribution is zero in
        *x*, *y*, and *z*. Typically, the virtual particle density and temperature are set
        to the same conditions as the solvent.

        The virtual particles will act as a weak thermostat on the fluid, and so energy
        is no longer conserved. Momentum will also be sunk into the walls.

        Example::

            slit_pore.set_filler(density=5.0, kT=1.0, seed=42)

        """
        type_id = hoomd.context.current.mpcd.particles.getTypeByName(type)
        T = hoomd.variant._setup_variant_input(kT)

        if self._filler is None:
            if not hoomd.context.current.device.mode == "gpu":
                fill_class = _mpcd.SlitPoreGeometryFiller
            else:
                fill_class = _mpcd.SlitPoreGeometryFillerGPU
            self._filler = fill_class(
                hoomd.context.current.mpcd.data,
                density,
                type_id,
                T.cpp_variant,
                seed,
                self._cpp.geometry,
            )
        else:
            self._filler.setDensity(density)
            self._filler.setType(type_id)
            self._filler.setTemperature(T.cpp_variant)
            self._filler.setSeed(seed)

    def remove_filler(self):
        """Remove the virtual particle filler.

        Example::

            slit_pore.remove_filler()

        """

        self._filler = None

    def set_params(self, H=None, L=None, boundary=None):
        """Set parameters for the slit geometry.

        Args:
            H (float): channel half-width
            L (float): pore half-length
            boundary (str): boundary condition at wall ("slip" or "no_slip"")

        Changing any of these parameters will require the geometry to be
        constructed and validated, so do not change these too often.

        Examples::

            slit_pore.set_params(H=15.0)
            slit_pore.set_params(L=10.0, boundary="no_slip")

        """

        if H is not None:
            self.H = H

        if L is not None:
            self.L = L

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self._cpp.geometry = _mpcd.SlitPoreGeometry(self.H, self.L, bc)
        if self._filler is not None:
            self._filler.setGeometry(self._cpp.geometry)
