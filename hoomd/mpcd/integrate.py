# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD integration methods

Defines bounce-back methods for integrating solutes (MD particles) embedded in an MPCD
solvent. The integration scheme is velocity Verlet (NVE) with bounce-back performed at
the solid boundaries defined by a geometry, as in :py:mod:`.mpcd.stream`. This gives a
simple approximation of the interactions required to keep a solute bounded in a geometry,
and more complex interactions can be specified, for example, by writing custom external fields.

Similar caveats apply to these methods as for the :py:mod:`.mpcd.stream` methods. In particular:

    1. The simulation box is periodic, but the geometry imposes inherently non-periodic boundary
       conditions. You must ensure that the box is sufficiently large to enclose the geometry
       and that all particles lie inside it, or an error will be raised at runtime.
    2. You must also ensure that particles do not self-interact through the periodic boundaries.
       This is usually achieved for simple pair potentials by padding the box size by the largest
       cutoff radius. Failure to do so may result in unphysical interactions.
    3. Bounce-back rules do not always enforce no-slip conditions at surfaces properly. It
       may still be necessary to add additional 'ghost' MD particles in the surface
       to achieve the right boundary conditions and reduce density fluctuations.

The integration methods defined here are not restricted to only MPCD simulations: they can be
used with both ``md.integrate.mode_standard`` and :py:class:`.mpcd.integrator`. For
example, the same integration methods might be used to run DPD simulations with surfaces.

These bounce-back methods do not support anisotropic integration because torques are currently
not computed for collisions with the boundary. Similarly, rigid bodies will also not be treated
correctly because the integrators are not aware of the extent of the particles; the surface
reflections are treated as point particles. An error will be raised if an anisotropic integration
mode is specified.

"""

import hoomd
from hoomd import _hoomd

from . import _mpcd


class _bounce_back():
    """ NVE integration with bounce-back rules.

    Args:
        group (``hoomd.group``): Group of particles on which to apply this method.

    :py:class:`_bounce_back` is a base class integration method. It must be used with
    ``md.integrate.mode_standard`` or :py:class:`.mpcd.integrator`.
    Deriving classes implement the specific geometry and valid parameters for those geometries.
    Currently, there is no mechanism to share geometries between multiple instances of the same
    integration method.

    A :py:class:`hoomd.md.compute.ThermodynamicQuantities` is automatically specified and associated with *group*.

    """

    def __init__(self, group):
        # initialize base class
        # hoomd.integrate._integration_method.__init__(self)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        # store metadata
        self.group = group
        self.boundary = None
        self.metadata_fields = ['group', 'boundary']

    def _process_boundary(self, bc):
        """ Process boundary condition string into enum

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
                "mpcd.integrate: boundary condition " + bc
                + " not recognized.\n")
            raise ValueError("Unrecognized streaming boundary condition")
            return None


class slit(_bounce_back):
    """ NVE integration with bounce-back rules in a slit channel.

    Args:
        group (``hoomd.group``): Group of particles on which to apply this method.
        H (float): channel half-width
        V (float): wall speed (default: 0)
        boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

    This integration method applies to particles in *group* in the parallel-plate channel geometry.
    This method is the MD analog of :py:class:`.stream.slit`, which documents additional details
    about the geometry.

    Examples::

        all = group.all()
        slit = mpcd.integrate.slit(group=all, H=5.0)
        slit = mpcd.integrate.slit(group=all, H=10.0, V=1.0)

    .. versionadded:: 2.7

    """

    def __init__(self, group, H, V=0.0, boundary="no_slip"):
        # initialize base class
        _bounce_back.__init__(self, group)
        self.metadata_fields += ['H', 'V']

        # initialize the c++ class
        if not hoomd.context.current.device.mode == 'gpu':
            cpp_class = _mpcd.BounceBackNVESlit
        else:
            cpp_class = _mpcd.BounceBackNVESlitGPU

        self.H = H
        self.V = V
        self.boundary = boundary

        bc = self._process_boundary(boundary)
        geom = _mpcd.SlitGeometry(H, V, bc)

        self.cpp_method = cpp_class(hoomd.context.current.system_definition,
                                    group.cpp_group, geom)
        self.cpp_method.validateGroup()

    def set_params(self, H=None, V=None, boundary=None):
        """ Set parameters for the slit geometry.

        Args:
            H (float): channel half-width
            V (float): wall speed (default: 0)
            boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

        Examples::

            slit.set_params(H=8.)
            slit.set_params(V=2.0)
            slit.set_params(boundary='slip')
            slit.set_params(H=5, V=0., boundary='no_slip')

        """

        if H is not None:
            self.H = H

        if V is not None:
            self.V = V

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self.cpp_method.geometry = _mpcd.SlitGeometry(self.H, self.V, bc)


class slit_pore(_bounce_back):
    """ NVE integration with bounce-back rules in a slit pore channel.

    Args:
        group (``hoomd.group``): Group of particles on which to apply this method.
        H (float): channel half-width.
        L (float): pore half-length.
        boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

    This integration method applies to particles in *group* in the parallel-plate (slit) pore geometry.
    This method is the MD analog of :py:class:`.stream.slit_pore`, which documents additional details
    about the geometry.

    Examples::

        all = group.all()
        slit_pore = mpcd.integrate.slit_pore(group=all, H=10.0, L=10.)

    .. versionadded:: 2.7

    """

    def __init__(self, group, H, L, boundary="no_slip"):
        # initialize base class
        _bounce_back.__init__(self, group)
        self.metadata_fields += ['H', 'L']

        # initialize the c++ class
        if not hoomd.context.current.device.mode == 'gpu':
            cpp_class = _mpcd.BounceBackNVESlitPore
        else:
            cpp_class = _mpcd.BounceBackNVESlitPoreGPU

        self.H = H
        self.L = L
        self.boundary = boundary

        bc = self._process_boundary(boundary)
        geom = _mpcd.SlitPoreGeometry(H, L, bc)

        self.cpp_method = cpp_class(hoomd.context.current.system_definition,
                                    group.cpp_group, geom)
        self.cpp_method.validateGroup()

    def set_params(self, H=None, L=None, boundary=None):
        """ Set parameters for the slit pore geometry.

        Args:
            H (float): channel half-width.
            L (float): pore half-length.
            boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

        Examples::

            slit_pore.set_params(H=8.)
            slit_pore.set_params(L=2.0)
            slit_pore.set_params(boundary='slip')
            slit_pore.set_params(H=5, L=4., boundary='no_slip')

        """
        if H is not None:
            self.H = H

        if L is not None:
            self.L = L

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self.cpp_method.geometry = _mpcd.SlitPoreGeometry(self.H, self.L, bc)
