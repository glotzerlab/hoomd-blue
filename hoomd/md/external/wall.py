# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Wall forces.

Wall potential classes compute forces, virials, and energies between all
particles and the given walls consistent with the energy:

.. math::

    U_\mathrm{wall} = \sum_{i=0}^{\mathrm{N_particles-1}}
                      \sum_{w \in walls} U_w(d_i),

where :math:`d_i` is the signed distance between particle :math:`i` and the wall
:math:`w`.

The potential :math:`U_w(d)` is a function of the signed cutoff distance between
the particle and a wall's surface :math:`d`. The resulting force :math:`\vec{F}`
is is parallel to :math:`\vec{d}`, the vector pointing from the closest point on
the wall's surface to the particle. :math:`U_w(d)` is related to a pair
potential :math:`U_{\mathrm{pair}}` as defined below and each subclass of
`WallPotential` implements a different functional form of
:math:`U_{\mathrm{pair}}`.

Walls are two-sided surfaces with positive signed distances to points on the
active side of the wall and negative signed distances to points on the inactive
side. Additionally, the wall's mode controls how forces and energies are
computed for particles on or near the inactive side. The ``inside`` flag (or
``normal`` in the case of `hoomd.wall.Plane`) determines which side of the
surface is active.

.. rubric:: Standard Mode

In the standard mode, when :math:`r_{\mathrm{extrap}} \le 0`, the potential
energy is only computed on the active side. :math:`U(d)` is evaluated in the
same manner as when the mode is shift for the analogous :py:mod:`pair
<hoomd.md.pair>` potentials within the boundaries of the active space:

.. math::

    U(d) = U_{\mathrm{pair}}(d) - U_{\mathrm{pair}}(r_{\mathrm{cut}})

For ``open=True`` spaces:

.. math::

    \vec{F} =
    \begin{cases}
    -\frac{\partial U}{\partial d}\hat{d} & 0 < d < r_{\mathrm{cut}} \\
    0 & d \ge r_{\mathrm{cut}} \\
    0 & d \le 0
    \end{cases}

For ``open=False`` (closed) spaces:

.. math::
    \vec{F} =
    \begin{cases}
    -\frac{\partial U}{\partial d}\hat{d} & 0 \le d < r_{\mathrm{cut}} \\
    0 & d \ge r_{\mathrm{cut}} \\
    0 & d < 0
    \end{cases}

Below we show the potential for a `hoomd.wall.Sphere` with radius 5 in 2D,
using the Gaussian potential with :math:`\epsilon=1, \sigma=1` and
``inside=True``:

.. image:: md-wall-potential.svg
    :alt: Example plot of wall potential.

When ``inside=False``, the potential becomes:

.. image:: md-wall-potential-outside.svg
    :alt: Example plot of an outside wall potential.

.. rubric:: Extrapolated Mode:

The wall potential can be linearly extrapolated starting at
:math:`d = r_{\mathrm{extrap}}` on the active side and continuing to the
inactive side. This can be useful to move particles from the inactive side
to the active side.

The extrapolated potential has the following form:

.. math::
    V_{\mathrm{extrap}}(d) =
    \begin{cases}
    U(d) & d > r_{\rm extrap} \\
    U(r_{\rm extrap}) + (r_{\rm extrap}-r)\vec{F}(r_{\rm extrap})
    \cdot \vec{n} & d \le r_{\rm extrap}
    \end{cases}

where :math:`\vec{n}` is such that the force points towards the active space
for repulsive potentials. This gives an effective force on the particle due
to the wall:

.. math::
    \vec{F}(d) =
    \begin{cases}
    \vec{F}(d) & d > r_{\rm extrap} \\
    \vec{F}(r_{\rm extrap}) & d \le r_{\rm extrap}
    \end{cases}

Below is an example of extrapolation with ``r_extrap=1.1`` for a LJ
potential with :math:`\epsilon=1, \sigma=1`.

.. image:: md-wall-extrapolate.svg
    :alt: Example plot demonstrating potential extrapolation.

To use extrapolated mode ``r_extrap`` must be set per particle type.

.. attention::
    Walls are fixed in space and do not adjust with the box size. For
    example, NPT simulations may not behave as expected.

Note:
    - The virial due to walls is computed, but the pressure computed and
      reported by `hoomd.md.compute.ThermodynamicQuantities` is not well
      defined. The volume (or area) of the box enters into the pressure
      computation, which is not correct in a confined system. It may not
      even be possible to define an appropriate volume with soft walls.
    - An effective use of wall forces **requires** considering the geometry
      of the system. Each wall is only evaluated in one simulation box and
      thus is not periodic. Forces will be evaluated and added to all
      particles from all walls. Additionally there are no
      safeguards requiring a wall to exist inside the box to have
      interactions. This means that an attractive force existing outside the
      simulation box would pull particles across the periodic boundary where
      they would immediately cease to have any interaction with that wall.
      It is therefore up to the user to use walls in a physically meaningful
      manner. This includes the geometry of the walls, their interactions,
      and as noted here their location.
    - When :math:`r_{\mathrm{cut}} \le 0` or is set to ``False`` the
      particle type wall interaction is excluded.
    - While wall potentials are based on the same potential energy
      calculations as pair potentials, features of pair potentials such as
      specified neighborlists, and alternative force shifting modes are not
      supported.
"""

from hoomd.md import force
from hoomd.data.array_view import _ArrayViewWrapper
from hoomd.md import _md
import hoomd


def _to_md_cpp_wall(wall):
    if isinstance(wall, hoomd.wall.Sphere):
        return _md.SphereWall(wall.radius, wall.origin.to_base(), wall.inside,
                              wall.open)
    if isinstance(wall, hoomd.wall.Cylinder):
        return _md.CylinderWall(wall.radius, wall.origin.to_base(),
                                wall.axis.to_base(), wall.inside, wall.open)
    if isinstance(wall, hoomd.wall.Cone):
        return _md.ConeWall(wall.radius1, wall.radius2, distance, wall.origin.to_base(),
                            wall.axis.to_base(), wall.inside, wall.open)
    if isinstance(wall, hoomd.wall.Plane):
        return _md.PlaneWall(wall.origin.to_base(), wall.normal.to_base(),
                             wall.open)
    raise TypeError(f"Unknown wall type encountered {type(wall)}.")


class _WallArrayViewFactory:

    def __init__(self, cpp_wall_potential, wall_type):
        self.cpp_obj = cpp_wall_potential
        self.func_name = {
            hoomd.wall.Sphere: "get_sphere_list",
            hoomd.wall.Cylinder: "get_cylinder_list",
            hoomd.wall.Cone: "get_cone_list",
            hoomd.wall.Plane: "get_plane_list"
        }[wall_type]

    def __call__(self):
        return getattr(self.cpp_obj.field, self.func_name)()


class WallPotential(force.Force):
    r"""Base class wall force.

    Warning:
        `WallPotential` should not be used directly.  It is a base class that
        provides features and documentation common to all standard wall
        potentials.
    """

    def __init__(self, walls):
        self._walls = None
        self.walls = hoomd.wall._WallsMetaList(walls, _to_md_cpp_wall)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def)
        self._walls._sync({
            hoomd.wall.Sphere:
                _ArrayViewWrapper(
                    _WallArrayViewFactory(self._cpp_obj, hoomd.wall.Sphere)),
            hoomd.wall.Cylinder:
                _ArrayViewWrapper(
                    _WallArrayViewFactory(self._cpp_obj, hoomd.wall.Cylinder)),
            hoomd.wall.Cone:
                _ArrayViewWrapper(
                    _WallArrayViewFactory(self._cpp_obj, hoomd.wall.Cone)),
            hoomd.wall.Plane:
                _ArrayViewWrapper(
                    _WallArrayViewFactory(self._cpp_obj, hoomd.wall.Plane)),
        })
        super()._attach()

    @property
    def walls(self):
        """`list` [`hoomd.wall.WallGeometry`]: \
            The walls associated with this wall potential."""
        return self._walls

    @walls.setter
    def walls(self, wall_list):
        if self._walls is wall_list:
            return
        self._walls = hoomd.wall._WallsMetaList(wall_list, _to_md_cpp_wall)
        if self._attached:
            self._walls._sync({
                hoomd.wall.Sphere:
                    _ArrayViewWrapper(
                        _WallArrayViewFactory(self._cpp_obj,
                                              hoomd.wall.Sphere)),
                hoomd.wall.Cylinder:
                    _ArrayViewWrapper(
                        _WallArrayViewFactory(self._cpp_obj,
                                              hoomd.wall.Cylinder)),
                hoomd.wall.Cone:
                    _ArrayViewWrapper(
                        _WallArrayViewFactory(self._cpp_obj,
                                              hoomd.wall.Cone)),
                hoomd.wall.Plane:
                    _ArrayViewWrapper(
                        _WallArrayViewFactory(self._cpp_obj, hoomd.wall.Plane)),
            })


class LJ(WallPotential):
    r"""Lennard-Jones wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Lennard-Jones force. See `hoomd.md.pair.LJ`
    for the functional form of the force and parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        lj = hoomd.md.external.wall.LJ(walls=walls)
        lj.params['A'] = {"sigma": 1.0, "epsilon": 1.0, "r_cut": 2.5}
        lj.params[['A','B']] = {"epsilon": 2.0, "sigma": 1.0, "r_cut": 2.8}
        lj.params["A"] = {"r_extrap": 1.1}

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential, defaults to 0
          :math:`[\mathrm{length}]`.

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialLJ"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)


class ExpandedLJ(WallPotential):
    r"""Expanded Lennard-Jones wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Expanded Lennard-Jones force. See `hoomd.md.pair.ExpandedLJ`
    for the functional form of the force and parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        expanded_lj = hoomd.md.external.wall.ExpandedLJ(walls=walls)
        expanded_lj.params['A'] = {"sigma": 1.0, "epsilon": 1.0, "delta": 1.0}
        expanded_lj.params[['A','B']] = {"epsilon": 2.0, "sigma": 1.0, "delta": 1.0}

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``delta`` (`float`, **required**) -
          radial shift :math:`\Delta` :math:`[\mathrm{length}]`.

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialExpandedLJ"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        delta=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class Gauss(WallPotential):
    r"""Gaussian wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Gaussian force.  See `hoomd.md.pair.Gauss`
    for the functional form of the force and parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        gaussian_wall = hoomd.md.external.wall.Gauss(walls=walls)
        gaussian_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0, "r_cut": 2.5}
        gaussian_wall.params[['A','B']] = {
            "epsilon": 2.0, "sigma": 1.0, "r_cut": 1.0}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential :math:`[\mathrm{length}]`,
          defaults to 0.

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialGauss"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)


class Yukawa(WallPotential):
    r"""Yukawa wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Yukawa force.  See `hoomd.md.pair.Yukawa`
    for the functional form of the force and parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        yukawa_wall = hoomd.md.external.wall.Yukawa(walls=walls)
        yukawa_wall.params['A'] = {
            "epsilon": 1.0, "kappa": 1.0, "r_cut": 3.0}
        yukawa_wall.params[['A','B']] = {
            "epsilon": 0.5, "kappa": 3.0, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential, defaults to 0
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialYukawa"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        kappa=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)


class Morse(WallPotential):
    r"""Morse wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Morse force.  See `hoomd.md.pair.Morse` for
    the functional form of the force and parameter definitions.

    Example::


        walls = [hoomd.wall.Sphere(radius=4.0)]
        morse_wall = hoomd.md.external.wall.Morse(walls=walls)
        morse_wall.params['A'] = {
            "D0": 1.0, "alpha": 1.0, "r0": 1.0, "r_cut": 3.0}
        morse_wall.params[['A','B']] = {
            "D0": 0.5, "alpha": 3.0, "r0": 1.0, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential, defaults to 0
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialMorse"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(D0=float,
                                                        r0=float,
                                                        alpha=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)


class ForceShiftedLJ(WallPotential):
    r"""Force-shifted Lennard-Jones wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Force-shifted Lennard-Jones force.  See
    `hoomd.md.pair.ForceShiftedLJ` for the functional form of the force and
    parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(
            walls=walls)
        shifted_lj_wall.params['A'] = {
            "epsilon": 1.0, "sigma": 1.0, "r_cut": 3.0}
        shifted_lj_wall.params[['A','B']] = {
            "epsilon": 0.5, "sigma": 3.0, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential, defaults to 0
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialForceShiftedLJ"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)


class Mie(WallPotential):
    r"""Mie wall force.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Mie force.  See `hoomd.md.pair.Mie` for the
    functional form of the force and parameter definitions.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        mie_wall = hoomd.md.external.wall.Mie(walls=walls)
        mie_wall.params['A'] = {
            "epsilon": 1.0, "sigma": 1.0, "n": 12, "m": 6, "r_cut": 3.0}
        mie_wall.params[['A','B']] = {
            "epsilon": 0.5, "sigma": 3.0, "n": 49, "m": 50, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential, defaults to 0
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [``particle_types``, `dict`]
    """

    _cpp_class_name = "WallsPotentialMie"

    def __init__(self, walls):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        m=float,
                                                        n=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)
