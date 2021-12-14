# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

r"""Wall potentials.

Wall potentials add forces to any particles within a certain distance,
:math:`r_{\mathrm{cut}}`, of each wall. In the extrapolated mode, all particles
outside of the wall boundary are included as well.

Wall geometries (`hoomd.wall`) are used to specify half-spaces. There are two
half spaces for each of the possible geometries included and each can be
selected using the inside parameter. In order to fully specify space, it is
necessary that one half space be closed and one open. Setting *inside=True* for
closed half-spaces and *inside=False* for open ones. See `hoomd.wall` for more
information on wall geometries and `WallPotential` for information about forces
and half-spaces.

.. attention::
    The current wall force implementation does not support NPT integrators.
"""

from hoomd.md import force
from hoomd.data.array_view import _ArrayViewWrapper
from hoomd.md import _md
import hoomd


def _to_md_cpp_wall(wall):
    if isinstance(wall, hoomd.wall.Sphere):
        return _md.SphereWall(wall.radius, wall.origin, wall.inside, wall.open)
    if isinstance(wall, hoomd.wall.Cylinder):
        return _md.CylinderWall(wall.radius, wall.origin, wall.axis,
                                wall.inside, wall.open)
    if isinstance(wall, hoomd.wall.Plane):
        return _md.PlaneWall(wall.origin, wall.normal, wall.open)
    raise TypeError(f"Unknown wall type encountered {type(wall)}.")


class _WallArrayViewFactory:

    def __init__(self, cpp_wall_potential, wall_type):
        self.cpp_obj = cpp_wall_potential
        self.func_name = {
            hoomd.wall.Sphere: "get_sphere_list",
            hoomd.wall.Cylinder: "get_cylinder_list",
            hoomd.wall.Plane: "get_plane_list"
        }[wall_type]

    def __call__(self):
        return getattr(self.cpp_obj.field, self.func_name)()


class WallPotential(force.Force):
    r"""Generic wall potential.

    Warning:
        `WallPotential` should not be used directly.  It is a base class that
        provides features and documentation common to all standard wall
        potentials.

    All wall potential classes specify that a given potential energy and
    force be computed on all particles in the system when the signed cutoff
    distance is near the wall surface: :math:`r < r_{\mathrm{cut}}`.
    The force :math:`\vec{F}` is in the direction of :math:`\vec{r}`, the vector
    pointing from the particle to closest point on the wall's surface and
    :math:`V_{\mathrm{pair}}(r)` is the pair potential specified by subclasses
    of `WallPotential`.  Walls are two-sided surfaces with positive signed
    distances to points on the active side of the wall and negative signed
    distances to points on the inactive side. Additionally, the wall's mode
    controls how forces and energies are computed for particles on or near the
    inactive side. The `inside` flag determines
    which side of the surface is active.

    .. rubric:: Standard Mode.

    In the standard mode, when :math:`r_{\mathrm{extrap}} \le 0`, the potential
    energy is only computed on the active side.
    :math:`V(r)` is evaluated in the same manner as when the mode is shift for
    the analogous :py:mod:`pair <hoomd.md.pair>` potentials within the
    boundaries of the half-space.

    .. math::

        V(r)  = V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}})

    For ``inside=True`` (closed) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 \le r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r < 0
        \end{eqnarray*}

    For ``inside=False`` (open) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 < r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r \le 0
        \end{eqnarray*}

    Below we show the potential for a `hoomd.wall.Sphere` with radius 5 in 2D,
    using the Gaussian potential with :math:`\epsilon=1, \sigma=1` and
    ``inside=True``.

    .. image:: md-wall-potential.svg

    When ``inside=False`` the potential becomes,

    .. image:: md-wall-potential-outside.svg

    .. rubric: Extrapolated Mode:

    The wall potential can be linearly extrapolated starting at
    :math:`r = r_{\mathrm{extrap}}` on the active side and continuing to the
    inactive side. This can be useful to move particles from the inactive side
    to the active side.

    The extrapolated potential has the following form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{extrap}}(r) =& V(r) &, r > r_{\rm extrap} \\
             =& V(r_{\rm extrap})
                + (r_{\rm extrap}-r)\vec{F}(r_{\rm extrap})
                \cdot \vec{n}&, r \le r_{\rm extrap}
        \end{eqnarray*}

    where :math:`\vec{n}` is the normal pointing toward the active side.
    This gives an effective force on the particle due to the wall:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}(r) =& \vec{F}_{\rm pair}(r) &, r > r_{\rm extrap} \\
                =& \vec{F}_{\rm pair}(r_{\rm extrap}) &, r \le r_{\rm extrap}
        \end{eqnarray*}

    where :math:`\vec{F}_{\rm pair}` is given by the gradient of the pair force

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}_{\rm pair}(r) =& -\nabla V_{\rm pair}(r) &, r < r_{\rm cut} \\
                           =& 0 &, r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Below is an example of
    extrapolation with ``r_extrap=1.1`` for a LJ potential with
    :math:`\epsilon=1, \sigma=1`.

    .. image:: md-wall-extrapolate.svg


    To use extrapolated mode ``r_extrap`` must be set per particle type.

    .. attention::
        Walls are fixed in space and do not adjust with the box size. For
        example, NPT simulations may not behave as expected.

    Note:
        - The virial due to walls is computed, but the pressure and reported by
          ``hoomd.md.compute.ThermodynamicQuantities`` is not well defined. The
          volume (or area) of the box enters into the pressure computation,
          which is not correct in a confined system. It may not even be possible
          to define an appropriate volume with soft walls.
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


class LJ(WallPotential):
    r"""Lennard-Jones wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Lennard-Jones potential.  See
    `hoomd.md.pair.LJ` for force details and base parameters and `WallPotential`
    for generalized wall potential implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        lj = hoomd.md.external.wall.LJ(walls=walls)
        # potential plotted below in red
        lj.params['A'] = {"sigma": 1.0, "epsilon": 1.0, "r_cut": 2.5}
        # set for both types "A" and "B"
        lj.params[['A','B']] = {"epsilon": 2.0, "sigma": 1.0, "r_cut": 2.8}
        # set to extrapolated mode
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
          The distance to extrapolate the potential, defauts to 0
          :math:`[\mathrm{length}]`
          .

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


class Gauss(WallPotential):
    r"""Gaussian wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Gaussian potential.  See
    `hoomd.md.pair.Gauss` for force details and base parameters and
    `WallPotential` for generalized wall potential implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        gaussian_wall=hoomd.md.external.wall.Gauss(walls=walls)
        gaussian_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0, "r_cut": 2.5}
        gaussian_wall.params[['A','B']] = {
            "epsilon": 2.0, "sigma": 1.0, "r_cut": 1.0}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

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
    r"""Yukawa wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Yukawa potential.  See `hoomd.md.pair.Yukawa`
    for force details and base parameters and `WallPotential` for generalized
    wall potential implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        yukawa_wall = hoomd.md.external.wall.Yukawa(walls=walls)
        yukawa_wall.params['A'] = {
            "epsilon": 1.0, "kappa": 1.0, "r_cut": 3.0}
        yukawa_wall.params[['A','B']] = {
            "epsilon": 0.5, "kappa": 3.0, "r_cut": 3.2}
        walls=wall.group()

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

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
    r"""Morse wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Morse potential.  See
    :py:class:`hoomd.md.pair.Morse` for force details and base parameters and
    :py:class:`WallPotential` for generalized wall potential implementation.

    Example::


        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        morse_wall=hoomd.md.external.wall.Morse(walls=walls)
        morse_wall.params['A'] = {
            "D0": 1.0, "alpha": 1.0, "r0": 1.0, "r_cut": 3.0}
        morse_wall.params[['A','B']] = {
            "D0": 0.5, "alpha": 3.0, "r0": 1.0, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

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
    r"""Force-shifted Lennard-Jones wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Force-shifted Lennard-Jones potential.  See
    `hoomd.md.pair.ForceShiftedLJ` for force details and base parameters and
    `WallPotential` for generalized wall potential implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        shifted_lj_wall=hoomd.md.external.wall.ForceShiftedLJ(
            walls=walls)
        shifted_lj_wall.params['A'] = {
            "epsilon": 1.0, "sigma": 1.0, "r_cut": 3.0}
        shifted_lj_wall.params[['A','B']] = {
            "epsilon": 0.5, "sigma": 3.0, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

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
    r"""Mie potential wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

    Wall force evaluated using the Mie potential.  See `hoomd.md.pair.Mie` for
    force details and base parameters and `WallPotential` for generalized wall
    potential implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        mie_wall=hoomd.md.external.wall.Mie(walls=walls)
        mie_wall.params['A'] = {
            "epsilon": 1.0, "sigma": 1.0, "n": 12, "m": 6, "r_cut": 3.0}
        mie_wall.params[['A','B']] = {
            "epsilon": 0.5, "sigma": 3.0, "n": 49, "m": 50, "r_cut": 3.2}

    Attributes:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.

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
