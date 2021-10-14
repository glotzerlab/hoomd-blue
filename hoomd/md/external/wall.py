# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

r"""Wall potentials.

Wall potentials add forces to any particles within a certain distance,
:math:`r_{\mathrm{cut}}`, of each wall. In the extrapolated mode, all particles
deemed outside of the wall boundary are included as well.

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
        return _md.SphereWall(wall.radius, wall.origin, wall.inside)
    if isinstance(wall, hoomd.wall.Cylinder):
        return _md.CylinderWall(wall.radius, wall.origin, wall.axis,
                                wall.inside)
    if isinstance(wall, hoomd.wall.Plane):
        return _md.PlaneWall(wall.origin, wall.normal, wall.inside)
    raise TypeError(f"Unknown wall type encountered {type(wall)}.")


class WallPotential(force.Force):
    r"""Generic wall potential.

    Warning:
        `WallPotential` should not be used directly.  It is a base class that
        provides features and documentation common to all standard wall
        potentials.

    All wall potential commands specify that a given potential energy and
    potential be computed on all particles in the system within a cutoff
    distance, :math:`r_{\mathrm{cut}}`, from each wall in the given wall group.
    The force :math:`\vec{F}` is in the direction of :math:`\vec{r}`, the vector
    pointing from the particle to the wall or half-space boundary and
    :math:`V_{\mathrm{pair}}(r)` is the pair potential specified by subclasses
    of `WallPotential`.  Wall forces are implemented with the concept of
    half-spaces in mind. There are two modes which are allowed currently in wall
    potentials: standard and extrapolated.

    .. rubric:: Standard Mode.

    In the standard mode, when :math:`r_{\mathrm{extrap}} \le 0`, the potential
    energy is only applied to the half-space specified in the wall group.
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

    .. rubric: Extrapolated Mode:

    The wall potential can be linearly extrapolated beyond a minimum separation
    from the wall :math:`r_{\mathrm{extrap}}` in the active half-space. This can
    be useful for bringing particles outside the half-space into the active
    half-space. It can also be useful for typical wall force usages by
    effectively limiting the maximum force experienced by the particle due to
    the wall. The potential is extrapolated into **both** half-spaces and the
    cutoff :math:`r_{\mathrm{cut}}` only applies in the active half-space. The
    user should then be careful using this mode with multiple nested walls. It
    is intended to be used primarily for initialization.

    The extrapolated potential has the following form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{extrap}}(r) =& V(r) &, r > r_{\rm extrap} \\
             =& V(r_{\rm extrap})
                + (r_{\rm extrap}-r)\vec{F}(r_{\rm extrap})
                \cdot \vec{n}&, r \le r_{\rm extrap}
        \end{eqnarray*}

    where :math:`\vec{n}` is the normal into the active half-space.
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

    In other words, if :math:`r_{\rm extrap}` is chosen so that the pair force
    would point into the active half-space, the extrapolated potential will push
    all particles into the active half-space. See `LJ` for a pictorial example.

    To use extrapolated mode ``r_extrap`` must be set per particle type.

    .. attention::
        The current wall force implementation does not support NPT integrators.

    Note:
        - The virial due to walls is computed, but the pressure and reported by
          ``hoomd.md.compute.ThermodynamicQuantities`` is not well defined. The
          volume (area) of the box enters into the pressure computation, which
          is not correct in a confined system. It may not even be possible to
          define an appropriate volume with soft walls.
        - An effective use of wall forces **requires** considering the geometry
          of the system. Each wall is only evaluated in one simulation box and
          thus is not periodic. Forces will be evaluated and added to all
          particles from all walls in the wall group. Additionally there are no
          safeguards requiring a wall to exist inside the box to have
          interactions. This means that an attractive force existing outside the
          simulation box would pull particles across the periodic boundary where
          they would immediately cease to have any interaction with that wall.
          It is therefore up to the user to use walls in a physically meaningful
          manner. This includes the geometry of the walls, their interactions,
          and as noted here their location.
        - When :math:`r_{\mathrm{cut}} \le 0` or is set to False the particle
          type wall interaction is excluded.
        - While wall potentials are based on the same potential energy
          calculations as pair potentials, Features of pair potentials such as
          specified neighborlists, and alternative force shifting modes are not
          supported.

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    def __init__(self, walls):
        self._walls = None
        if walls is None:
            walls = []
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
        """`list`[ `hoomd.wall.WallGeometry`]: \
            The walls associated with this wall potential."""
        return self._walls

    @walls.setter
    def walls(self, wall_list):
        if self._walls is wall_list:
            return
        self._walls = hoomd.wall._WallsMetaList(wall_list, _to_md_cpp_wall)


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
        lj = hoomd.md.wall.LJ(walls, default_r_cut=3.0)
        # potential plotted below in red
        lj.params['A'] = {"sigma": 1.0, "epsilon": 1.0}
        lj.r_cut['B'] = 2.0**(1.0 / 2.0)
        # set for both types "A" and "B"
        lj.params['A','B'] = {"epsilon": 2.0, "sigma": 1.0}
        # set to extrapolated mode
        lj.r_extrap.default = 1.1

    V(r) plot:

    .. image:: wall_extrap.png

    .. py:attribute:: params

        The potential parameters per type. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialLJ"

    def __init__(self, walls=None):

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
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Gaussian potential.  See
    `hoomd.md.pair.Gauss` for force details and base parameters and
    `WallPotential` for generalized wall potential implementation

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        gaussian_wall=hoomd.md.wall.Gauss(walls, default_r_cut=3.0)
        gaussian_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0}
        gaussian_wall.r_cut['A'] = 3.0
        gaussian_wall.params['A','B'] = {
            "epsilon": 2.0, "sigma": 1.0, "alpha": 1.0}

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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialGauss"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class SLJ(WallPotential):
    r"""Shifted Lennard-Jones wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Shifted Lennard-Jones potential.  Note that
    because `SLJ` is dependent upon particle diameters the following correction
    is necessary to the force details in the :py:class:`hoomd.md.pair.SLJ`
    description.

    :math:`\Delta = d_i/2 - 1` where :math:`d_i` is the diameter of particle
    :math:`i`.  See :py:class:`hoomd.md.pair.SLJ` for force details and base
    parameters and :py:class:`WallPotential` for generalized wall potential
    implementation

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        slj_wall=hoomd.md.wall.SLJ(walls, default_r_cut=3.0)
        slj_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0}
        slj_wall.r_cut['A'] = 3.0
        slj_wall.params['A','B'] = {"epsilon": 2.0, "sigma": 1.0}

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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialSLJ"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class Yukawa(WallPotential):
    r"""Yukawa wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Yukawa potential.  See
    :py:class:`hoomd.md.pair.Yukawa` for force details and base parameters and
    :py:class:`WallPotential` for generalized wall potential implementation

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        yukawa_wall=hoomd.md.wall.Yukawa(walls, default_r_cut=3.0)
        yukawa_wall.params['A'] = {"epsilon": 1.0, "kappa": 1.0}
        yukawa_wall.r_cut['A'] = 3.0
        yukawa_wall.params['A','B'] = {"epsilon": 0.5, "kappa": 3.0}
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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialYukawa"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        kappa=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class Morse(WallPotential):
    r"""Morse wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Morse potential.  See
    :py:class:`hoomd.md.pair.Morse` for force details and base parameters and
    :py:class:`WallPotential` for generalized wall potential implementation

    Example::


        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        morse_wall=hoomd.md.wall.Morse(walls, default_r_cut=3.0)
        morse_wall.params['A'] = {"D0": 1.0, "alpha": 1.0, "r0": 1.0}
        morse_wall.r_cut['A'] = 3.0
        morse_wall.params['A','B'] = {"D0": 0.5, "alpha": 3.0, "r0": 1.0}

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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialMorse"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class ForceShiftedLJ(WallPotential):
    r"""Force-shifted Lennard-Jones wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Force-shifted Lennard-Jones potential.  See
    :py:class:`hoomd.md.pair.ForceShiftedLJ` for force details and base
    parameters and :py:class:`WallPotential` for generalized wall potential
    implementation.

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        force_shifted_lj_wall=hoomd.md.wall.ForceShiftedLJ(
            walls, default_r_cut=3.0)
        force_shifted_lj_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0}
        force_shifted_lj_wall.r_cut['A'] = 3.0
        force_shifted_lj_wall.params['A','B'] = {"epsilon": 0.5, "sigma": 3.0}

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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialForceShiftedLJ"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        len_keys=1))
        self._add_typeparam(params)


class Mie(WallPotential):
    r"""Mie potential wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

    Wall force evaluated using the Mie potential.  See
    :py:class:`hoomd.md.pair.Mie` for force details and base parameters and
    :py:class:`WallPotential` for generalized wall potential implementation

    Example::

        walls = [hoomd.wall.Sphere(radius=4.0)]
        # add walls to interact with
        mie_wall=hoomd.md.wall.Mie(walls, default_r_cut=3.0)
        mie_wall.params['A'] = {"epsilon": 1.0, "sigma": 1.0, "n": 12, "m": 6}
        mie_wall.r_cut['A'] = 3.0
        mie_wall.params['A','B'] = {
            "epsilon": 0.5, "sigma": 3.0, "n": 49, "m": 50}

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

        Type: `TypeParameter` [``particle_types``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_types``, `float` ]
    """

    _cpp_class_name = "WallsPotentialMie"

    def __init__(self, walls=None):

        # initialize the base class
        super().__init__(walls)

        params = hoomd.data.typeparam.TypeParameter(
            "params", "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        m=float,
                                                        n=float,
                                                        len_keys=1))
        self._add_typeparam(params)
