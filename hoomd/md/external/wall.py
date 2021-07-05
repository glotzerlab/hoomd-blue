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

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import external
import hoomd
import math


class WallPotential(external.field.Field):
    r"""Generic wall potential.

    :py:class:`WallPotential` should not be used directly.
    It is a base class that provides features and documentation common to all
    standard wall potentials.

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

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut, name=""):
        external._external_force.__init__(self, name)
        self.field_coeff = walls
        self.required_coeffs = ["r_cut", "r_extrap"]
        self.force_coeff.set_default_coeff('r_extrap', 0.0)

        # convert r_cut False to a floating point type
        if (r_cut == False):
            r_cut = 0.0
        self.global_r_cut = r_cut
        self.force_coeff.set_default_coeff('r_cut', self.global_r_cut)

    def process_field_coeff(self, coeff):
        return _md.make_wall_field_params(
            coeff, hoomd.context.current.device.cpp_exec_conf)

    def update_coeffs(self):
        if not self.force_coeff.verify(self.required_coeffs):
            raise RuntimeError('Error updating force coefficients')

        ntypes = hoomd.context.current.system_definition.getParticleData(
        ).getNTypes()
        for i in range(0, ntypes):
            type = hoomd.context.current.system_definition.getParticleData(
            ).getNameByType(i)
            if self.force_coeff.values[str(type)]['r_cut'] <= 0:
                self.force_coeff.values[str(type)]['r_cut'] = 0
        external._external_force.update_coeffs(self)


class LJ(WallPotential):
    r"""Lennard-Jones wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the potential.
        default_r_cut (float): The default cut off radius for the potential
            :math:`[\mathrm{length}]`.
        default_r_extrap (float): The default ``r_extrap`` value to use.
            Defaults to 0. This only has an effect in the extrapolated mode
            :math:`[\mathrm{length}]`.

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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialLJ(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialLJ
        else:

            self.cpp_force = _md.WallsPotentialLJGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialLJGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha']
        self.force_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0)
        return _md.make_wall_lj_params(_hoomd.make_scalar2(lj1, lj2),
                                       coeff['r_cut'] * coeff['r_cut'],
                                       coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)
        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialGauss(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialGauss
        else:

            self.cpp_force = _md.WallsPotentialGaussGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialGaussGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma']

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        return _md.make_wall_gauss_params(_hoomd.make_scalar2(epsilon, sigma),
                                          coeff['r_cut'] * coeff['r_cut'],
                                          coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, d_max=None, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # update the neighbor list
        if d_max is None:
            sysdef = hoomd.context.current.system_definition
            d_max = sysdef.getParticleData().getMaxDiameter()
            hoomd.context.current.device.cpp_msg.notice(
                2, "Notice: slj set d_max=" + str(d_max) + "\n")

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialSLJ(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialSLJ
        else:

            self.cpp_force = _md.WallsPotentialSLJGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialSLJGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha']
        self.force_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0)
        return _md.make_wall_slj_params(_hoomd.make_scalar2(lj1, lj2),
                                        coeff['r_cut'] * coeff['r_cut'],
                                        coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialYukawa(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialYukawa
        else:
            self.cpp_force = _md.WallsPotentialYukawaGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialYukawaGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'kappa']

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        kappa = coeff['kappa']
        return _md.make_wall_yukawa_params(_hoomd.make_scalar2(epsilon, kappa),
                                           coeff['r_cut'] * coeff['r_cut'],
                                           coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialMorse(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialMorse
        else:

            self.cpp_force = _md.WallsPotentialMorseGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialMorseGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['D0', 'alpha', 'r0']

    def process_coeff(self, coeff):
        D0 = coeff['D0']
        alpha = coeff['alpha']
        r0 = coeff['r0']

        return _md.make_wall_morse_params(
            _hoomd.make_scalar4(D0, alpha, r0, 0.0),
            coeff['r_cut'] * coeff['r_cut'], coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialForceShiftedLJ(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialForceShiftedLJ
        else:

            self.cpp_force = _md.WallsPotentialForceShiftedLJGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialForceShiftedLJGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha']
        self.force_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0)
        return _md.make_wall_force_shift_lj_params(
            _hoomd.make_scalar2(lj1, lj2), coeff['r_cut'] * coeff['r_cut'],
            coeff['r_extrap'])


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

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. py:attribute:: r_cut

        The cut off distance for the wall potential per particle type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]

    .. py:attribute:: r_extrap

        The distance to extrapolate the potential per type
        :math:`[\mathrm{length}]`.

        Type: `hoomd.data.TypeParameter` [``particle_type``, `float` ]
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        WallPotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialMie(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialMie
        else:

            self.cpp_force = _md.WallsPotentialMieGPU(
                hoomd.context.current.system_definition, self.name)
            self.cpp_class = _md.WallsPotentialMieGPU

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'n', 'm']

    def process_coeff(self, coeff):
        epsilon = float(coeff['epsilon'])
        sigma = float(coeff['sigma'])
        n = float(coeff['n'])
        m = float(coeff['m'])

        mie1 = epsilon * math.pow(sigma, n) * (n / (n - m)) * math.pow(
            n / m, m / (n - m))
        mie2 = epsilon * math.pow(sigma, m) * (n / (n - m)) * math.pow(
            n / m, m / (n - m))
        mie3 = n
        mie4 = m
        return _md.make_wall_mie_params(
            _hoomd.make_scalar4(mie1, mie2, mie3, mie4),
            coeff['r_cut'] * coeff['r_cut'], coeff['r_extrap'])
