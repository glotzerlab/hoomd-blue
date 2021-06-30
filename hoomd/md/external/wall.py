# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: jproc

R""" Wall potentials.

Wall potentials add forces to any particles within a certain distance,
:math:`r_{\mathrm{cut}}`, of each wall. In the extrapolated
mode, all particles deemed outside of the wall boundary are included as well.

Wall geometries are used to specify half-spaces. There are two half spaces for
each of the possible geometries included and each can be selected using the
inside parameter. In order to fully specify space, it is necessary that one half
space be closed and one open. Setting *inside=True* for closed half-spaces
and *inside=False* for open ones. See :py:class:`wallpotential` for more
information on how the concept of half-spaces are used in implementing wall forces.

.. attention::
    The current wall force implementation does not support NPT integrators.

Wall groups (:py:class:`group`) are used to pass wall geometries to wall forces.
By themselves, wall groups do nothing. Only when you specify a wall force
(i.e. :py:class:`hoomd.md.wall.lj`),  are forces actually applied between the wall and the
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import external
import hoomd
import math

#           *** Helpers ***

#           *** Potentials ***


class wallpotential(external.field.Field):
    R""" Generic wall potential.

    :py:class:`wallpotential` should not be used directly.
    It is a base class that provides common features to all standard wall
    potentials. Rather than repeating all of that documentation in many different
    places, it is collected here.

    All wall potential commands specify that a given potential energy and potential be
    computed on all particles in the system within a cutoff distance,
    :math:`r_{\mathrm{cut}}`, from each wall in the given wall group.
    The force :math:`\vec{F}` is in the direction of :math:`\vec{r}`, the vector
    pointing from the particle to the wall or half-space boundary and
    :math:`V_{\mathrm{pair}}(r)` is the specific pair potential chosen by the
    respective command. Wall forces are implemented with the concept of half-spaces
    in mind. There are two modes which are allowed currently in wall potentials:
    standard and extrapolated.

    .. rubric:: Standard Mode.

    In the standard mode, when :math:`r_{\mathrm{extrap}} \le 0`, the potential
    energy is only applied to the half-space specified in the wall group. :math:`V(r)`
    is evaluated in the same manner as when the mode is shift for the analogous :py:mod:`pair <hoomd.md.pair>`
    potentials within the boundaries of the half-space.

    .. math::

        V(r)  = V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}})

    For inside=True (closed) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 \le r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r < 0
        \end{eqnarray*}

    For inside=False (open) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 < r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r \le 0
        \end{eqnarray*}

    .. rubric: Extrapolated Mode:

    The wall potential can be linearly extrapolated beyond a minimum separation from the wall
    :math:`r_{\mathrm{extrap}}` in the active half-space. This can be useful for bringing particles outside the
    half-space into the active half-space. It can also be useful for typical wall force usages by
    effectively limiting the maximum force experienced by the particle due to the wall. The potential is extrapolated into **both**
    half-spaces and the cutoff :math:`r_{\mathrm{cut}}` only applies in the active half-space. The user should
    then be careful using this mode with multiple nested walls. It is intended to be used primarily for initialization.

    The extrapolated potential has the following form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{extrap}}(r) =& V(r) &, r > r_{\rm extrap} \\
             =& V(r_{\rm extrap}) + (r_{\rm extrap}-r)\vec{F}(r_{\rm extrap}) \cdot \vec{n}&, r \le r_{\rm extrap}
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

    In other words, if :math:`r_{\rm extrap}` is chosen so that the pair force would point into the active half-space,
    the extrapolated potential will push all particles into the active half-space. See :py:class:`lj` for a
    pictorial example.

    To use extrapolated mode, the following coefficients must be set per unique particle types:

    - All parameters required by the pair potential base for the wall potential
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units) - *Optional: Defaults to global r_cut for the force if given or 0.0 if not*
    - :math:`r_{\mathrm{extrap}}` - *r_extrap* (in distance units) - *Optional: Defaults to 0.0*


    .. rubric:: Generic Example

    Note that the walls object below must be created before it is given as an
    argument to the force object. However, walls can be modified at any time before
    ```hoomd.run``` is called and it will update itself appropriately. See
    :py:class:`group` for more details about specifying the walls to be used::

        walls=wall.group()
        # Edit walls
        my_force=wall.pairpotential(walls)
        my_force.force_coeff.set('A', all required arguments)
        my_force.force_coeff.set(['B','C'],r_cut=0.3, all required arguments)
        my_force.force_coeff.set(['B','C'],r_extrap=0.3, all required arguments)

    A specific example can be found in :py:class:`lj`

    .. attention::
        The current wall force implementation does not support NPT integrators.

    Note:
        The virial due to walls is computed, but the pressure and reported by ``hoomd.analyze.log``
        is not well defined. The volume (area) of the box enters into the pressure computation, which is
        not correct in a confined system. It may not even be possible to define an appropriate volume with
        soft walls.

    Note:
        An effective use of wall forces **requires** considering the geometry of the
        system. Each wall is only evaluated in one simulation box and thus is not
        periodic. Forces will be evaluated and added to all particles from all walls in
        the wall group. Additionally there are no safeguards
        requiring a wall to exist inside the box to have interactions. This means that
        an attractive force existing outside the simulation box would pull particles
        across the periodic boundary where they would immediately cease to have any
        interaction with that wall. It is therefore up to the user to use walls in a
        physically meaningful manner. This includes the geometry of the walls, their
        interactions, and as noted here their location.

    Note:
        When :math:`r_{\mathrm{cut}} \le 0` or is set to False the particle type
        wall interaction is excluded.

    Note:
        While wall potentials are based on the same potential energy calculations
        as pair potentials, Features of pair potentials such as specified neighborlists,
        and alternative force shifting modes are not supported.
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

    ## \internal
    # \brief passes the wall field
    def process_field_coeff(self, coeff):
        return _md.make_wall_field_params(
            coeff, hoomd.context.current.device.cpp_exec_conf)

    ## \internal
    # \brief Fixes negative values to zero before squaring
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


class lj(wallpotential):
    R""" Lennard-Jones wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Lennard-Jones potential.
    See :py:class:`hoomd.md.pair.LJ` for force details and base parameters and
    :py:class:`wallpotential` for generalized wall potential implementation

    Standard mode::

        walls=wall.group()
        #add walls
        lj=wall.lj(walls, r_cut=3.0)
        lj.force_coeff.set('A', sigma=1.0,epsilon=1.0)  #plotted below in red
        lj.force_coeff.set('B', sigma=1.0,epsilon=1.0, r_cut=2.0**(1.0/2.0))
        lj.force_coeff.set(['A','B'], epsilon=2.0, sigma=1.0, alpha=1.0, r_cut=3.0)

    Extrapolated mode::

        walls=wall.group()
        #add walls
        lj_extrap=wall.lj(walls, r_cut=3.0)
        lj_extrap.force_coeff.set('A', sigma=1.0,epsilon=1.0, r_extrap=1.1) #plotted in blue below

    V(r) plot:

    .. image:: wall_extrap.png
    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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


class gauss(wallpotential):
    R""" Gaussian wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Gaussian potential.
    See :py:class:`hoomd.md.pair.Gauss` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_gauss=wall.gauss(walls, r_cut=3.0)
        wall_force_gauss.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_gauss.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
        wall_force_gauss.force_coeff.set(['C', 'D'], epsilon=3.0, sigma=0.5)

    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)
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


class slj(wallpotential):
    R""" Shifted Lennard-Jones wall potential

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Shifted Lennard-Jones potential.
    Note that because slj is dependent upon particle diameters the following
    correction is necessary to the force details in the :py:class:`hoomd.md.pair.SLJ` description.

    :math:`\Delta = d_i/2 - 1` where :math:`d_i` is the diameter of particle :math:`i`.
    See :py:class:`hoomd.md.pair.SLJ` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_slj=wall.slj(walls, r_cut=3.0)
        wall_force_slj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_slj.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
        wall_force_slj.force_coeff.set('B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0))

    """

    def __init__(self, walls, r_cut=False, d_max=None, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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


class yukawa(wallpotential):
    R""" Yukawa wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Yukawa potential.
    See :py:class:`hoomd.md.pair.Yukawa` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_yukawa=wall.yukawa(walls, r_cut=3.0)
        wall_force_yukawa.force_coeff.set('A', epsilon=1.0, kappa=1.0)
        wall_force_yukawa.force_coeff.set('A', epsilon=2.0, kappa=0.5, r_cut=3.0)
        wall_force_yukawa.force_coeff.set(['C', 'D'], epsilon=0.5, kappa=3.0)

    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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


class morse(wallpotential):
    R""" Morse wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Morse potential.
    See :py:class:`hoomd.md.pair.Morse` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_morse=wall.morse(walls, r_cut=3.0)
        wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0)
        wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0)
        wall_force_morse.force_coeff.set(['C', 'D'], D0=1.0, alpha=3.0)

    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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


class force_shifted_lj(wallpotential):
    R""" Force-shifted Lennard-Jones wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Force-shifted Lennard-Jones potential.
    See :py:class:`hoomd.md.pair.ForceShiftedLJ` for force details and base parameters and :py:class:`wallpotential`
    for generalized wall potential implementation.

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_fslj=wall.force_shifted_lj(walls, r_cut=3.0)
        wall_force_fslj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_fslj.force_coeff.set('B', epsilon=1.5, sigma=3.0, r_cut = 8.0)
        wall_force_fslj.force_coeff.set(['C','D'], epsilon=1.0, sigma=1.0, alpha = 1.5)

    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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


class mie(wallpotential):
    R""" Mie potential wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the log files.

    Wall force evaluated using the Mie potential.
    See :py:class:`hoomd.md.pair.Mie` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_mie=wall.mie(walls, r_cut=3.0)
        wall_force_mie.force_coeff.set('A', epsilon=1.0, sigma=1.0, n=12, m=6)
        wall_force_mie.force_coeff.set('A', epsilon=2.0, sigma=1.0, n=14, m=7, r_cut=3.0)
        wall_force_mie.force_coeff.set('B', epsilon=1.0, sigma=1.0, n=15.1, m=6.5, r_cut=2**(1.0/6.0))

    """

    def __init__(self, walls, r_cut=False, name=""):

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name)

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
