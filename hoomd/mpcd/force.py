# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD solvent forces.

MPCD can apply a body force to each MPCD particle as a function of position.
The external force should be compatible with the chosen `~hoomd.mpcd.geometry`.
Global momentum conservation is typically broken by adding a solvent force, so
care should be chosen that the entire model is designed so that the system
does not have net acceleration. For example, solid boundaries can be used to
dissipate momentum, or a balancing force can be applied to particles that are
coupled to the solvent through the collision step. Additionally, a thermostat
will likely be required to maintain temperature control in the driven system.

"""

from hoomd import _hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.operation import _HOOMDBaseObject


class SolventForce(_HOOMDBaseObject):
    """Solvent force.

    The SolventForce is a body force applied to each solvent particle. This
    class should not be instantiated directly. It exists for type checking.

    """
    pass


class BlockForce(SolventForce):
    r"""Block force.

    Args:
        force (float): Magnitude of the force in *x* per particle.
        half_separation (float): Half the distance between the centers of the
            blocks.
        half_width (float): Half the width of each block.

    The ``force`` magnitude *F* is applied in the *x* direction on the particles
    in blocks defined along the *z* direction by the ``half_separation`` *H* and
    the ``half_width`` *w*. The force in *x* is :math:`+F` in the upper block,
    :math:`-F` in the lower block, and zero otherwise.

    .. math::
        :nowrap:

        \begin{equation}
        \mathbf{F} = \begin{cases}
        +F \mathbf{e}_x & |r_z - H| < w \\
        -F \mathbf{e}_x & |r_z + H| < w \\
           \mathbf{0}   & \mathrm{otherwise}
        \end{cases}
        \end{equation}

    The BlockForce can be used to implement the double-parabola method for measuring
    viscosity by setting :math:`H = L_z/4` and :math:`w = L_z/4`, where :math:`L_z` is
    the size of the simulation box in *z*.

    Warning:
        You should define the blocks to lie fully within the simulation box and
        to not overlap each other.

    Attributes:
        force (float): Magnitude of the force in *x* per particle.

        half_separation (float): Half the distance between the centers of the
            blocks.

        half_width (float): Half the width of each block.

    """

    def __init__(self, force, half_separation=None, half_width=None):
        super().__init__()

        param_dict = ParameterDict(
            force=float(force),
            half_separation=float(half_separation),
            half_width=float(half_width),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.BlockForce(self.force, self.half_separation,
                                         self.half_width)
        super()._attach_hook()


class ConstantForce(SolventForce):
    r"""Constant force.

    Args:
        force (`tuple` [`float`, `float`, `float`]): Force vector per particle.

    The same constant force is applied to all particles, independently of time
    and their positions. This force is useful for simulating pressure-driven
    flow in conjunction with a confined geometry having no-slip boundary conditions.
    It is also useful for measuring diffusion coefficients with nonequilibrium
    methods.

    Attributes:
        force (`tuple` [`float`, `float`, `float`]): Force vector per particle.

    """

    def __init__(self, force):
        super().__init__()

        param_dict = ParameterDict(force=(float, float, float))
        param_dict["force"] = force
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.ConstantForce(
            _hoomd.make_scalar3(self.force[0], self.force[1], self.force[2]))
        super()._attach_hook()


class SineForce(SolventForce):
    r"""Sine force.

    Args:
        amplitude (float): Amplitude of the sinusoid.
        wavenumber (float): Wavenumber for the sinusoid.

    SineForce applies a force with amplitude *F* in *x* that is sinusoidally
    varying in *z* with wavenumber *k*:

    .. math::

        \mathbf{F}(\mathbf{r}) = F \sin (k r_z) \mathbf{e}_x

    Typically, the wavenumber should be something that is commensurate
    with the simulation box. For example, :math:`k = 2\pi/L_z` will generate
    one period of the sine.

    Attributes:
        amplitude (float): Amplitude of the sinusoid.

        wavenumber (float): Wavenumber for the sinusoid.

    """

    def __init__(self, amplitude, wavenumber):
        super().__init__()

        param_dict = ParameterDict(amplitude=float(amplitude),
                                   wavenumber=float(wavenumber))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.SineForce(self.amplitude, self.wavenumber)
        super()._attach_hook()
