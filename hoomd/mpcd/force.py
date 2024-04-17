# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD particle body forces.

MPCD can apply a body force to each MPCD particle as a function of position.
The external force should be compatible with the chosen
:class:`~hoomd.mpcd.geometry.Geometry`. Global momentum conservation can be
broken by adding a body force, so care should be chosen that the entire model
is designed so that the system does not have net acceleration. For example,
solid boundaries can be used to dissipate momentum, or a balancing force can be
applied to particles that are embedded in the MPCD particles through the
collision step. Additionally, a thermostat will likely be required to maintain
temperature control in the driven system.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

from hoomd import _hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.operation import _HOOMDBaseObject


class BodyForce(_HOOMDBaseObject):
    """Body force.

    The `BodyForce` is a body force applied to each MPCD particle. This
    class should not be instantiated directly. It exists for type checking.

    """

    pass


class BlockForce(BodyForce):
    r"""Block force.

    Args:
        force (float): Magnitude of the force in *x* per particle.
        separation (float): Distance between the centers of the blocks.
        width (float): Width of each block.

    The `force` magnitude *F* is applied in the *x* direction on the MPCD
    particles in blocks defined along the *y* direction by the `separation`
    :math:`2H` and the `width` :math:`2w`. The force in *x* is :math:`+F` in the
    upper block, :math:`-F` in the lower block, and zero otherwise.

    .. math::
        :nowrap:

        \begin{equation}
        \mathbf{F} = \begin{cases}
        +F \mathbf{e}_x & |r_y - H| < w \\
        -F \mathbf{e}_x & |r_y + H| < w \\
           \mathbf{0}   & \mathrm{otherwise}
        \end{cases}
        \end{equation}

    The `BlockForce` can be used to implement the double-parabola method for
    measuring viscosity using separation :math:`L_y/2` and width :math:`L_y/2`,
    where :math:`L_y` is the size of the simulation box in *y*.

    Warning:
        You should define the blocks to lie fully within the simulation box and
        to not overlap each other.

    .. rubric:: Example:

    Block force for double-parabola method.

    .. code-block:: python

        Ly = simulation.state.box.Ly
        force = hoomd.mpcd.force.BlockForce(
            force=1.0, separation=Ly/2, width=Ly/2)
        stream = hoomd.mpcd.stream.Bulk(period=1, mpcd_particle_force=force)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        force (float): Magnitude of the force in *x* per particle.

            .. rubric:: Example:

            .. code-block:: python

                force.force = 1.0

        separation (float): Distance between the centers of the blocks.

            .. rubric:: Example:

            .. code-block:: python

                Ly = simulation.state.box.Ly
                force.separation = Ly / 2

        width (float): Width of each block.

            .. rubric:: Example:

            .. code-block:: python

                Ly = simulation.state.box.Ly
                force.width = Ly / 2

    """

    def __init__(self, force, separation=None, width=None):
        super().__init__()

        param_dict = ParameterDict(
            force=float(force),
            separation=float(separation),
            width=float(width),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.BlockForce(self.force, self.separation,
                                         self.width)
        super()._attach_hook()


class ConstantForce(BodyForce):
    r"""Constant force.

    Args:
        force (`tuple` [`float`, `float`, `float`]): Force vector per particle.

    The same constant force is applied to all MPCD particles, independently
    of time and position. This force is useful for simulating pressure-driven
    flow in conjunction with a confined geometry having no-slip boundary
    conditions. It is also useful for measuring diffusion coefficients with
    nonequilibrium methods.

    .. rubric:: Example:

    .. code-block:: python

        force = hoomd.mpcd.force.ConstantForce((1.0, 0, 0))
        stream = hoomd.mpcd.stream.Bulk(period=1, mpcd_particle_force=force)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        force (`tuple` [`float`, `float`, `float`]): Force vector per particle.

            .. rubric:: Example:

            .. code-block:: python

                force.force = (1.0, 0.0, 0.0)

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


class SineForce(BodyForce):
    r"""Sine force.

    Args:
        amplitude (float): Amplitude of the sinusoid.
        wavenumber (float): Wavenumber for the sinusoid.

    `SineForce` applies a force with amplitude *F* in *x* that is sinusoidally
    varying in *y* with wavenumber *k* to all MPCD particles:

    .. math::

        \mathbf{F}(\mathbf{r}) = F \sin (k r_y) \mathbf{e}_x

    Typically, the wavenumber should be something that is commensurate
    with the simulation box. For example, :math:`k = 2\pi/L_y` will generate
    one period of the sine.

    .. rubric:: Example:

    Sine force with one period.

    .. code-block:: python

        Ly = simulation.state.box.Ly
        force = hoomd.mpcd.force.SineForce(
            amplitude=1.0,
            wavenumber=2 * numpy.pi / Ly)
        stream = hoomd.mpcd.stream.Bulk(period=1, mpcd_particle_force=force)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        amplitude (float): Amplitude of the sinusoid.

            .. rubric:: Example:

            .. code-block:: python

                force.amplitude = 1.0

        wavenumber (float): Wavenumber for the sinusoid.

            .. rubric:: Example:

            .. code-block:: python

                Ly = simulation.state.box.Ly
                force.wavenumber = 2 * numpy.pi / Ly

    """

    def __init__(self, amplitude, wavenumber):
        super().__init__()

        param_dict = ParameterDict(amplitude=float(amplitude),
                                   wavenumber=float(wavenumber))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.SineForce(self.amplitude, self.wavenumber)
        super()._attach_hook()
