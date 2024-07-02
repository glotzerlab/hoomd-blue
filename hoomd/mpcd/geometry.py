# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD geometries.

A geometry defines solid boundaries that cannot be penetrated. These
geometries are used for various operations in the MPCD algorithm including:

* Bounce-back streaming for MPCD particles
  (:class:`hoomd.mpcd.stream.BounceBack`)
* Bounce-back integration for MD particles
  (:class:`hoomd.mpcd.methods.BounceBack`)
* Virtual particle filling (:class:`hoomd.mpcd.fill.GeometryFiller`)

Each geometry may put constraints on the size of the simulation and where
particles are allowed. These constraints will be documented by each object.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.operation import _HOOMDBaseObject


class Geometry(_HOOMDBaseObject):
    r"""Geometry.

    Args:
        no_slip (bool): If True, surfaces have a no-slip boundary condition.
            Otherwise, they have a slip boundary condition.

    .. invisible-code-block: python

        geometry = hoomd.mpcd.geometry.Geometry(no_slip=True)

    Attributes:
        no_slip (bool): If True, plates have a no-slip boundary condition.
            Otherwise, they have a slip boundary condition (*read only*).

            A no-slip boundary condition means that the average velocity is
            zero at the surface. A slip boundary condition means that the
            average *normal* velocity is zero at the surface, but there
            is no friction against the *tangential* velocity.

    """

    def __init__(self, no_slip):
        super().__init__()

        param_dict = ParameterDict(no_slip=bool(no_slip),)
        self._param_dict.update(param_dict)


class CosineChannel(Geometry):
    r"""Serpentine (sinusoidal) channel.

    Args:
        amplitude (float): Amplitude of cosine.
        wavenumber (float): Wavenumber of cosine.
        separation (float): Distance between channel walls.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `CosineChannel` models a fluid confined in :math:`y` between two
    walls described by a sinusoidal profile with equations

    .. math::

        y(x) = A \cos(k x) \pm H

    where :math:`A` is the `amplitude`, :math:`k` is the `wavenumber`, and
    :math:`2H` is the `separation`.

    .. rubric:: Example:

    .. code-block:: python

        channel = hoomd.mpcd.geometry.CosineChannel(
            amplitude=2.0,
            separation=4.0,
            wavenumber=2.0 * numpy.pi / 10.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=channel)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        amplitude (float): Amplitude of cosine (*read only*).

        wavenumber (float): Wavenumber of cosine (*read only*).

        separation (float): Distance between walls (*read only*).


    """

    def __init__(self, amplitude, wavenumber, separation, no_slip=True):
        super().__init__(no_slip)

        param_dict = ParameterDict(
            amplitude=float(amplitude),
            wavenumber=float(wavenumber),
            separation=float(separation),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.CosineChannel(self.amplitude, self.wavenumber,
                                            self.separation, self.no_slip)
        super()._attach_hook()


class CosineExpansionContraction(Geometry):
    r"""Channel with sinusoidal expansion and contraction.

    Args:
        expansion_separation (float): Maximum distance between channel walls.
        contraction_separation (float): Minimum distance between channel walls.
        wavenumber (float): Wavenumber of cosine.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `CosineExpansionContraction` models a fluid confined in :math:`y` between
    two walls described by a sinusoidal profile with equations

    .. math::

        y(x) = \pm\left[ \frac{H_{\rm e}-H_{\rm c}}{2}
                         \left(1 + \cos(k x)\right) +  H_{\rm c} \right]

    where :math:`2 H_{\rm e}` is the `expansion_separation`, :math:`2 H_{\rm c}`
    is the `contraction_separation`, and :math:`k` is the `wavenumber`.

    .. rubric:: Example:

    .. code-block:: python

        channel = hoomd.mpcd.geometry.CosineExpansionContraction(
            expansion_separation=6.0,
            contraction_separation=3.0,
            wavenumber=2.0 * numpy.pi / 10.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=channel)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        contraction_separation (float): Distance between channel walls at
            the minimum contraction (*read only*).

        expansion_separation (float): Distance between channel walls at the
            maximum expansion (*read only*).

        wavenumber (float): Wavenumber of cosine (*read only*).

    """

    def __init__(self,
                 expansion_separation,
                 contraction_separation,
                 wavenumber,
                 no_slip=True):
        super().__init__(no_slip)

        param_dict = ParameterDict(
            expansion_separation=float(expansion_separation),
            contraction_separation=float(contraction_separation),
            wavenumber=float(wavenumber),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.CosineExpansionContraction(
            self.expansion_separation, self.contraction_separation,
            self.wavenumber, self.no_slip)
        super()._attach_hook()


class ParallelPlates(Geometry):
    r"""Parallel-plate channel.

    Args:
        separation (float): Distance between plates.
        speed (float): Wall speed.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `ParallelPlates` confines particles between two infinite parallel plates
    centered around the origin. The plates are placed at :math:`y=-H` and
    :math:`y=+H`, where the total `separation` is :math:`2H`. The plates may be
    put into motion with `speed` *V*, having velocity :math:`-V` and :math:`+V`
    in the *x* direction, respectively. If combined with a no-slip boundary
    condition, this motion can be used to generate simple shear flow.

    .. rubric:: Examples:

    Stationary parallel plates with no-slip boundary condition.

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(separation=6.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=plates)
        simulation.operations.integrator.streaming_method = stream

    Stationary parallel plates with slip boundary condition.

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(
            separation=6.0, no_slip=False)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=plates)
        simulation.operations.integrator.streaming_method = stream

    Moving parallel plates.

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(
            separation=6.0, speed=1.0, no_slip=True)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=plates)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        separation (float): Distance between plates (*read only*).

        speed (float): Wall speed (*read only*).

            `speed` will have no effect if `no_slip` is False because the slip
            surface cannot generate shear stress.

    """

    def __init__(self, separation, speed=0.0, no_slip=True):
        super().__init__(no_slip)
        param_dict = ParameterDict(
            separation=float(separation),
            speed=float(speed),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.ParallelPlates(self.separation, self.speed,
                                             self.no_slip)
        super()._attach_hook()


class PlanarPore(Geometry):
    r"""Pore with parallel plate opening.

    Args:
        separation (float): Distance between pore walls.
        length (float): Pore length.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `PlanarPore` is a finite-length version of `ParallelPlates`. The
    geometry is similar, except that the plates extend from :math:`x=-L` to
    :math:`x=+L` (total `length` *2L*). Additional solid walls
    with normals in *x* prevent penetration into the regions above / below the
    plates. The plates are infinite in *z*. Outside the pore, the simulation box
    has full periodic boundaries; it is not confined by any walls. This model
    hence mimics a narrow pore in, e.g., a membrane.

    .. rubric:: Example:

    .. code-block:: python

        pore = hoomd.mpcd.geometry.PlanarPore(separation=6.0, length=4.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=pore)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        separation (float): Distance between pore walls (*read only*).

        length (float): Pore length (*read only*).

    """

    def __init__(self, separation, length, no_slip=True):
        super().__init__(no_slip)

        param_dict = ParameterDict(
            separation=float(separation),
            length=float(length),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.PlanarPore(self.separation, self.length,
                                         self.no_slip)
        super()._attach_hook()


class Sphere(Geometry):
    r"""Spherical confinement.

    Args:
        radius (float): Radius of sphere.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `Sphere` confines particles inside a sphere of `radius` :math:`R`
    centered at the origin.

    .. rubric:: Examples:

    Sphere with no-slip boundary condition.

    .. code-block:: python

        sphere = hoomd.mpcd.geometry.Sphere(radius=5.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=sphere)
        simulation.operations.integrator.streaming_method = stream

    Sphere with slip boundary condition.

    .. code-block:: python

        sphere = hoomd.mpcd.geometry.Sphere(radius=5.0, no_slip=False)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=sphere)
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        radius (float): Radius of sphere (*read only*).

    """

    def __init__(self, radius, no_slip=True):
        super().__init__(no_slip)
        param_dict = ParameterDict(radius=float(radius),)
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.Sphere(self.radius, self.no_slip)
        super()._attach_hook()
