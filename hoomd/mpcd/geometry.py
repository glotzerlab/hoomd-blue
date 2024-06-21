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


class CosineExpansionContraction(Geometry):
    r"""Cosine Expansion-Contraction Geometry (Symmetric).

    Args:
        expansion_separation (float): Maximum distance between channel walls.
        contraction_separation (float): Minimum distance between channel walls.
        wavenumber (float): Wavenumber of cosine.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `CosineExpansionContraction` geometry represents a fluid confined between
    two walls described by a sinusoidal profile with equations

    .. math::
        +/-(A cos(x*wavenumber) + A + {contraction\_separation}/2)

    where

    .. math::
        A = 0.25*({expansion\_separation}-{contraction\_separation})

    is the amplitude.
    The channel is axis-symmetric around the origin in
    *y* direction. The two symmetric cosine walls create a periodic series of

    .. math::
        p = wavenumber*Lx / 2\pi

    constrictions and expansions.
    The parameter :math:`expansion\_separation` gives the
    channel width at its widest and :math:`contraction\_separation` is the
    channel width at its narrowest point.

    TODO:  Install sybil to test your examples in docs.

    Example::

        def make_pos(density,H_w,h_s,period,Lx,Lz):
            A = (H_w-h_s)/2.
            channel_volume = 2*(h_s + A)*Lz*Lx
            N = int(density*channel_volume)
            buffer = (2*H*Lx*Lz/channel_volume)*1.5

            coords_x = np.random.uniform(-Lx/2.,+Lx/2.,size=int(N*buffer))
            coords_z = np.random.uniform(-Lz/2.,+Lz/2.,size=int(N*buffer))
            coords_y = np.random.uniform(-H,+H,size=int(N*buffer))

            cutoff = A*np.cos(coords_x*2*np.pi/Lx*period)+A+h_s

            coords_y_inside = coords_z[np.abs(coords_y)<cutoff-1e-5][0:N]
            coords_x_inside = coords_x[np.abs(coords_y)<cutoff-1e-5][0:N]
            coords_z_inside = coords_y[np.abs(coords_y)<cutoff-1e-5][0:N]

            p = np.vstack((coords_x_inside,coords_y_inside,coords_z_inside)).T
            return p

        kT = 1.0    # temperature
        rho = 3.0   # density

        Lx = 20.0   # box length
        Lz = 10.0    # box width
        H = 5.0     # half largest distance between walls
        h = 2.0     # half smallest distance between walls
        p = 1        # periodicty of sine wall, integer > 0
        Ly = 2*H+5

        pos = make_pos(rho,H,h,p,Lx,Lz)

        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

        snapshot.mpcd.types = ["A"]
        snapshot.mpcd.N = len(pos)
        snapshot.mpcd.position[:] = pos

        device = hoomd.device.CPU()
        simulation = hoomd.Simulation(device=device,seed=15)
        simulation.create_state_from_snapshot(snapshot)

        integrator = hoomd.mpcd.Integrator(dt=0.1)
        integrator.collision_method = hoomd.mpcd.collide.../
        StochasticRotationDynamics(period=1, angle=130, kT=kT)

        fx = 0.004

        geometry=hoomd.mpcd.geometry.CosineExpansionContraction(
            expansion_separation=2*H,
            contraction_separation=2*h,
            wavenumber=2.0 * np.pi / Lx,
            no_slip=False)


        integrator.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=geometry,
            mpcd_particle_force=hoomd.mpcd.force.ConstantForce(force=(fx, 0, 0))
        )

        filler = hoomd.mpcd.fill.GeometryFiller(
            type="A", density=5.0, kT=kT, geometry=geometry)

        integrator.virtual_particle_fillers.append(filler)

        integrator.solvent_sorter = hoomd.mpcd.tune.ParticleSorter(trigger=20)
        simulation.operations.integrator = integrator

        simulation.run(100)

    Attributes:
        expansion_separation (float): Maximum distance between channel walls
        (*read only*).
        contraction_separation (float): Minimum distance between channel walls
        (*read only*).
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


class CosineChannel(Geometry):
    r"""Cosine Channel Geometry (Anti-symmetric).

    Args:
        amplitude (float): Amplitude of cosine.
        wavenumber (float): Wavenumber of cosine.
        separation (float): Distance between channel walls.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `CosineChannel` geometry represents a fluid confined between two
    walls described by a sinusoidal profile with equations

    .. math::
        amplitude*cos(x*wavenumber) +/- {separation}/2

    The channel is
    anti-symmetric around the origin in *y* direction. The cosines of top and
    bottom are running in parallel and create a "wavy" channel with


    .. math::
        p = wavenumber*L_{x} / 2\pi

    repetitions.
    The parameter :math:`separation` gives the
    channel width.

    TODO:  Install sybil to test your examples in docs.

    Example::

        def make_pos(density,A,h_narrow,repetitions,Lx,Lz):
            Ly = 2*(A+h_narrow+5.0)
            total_volume = Lz*Ly*Lx
            N = int(density*total_volume)
            posx = np.random.uniform(-Lx/2.,+Lx/2.,size=int(N))
            posy = np.random.uniform(-Ly/2.,+Ly/2.,size=int(N))
            posz = np.random.uniform(-Lz/2.,+Lz/2.,size=int(N))

            pos = np.column_stack((posx,posy,posz))

            pos = pos[np.abs(pos[:,1] - A*np.cos(2.0*np.pi*repetitions*../
            pos[:,0]/Lx))<h_narrow]

            return Ly,pos

        kT = 1.0    # temperature
        rho = 3.0   # density

        Lx = 20.0   # box length
        Lz = 10.0    # box width
        A = 5.0     # amplitude of the wave
        h = 1.0     # half smallest distance between walls
        p = 1        # periodicty of sine wall, integer > 0


        Ly, pos = make_pos(rho,A,h,p,Lx,Lz)

        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = [Lx, Ly, Lz, 0, 0, 0]

        snapshot.mpcd.types = ["A"]
        snapshot.mpcd.N = len(pos)
        snapshot.mpcd.position[:] = pos

        device = hoomd.device.CPU()
        simulation = hoomd.Simulation(device=device,seed=15)
        simulation.create_state_from_snapshot(snapshot)

        integrator = hoomd.mpcd.Integrator(dt=0.1)
        integrator.collision_method = hoomd.mpcd.collide../
        .StochasticRotationDynamics(period=1, angle=130, kT=kT)

        fx = 0.004

        geometry=hoomd.mpcd.geometry.CosineChannel(
            amplitude=A,
            separation=2*h,
            wavenumber=2.0 * np.pi / Lx,
            no_slip=False)


        integrator.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=geometry,
            mpcd_particle_force=hoomd.mpcd.force.ConstantForce(force=(fx, 0, 0))
        )

        filler = hoomd.mpcd.fill.GeometryFiller(
            type="A", density=5.0, kT=kT, geometry=geometry)

        integrator.virtual_particle_fillers.append(filler)

        integrator.solvent_sorter = hoomd.mpcd.tune.ParticleSorter(trigger=20)
        simulation.operations.integrator = integrator

        simulation.run(100)


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
