import numpy as np
from functools import partial
import hoomd._hoomd as _hoomd


def make_three_vec(vec, func, scalar_conversion=None, is2D=False, name=None):
    if scalar_conversion is None:
        def sc(x):
            return x
    else:
        sc = scalar_conversion
    if len(vec) == 2 and is2D:
        return func(sc(vec[0]), sc(vec[1]), sc(0))
    elif len(vec) == 3:
        return func(sc(vec[0]), sc(vec[1]), sc(vec[2]))
    else:
        if name is None:
            name = "vector"
        if not is2D and len(vec) == 2:
            raise ValueError("Box is 3 dimensions, but {name} was length 2. "
                             "Give a 3 dimensional vector for {name}."
                             "".format(name=name))
        else:
            raise ValueError("Length of the {} must be 2 or 3 dimensions."
                             "".format(name))


make_scalar3 = partial(make_three_vec, func=_hoomd.make_scalar3,
                       scalar_conversion=float)

make_int3 = partial(make_three_vec, func=_hoomd.make_int3,
                    scalar_conversion=int)

make_uint3 = partial(make_three_vec, func=_hoomd.make_uint3,
                     scalar_conversion=int)

make_char3 = partial(make_three_vec, func=_hoomd.make_char3,
                     scalar_conversion=int)


def to_three_array(vec):
    return np.array((vec.x, vec.y, vec.z))


class LatticeVectors:
    def __init__(self, cpp_box):
        self._cpp_obj = cpp_box

    def __getitem__(self, index):
        if index < 0 or index > 2:
            raise ValueError("The index for the lattice vector must be 0, 1, "
                             "or 2.")

        return to_three_array(self._cpp_obj.getLatticeVector(index))


class Box:
    R""" Define box dimensions.

    Args:
        Lx (float): box extent in the x direction (distance units)
        Ly (float): box extent in the y direction (distance units)
        Lz (float): box extent in the z direction (distance units)
        xy (float): tilt factor xy (dimensionless)
        xz (float): tilt factor xz (dimensionless)
        yz (float): tilt factor yz (dimensionless)
        dimensions (int): Number of dimensions in the box (2 or 3).
        L (float): shorthand for specifying Lx=Ly=Lz=L (distance units)
        volume (float): Scale the given box dimensions up to the this volume (area if dimensions=2)

    Simulation boxes in hoomd are specified by six parameters, *Lx*, *Ly*, *Lz*,
    *xy*, *xz* and *yz*. For full details, see TODO: ref page. A boxdim provides
    a way to specify all six parameters for a given box and perform some common
    operations with them. Modifying a boxdim does not modify the underlying
    simulation box in hoomd. A boxdim can be passed to an initialization method
    or to assigned to a saved sysdef variable (`system.box = new_box`) to set
    the simulation box.

    Access attributes directly::

        b = data.boxdim(L=20);
        b.xy = 1.0;
        b.yz = 0.5;
        b.Lz = 40;


    .. rubric:: Two dimensional systems

    2D simulations in hoomd are embedded in 3D boxes with short heights in the z
    direction. To create a 2D box, set dimensions=2 when creating the boxdim.
    This will force Lz=1 and xz=yz=0. init commands that support 2D boxes will
    pass the dimensionality along to the system. When you assign a new boxdim to
    an already initialized system, the dimensionality flag is ignored. Changing
    the number of dimensions during a simulation run is not supported.

    In 2D boxes, *volume* is in units of area.

    .. rubric:: Shorthand notation

    data.boxdim accepts the keyword argument *L=x* as shorthand notation for
    `Lx=x, Ly=x, Lz=x` in 3D and `Lx=x, Ly=z, Lz=1` in 2D. If you specify both
    `L=` and `Lx,Ly, or Lz`, then the value for `L` will override the others.

    Examples:

    * Cubic box with given volume: `data.boxdim(volume=V)`
    * Triclinic box in 2D with given area: `data.boxdim(xy=1.0, dimensions=2,
      volume=A)`
    * Rectangular box in 2D with given area and aspect ratio: `data.boxdim(Lx=1,
      Ly=aspect, dimensions=2, volume=A)`
    * Cubic box with given length: `data.boxdim(L=10)`
    * Fully define all box parameters: `data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0,
      xz=0.5, yz=0.1)`
    """

    def __init__(self, Lx=1.0, Ly=1.0, Lz=1.0, xy=0.0, xz=0.0, yz=0.0):
        self._cpp_obj = _hoomd.BoxDim(Lx, Ly, Lz)
        self._cpp_obj.setTiltFactors(xy, xz, yz)
        self._lattice_vectors = LatticeVectors(self._cpp_obj)

    # Dimension based properties
    @property
    def dimension(self):
        return 2 if self.Lz == 0 else 3

    @property
    def is2D(self):
        return self.Lz == 0

    # Length based properties
    @property
    def L(self):
        return to_three_array(self._cpp_obj.getL())

    @L.setter
    def L(self, new_L):
        if len(new_L) == 2:
            new_L = [new_L[0], new_L[1], 0]
        self._cpp_obj.setL(make_scalar3(new_L))

    @property
    def Lx(self):
        return self.L[0]

    @Lx.setter
    def Lx(self, value):
        L = self._cpp_obj.getL()
        L.x = float(value)
        self.L = L

    @property
    def Ly(self):
        return self.L[1]

    @Ly.setter
    def Ly(self, value):
        L = self._cpp_obj.getL()
        L.y = float(value)
        self.L = L

    @property
    def Lz(self):
        return self.L[2]

    @Lz.setter
    def Lz(self, value):
        L = self._cpp_obj.getL()
        L.z = float(value)
        self.L = L

    # Box tilt based properties
    @property
    def tilts(self):
        return np.array([self.xy, self.xz, self.yz])

    @tilts.setter
    def tilts(self, new_tilts):
        if isinstance(new_tilts, (float, int)) or len(new_tilts) == 1:
            if self.is2D:
                self._cpp_obj.setTiltFactors(new_tilts[0], 0, 0)
            else:
                raise ValueError("Must specify 3 tilt factors for 3D box.")
        elif len(new_tilts) == 3:
            self._cpp_obj.setTiltFactors(new_tilts[0], new_tilts[1],
                                         new_tilts[2])
        else:
            raise ValueError("Tilt array must be either the box dimension "
                             "or three.")

    @property
    def xy(self):
        return self._cpp_obj.getTiltFactorXY()

    @xy.setter
    def xy(self, xy):
        self.tilts = [xy, self.xz, self.yz]

    @property
    def xz(self):
        return self._cpp_obj.getTiltFactorXZ()

    @xz.setter
    def xz(self, xz):
        self.tilts = [self.xy, xz, self.yz]

    @property
    def yz(self):
        return self._cpp_obj.getTiltFactorYZ()

    @yz.setter
    def yz(self, yz):
        self.tilts = [self.xy, self.xz, yz]

    # Misc. properties
    @property
    def periodicity(self):
        return to_three_array(self._cpp_obj.getPeriodic())

    @property
    def lattice_vectors(self):
        """Box lattice vectors"""
        return self._lattice_vectors

    @property
    def volume(self):
        return self._cpp_obj.getVolume(self.is2D)

    @volume.setter
    def volume(self, vol):
        R""" Set the box volume.

        Scale the box to the given volume (or area).

        Args:
            volume (float): new box volume (area if dimensions=2)

        Returns:
            A reference to the modified box.
        """
        cur_vol = self.volume

        if self.dimension == 3:
            s = (vol / cur_vol) ** (1.0 / 3.0)
        else:
            s = (vol / cur_vol) ** (1.0 / 2.0)
        self.scale(s)

    @property
    def matrix(self):
        Lx, Ly, Lz = self.L
        xy, xz, yz = self.tilts
        return np.array([[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz], [0, 0, Lz]])

    @matrix.setter
    def matrix(self, box_matrix):
        if not isinstance(box_matrix, np.ndarray):
            box_matrix = np.array(box_matrix)
        # Error checking
        if box_matrix.shape == (2, 2):
            if box_matrix[1, 0] != 0:
                raise ValueError("Box matrix must be upper triangular.")
        elif box_matrix.shape == (3, 3):
            if box_matrix[1, 0] != 0 or any(box_matrix[2, :-1] != 0):
                raise ValueError("Box matrix must be upper triangular.")
        else:
            raise ValueError("Box matrix must be shape (2, 2) or (3, 3)")

        L = np.diag(box_matrix)
        self.L = L
        self.xy = box_matrix[0, 1] / L[1]
        if box_matrix.shape == (3, 3):
            self.xz = box_matrix[0, 2] / L[2]
            self.yz = box_matrix[1, 2] / L[2]

    def scale(self, *args):
        R""" Scale box dimensions.

        Scales the box by the given scale factors.Tilt factors are not
        modified.

        Args:
            args (float): scale factors in each dimension. If box is two
            dimensions then two scale factors is acceptable, but if the box is
            three dimensions this will error. One scaling factor can be given
            and used to scale all dimensions.

        Returns:
            A reference to the modified box.
        """
        Nargs = len(args)
        if Nargs != 1 and Nargs != self.dimension:
            raise ValueError("The number of scaling factors must be 1 or {}. "
                             "Given scaling factors {}".format(self.dimension,
                                                               Nargs))
        if Nargs == 1:
            new_L = args[0] * self.L
        else:
            new_L = np.array(args) * self.L[:Nargs]
        self.L = new_L
        return self

    def wrap(self, v, image=(0, 0, 0)):
        R""" Wrap a vector using the periodic boundary conditions.

        Args:
            v (tuple): The vector to wrap
            image (tuple): A vector of integer image flags that will be updated
                (optional)

        Returns:
            The wrapped vector and the image flags as two numpy arrays.
        """
        u = make_scalar3(v, is2D=self.is2D, name='v')
        image = make_int3(image, is2D=self.is2D, name="img")
        c = make_char3([0, 0, 0])
        self._cpp_obj.wrap(u, image, c)
        return to_three_array(u), to_three_array(image)

    def min_image(self, v):
        R""" Apply the minimum image convention to a vector.

        Args:
            v (tuple): The vector to apply minimum image to

        Returns:
            The minimum image as a tuple.

        """
        u = make_scalar3(v, is2D=self.is2D, name="v")
        return to_three_array(self._cpp_obj.minImage(u))

    def make_fraction(self, v):
        R""" Scale a vector to fractional coordinates.

        Args:
            v (tuple): The vector to convert to fractional coordinates

        make_fraction takes a vector in a box and computes a vector where all
        components are between 0 and 1.

        Returns:
            The scaled vector.
        """
        u = make_scalar3(v, is2D=self.is2D, name="v")
        w = make_scalar3([0., 0., 0.])
        return to_three_array(self._cpp_obj.makeFraction(u, w))

    def __repr__(self):
        return "Box(Lx={}, Ly={}, Lz={}, xy={}, xz={}, yz={})".format(
            self.Lx, self.Ly, self.Lz, self.xy, self.xz, self.yz)

    @classmethod
    def cube(cls, L, dimension=3):
        if dimension == 3:
            return cls(L, L, L, 0, 0, 0)
        else:
            return cls(L, L, 0, 0, 0, 0)

    @classmethod
    def from_matrix(cls, box_matrix):
        b = cls()
        b.matrix = box_matrix
        return b

    @classmethod
    def _from_cpp(self, cpp_obj):
        b = Box()
        b._cpp_obj = cpp_obj
        return b

    def __copy__(self):
        return type(self)(*self.L, *self.tilts)

    # \internal
    # \brief Get a dictionary representation of the box dimensions
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['d'] = self.dimensions
        data['Lx'] = self.Lx
        data['Ly'] = self.Ly
        data['Lz'] = self.Lz
        data['xy'] = self.xy
        data['xz'] = self.xz
        data['yz'] = self.yz
        data['V'] = self.get_volume()
        return data
