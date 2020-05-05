import numpy as np
from functools import partial
import hoomd._hoomd as _hoomd


def _make_vec_three(vec, vec_factory, scalar_type):
    """A helper function for converting Python sequences to HOOMD T3 classes.

    Args:
        vec (Sequence[X] or X): A sequence of type X where X is the type
            returned by scalar_type.
        vec_factory (function): A function from `hoomd._hoomd` that makes a T3
            class (e.g. Scalar3, Int3).
        scalar_type (function): A function to convert a Python value into the
            base type for the vec_factory function. For `Scalar3` this would be
            `float`.
    """
    try:
        l_vec = len(vec)
    except TypeError:
        try:
            v = scalar_type(vec)
        except (ValueError, TypeError):
            raise ValueError("Expected value of type {}.".format(scalar_type))
        else:
            return vec_factory(v, v, v)
    if l_vec == 3:
        try:
            return vec_factory(scalar_type(vec[0]),
                               scalar_type(vec[1]),
                               scalar_type(vec[2]))
        except (ValueError, TypeError):
            raise ValueError("Expected values of type {}.".format(scalar_type))
    else:
        raise ValueError("Expected three or one value received {}.".format(
            len(vec)))


_make_scalar3 = partial(_make_vec_three,
                        vec_factory=_hoomd.make_scalar3,
                        scalar_type=float)

_make_int3 = partial(_make_vec_three,
                     vec_factory=_hoomd.make_int3,
                     scalar_type=int)

_make_char3 = partial(_make_vec_three,
                      vec_factory=_hoomd.make_char3,
                      scalar_type=int)


def _to_three_array(vec, dtype=None):
    return np.array((vec.x, vec.y, vec.z), dtype=dtype)


class Box:
    R""" Define box dimensions.

    Args:
        Lx (float): box extent in the x direction (distance units).
        Ly (float): box extent in the y direction (distance units).
        Lz (float): box extent in the z direction (distance units).
        xy (float): tilt factor xy (dimensionless).
        xz (float): tilt factor xz (dimensionless).
        yz (float): tilt factor yz (dimensionless).

    Simulation boxes in hoomd are specified by six parameters, ``Lx``, ``Ly``,
    ``Lz``, ``xy``, ``xz``, and ``yz``. `Box` provides a way to specify all
    six parameters for a given box and perform some common operations with them.
    A `Box` can be passed to an initialization method or to assigned to a
    saved :py:class:`State` variable (``state.box = new_box``) to set the
    simulation box.

    Access attributes directly::

        b = hoomd.box.Box(L=20)
        b.xy = 1.0
        b.yz = 0.5
        b.Lz = 40

    .. rubric:: Two dimensional systems

    2D simulations in HOOMD use boxes with ``Lz == 0``. 2D boxes ignore ``xz``
    and ``yz``. If a new `Box` is assigned to a system with differnt
    dimensionality, a warning will be shown.

    In 2D boxes, *volume* is in units of area.

    .. rubric:: Factory Methods

    `Box` has factory methods to enable easier creation of boxes:
    `cube` and `from_matrix`. See the method documentation for usage.

    Examples:

    * Cubic box with given length: ``Box.cube(L=1)``
    * Square box with given length: ``Box.cube(L=1, dimensions=2)``
    * From an upper triangular matrix: ``Box.from_matrix(matrix)``
    * Full spec: ``Box(Lx=1, Ly=2, Lz=3, xy=1., xz=2., yz=3.)``
    """

    # Constructors
    def __init__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0):
        if Lz != 0 and (xz != 0 or yz !=0):
            raise ValueError("Cannot set the xz or yz tilt factor on a 2D box.")
        self._cpp_obj = _hoomd.BoxDim(Lx, Ly, Lz)
        self._cpp_obj.setTiltFactors(xy, xz, yz)

    @classmethod
    def cube(cls, L):
        """Create a cube with side lengths ``L``.

        Args:
            L (float): The box side length (distance units).

        Returns:
            box (hoomd.Box): The created box.
        """
        return cls(L, L, L, 0, 0, 0)

    @classmethod
    def square(cls, L):
        """Create a square with side lengths ``L``.

        Args:
            L (float): The box side length (distance units).

        Returns:
            box (hoomd.Box): The created 2D box.
        """
        return cls(L, L, 0, 0, 0, 0)

    @classmethod
    def from_matrix(cls, box_matrix):
        """Create a box from an upper triangular matrix.

        Args:
            box_matrix ((3, 3) `numpy.ndarray` of `float`): An upper
                triangular matrix representing a box. The values for ``Lx``,
                ``Ly``, ``Lz``, ``xy``, ``xz``, and ``yz`` are related to the
                matrix by the following expressions.

                .. code-block:: python

                    [[Lx, Ly * xy, Lz * xz],
                    [0,  Ly,      Lz * yz],
                    [0,  0,       Lz]]


        Returns:
            box (hoomd.Box): The created box.
        """
        b = cls(0, 0)
        b.matrix = box_matrix
        return b

    @classmethod
    def _from_cpp(self, cpp_obj):
        b = Box()
        b._cpp_obj = cpp_obj
        return b

    # Dimension based properties
    @property
    def dimensions(self):
        """int: The dimensionality of the box.

        If ``Lz == 0``, the box is treated as 2D, otherwise it is 3D. This
        property is not settable.
        """
        return 2 if self.is2D else 3

    @property
    def is2D(self):
        """bool: Flag whether the box is 2D.

        If ``Lz == 0``, the box is treated as 2D, otherwise it is 3D. This
        property is not settable.
        """
        return self.Lz == 0

    # Length based properties
    @property
    def L(self):
        """(3) `numpy.ndarray` of `float`: The box lengths,
        ``[Lx, Ly, Lz]``.

        Can be set with a float which sets all lengths, or a length 3 vector.
        """
        return _to_three_array(self._cpp_obj.getL())

    @L.setter
    def L(self, new_L):
        newL = _make_scalar3(new_L)
        if newL.z == 0 and not self.is2D:
            self.tilts = [self.xy, 0, 0]
        self._cpp_obj.setL(newL)

    @property
    def Lx(self):
        """float: The length of the box in the x dimension."""
        return self.L[0]

    @Lx.setter
    def Lx(self, value):
        L = self.L
        L[0] = float(value)
        self.L = L

    @property
    def Ly(self):
        """float: The length of the box in the y dimension."""
        return self.L[1]

    @Ly.setter
    def Ly(self, value):
        L = self.L
        L[1] = float(value)
        self.L = L

    @property
    def Lz(self):
        """float: The length of the box in the z dimension."""
        return self.L[2]

    @Lz.setter
    def Lz(self, value):
        L = self.L
        L[2] = float(value)
        self.L = L

    # Box tilt based properties
    @property
    def tilts(self):
        """(3) `numpy.ndarray` of `float`: The box tilts, ``[xy, xz, yz]``.

        Can be set using one tilt for all axes or three tilts. If the box is 2D
        ``xz`` and ``yz`` will automatically be set to zero."""
        return np.array([self.xy, self.xz, self.yz])

    @tilts.setter
    def tilts(self, new_tilts):
        new_tilts = _make_scalar3(new_tilts)
        if self.is2D and (new_tilts.y != 0 or new_tilts.z != 0):
            raise ValueError("Cannot set the xz or yz tilt factor on a 2D box.")
        self._cpp_obj.setTiltFactors(new_tilts.x, new_tilts.y, new_tilts.z)

    @property
    def xy(self):
        """float: The tilt for the xy plane."""
        return self._cpp_obj.getTiltFactorXY()

    @xy.setter
    def xy(self, xy):
        self.tilts = [xy, self.xz, self.yz]

    @property
    def xz(self):
        """float: The tilt for the xz plane."""
        return self._cpp_obj.getTiltFactorXZ()

    @xz.setter
    def xz(self, xz):
        if self.is2D:
            raise ValueError("Cannot set xz tilt factor on a 2D box.")
        self.tilts = [self.xy, xz, self.yz]

    @property
    def yz(self):
        """float: The tilt for the yz plane."""
        return self._cpp_obj.getTiltFactorYZ()

    @yz.setter
    def yz(self, yz):
        if self.is2D:
            raise ValueError("Cannot set yz tilt factor on a 2D box.")
        self.tilts = [self.xy, self.xz, yz]

    # Misc. properties
    @property
    def periodic(self):
        """(3) `numpy.ndarray` of `bool`: The periodicity of
        each dimension."""
        return _to_three_array(self._cpp_obj.getPeriodic(), np.bool)

    @property
    def lattice_vectors(self):
        """(3, 3) `numpy.ndarray` of `float`: Box lattice vectors.

        The lattice vectors are read-only.
        """
        return np.concatenate(
            [_to_three_array(self._cpp_obj.getLatticeVector(i))
             for i in range(3)]).reshape(3, 3)

    @property
    def volume(self):
        """float: The current volume (area in 2D) of the box.

        When setting volume the aspect ratio of the box is maintained while the
        lengths are changed.
        """
        return self._cpp_obj.getVolume(self.is2D)

    @volume.setter
    def volume(self, volume):
        self.scale((volume / self.volume)**(1 / self.dimensions))

    @property
    def matrix(self):
        """(3, 3) `numpy.ndarray` `float`: The upper triangular matrix that
        defines the box.

        Can be used to set the box to one defined by an upper triangular
        matrix.

        .. code-block:: python

            [[Lx, Ly * xy, Lz * xz],
             [0,  Ly,      Lz * yz],
             [0,  0,       Lz]]

        """
        Lx, Ly, Lz = self.L
        xy, xz, yz = self.tilts
        return np.array([[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz], [0, 0, Lz]])

    @matrix.setter
    def matrix(self, box_matrix):
        box_matrix = np.asarray(box_matrix)
        if not np.allclose(box_matrix, np.triu(box_matrix)):
            raise ValueError("Box matrix must be upper triangular.")
        if box_matrix.shape != (3, 3):
            raise ValueError("Box matrix must be a 3x3 matrix.")
        L = np.diag(box_matrix)
        self.L = L
        self.xy = box_matrix[0, 1] / L[1]
        self.xz = box_matrix[0, 2] / L[2]
        self.yz = box_matrix[1, 2] / L[2]

    def scale(self, s):
        R"""Scale box dimensions.

        Scales the box by the given scale factors. Tilt factors are not
        modified.

        Args:
            s (float or Sequence[float]): scale factors in each dimension. If a
                single float is given then scale all dimensions by s; otherwise,
                s must be a sequence of 3 values used to scale each dimension.
        """
        s = np.asarray(s, dtype=np.float)
        self.L *= s

    # Magic Methods
    def __repr__(self):
        return "hoomd.box.Box(Lx={}, Ly={}, Lz={}, xy={}, xz={}, yz={})".format(
            self.Lx, self.Ly, self.Lz, self.xy, self.xz, self.yz)

    def __eq__(self, other):
        return self._cpp_obj == other._cpp_obj

    def __neq__(self, other):
        return self._cpp_obj != other._cpp_obj

#     def wrap(self, v, image=(0, 0, 0)):
#         R""" Wrap a vector using the periodic boundary conditions.

#         Args:
#             v (Sequence[float]): The vector to wrap of length 3.
#             image (Sequence[float]): A vector of integer image flags that will
#                 be updated (optional).

#         Returns:
#             The wrapped vector and the image flags as two numpy arrays.
#         """
#         u = _make_scalar3(v)
#         image = _make_int3(image)
#         c = _make_char3([0, 0, 0])
#         self._cpp_obj.wrap(u, image, c)
#         return _to_three_array(u), _to_three_array(image)

#     def min_image(self, v):
#         R""" Apply the minimum image convention to a vector.

#         Args:
#             v (Sequence[float]): The vector to apply minimum image to.

#         Returns:
#             The minimum image as a tuple.
#         """
#         u = _make_scalar3(v)
#         return _to_three_array(self._cpp_obj.minImage(u))

#     def make_fraction(self, v):
#         R""" Scale a vector to fractional coordinates.

#         make_fraction takes a vector in a box and computes a vector where all
#         components are between 0 and 1 representing their scaled position.

#         Args:
#             v (Sequence[float]): The vector to convert to fractional
#                 coordinates.

#         Returns:
#             The scaled vector.
#         """
#         u = _make_scalar3(v)
#         w = _make_scalar3([0., 0., 0.])
#         return _to_three_array(self._cpp_obj.makeFraction(u, w))
