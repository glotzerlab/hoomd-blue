import numpy as np
from functools import partial
import hoomd._hoomd as _hoomd


def _make_vec3(vec, vec_factory, scalar_type):
    """Converts Python types to HOOMD T3 classes (e.g. Scalar3, Int3).

    Args:
        vec (Sequence[T] or T): A sequence or scalar of type ``scalar_type``.
        vec_factory (function): A function from `hoomd._hoomd` that makes a T3
            class (e.g. Scalar3, Int3).
        scalar_type (class): A class defining the base type ``T`` for the
            vec_factory function. For `Scalar3` this would be `float`.
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
        raise ValueError("Expected a sequence of three values or a single "
                         "value. Received {} values.".format(len(vec)))


_make_scalar3 = partial(_make_vec3,
                        vec_factory=_hoomd.make_scalar3,
                        scalar_type=float)

_make_int3 = partial(_make_vec3,
                     vec_factory=_hoomd.make_int3,
                     scalar_type=int)

_make_char3 = partial(_make_vec3,
                      vec_factory=_hoomd.make_char3,
                      scalar_type=int)


def _vec3_to_array(vec, dtype=None):
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

        box = hoomd.Box.cube(L=20)
        box.xy = 1.0
        box.yz = 0.5
        box.Lz = 40

    .. rubric:: Two dimensional systems

    2D simulations in HOOMD use boxes with ``Lz == 0``. 2D boxes ignore ``xz``
    and ``yz``. If a new `Box` is assigned to a system with different
    dimensionality, a warning will be shown.

    In 2D boxes, *volume* is in units of area.

    .. rubric:: Factory Methods

    `Box` has factory methods to enable easier creation of boxes: `cube`,
    `square`, `from_matrix`, and `from_box`. See the method documentation for
    usage.

    Examples:

    * Cubic box with given length: ``hoomd.Box.cube(L=1)``
    * Square box with given length: ``hoomd.Box.square(L=1)``
    * From an upper triangular matrix: ``hoomd.Box.from_matrix(matrix)``
    * Specify all values: ``hoomd.Box(Lx=1., Ly=2., Lz=3., xy=1., xz=2.,
      yz=3.)``
    """

    # Constructors
    def __init__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0):
        if Lz == 0 and (xz != 0 or yz !=0):
            raise ValueError("Cannot set the xz or yz tilt factor on a 2D box.")
        self._cpp_obj = _hoomd.BoxDim(Lx, Ly, Lz)
        self._cpp_obj.setTiltFactors(xy, xz, yz)

    @classmethod
    def cube(cls, L):
        """Create a cube with side lengths ``L``.

        Args:
            L (float): The box side length (distance units).

        Returns:
            hoomd.Box: The created 3D box.
        """
        return cls(L, L, L, 0, 0, 0)

    @classmethod
    def square(cls, L):
        """Create a square with side lengths ``L``.

        Args:
            L (float): The box side length (distance units).

        Returns:
            hoomd.Box: The created 2D box.
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
            hoomd.Box: The created box.
        """
        b = cls(0, 0)
        b.matrix = box_matrix
        return b

    @classmethod
    def _from_cpp(self, cpp_obj):
        """Wrap a C++ BoxDim.

        Does not copy the C++ object.
        """
        b = Box(0, 0)
        b._cpp_obj = cpp_obj
        return b

    @classmethod
    def from_box(cls, box):
        R"""Initialize a Box instance from a box-like object.

        Args:
            box:
                A box-like object

        .. note:: Objects that can be converted to HOOMD-blue boxes include
                  lists like :code:`[Lx, Ly, Lz, xy, xz, yz]`,
                  dictionaries with keys
                  :code:`'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz',
                  objects with attributes
                  :code:`Lx, Ly, Lz, xy, xz, yz,
                  3x3 matrices (see `from_matrix`),
                  or existing :class:`hoomd.Box` objects.

                  If any of :code:`Lz, xy, xz, yz` are not provided, they will
                  be set to 0.

                  If all values are provided, a triclinic box will be
                  constructed. If only :code:`Lx, Ly, Lz` are provided, an
                  orthorhombic box will be constructed. If only :code:`Lx, Ly`
                  are provided, a rectangular (2D) box will be constructed.

        Returns:
            :class:`hoomd.Box`: The resulting box object.
        """
        if np.asarray(box).shape == (3, 3):
            # Handles 3x3 matrices
            return cls.from_matrix(box)
        try:
            # Handles hoomd.box.Box and objects with attributes
            Lx = box.Lx
            Ly = box.Ly
            Lz = getattr(box, 'Lz', 0)
            xy = getattr(box, 'xy', 0)
            xz = getattr(box, 'xz', 0)
            yz = getattr(box, 'yz', 0)
        except AttributeError:
            try:
                # Handle dictionary-like
                Lx = box['Lx']
                Ly = box['Ly']
                Lz = box.get('Lz', 0)
                xy = box.get('xy', 0)
                xz = box.get('xz', 0)
                yz = box.get('yz', 0)
            except (IndexError, KeyError, TypeError):
                if not len(box) in [2, 3, 6]:
                    raise ValueError(
                        "List-like objects must have length 2, 3, or 6 to be "
                        "converted to freud.box.Box.")
                # Handle list-like
                Lx = box[0]
                Ly = box[1]
                Lz = box[2] if len(box) > 2 else 0
                xy, xz, yz = box[3:6] if len(box) == 6 else (0, 0, 0)
        except:  # noqa
            raise

        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)

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
        return _vec3_to_array(self._cpp_obj.getL())

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
        return _vec3_to_array(self._cpp_obj.getPeriodic(), bool)

    @property
    def lattice_vectors(self):
        """(3, 3) `numpy.ndarray` of `float`: Box lattice vectors.

        The lattice vectors are read-only.
        """
        return np.concatenate(
            [_vec3_to_array(self._cpp_obj.getLatticeVector(i))
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
        s = np.asarray(s, dtype=float)
        self.L *= s

    # Magic Methods
    def __repr__(self):
        return "hoomd.box.Box(Lx={}, Ly={}, Lz={}, xy={}, xz={}, yz={})".format(
            self.Lx, self.Ly, self.Lz, self.xy, self.xz, self.yz)

    def __eq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return self._cpp_obj == other._cpp_obj

    def __neq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return self._cpp_obj != other._cpp_obj

    def __reduce__(self):
        return (type(self), (*self.L, *self.tilts))
