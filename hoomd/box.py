# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Box."""

import abc
import typing

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
            return vec_factory(scalar_type(vec[0]), scalar_type(vec[1]),
                               scalar_type(vec[2]))
        except (ValueError, TypeError):
            raise ValueError("Expected values of type {}.".format(scalar_type))
    else:
        raise ValueError("Expected a sequence of three values or a single "
                         "value. Received {} values.".format(len(vec)))


_make_scalar3 = partial(_make_vec3,
                        vec_factory=_hoomd.make_scalar3,
                        scalar_type=float)

_make_int3 = partial(_make_vec3, vec_factory=_hoomd.make_int3, scalar_type=int)

_make_char3 = partial(_make_vec3,
                      vec_factory=_hoomd.make_char3,
                      scalar_type=int)


def _vec3_to_array(vec, dtype=None):
    return np.array((vec.x, vec.y, vec.z), dtype=dtype)


class Box:
    """Define box dimensions.

    Args:
        Lx (float): box extent in the x direction :math:`[\\mathrm{length}]`.
        Ly (float): box extent in the y direction :math:`[\\mathrm{length}]`.
        Lz (float): box extent in the z direction :math:`[\\mathrm{length}]`.
        xy (float): tilt factor xy :math:`[\\mathrm{dimensionless}]`.
        xz (float): tilt factor xz :math:`[\\mathrm{dimensionless}]`.
        yz (float): tilt factor yz :math:`[\\mathrm{dimensionless}]`.

    .. image:: box.svg
       :alt: Example simulation box labelled with lengths and vectors.

    Particles in a simulation exist in a triclinic box with
    periodic boundary conditions. A triclinic box is defined by six values: the
    extents :math:`L_x`, :math:`L_y` and :math:`L_z` of the box in the three
    directions, and three tilt factors :math:`xy`, :math:`xz` and :math:`yz`.

    The parameter matrix is defined in terms of the lattice
    vectors :math:`\\vec a_1`, :math:`\\vec a_2` and :math:`\\vec a_3`:

    .. math::

        \\left( \\vec a_1, \\vec a_2, \\vec a_3 \\right)

    The first lattice vector :math:`\\vec a_1` is parallel to the unit vector
    :math:`\\vec e_x = (1,0,0)`. The tilt factor :math:`xy` indicates how the
    second lattice vector :math:`\\vec a_2` is tilted with respect to the first
    one. Similarly, :math:`xz` and :math:`yz` indicate the tilt of the third
    lattice vector :math:`\\vec a_3` with respect to the first and second
    lattice vector.

    The full cell parameter matrix is:

    .. math::
        \\left(\\begin{array}{ccc} L_x & xy L_y & xz L_z \\\\
                                   0   & L_y    & yz L_z \\\\
                                   0   & 0      & L_z    \\\\
                \\end{array}\\right)

    The tilt factors :math:`xy`, :math:`xz` and :math:`yz` are dimensionless.
    The relationships between the tilt factors and the box angles
    :math:`\\alpha`, :math:`\\beta` and :math:`\\gamma` are as follows:

    .. math::
        \\cos\\gamma &= \\cos(\\angle\\vec a_1, \\vec a_2) &=&
            \\frac{xy}{\\sqrt{1+xy^2}}\\\\
        \\cos\\beta &= \\cos(\\angle\\vec a_1, \\vec a_3) &=&
            \\frac{xz}{\\sqrt{1+xz^2+yz^2}}\\\\
        \\cos\\alpha &= \\cos(\\angle\\vec a_2, \\vec a_3) &=&
            \\frac{xy \\cdot xz + yz}{\\sqrt{1+xy^2} \\sqrt{1+xz^2+yz^2}}

    Given an arbitrarily oriented lattice with box vectors :math:`\\vec v_1,
    \\vec v_2, \\vec v_3`, the parameters for the rotated box can be found as
    follows:

    .. math::
        L_x &= v_1\\\\
        a_{2x} &= \\frac{\\vec v_1 \\cdot \\vec v_2}{v_1}\\\\
        L_y &= \\sqrt{v_2^2 - a_{2x}^2}\\\\
        xy &= \\frac{a_{2x}}{L_y}\\\\
        L_z &= \\vec v_3 \\cdot \\frac{\\vec v_1 \\times \\vec v_2}{\\left|
            \\vec v_1 \\times \\vec v_2 \\right|}\\\\
        a_{3x} &= \\frac{\\vec v_1 \\cdot \\vec v_3}{v_1}\\\\
        xz &= \\frac{a_{3x}}{L_z}\\\\
        yz &= \\frac{\\vec v_2 \\cdot \\vec v_3 - a_{2x}a_{3x}}{L_y L_z}

    .. rubric:: Box images

    HOOMD-blue always stores particle positions :math:`\\vec{r}` inside the
    primary box image which includes the origin at the center. The primary box
    image include the left, bottom, and back face while excluding the right,
    top, and front face. In cubic boxes, this implies that the particle
    coordinates in the primary box image are in the interval :math:`\\left[
    -\\frac{L}{2},\\frac{L}{2} \\right)`.

    Unless otherwise noted in the documentation, operations apply the
    minimum image convention when computing pairwise interactions between
    particles:

    .. math::

        \\vec{r}_{ij} = \\mathrm{minimum\\_image}(\\vec{r}_j - \\vec{r}_i)

    When running simulations with a fixed box size, use the particle images
    :math:`\\vec{n}` to compute the unwrapped coordinates:

    .. math::

        \\vec{r}_\\mathrm{unwrapped} = \\vec{r} + n_x \\vec{a}_1
            + n_y \\vec{a}_2 + n_z \\vec{a}_3

    .. rubric:: Two dimensional systems

    Set ``Lz == 0`` to make the box 2D. 2D boxes ignore ``xz`` and ``yz``.
    Changing the box dimensionality from 2D to 3D (or from 3D to 2D) during
    a simulation will result in undefined behavior.

    In 2D boxes, *volume* is in units of :math:`[\\mathrm{length}]^2`.

    .. rubric:: Factory Methods

    `Box` has factory methods to enable easier creation of boxes: `cube`,
    `square`, `from_matrix`, `from_basis_vectors`, and `from_box`. See each
    method's documentation for more details.

    .. rubric:: Example:

    .. code-block:: python

        box = hoomd.Box(Lx=10, Ly=20, Lz=30, xy=0.5, xz=0.2, yz=0.1)

    See Also:
        `hoomd.State.box`

        `hoomd.State.set_box`
    """

    # Constructors
    def __init__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0):
        if Lz == 0 and (xz != 0 or yz != 0):
            raise ValueError("Cannot set the xz or yz tilt factor on a 2D box.")
        self._cpp_obj = _hoomd.BoxDim(Lx, Ly, Lz)
        self._cpp_obj.setTiltFactors(xy, xz, yz)

    @classmethod
    def cube(cls, L):
        """Create a cube with side lengths ``L``.

        Args:
            L (float): The box side length :math:`[\\mathrm{length}]`.

        Returns:
            hoomd.Box: The created 3D box.

        .. rubric:: Example:

        .. code-block:: python

            box = hoomd.Box.cube(L=13)
        """
        return cls(L, L, L, 0, 0, 0)

    @classmethod
    def square(cls, L):
        """Create a square with side lengths ``L``.

        Args:
            L (float): The box side length :math:`[\\mathrm{length}]`.

        Returns:
            hoomd.Box: The created 2D box.

        .. code-block:: python

            box = hoomd.Box.square(L=128)
        """
        return cls(L, L, 0, 0, 0, 0)

    @classmethod
    def from_basis_vectors(cls, box_matrix):
        r"""Initialize a Box instance from a box matrix.

        Args:
            box_matrix ((3, 3) `numpy.ndarray` of `float`): A 3x3 matrix
                or list of lists representing a set of lattice basis vectors.

        Note:
           The created box will be rotated with respect to the lattice basis. As
           a consequence the output of `to_matrix` will not be the same as the
           input provided to this function. This function also returns a
           rotation matrix comensurate with this transformation. Using this
           rotation matrix users can rotate the original points into the new box
           by applying the rotation to the points.

        Note:
           When passing a 2D basis vectors, the third vector should be set to
           all zeros, while first two vectors should have the last element set
           to zero.

        Returns:
            tuple[hoomd.Box, numpy.ndarray]: A tuple containing:
                - hoomd.Box: The created box configured according to the given
                  basis vectors.
                - numpy.ndarray: A 3x3 floating-point rotation matrix that can
                  be used to transform the original basis vectors to align with
                  the new box basis vectors.

        .. rubric:: Example:

        .. code-block:: python

            points = np.array([[0, 0, 0], [0.5, 0, 0], [0.25, 0.25, 0]])
            box, rotation = hoomd.Box.from_basis_vectors(
                box_matrix = [[ 1,  1,  0],
                              [ 1, -1,  0],
                              [ 0,  0,  1]])
            rotated_points = rotation @ points
        """
        box_matrix = np.asarray(box_matrix, dtype=np.float64)
        if box_matrix.shape != (3, 3):
            raise ValueError("Box matrix must be a 3x3 matrix.")
        v0 = box_matrix[:, 0]
        v1 = box_matrix[:, 1]
        v2 = box_matrix[:, 2]
        Lx = np.sqrt(np.dot(v0, v0))
        a2x = np.dot(v0, v1) / Lx
        Ly = np.sqrt(np.dot(v1, v1) - a2x * a2x)
        xy = a2x / Ly
        v0xv1 = np.cross(v0, v1)
        v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
        Lz = np.dot(v2, v0xv1) / v0xv1mag
        if Lz != 0:
            a3x = np.dot(v0, v2) / Lx
            xz = a3x / Lz
            yz = (np.dot(v1, v2) - a2x * a3x) / (Ly * Lz)
            upper_triangular_box_matrix = np.array([[Lx, Ly * xy, Lz * xz],
                                                    [0, Ly, Lz * yz],
                                                    [0, 0, Lz]])
        else:
            xz = yz = 0
            if not (np.allclose(v2, [0, 0, 0]) and np.allclose(v0[2], 0)
                    and np.allclose(v1[2], 0)):
                error_string = ("A 2D box matrix must have a third vector and"
                                "third component of first two vectors set to"
                                "zero.")
                raise ValueError(error_string)
            upper_triangular_box_matrix = np.array([[Lx, Ly * xy], [0, Ly]])
            box_matrix = box_matrix[:2, :2]

        rotation = np.linalg.solve(upper_triangular_box_matrix, box_matrix)

        if Lz == 0:
            rotation = np.zeros((3, 3))
            rotation[:2, :2] = box_matrix
            rotation[2, 2] = 1

        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz), rotation

    @classmethod
    def from_matrix(cls, box_matrix):
        r"""Create a box from an upper triangular matrix.

        Args:
            box_matrix ((3, 3) `numpy.ndarray` of `float`): An upper
                triangular matrix representing a box. The values for ``Lx``,
                ``Ly``, ``Lz``, ``xy``, ``xz``, and ``yz`` are related to the
                matrix:


                .. math::

                    \begin{bmatrix}
                    L_x & L_y \cdot xy & L_z \cdot xz \\
                    0 & L_y & L_z \cdot yz \\
                    0 & 0 & L_z
                    \end{bmatrix}

        Returns:
            hoomd.Box: The created box.

        .. rubric:: Example:

        .. code-block:: python

            box = hoomd.Box.from_matrix(
                box_matrix = [[10, 12, 14],
                              [0, 8, 16],
                              [0, 0, 18]])
        """
        box_matrix = np.asarray(box_matrix, dtype=np.float64)
        if box_matrix.shape != (3, 3):
            raise ValueError("Box matrix must be a 3x3 matrix.")
        if not np.allclose(box_matrix, np.triu(box_matrix)):
            raise ValueError("Box matrix must be upper triangular.")
        L = np.diag(box_matrix)
        return cls(*L, box_matrix[0, 1] / L[1], box_matrix[0, 2] / L[2],
                   box_matrix[1, 2] / L[2])

    @classmethod
    def _from_cpp(cls, cpp_obj):
        """Wrap a C++ BoxDim.

        Does not copy the C++ object.
        """
        b = Box(0, 0)
        b._cpp_obj = cpp_obj
        return b

    @classmethod
    def from_box(cls, box):
        r"""Initialize a Box instance from a box-like object.

        Args:
            box (box_like): A box-like object.

        Note:
           If all values are provided, a triclinic box will be constructed.
           If only ``Lx, Ly, Lz`` are provided, an orthorhombic box will
           be constructed. If only ``Lx, Ly`` are provided, a rectangular
           (2D) box will be constructed.

        Returns:
            hoomd.Box: The resulting box object.

        .. rubric:: Example:

        .. code-block:: python

            box = hoomd.Box.from_box(box=[10, 20, 30, 0.5, 0.2, 0.1])
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
                        "converted to hoomd.Box.")
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

        .. rubric:: Example:

        .. code-block:: python

            if box.dimensions == 2:
                pass
        """
        return 2 if self.is2D else 3

    @property
    def is2D(self):  # noqa: N802 - allow function name
        """bool: Flag whether the box is 2D.

        If ``Lz == 0``, the box is treated as 2D, otherwise it is 3D. This
        property is not settable.

        .. rubric:: Example:

        .. code-block:: python

            if box.is2D:
                pass
        """
        return self.Lz == 0

    # Length based properties
    @property
    def L(self):  # noqa: N802 - allow function name
        """(3, ) `numpy.ndarray` of `float`: The box lengths, ``[Lx, Ly, Lz]`` \
        :math:`[\\mathrm{length}]`.

        Can be set with a float which sets all lengths, or a length 3 vector.

        .. rubric:: Example:

        .. code-block:: python

            box.L = (15, 30, 60)
        """
        return _vec3_to_array(self._cpp_obj.getL())

    @L.setter
    def L(self, new_L):  # noqa: N802: Allow function name
        newL = _make_scalar3(new_L)
        if newL.z == 0 and not self.is2D:
            self.tilts = [self.xy, 0, 0]
        self._cpp_obj.setL(newL)

    @property
    def Lx(self):  # noqa: N802: Allow function name
        """float: The length of the box in the x dimension \
        :math:`[\\mathrm{length}]`.

        .. rubric:: Example:

        .. code-block:: python

            box.Lx = 15
        """
        return self.L[0]

    @Lx.setter
    def Lx(self, value):  # noqa: N802: Allow function name
        L = self.L
        L[0] = float(value)
        self.L = L

    @property
    def Ly(self):  # noqa: N802: Allow function name
        """float: The length of the box in the y dimension \
        :math:`[\\mathrm{length}]`.

        .. rubric:: Example:

        .. code-block:: python

            box.Ly = 30
        """
        return self.L[1]

    @Ly.setter
    def Ly(self, value):  # noqa: N802: Allow function name
        L = self.L
        L[1] = float(value)
        self.L = L

    @property
    def Lz(self):  # noqa: N802: Allow function name
        """float: The length of the box in the z dimension \
        :math:`[\\mathrm{length}]`.

        .. rubric:: Example:

        .. code-block:: python

            box.Lz = 60
        """
        return self.L[2]

    @Lz.setter
    def Lz(self, value):  # noqa: N802: Allow function name
        L = self.L
        L[2] = float(value)
        self.L = L

    # Box tilt based properties
    @property
    def tilts(self):
        """(3, ) `numpy.ndarray` of `float`: The box tilts, ``[xy, xz, yz]``.

        Can be set using one tilt for all axes or three tilts. If the box is 2D
        ``xz`` and ``yz`` will automatically be set to zero.

        .. rubric:: Example:

        .. code-block:: python

            box.tilts = (1.1, 0.8, 0.2)
        """
        return np.array([self.xy, self.xz, self.yz])

    @tilts.setter
    def tilts(self, new_tilts):
        new_tilts = _make_scalar3(new_tilts)
        if self.is2D and (new_tilts.y != 0 or new_tilts.z != 0):
            raise ValueError("Cannot set the xz or yz tilt factor on a 2D box.")
        self._cpp_obj.setTiltFactors(new_tilts.x, new_tilts.y, new_tilts.z)

    @property
    def xy(self):
        """float: The tilt for the xy plane.

        .. rubric:: Example:

        .. code-block:: python

            box.xy = 1.1
        """
        return self._cpp_obj.getTiltFactorXY()

    @xy.setter
    def xy(self, xy):
        self.tilts = [xy, self.xz, self.yz]

    @property
    def xz(self):
        """float: The tilt for the xz plane.

        .. rubric:: Example:

        .. code-block:: python

            box.xz = 0.8
        """
        return self._cpp_obj.getTiltFactorXZ()

    @xz.setter
    def xz(self, xz):
        if self.is2D:
            raise ValueError("Cannot set xz tilt factor on a 2D box.")
        self.tilts = [self.xy, xz, self.yz]

    @property
    def yz(self):
        """float: The tilt for the yz plane.

        .. rubric:: Example:

        .. code-block:: python

            box.yz = 0.2
        """
        return self._cpp_obj.getTiltFactorYZ()

    @yz.setter
    def yz(self, yz):
        if self.is2D:
            raise ValueError("Cannot set yz tilt factor on a 2D box.")
        self.tilts = [self.xy, self.xz, yz]

    # Misc. properties
    @property
    def periodic(self):
        """(3, ) `numpy.ndarray` of `bool`: The periodicity of each \
        dimension.

        `periodic` is always ``(True, True, True)`` for the box associated with
        the simulation `State`. Some components of `periodic` may be `False` in
        the `hoomd.data.LocalSnapshot` box attribute in MPI domain decomposition
        simulations. This indicates which box directions are communicated with
        neighboring ranks (`False`) and which are not (`True`).
        """
        return _vec3_to_array(self._cpp_obj.getPeriodic(), bool)

    @property
    def volume(self):
        """float: Volume of the box.

        :math:`[\\mathrm{length}]^{2}` in 2D and
        :math:`[\\mathrm{length}]^{3}` in 3D.

        When setting volume the aspect ratio of the box is maintained while the
        lengths are changed.

        .. rubric:: Example:

        .. code-block:: python

            box.volume = 2000
        """
        return self._cpp_obj.getVolume(self.is2D)

    @volume.setter
    def volume(self, volume):
        self.scale((volume / self.volume)**(1 / self.dimensions))

    def to_matrix(self):
        """(3, 3) `numpy.ndarray` `float`: The upper triangular matrix that \
        defines the box.

        .. code-block::

            [[Lx, Ly * xy, Lz * xz],
             [0,  Ly,      Lz * yz],
             [0,  0,       Lz]]

        .. rubric:: Example:

        .. code-block:: python

            matrix = box.to_matrix()
        """
        Lx, Ly, Lz = self.L
        xy, xz, yz = self.tilts
        return np.array([[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz], [0, 0, Lz]])

    def scale(self, s):
        r"""Scale box dimensions.

        Scales the box in place by the given scale factors. Tilt factors are not
        modified.

        Args:
            s (float or list[float]): scale factors in each dimension. If a
                single float is given then scale all dimensions by s; otherwise,
                s must be a sequence of 3 values used to scale each dimension.

        Returns:
            ``self``

        .. rubric:: Examples:

        .. code-block:: python

            box.scale(2)

        .. code-block:: python

            box.scale((1, 2, 4))
        """
        s = np.asarray(s, dtype=float)
        self.L *= s
        return self

    # Magic Methods
    def __repr__(self):
        """Executable representation of the object."""
        return "hoomd.box.Box(Lx={}, Ly={}, Lz={}, xy={}, xz={}, yz={})".format(
            self.Lx, self.Ly, self.Lz, self.xy, self.xz, self.yz)

    def __eq__(self, other):
        """Test if boxes are equal."""
        if not isinstance(other, Box):
            return NotImplemented
        return self._cpp_obj == other._cpp_obj

    def __neq__(self, other):
        """Test if boxes are not equal."""
        if not isinstance(other, Box):
            return NotImplemented
        return self._cpp_obj != other._cpp_obj

    def __reduce__(self):
        """Reduce values to picklable format."""
        return (type(self), (*self.L, *self.tilts))


class BoxInterface(abc.ABC):
    """The class interface which HOOMD considers to be a box-like object.

    Note:
        This class is exclusively used for help with typing and documentation in
        HOOMD, and is not meant to be used.
    """

    @property
    @abc.abstractmethod
    def Lx(self) -> float:  # noqa: N802: Allow function name
        """Length in x direction."""
        pass

    @property
    @abc.abstractmethod
    def Ly(self) -> float:  # noqa: N802: Allow function name
        """Length in y direction."""
        pass

    @property
    @abc.abstractmethod
    def Lz(self) -> float:  # noqa: N802: Allow function name
        """Length in z direction."""
        pass

    @property
    @abc.abstractmethod
    def xy(self) -> float:
        """Tilt factor in the xy plane."""
        pass

    @property
    @abc.abstractmethod
    def xz(self) -> float:
        """Tilt factor in the xy plane."""
        pass

    @property
    @abc.abstractmethod
    def yz(self) -> float:
        """Tilt factor in the xy plane."""
        pass


box_like = typing.Union[Box, BoxInterface, typing.Sequence[float],
                        typing.Mapping[str, float], np.ndarray]
"""Objects that are or can be converted to `Box`.

This includes

* `hoomd.Box` objects.
* Objects with attributes ``Lx, Ly, Lz, xy, xz, yz``.
* Lists like ``[Lx, Ly, Lz, xy, xz, yz]``.
* Dictionaries with keys ``'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'``.
* 3x3 NumPy arrays or objects convertible to a 3x3 array (see
  `hoomd.Box.from_matrix`).

Note:
    If any of ``Lz, xy, xz, yz`` for these different types are not provided,
    they are considered 0.
"""
