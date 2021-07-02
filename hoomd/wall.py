"""Implement data classes for support HOOMD wall geometries."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence


class WallGeometry(ABC):
    """Abstract base class for a HOOMD wall geometry.

    Walls are used in both HPMC and MD subpackages. Subclass of `WallGeometry`
    abstract over the wall geometries for both use cases.
    """

    @abstractmethod
    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the geometry.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        pass


class Sphere(WallGeometry):
    """Define a circle/sphere in 2D/3D Euclidean space.

    Whether the wall is interpreted as a sphere or circle is dependent on the
    dimension of the system the wall is applied to. For 2D systems the
    z-component of th origin should be zero.

    Args:
        radius (`float`, optional):
            The radius of the sphere.
        origin (`tuple` [`float`,`float`,`float`], optional):
            The origin of the sphere.
        inside (`bool`, optional):
            Whether particles are restricted to the space inside or outside the
            sphere.

    Attributes:
        radius (float):
            The radius of the sphere.
        origin (`tuple` [`float`,`float`,`float`]):
            The origin of the sphere.
        inside (bool):
            Whether particles are restricted to the space inside or outside the
            sphere.
    """

    def __init__(self, radius=0.0, origin=(0.0, 0.0, 0.0), inside=True):
        self.radius = radius
        self.origin = origin
        self.inside = inside

    def __str__(self):
        """A string representation of the Sphere."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Sphere."""
        return f"Sphere(radius={self.radius}, origin={self.origin}, "
        f"inside={self.inside})"


class Cylinder(WallGeometry):
    """Define a cylinder in 3D Euclidean space.

    Args:
        radius (`float`, optional):
            The radius of the circle faces of the cylinder.
        origin (`tuple` [`float`,`float`,`float`], optional):
            The origin of the cylinder defined as the center of the bisecting
            circle along the cylinder's axis.
        axis (`tuple` [`float`,`float`,`float`], optional):
            A vector perpendicular to the circular faces.
        inside (`bool`, optional):
            Whether particles are restricted to the space inside or outside the
            cylinder.

    Attributes:
        radius (float):
            The radius of the circle faces of the cylinder.
        origin (`tuple` [`float`,`float`,`float`]):
            The origin of the cylinder defined as the center of the bisecting
            circle along the cylinder's axis.
        axis (`tuple` [`float`,`float`,`float`]):
            A vector perpendicular to the circular faces.
        inside (bool):
            Whether particles are restricted to the space inside or outside the
            cylinder.
    """

    def __init__(self,
                 radius=0.0,
                 origin=(0.0, 0.0, 0.0),
                 axis=(0.0, 0.0, 1.0),
                 inside=True):
        self.radius = radius
        self.origin = origin
        self.axis = axis
        self.inside = inside

    def __str__(self):
        """A string representation of the Cylinder."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Cylinder."""
        return f"Cylinder(radius={self.radius}, origin=self.origin, "
        f"axis={self.axis}, inside={self.inside})"


class Plane(WallGeometry):
    """Define a Plane in 3D Euclidean space.

    Args:
        origin (`tuple` [`float`,`float`,`float`], optional):
            A point that lies on the plane used with ``normal`` to fully specify
            the plane.
        normal (`tuple` [`float`,`float`,`float`], optional):
            The normal vector to the plane.
        inside (`bool`, optional):
            Whether particles are restricted to the space inside or outside the
            cylinder. Inside is the side of the plane the normal points to, and
            outside constitutes the other side.

    Attributes:
        origin (`tuple` [`float`,`float`,`float`]):
            A point that lies on the plane used with ``normal`` to fully specify
            the plane.
        normal (`tuple` [`float`,`float`,`float`]):
            The normal vector to the plane.
        inside (bool):
            Whether particles are restricted to the space inside or outside the
            cylinder. Inside is the side of the plane the normal points to, and
            outside constitutes the other side.
    """

    def __init__(self,
                 origin=(0.0, 0.0, 0.0),
                 normal=(0.0, 0.0, 1.0),
                 inside=True):
        self.origin = origin
        self.normal = normal
        self.inside = inside

    def __str__(self):
        """A string representation of the Plane."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Plane."""
        return f"Plane(origin={self.origin}, normal={self.normal}, "
        f"inside={self.inside})"


class _WallsMetaList(MutableSequence):

    def __init__(self, attach_method, walls):
        self.walls = []
        # self._walls = {
        #     Sphere:_SyncedList(Sphere,attach_method),
        #     Cylinder:_SyncedList(Cylinder,attach_method),
        #     Plane:_SyncedList(Plane,attach_method)}
        self._walls = {Sphere: [], Cylinder: [], Plane: []}
        self.index = {Sphere: [], Cylinder: [], Plane: []}

        for wall in walls:
            self.append(wall)

    def __getitem__(self, index):
        return self.walls[index]

    def __setitem__(self, index, wall):
        old = self.walls[index]
        self.index[type(old)].remove(index)
        self._walls[type(old)].remove(old)
        self.index[type(wall)].append(wall)
        self._walls[type(wall)].append(index)
        self.walls[index] = wall

    def __delitem__(self, index):
        if isinstance(index, slice):
            for i in reversed(
                    sorted(
                        list(
                            range(index.start or 0, index.stop or len(self),
                                  index.step or 1)))):
                self.__delitem__(i)
        else:
            self._walls[type(self.walls[index])].remove(self.walls[index])
            self.index[type(self.walls[index])].remove(index)
            for k in self.index.keys():
                self.index[k] = list(
                    map(lambda i: i if i < index else i - 1, self.index[k]))
            del self.walls[index]

    def __len__(self):
        return len(self.walls)

    def insert(self, index, wall):
        if not wall in self.walls:
            for k in self.index.keys():
                self.index[k] = list(
                    map(lambda i: i if i < index else i + 1, self.index[k]))
            self._walls[type(wall)].append(wall)
            self.index[type(wall)].append(index)
            self.walls.insert(index, wall)
