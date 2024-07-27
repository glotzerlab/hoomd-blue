# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Wall geometries.

Walls define an oriented surface in space. Walls exist only in the primary box
image and are not replicated across the periodic boundary conditions. Points on
one side of the surface have a positive signed distance to that surface, and
points on the other side have a negative signed distance.

Define individual walls with `Cylinder`, `Plane`, and `Sphere`. Create lists of
these `WallGeometry` objects to describe more complex geometries. Use walls to
confine particles to specific regions of space in HPMC and MD simulations.

See Also:
    `hoomd.hpmc.external.wall`

    `hoomd.md.external.wall`
"""

from abc import ABC, abstractmethod
from copy import copy
from collections.abc import MutableSequence
from hoomd.data.syncedlist import identity, SyncedList

from hoomd.operation import _HOOMDGetSetAttrBase
from hoomd.data.parameterdicts import ParameterDict


class WallGeometry(ABC, _HOOMDGetSetAttrBase):
    """Abstract base class for a HOOMD wall geometry.

    Walls are used in both HPMC and MD subpackages. Subclasses of `WallGeometry`
    abstract over the wall geometries for both use cases.
    """

    @abstractmethod
    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the geometry.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        pass

    def _setattr_param(self, attr, value):
        """Make WallGeometry objects effectively immutable."""
        raise ValueError(f"Cannot set {attr} after construction as "
                         f"{self.__class__} objects are immutable")


class Sphere(WallGeometry):
    r"""A sphere.

    Args:
        radius (float):
            The radius of the sphere :math:`[\mathrm{length}]`.
        origin (`tuple` [`float`, `float`, `float`], optional):
            The origin of the sphere, defaults to ``(0, 0, 0)``
            :math:`[\mathrm{length}]`.
        inside (`bool`, optional):
            Whether positive signed distances are inside or outside the
            sphere, defaults to ``True``.
        open (`bool`, optional):
            Whether to include the surface of the sphere in the space. ``True``
            means do not include the surface, defaults to ``True``.

    The signed distance from the wall surface is:

    .. math::

        d = \left( R - \lvert \vec{r} - \vec{r}_o \rvert \right)

    for ``inside=True``, where :math:`r` is the particle position, :math:`r_o`
    is the origin of the sphere, and :math:`R` is the sphere's radius. The
    distance is negated when ``inside=False``.

    Warning:
        When running MD simulations in 2D simulation boxes, set
        ``origin[2]=(x,y,0)``. Otherwise, the wall force will push particles off
        the xy plane.

    Note:
        `Sphere` objects are immutable.

    Attributes:
        radius (float):
            The radius of the sphere :math:`[\mathrm{length}]`.
        origin (`tuple` [`float`, `float`, `float`]):
            The origin of the sphere :math:`[\mathrm{length}]`.
        inside (bool):
            Whether positive signed distances are inside or outside the
            sphere.
        open (bool):
            Whether to include the surface of the sphere in the space. Open
            means do not include the surface.
    """

    def __init__(self, radius, origin=(0.0, 0.0, 0.0), inside=True, open=True):
        param_dict = ParameterDict(radius=float,
                                   origin=(float, float, float),
                                   inside=bool,
                                   open=bool)
        param_dict["radius"] = radius
        param_dict["origin"] = origin
        param_dict["inside"] = inside
        param_dict["open"] = open
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Sphere."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Sphere."""
        return f"Sphere(radius={self.radius}, origin={self.origin}, "
        f"inside={self.inside}, open={self.open})"

    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the sphere.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        return {
            "radius": self.radius,
            "origin": self.origin,
            "inside": self.inside,
            "open": self.open
        }


class Cylinder(WallGeometry):
    r"""A right circular cylinder.

    Args:
        radius (float):
            The radius of the circle faces of the cylinder
            :math:`[\mathrm{length}]`.
        axis (`tuple` [`float`, `float`, `float`]):
            A vector perpendicular to the circular faces.
        origin (`tuple` [`float`, `float`, `float`], optional):
            The origin of the cylinder defined as the center of the circle along
            the cylinder's axis :math:`[\mathrm{length}]`.
        inside (`bool`, optional):
            Whether positive signed distances are inside or outside the
            cylinder.
        open (`bool`, optional):
            Whether to include the surface of the cylinder in the space.
            ``True`` means do not include the surface, defaults to ``True``.

    Cylinder walls in HOOMD span the simulation box in the direction given by
    the ``axis`` attribute.

    The signed distance from the wall surface is

    .. math::

        d = \left( R - \lvert \left( \vec{r} - \vec{r}_o \right)
            - \left( \left( \vec{r} - \vec{r}_o \right) \cdot \hat{n}
            \right) \hat{n} \rvert \right)

    for ``inside=True``, where :math:`r` is the particle position,
    :math:`\vec{r}_o` is the origin of the cylinder, :math:`\hat{n}` is the
    cylinder's unit axis, and :math:`R` is the cylinder's radius. The distance
    is negated when ``inside=False``.

    Warning:
        When running MD simulations in 2D simulation boxes, set
        ``axis=(0,0,1)``. Otherwise, the wall force will push particles off the
        xy plane.

    Note:
        `Cylinder` objects are immutable.

    Attributes:
        radius (float):
            The radius of the circle faces of the cylinder
            :math:`[\mathrm{length}]`.
        origin (`tuple` [`float`, `float`, `float`]):
            The origin of the cylinder defined as the center of the circle along
            the cylinder's axis :math:`[\mathrm{length}]`.
        axis (`tuple` [`float`, `float`, `float`]):
            A vector perpendicular to the circular faces.
        inside (bool):
            Whether positive signed distances are inside or outside the
            cylinder.
        open (`bool`, optional):
            Whether to include the surface of the cylinder in the space.
            ``True`` means do not include the surface.
    """

    def __init__(self,
                 radius,
                 axis,
                 origin=(0.0, 0.0, 0.0),
                 inside=True,
                 open=True):
        param_dict = ParameterDict(radius=float,
                                   origin=(float, float, float),
                                   axis=(float, float, float),
                                   inside=bool,
                                   open=bool)
        param_dict["radius"] = radius
        param_dict["origin"] = origin
        param_dict["axis"] = axis
        param_dict["inside"] = inside
        param_dict["open"] = open
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Cylinder."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Cylinder."""
        return f"Cylinder(radius={self.radius}, origin={self.origin}, "
        f"axis={self.axis}, inside={self.inside}, open={self.open})"

    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the cylinder.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        return {
            "radius": self.radius,
            "origin": self.origin,
            "axis": self.axis,
            "inside": self.inside,
            "open": self.open
        }


class Cone(WallGeometry):
    r"""A truncated cone shape wall.

    Args:
        radius1 (float):
            The radius of the smaller circle face of the cylinder
            :math:`[\mathrm{length}]`.
        radius2 (float):
            The radius of the larger circle face of the cylinder
            :math:`[\mathrm{length}]`.
        distance (float):
            The distance between two radius, radius1 and radius2
            :math:`[\mathrm{length}]`.
        axis (`tuple` [`float`, `float`, `float`]):
            A vector perpendicular to the circular faces.
        origin (`tuple` [`float`, `float`, `float`], optional):
            The origin of the Cone defined as the center of the circle along
            the cylinder's axis :math:`[\mathrm{length}]`.
        inside (`bool`, optional):
            Whether positive signed distances are inside or outside the
            Cone.
        open (`bool`, optional):
            Whether to include the surface of the Cone in the space.
            ``True`` means do not include the surface, defaults to ``True``.

    Cone walls in HOOMD span the ``distance`` in the direction given by
    the ``axis`` attribute.

    The signed distance from the wall surface is

    .. math::

        d = \left( R - \lvert \left( \vec{r} - \vec{r}_o \right)
            - \left( \left( \vec{r} - \vec{r}_o \right) \cdot \hat{n}
            \right) \hat{n} \rvert \right)

    for ``inside=True``, where :math:`r` is the particle position,
    :math:`\vec{r}_o` is the origin of the cylinder, :math:`\hat{n}` is the
    cylinder's unit axis, and :math:`R` is the cylinder's radius. The distance
    is negated when ``inside=False``.

    Warning:
        When running MD simulations in 2D simulation boxes, set
        ``axis=(0,0,1)``. Otherwise, the wall force will push particles off the
        xy plane.

    Note:
        `Cone` objects are immutable.

    Attributes:
        radius1 (float):
            The radius of the smaller circle face of the cylinder
            :math:`[\mathrm{length}]`.
        radius2 (float):
            The radius of the larter circle face of the cylinder
            :math:`[\mathrm{length}]`.
        distance (float):
            The distance between two radius, radius1 and radius2
            :math:`[\mathrm{length}]`.
        origin (`tuple` [`float`, `float`, `float`]):
            The origin of the Cone defined as the center of the circle along
            the cylinder's axis :math:`[\mathrm{length}]`.
        axis (`tuple` [`float`, `float`, `float`]):
            A vector perpendicular to the circular faces.
        inside (bool):
            Whether positive signed distances are inside or outside the
            Cone.
        open (`bool`, optional):
            Whether to include the surface of the Cone in the space.
            ``True`` means do not include the surface.
    """

    def __init__(self,
                 radius1,
                 radius2,
                 distance,
                 axis,
                 origin=(0.0, 0.0, 0.0),
                 inside=True,
                 open=True):
        param_dict = ParameterDict(radius1=float,
                                   radius2=float,
                                   distance=float,
                                   origin=(float, float, float),
                                   axis=(float, float, float),
                                   inside=bool,
                                   open=bool)
        param_dict["radius1"] = radius1
        param_dict["radius2"] = radius2
        param_dict["distance"] = distance
        param_dict["origin"] = origin
        param_dict["axis"] = axis
        param_dict["inside"] = inside
        param_dict["open"] = open
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Cone."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Cone."""
        return f"Cone(radius1={self.radius1}, radius2={self.radius2},distance={self.distance},origin={self.origin}, "
        f"axis={self.axis}, inside={self.inside}, open={self.open})"

    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the cylinder.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        return {
            "radius1": self.radius1,
            "radius2": self.radius2,
            "distance": self.distance,
            "origin": self.origin,
            "axis": self.axis,
            "inside": self.inside,
            "open": self.open
        }


class Plane(WallGeometry):
    r"""A plane.

    Args:
        origin (`tuple` [`float`, `float`, `float`]):
            A point that lies on the plane :math:`[\mathrm{length}]`.
        normal (`tuple` [`float`, `float`, `float`]):
            The normal vector to the plane. The vector will be converted to a
            unit vector :math:`[\mathrm{dimensionless}]`.
        open (`bool`, optional):
            Whether to include the surface of the plane in the space. ``True``
            means do not include the surface, defaults to ``True``.

    The signed distance from the wall surface is:

    .. math::

        d = \hat{n} \cdot \left( \vec{r} - \vec{r}_o \right)

    where :math:`\vec{r}` is the particle position, :math:`\vec{r}_o` is the
    origin of the plane, and :math:`\hat{n}` is the plane's unit normal.
    The normal points toward the points with a positive signed distance to the
    plane.

    Warning:
        When running MD simulations in 2D simulation boxes, set
        ``normal=(nx,ny,0)``. Otherwise, the wall force will push particles off
        the xy plane.

    Note:
        `Plane` objects are immutable.

    Attributes:
        origin (`tuple` [`float`, `float`, `float`]):
            A point that lies on the plane :math:`[\mathrm{length}]`.
        normal (`tuple` [`float`, `float`, `float`]):
            The unit normal vector to the plane.
        open (bool):
            Whether to include the surface of the plane in the space. ``True``
            means do not include the surface.
    """

    def __init__(self, origin, normal, open=True):
        param_dict = ParameterDict(origin=(float, float, float),
                                   normal=(float, float, float),
                                   open=bool)
        param_dict["origin"] = origin
        param_dict["normal"] = normal
        param_dict["open"] = open
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Plane."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Plane."""
        return f"Plane(origin={self.origin}, normal={self.normal}, "
        f"open={self.open})"

    def to_dict(self):
        """Convert the wall geometry to a dictionary defining the plane.

        Returns:
            dict: The geometry in a Python dictionary.
        """
        return {"origin": self.origin, "normal": self.axis, "open": self.open}


class _MetaListIndex:
    """Index and type information between frontend and backend lists.

    This class facilitates maintaining order between the user exposed list in
    `_WallsMetaList` and the backend lists used in C++. This is essentially a
    dataclass (we cannot use a dataclass since it requires Python 3.7 and we
    support prior versions.
    """

    def __init__(self, type, index=0):
        self.index = index
        self.type = type

    def __repr__(self):
        return f"_MetaListIndex(type={self.type}, index={self.index})"


class _WallsMetaList(MutableSequence):
    """Creates a lists that sieves its items into multiple backend lists.

    The class redirects and manages each contained object into one of a number
    of 'backend' lists based on some condition (here object type). This is to
    provide the interface of a single list while allowing for the necessity of
    separate lists for a given set of items (e.g. for C++ type checking). This
    is managed by mantaining the frontend list, multiple backend lists, and an
    index list of `_MetaListIndex` which links items in the frontend list to
    their equivalent in the backend list. Most mutative operations on the list
    require the careful manipulation of the backend indices.

    The time-complexity of most operations given the requirements is
    :math:`O(n)`. However, the amortized complexity of ``append`` and ``extend``
    is :math:`O(1)` and :math:`O(k)` respectively (where :math:`k` is the number
    of items to extend the list by). This means that common usage should not be
    unreasonably slow, and should be asymptotically comparable to a standard
    Python list.

    `_WallsMetaList` maintains ordering of the constituent objects between
    lists (e.g. for two backend list chosen on whether a character is a vowel,
    the order of the backend lists for the sequence "abdefg" would be "bdfg" and
    "ae"). If this is not necessary, the class could be sped up and simplified
    by using `dict` objects to maintain references to frontend elements and
    always appending to backend lists despite the behavior on the front end.

    Attributes:
        _walls (`list` [`WallGeometry`]): The list of walls exposed to the user.
        _backend_list_index (`list` [`_MetaListIndex`]): The list of type, index
            pairs that connects ``_walls`` to the lists in ``_backend_lists``.
        _backend_lists (`dict` [`type`, `hoomd.data.SyncedList` \
                [`WallGeometry`]): A dictionary mapping wall type with the
                `hoomd.data.SyncedList` instance used to sync the Python with
                C++ wall lists.
    """

    def __init__(self, walls=None, to_cpp=identity):
        self._walls = []
        self._backend_list_index = []
        self._backend_lists = {
            Sphere:
                SyncedList(Sphere, to_synced_list=to_cpp, attach_members=False),
            Cylinder:
                SyncedList(Cylinder,
                           to_synced_list=to_cpp,
                           attach_members=False),
            Plane:
                SyncedList(Plane, to_synced_list=to_cpp, attach_members=False)
        }

        if walls is None:
            return
        self.extend(walls)

    def __getitem__(self, index):
        return self._walls[index]

    def __setitem__(self, index, wall):
        self._walls[index] = wall

        # handle backend list indices
        old_backend_index = self._backend_list_index[index]
        new_type = type(wall)
        old_type = old_backend_index.type
        # If the old type at index matches the new wall type then we just swap
        # on the backend. Also this is a necessary short-circuit as
        # _get_obj_backend_index would incorrectly increment all later indices
        # of the same type as new_type.
        if old_type == new_type:
            self._backend_lists[new_type][old_backend_index.index] = wall
            return

        new_backend_index = self._get_obj_backend_index(index + 1, new_type,
                                                        old_type)
        self._backend_list_index[index] = new_backend_index

        # Add/remove the new/old walls from their respective backend lists
        self._backend_lists[new_type].insert(new_backend_index.index, wall)
        del self._backend_lists[old_type][old_backend_index.index]

    def __delitem__(self, index):
        if isinstance(index, slice):
            for i in reversed(sorted(range(len(self))[index])):
                self.__delitem__(i)
            return
        del self._walls[index]
        backend_index = self._backend_list_index.pop(index)
        wall_type = backend_index.type
        del self._backend_lists[wall_type][backend_index.index]
        # Decrement backend index for all indices of the deleted type
        # First handle the case where the last item was deleted which requires
        # no updating.
        if index == -1 or index == len(self._walls):
            return
        # Now handle the case where [index:] would include one wall before the
        # deleted index (since negative index count from the back).
        if index < 0:
            index += 1
        for bi in self._backend_list_index[index:]:
            if wall_type is bi.type:
                bi.index -= 1

    def __len__(self):
        return len(self._walls)

    def insert(self, index, wall):
        self._walls.insert(index, wall)
        new_type = type(wall)
        new_index = self._get_obj_backend_index(index, new_type)
        self._backend_lists[new_type].insert(new_index.index, wall)
        self._backend_list_index.insert(index, new_index)

    def append(self, wall):
        # While not required we overwrite the default append to increase the
        # efficiency of appending and extending as those are common operations.
        # In CPython extend calls append.
        self._walls.append(wall)

        wall_type = type(wall)
        index = len(self._backend_lists[wall_type])
        self._backend_lists[wall_type].append(wall)
        self._backend_list_index.append(_MetaListIndex(wall_type, index))

    def _sync(self, sync_lists):
        """Sync backend list with associated C++ wall lists.

        Args:
            sync_lists (dict[type, list[WallData]]): A dictionary of Python wall
                types to C++ lists (something like an
                `hoomd.data.array_view._ArrayView` or pybind11 exported
                std::vector).
        """
        for wall_type, wall_list in sync_lists.items():
            # simulation is unnecessary here since the SyncedList instance is
            # not user facing, and unique membership of frontend items not
            # required.
            self._backend_lists[wall_type]._sync(None, wall_list)

    def _unsync(self):
        for wall_list in self._backend_lists.values():
            wall_list._unsync()

    def _get_obj_backend_index(self, frontend_index, new_type, old_type=None):
        """Find the correct backend index while adjusting other indices.

        The method increments all backend indices of the same type that come
        after ``frontend_index``, and decrements all indices of the same type as
        ``old_type`` if provided.
        """
        backend_index = None
        # Check for next index that is of the same type as the new wall,
        # while incrementing or decrementing the indices of the appropriate
        # type.
        for bi in self._backend_list_index[frontend_index:]:
            if bi.type == new_type:
                if backend_index is None:
                    backend_index = copy(bi)
                bi.index += 1
            elif old_type is not None and bi.type == old_type:
                bi.index -= 1
        # If we did not find a _MetaListIndex of the appropriate type check
        # before the index in the list for a _MetaListIndex of the correct type.
        if backend_index is not None:
            return backend_index

        for bi in self._backend_list_index[frontend_index - 1::-1]:
            if bi.type == new_type:
                backend_index = copy(bi)
                backend_index.index += 1
                return backend_index
        # No other object of this wall type currently exists create a new
        # index object to use.
        else:
            return _MetaListIndex(new_type)
