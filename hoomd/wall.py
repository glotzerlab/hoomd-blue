"""Implement data classes for support HOOMD wall geometries."""

from abc import ABC, abstractmethod
from copy import copy
from collections.abc import MutableSequence
from hoomd.data.syncedlist import SyncedList

from hoomd.operation import _HOOMDGetSetAttrBase
from hoomd.data.parameterdicts import ParameterDict


class WallGeometry(ABC, _HOOMDGetSetAttrBase):
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

    def _setattr_param(self, attr, value):
        """Make WallGeometry objects effectively immutable."""
        raise ValueError(f"Cannot set {attr} after construction as "
                         f"{self.__class__} objects are immutable")


class Sphere(WallGeometry):
    """Define a circle/sphere in 2D/3D Euclidean space.

    Whether the wall is interpreted as a sphere or circle is dependent on the
    dimension of the system the wall is applied to. For 2D systems the
    z-component of the origin should be zero.

    Note:
        `Sphere` objects are immutable.

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
        param_dict = ParameterDict(radius=float,
                                   origin=(float, float, float),
                                   inside=bool)
        param_dict["radius"] = radius
        param_dict["origin"] = origin
        param_dict["inside"] = inside
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Sphere."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Sphere."""
        return f"Sphere(radius={self.radius}, origin={self.origin}, "
        f"inside={self.inside})"

    def to_dict(self):
        """Return a dictionary specifying the sphere."""
        return {
            "radius": self.radius,
            "origin": self.origin,
            "inside": self.inside
        }


class Cylinder(WallGeometry):
    """Define a cylinder in 3D Euclidean space.

    Note:
        `Cylinder` objects are immutable.

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
        param_dict = ParameterDict(radius=float,
                                   origin=(float, float, float),
                                   axis=(float, float, float),
                                   inside=bool)
        param_dict["radius"] = radius
        param_dict["origin"] = origin
        param_dict["axis"] = axis
        param_dict["inside"] = inside
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Cylinder."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Cylinder."""
        return f"Cylinder(radius={self.radius}, origin=self.origin, "
        f"axis={self.axis}, inside={self.inside})"

    def to_dict(self):
        """Return a dictionary specifying the cylinder."""
        return {
            "radius": self.radius,
            "origin": self.origin,
            "axis": self.axis,
            "inside": self.inside
        }


class Plane(WallGeometry):
    """Define a Plane in 3D Euclidean space.

    Note:
        `Plane` objects are immutable.

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
        param_dict = ParameterDict(origin=(float, float, float),
                                   normal=(float, float, float),
                                   inside=bool)
        param_dict["origin"] = origin
        param_dict["normal"] = normal
        param_dict["inside"] = inside
        self._param_dict = param_dict

    def __str__(self):
        """A string representation of the Plane."""
        return self.__repr__()

    def __repr__(self):
        """A string representation of the Plane."""
        return f"Plane(origin={self.origin}, normal={self.normal}, "
        f"inside={self.inside})"

    def to_dict(self):
        """Return a dictionary specifying the plane."""
        return {
            "origin": self.origin,
            "normal": self.axis,
            "inside": self.inside
        }


class _MetaListIndex:
    """Index and type information between frontend and backend lists.

    This faciliates mantaining order between the user exposed list in
    `_WallsMetaList` and the backend lists used in C++. This is essentially a
    dataclass (we cannot use a dataclass since it requires Python 3.7 and we
    support prior versions.
    """

    def __init__(self, type, index=0):
        self.index = index
        self.type = type

    def __repr__(self):
        return f"_MetaListIndex(type={self.type}, index={self.index})"


# TODO: remove before merging (implemented in another PR)
def _islice_index(sequence, *args):
    if len(args) == 1:
        start, stop, step = args[0].start, args[0].stop, args[0].step
    elif len(args) < 3:
        args.extend((None,) * 3 - len(args))
        start, stop, step = args
    if step is None:
        step = 1
    if start is None:
        start = 0 if step > 0 else len(sequence)
    if stop is None:
        stop = 0 if step < 0 else len(sequence)
    yield from range(start, stop, step)


# TODO: remove before merging (implemented in another PR)
def _islice(sequence, *args):
    for i in _islice_index(sequence, *args):
        yield sequence[i]


def _to_cpp_wall(wall):
    """Converts a Python `WallGeometry` object to a C++ wall object."""
    pass


class _WallsMetaList(MutableSequence):
    """Creates a lists that sieves its items into multiple backend lists.

    The class redirects and manages each contained object into one of a number
    of 'backend' lists based on some condition (here object type). This is to
    proivde the interface of a single list while allowing for the necessity of
    separate lists for a given set of items (e.g. for C++ type checking). This
    is managed by mantaining the frontend list, mutliple backend lists, and an
    index list of `_MetaListIndex` which links items in the frontend list to
    their equivalent in the backend list. Most mutative operations on the list
    require the careful manipulation of the backend indices.

    The time-complexity of most operations given the requirements is
    :math:`O(n)`. However, the amortized complexity of ``append`` and ``extend``
    is :math:`O(1)` and :math:`O(k)` respectively (where :math:`k` is the number
    of items to extend the list by). This means that common usage should not be
    unreasonably slow, and should be asymptotically comparable to a standard
    Python list.

    `_WallsMetaList` maintains ordering of the constitutent objects between
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

    def __init__(self, walls=None):
        self._walls = []
        self._backend_list_index = []
        self._backend_lists = {
            Sphere:
                SyncedList(Sphere,
                           to_synced_list=_to_cpp_wall,
                           attach_members=False),
            Cylinder:
                SyncedList(Cylinder,
                           to_synced_list=_to_cpp_wall,
                           attach_members=False),
            Plane:
                SyncedList(Plane,
                           to_synced_list=_to_cpp_wall,
                           attach_members=False)
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

        new_backend_index = self._get_obj_backend_index(index, new_type,
                                                        old_type)

        # Add/remove the new/old walls from their respective backend lists
        self._backend_lists[new_type].insert(new_backend_index.index, wall)
        del self._backend_lists[old_type][old_backend_index.index]

    def __delitem__(self, index):
        if isinstance(index, slice):
            for i in _islice_index(self, slice.start, slice.stop, slice.step):
                self.__delitem__(i)
            return
        del self._walls[index]
        backend_index = self._backend_list_index.pop(index)
        wall_type = backend_index.type
        del self._backend_lists[wall_type][backend_index.index]
        # Decrement backend index for all indices of the deleted type
        for bi in self._backend_list_index[index:]:
            if wall_type == bi.type:
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

    def _sync(self, cpp_obj):
        for wall_type, wall_list in self._backend_lists.items():
            wall_list._sync(
                None, getattr(cpp_obj, self._type_to_list_name[wall_type]))

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
        # type. Don't use _islice here since we have to iterate over the
        # entire slice and would only increase run-time by adding a layer of
        # abstraction.
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

        for bi in _islice(self._backend_list_index,
                          start=frontend_index - 1,
                          step=-1):
            if bi.type == new_type:
                backend_index = copy(bi)
                backend_index.index += 1
                return backend_index
        # No other object of this wall type currently exists create a new
        # index object to use.
        else:
            return _MetaListIndex(new_type)
