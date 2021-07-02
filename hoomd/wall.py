"""Implement data classes for support HOOMD wall geometries."""

from abc import ABC, abstractmethod
from collections.abc import MutableSequence


class WallGeometry(object):
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

    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), inside=True):
        self.r = r
        self.origin = origin
        self.inside = inside
        super().__init__()

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tInside=%s" % (str(self.r), str(
            self.origin), str(self.inside))

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'inside': %s}" % (str(
            self.r), str(self.origin), str(self.inside))


class Cylinder(WallGeometry):

    def __init__(self,
                 r=0.0,
                 origin=(0.0, 0.0, 0.0),
                 axis=(0.0, 0.0, 1.0),
                 inside=True):
        self.r = r
        self.origin = origin
        self.axis = axis
        self.inside = inside
        super().__init__()

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tAxis=%s\tInside=%s" % (str(
            self.r), str(self.origin), str(self.axis), str(self.inside))

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'axis': %s, 'inside': %s}" % (str(
            self.r), str(self.origin), str(self.axis), str(self.inside))


class Plane(WallGeometry):

    def __init__(self,
                 origin=(0.0, 0.0, 0.0),
                 normal=(0.0, 0.0, 1.0),
                 inside=True):
        self.origin = origin
        self.normal = normal
        self.inside = inside
        super().__init__()

    def __str__(self):
        return "Origin=%s\tNormal=%s\tInside=%s" % (str(
            self.origin), str(self.normal), str(self.inside))

    def __repr__(self):
        return "{'origin':%s, 'normal': %s, 'inside': %s}" % (str(
            self.origin), str(self.normal), str(self.inside))


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
