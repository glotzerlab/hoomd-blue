from hoomd.filter.filter_ import _ParticleFilter
from hoomd._hoomd import ParticleFilterTags
import numpy as np


class Tags(_ParticleFilter, ParticleFilterTags):
    def __init__(self, tags):
        self._tags = np.ascontiguousarray(np.unique(tags), dtype=np.uint32)
        ParticleFilterTags.__init__(self, tags)

    def __hash__(self):
        if not hasattr(self, '_id'):
            self._id = hash(self._tags.tobytes())
        return self._id

    def __eq__(self, other):
        return type(self) == type(other) and all(self.tags == other.tags)

    @property
    def tags(self):
        return self._tags
