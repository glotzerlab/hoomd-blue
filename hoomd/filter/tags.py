from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterTags
import numpy as np


class Tags(ParticleFilter, ParticleFilterTags):
    def __init__(self, tags):
        if isinstance(tags, np.ndarray):
            tags = tags.astype(np.uint32)
        else:
            tags = np.array(tags, dtype=np.uint32)
        tags = np.unique(tags)
        self._tags = tags
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
