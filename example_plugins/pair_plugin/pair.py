
import hoomd
from hoomd.md import pair, _md
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter

from hoomd.pair_plugin import _pair_plugin


class ExamplePair(pair.Pair):

    _cpp_class_name = "PotentialPairExample"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(alpha=float, len_keys=2))
        self._add_typeparam(params)

    def _attach(self):
        """Slightly modified with regard to the base class `md.Pair`.
        
        In particular, we search for `PotentialPairExample` in `hoomd.pair_plugin._pair_plugin`
        instead of `hoomd.md._md`.
        """
        # create the c++ mirror class
        if not self.nlist._added:
            self.nlist._add(self._simulation)
        else:
            if self._simulation != self.nlist._simulation:
                raise RuntimeError("{} object's neighbor list is used in a "
                                   "different simulation.".format(type(self)))
        if not self.nlist._attached:
            self.nlist._attach()
        # Find definition of _cpp_class_name in _pair_plugin
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_pair_plugin, self._cpp_class_name)
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.half)
        else:
            cls = getattr(_pair_plugin, self._cpp_class_name + "GPU")
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)
        
        grandparent = super(pair.Pair, self)
        grandparent._attach()
