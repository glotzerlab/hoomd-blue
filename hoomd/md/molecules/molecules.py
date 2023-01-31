from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.filter import ParticleFilter


class Molecules(_HOOMDBaseObject):
    def __init__(self, filter, include_all_bonded=True):
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   include_all_bonded=bool)
        param_dict.update(
            dict(filter=filter,
                 include_all_bonded=include_all_bonded))
        # set defaults
        self._param_dict.update(param_dict)