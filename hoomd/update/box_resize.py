from hoomd.operation import _Updater
from hoomd.box import Box
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.variant import _Variant
from hoomd.util import variant_preprocessing


class BoxResize(_Updater):
    def __init__(self, box1, box2, variant, trigger):
        params = ParameterDict(
            box1=OnlyType(Box), box2=OnlyType(Box),
            variant=OnlyType(_Variant, variant_preprocessing))
        self._param_dict.update(params)
        super().__init__(trigger)
