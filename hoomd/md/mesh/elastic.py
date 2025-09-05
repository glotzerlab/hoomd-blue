from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.parameterdicts import TypeparameterDict
from hoomd.data.typeparam import TypeParameter
import hoomd

import numpy
import inspect

class Elastic(Force):
    """ 
    """

    _ext_module = _md

    ## TO DO: need to input mesh
    def __init__(self,reference_positions):
        super().__init__()

        params = TypeParameter(
            "params",
            "types", 
            TypeParameterDict(C_xxxx=float, C_xxyy=float, C_xyxy=float, len_keys=1)
            )
        
        self._add_typeparam(params)
        self._reference_positions = reference_positions

    def _attach_hook(self):
        """ Create the c++ mirror class."""
        ## TO DO

        self._cpp_obj = self._ext_module.Elastic(
            self._simulation.state._cpp_sys_def, 
            #TODO tetrahedra data, 
            self._reference_positions
        )

## Reference mesh/potential.py, and Philipp's bending and helfrich potential to try to match. Another objcet to create that has a python interface. Writing Python interface. 

