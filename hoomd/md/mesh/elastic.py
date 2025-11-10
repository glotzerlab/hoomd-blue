# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import warnings
import copy


class Elastic(Force):
    """ """

    _ext_module = _md

    def __init__(self, mesh, reference_tags):
        super().__init__()

        params = TypeParameter(
            "params",
            "_types",
            TypeParameterDict(C_xxxx=float, C_xxyy=float, C_xyxy=float, len_keys=1),
        )

        self._add_typeparam(params)
        self._mesh = mesh
        self._reference_tags = reference_tags

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if self._mesh._attached and self._simulation != self._mesh._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent mesh structure."
                f" This is happending since the force is moving to a new "
                f"simulation. To suppress the warning explicitly set new mesh.",
                RuntimeWarning,
            )
            self._mesh = copy.deepcopy(self._mesh)

        self._mesh._attach(self._simulation)

        self._cpp_obj = self._ext_module.Elastic(
            self._simulation.state._cpp_sys_def,
            self._mesh._cpp_obj,
            self._mesh._reference_positions,
            self._reference_tags,
        )

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, self._mesh)
            except ValueError as err:
                raise err.__class__(
                    f"For {type(self)} in TypeParameter {typeparam.name} {err!s}"
                )


## Reference mesh/potential.py, and Philipp's bending and helfrich potential to try to match. Another objcet to create that has a python interface. Writing Python interface.
