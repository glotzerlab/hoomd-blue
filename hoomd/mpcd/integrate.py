# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.md.integrate import Integrator as _MDIntegrator
from hoomd.mpcd import _mpcd
from hoomd.mpcd.collide import CellList, CollisionMethod
from hoomd.mpcd.stream import StreamingMethod


@hoomd.logging.modify_namespace(("mpcd", "Integrator"))
class Integrator(_MDIntegrator):
    """MPCD integrator.

    The MPCD `Integrator` enables the MPCD algorithm concurrently with standard
    MD methods.

    In MPCD simulations, `dt` defines the amount of time that the system is
    advanced forward every time step. MPCD streaming and collision steps can be
    defined to occur in multiples of `dt`. In these cases, any MD particle data
    will be updated every `dt`, while the MPCD particle data is updated
    asynchronously for performance. For example, if MPCD streaming happens every
    5 steps, then the particle data will be updated as follows::

                0     1     2     3     4     5
        MD:     |---->|---->|---->|---->|---->|
        MPCD:   |---------------------------->|

    If the MPCD particle data is accessed via the snapshot interface at time
    step 3, it will actually contain the MPCD particle data for time step 5.
    The MD particles can be read at any time step because their positions
    are updated every step.

    """

    def __init__(
        self,
        dt,
        integrate_rotational_dof=False,
        forces=None,
        constraints=None,
        methods=None,
        rigid=None,
        half_step_hook=None,
        streaming_method=None,
        collision_method=None,
    ):
        super().__init__(
            dt,
            integrate_rotational_dof,
            forces,
            constraints,
            methods,
            rigid,
            half_step_hook,
        )

        param_dict = ParameterDict(
            streaming_method=OnlyTypes(StreamingMethod, allow_none=True),
            collision_method=OnlyTypes(CollisionMethod, allow_none=True),
        )
        param_dict.update(
            dict(streaming_method=streaming_method,
                 collision_method=collision_method))
        self._param_dict.update(param_dict)

        self._cell_list = CellList(cell_size=1.0, shift=True)

    @property
    def cell_list(self):
        """hoomd.mpcd.CellList: Collision cell list.

        A `~hoomd.mpcd.CellList` is automatically created with each `Integrator`
        using typical defaults of cell size 1 and random grid shifting enabled.
        You can change this configuration if desired.

        """
        return self._cell_list

    def _attach_hook(self):
        self._cell_list._attach()
        if self.streaming_method is not None:
            self.streaming_method._attach()
        if self.collision_method is not None:
            self.collision_method._attach()

        self._cpp_obj = _mpcd.Integrator(self._simulation.state._cpp_sys_def,
                                         self.dt)
        self._cpp_obj.setCellList(self._cell_list._cpp_obj)

        super()._attach_hook()

    def _detach_hook(self):
        self._cell_list._detach()
        if self.streaming_method is not None:
            self.streaming_method._detach()
        if self.collision_method is not None:
            self.collision_method._detach()

        super()._detach_hook()

    def _setattr_param(self, attr, value):
        if attr == "streaming_method":
            self._set_streaming_method(value)
        elif attr == "collision_method":
            self._set_collision_method(value)
        else:
            super()._setattr_param(attr, value)

    def _set_streaming_method(self, streaming_method):
        if streaming_method is self.streaming_method:
            return

        if streaming_method is not None and streaming_method._attached:
            raise ValueError(
                "Cannot attach streaming method to multiple integrators.")

        # if already attached, change out which is attached, then set parameter
        if self._attached:
            if self.streaming_method is not None:
                self.streaming_method._detach()
            if streaming_method is not None:
                streaming_method._attach(self._simulation)
        self._param_dict["streaming_method"] = streaming_method

    def _set_collision_method(self, collision_method):
        if collision_method is self.collision_method:
            return

        if collision_method is not None and collision_method._attached:
            raise ValueError(
                "Cannot attach collision method to multiple integrators.")

        # if already attached, change out which is attached, then set parameter
        if self._attached:
            if self.collision_method is not None:
                self.collision_method._detach()
            if collision_method is not None:
                collision_method._attach(self._simulation)
        self._param_dict["collision_method"] = collision_method
