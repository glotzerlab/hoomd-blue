# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""User-defined external fields for HPMC simulations."""

import hoomd
from hoomd import _compile
from hoomd.hpmc import integrate
if hoomd.version.llvm_enabled:
    from hoomd.hpmc import _jit
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log


class CPPExternalField(_HOOMDBaseObject):
    """Define an external field imposed on all particles in the system.

    Args:
        code (str): C++ function body to compile.

    Potentials added using external.CPPExternalField are added to the total
    energy calculation in :py:mod:`hpmc <hoomd.hpmc>` integrators. The
    :py:class:`CPPExternalField` external field takes C++ code, compiles it
    at runtime, and executes the code natively in the MC loop with full
    performance. It enables researchers to quickly and easily implement custom
    energetic field intractions without the need to modify and recompile HOOMD.

    .. rubric:: C++ code

    Supply C++ code to the *code* argument and :py:class:`CPPExternalField` will
    compile the code and call it to evaluate the energy.
    The text provided in *code* is the body of a function with the following
    signature:

    .. code::

        float eval(const BoxDim& box,
                   unsigned int type_i,
                   const vec3<Scalar>& r_i,
                   const quat<Scalar>& q_i
                   Scalar diameter,
                   Scalar charge
        )

    * *box* is the system box.
    * *type_i* is the (integer) particle type.
    * *r_i* is the particle position
    * *q_i* the quaternion representing the particle orientation.
    * *diameter* the particle diameter.
    * *charge* the particle charge.
    * Your code *must* return a value.
    * `BoxDim` is defined in :file:`BoxDim.h` in the HOOMD-blue source code.
    * ``vec3`` and ``quat`` are defined in :file:`HOOMDMath.h` in the \
            HOOMD-blue source code.

    Example:
        .. code-block:: python

            grav_code = "return r_i.z + box.getL().z/2;"
            gravity = hoomd.hpmc.external.user.CPPExternalField(code=grav_code)
            mc.field = gravity

    Note:
        `CPPExternalField` does not support execution on GPUs.

    Warning:
        ``CPPExternalField`` is **experimental** and subject to change in future
        minor releases.

    """

    def __init__(self, code):
        self._code = code

    def _wrap_cpu_code(self, code):
        """Helper function to wrap the provided code into a function \
            with the expected signature.

        Args:
            code (`str`): Body of the C++ function
        """
        cpp_function = """
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"
                        #include "hoomd/BoxDim.h"

                        using namespace hoomd;

                        extern "C"
                        {

                        float eval(const BoxDim& box,
                        unsigned int type_i,
                        const vec3<Scalar> r_i,
                        const quat<Scalar>& q_i,
                        Scalar diameter,
                        Scalar charge
                        )
                            {
                        """
        cpp_function += code
        cpp_function += """
                            }
                        }
                        """
        return cpp_function

    def _attach(self):
        integrator_pairs = {
            integrate.Sphere:
                _jit.ExternalFieldJITSphere,
            integrate.ConvexPolygon:
                _jit.ExternalFieldJITConvexPolygon,
            integrate.SimplePolygon:
                _jit.ExternalFieldJITSimplePolygon,
            integrate.ConvexPolyhedron:
                _jit.ExternalFieldJITConvexPolyhedron,
            integrate.ConvexSpheropolyhedron:
                _jit.ExternalFieldJITSpheropolyhedron,
            integrate.Ellipsoid:
                _jit.ExternalFieldJITEllipsoid,
            integrate.ConvexSpheropolygon:
                _jit.ExternalFieldJITSpheropolygon,
            integrate.FacetedEllipsoid:
                _jit.ExternalFieldJITFacetedEllipsoid,
            integrate.Polyhedron:
                _jit.ExternalFieldJITPolyhedron,
            integrate.Sphinx:
                _jit.ExternalFieldJITSphinx
        }

        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if (isinstance(self._simulation.device, hoomd.device.GPU)):
            msg = 'User-defined external fields are not supported on the GPU.'
            raise NotImplementedError(msg)

        cpp_cls = integrator_pairs.get(
            self._simulation.operations.integrator.__class__, None)
        if cpp_cls is None:
            raise RuntimeError("Unsupported integrator.\n")

        cpu_code = self._wrap_cpu_code(self._code)
        cpu_include_options = _compile.get_cpu_include_options()

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self._simulation.device._cpp_exec_conf,
                                cpu_code, cpu_include_options)
        super()._attach()

    @log(requires_run=True)
    def energy(self):
        """float: Total field energy of the system in the current state.

        Returns `None` when the patch object and integrator are not attached.
        """
        timestep = self._simulation.timestep
        return self._cpp_obj.computeEnergy(timestep)

    @property
    def code(self):
        """str: The C++ code defines the external field.

        This returns the code that was passed into the class constructor, which
        contains only the body of the external field energy kernel.
        """
        return self._code

    @code.setter
    def code(self, code):
        if self._attached:
            raise AttributeError("This attribute can only be set before the \
                                  object is attached.")
        else:
            self._code = code
