# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd
from hoomd import _compile
from hoomd.hpmc import integrate
from hoomd.hpmc import _jit
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log


class CPPExternalField(_HOOMDBaseObject):
    r'''Define an external field imposed on all particles in the system.

    Args:
        code (str): C++ function body to compile.
        clang_exec (str, optional): The clang executable to use, defaults to
        ``'clang'``.

    Potentials added using external.CPPExternalField are added to the total
    energy calculation in :py:mod:`hpmc <hoomd.hpmc>` integrators. The
    :py:class:`CPPExternalField` external field takes C++ code, JIT compiles it
    at run time and executes the code natively in the MC loop with full
    performance. It enables researchers to quickly and easily implement custom
    energetic field intractions without the need to modify and recompile HOOMD.

    .. rubric:: C++ code

    Supply C++ code to the *code* argument and :py:class:`CPPExternalField` will
    compile the code and call it to evaluate forces. Compilation assumes that a
    recent ``clang`` installation is on your PATH.

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

    * ``vec3`` and ``quat`` are defined in HOOMDMath.h.
    * *box* is the system box.
    * *type_i* is the particle type.
    * *r_i* is the particle position
    * *q_i* the particle orientation.
    * *diameter* the particle diameter.
    * *charge* the particle charge.
    * Your code *must* return a value.

    Example:

    .. code-block:: python

        gravity = "return r_i.z + box.getL().z/2;"
        external = hoomd.jit.external.CPPExternalField(code=gravity)

    Note:
        CPPExternalField does not support execution on GPUs.
    '''
    _integrator_pairs = {
        integrate.Sphere: _jit.ExternalFieldJITSphere,
        integrate.ConvexPolygon: _jit.ExternalFieldJITConvexPolygon,
        integrate.SimplePolygon: _jit.ExternalFieldJITSimplePolygon,
        integrate.ConvexPolyhedron: _jit.ExternalFieldJITConvexPolyhedron,
        integrate.ConvexSpheropolyhedron: _jit.ExternalFieldJITSpheropolyhedron,
        integrate.Ellipsoid: _jit.ExternalFieldJITEllipsoid,
        integrate.ConvexSpheropolygon: _jit.ExternalFieldJITSpheropolygon,
        integrate.FacetedEllipsoid: _jit.ExternalFieldJITFacetedEllipsoid,
        integrate.Polyhedron: _jit.ExternalFieldJITPolyhedron,
        integrate.Sphinx: _jit.ExternalFieldJITSphinx
    }

    def __init__(self, code, clang_exec='clang'):
        self._llvm_ir = _compile.to_llvm_ir(self._wrap_cpu_(code))
        self._code = code

    def _wrap_cpu_code(self, code):
        r"""Helper function to wrap the provided code into a function
            with the expected signature.

        Args:
            code (`str`): Body of the C++ function
        """
        cpp_function = """
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"
                        #include "hoomd/BoxDim.h"

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
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if (isinstance(self._simulation.device, hoomd.device.GPU)):
            raise RuntimeError("JIT forces are not supported on the GPU.")

        cpp_cls = self._integrator_pairs.get(
            self._simulation.operations.integrator.__class__, None)
        if cpp_cls is None:
            raise RuntimeError("Unsupported integrator.\n")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self._simulation.device._cpp_exec_conf,
                                self._llvm_ir)
        super()._attach()

    @log
    def energy(self):
        """float: Total field energy of the system in the current state.

        Returns `None` when the patch object and integrator are not attached.
        """
        if self._attached:
            timestep = self._simulation.timestep
            return self._cpp_obj.computeEnergy(timestep)
        else:
            return None
