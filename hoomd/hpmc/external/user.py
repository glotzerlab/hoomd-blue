# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""User-defined external fields for HPMC simulations.

Set :math:`U_{\\mathrm{external},i}` evaluated in
`hoomd.hpmc.integrate.HPMCIntegrator` to a user-defined expression.

See Also:
    :doc:`features` explains the compile time options needed for user defined
    external potentials.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    hpmc_integrator = hoomd.hpmc.integrate.Sphere()
    hpmc_integrator.shape['A'] = dict(diameter=1.0)
    simulation.operations.integrator = hpmc_integrator
    logger = hoomd.logging.Logger()
"""

import hoomd
from hoomd import _compile
from hoomd.hpmc import integrate
from hoomd.hpmc.external.field import ExternalField
if hoomd.version.llvm_enabled:
    from hoomd.hpmc import _jit
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.logging import log
import numpy as np
import warnings


class CPPExternalPotential(ExternalField):
    """Define an external potential energy field imposed on all particles in \
            the system.

    Args:
        code (str): C++ function body to compile.
        param_array (list[float]): Parameter values to make available in
            ``float *param_array`` in the compiled code. If no adjustable
            parameters are needed in the C++ code, pass an empty array.

    Potentials added using `CPPExternalPotential` are added to the total energy
    calculation in `hoomd.hpmc.integrate.HPMCIntegrator`. `CPPExternalPotential`
    takes C++ code, compiles it at runtime, and executes the code natively in
    the MC loop with full performance. It enables researchers to quickly and
    easily implement custom energetic field intractions without the need to
    modify and recompile HOOMD.

    Adjust parameters within the code with the `param_array` attribute without
    requiring a recompile. These arrays are **read-only** during function
    evaluation.

    .. rubric:: C++ code

    Supply C++ code to the *code* argument and `CPPExternalPotential` will
    compile the code and call it to evaluate the energy. The text provided in
    *code* is the body of a function with the following signature:

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

    Note:
        ``vec3`` and ``quat`` are defined in the file `VectorMath.h`_ in the \
                HOOMD-blue source code, and ``BoxDim`` is defined in he file \
                `BoxDim.h`_ in the HOOMD-blue source code.

    Note:
        Your code *must* return a value.

    .. _VectorMath.h: https://github.com/glotzerlab/hoomd-blue/blob/\
            v4.8.2/hoomd/VectorMath.h
    .. _BoxDim.h: https://github.com/glotzerlab/hoomd-blue/blob/\
            v4.8.2/hoomd/BoxDim.h

    .. rubric:: Example:

    .. skip: next if(llvm_not_available)

    .. code-block:: python

        gravity_code = '''
            float gravity_constant = param_array[0];
            return gravity_constant * (r_i.z + box.getL().z/2);
        '''
        cpp_external_potential = hoomd.hpmc.external.user.CPPExternalPotential(
            code=gravity_code, param_array=[9.8])
        hpmc_integrator.external_potential = cpp_external_potential

    Note:
        `CPPExternalPotential` does not support execution on GPUs.

    Warning:
        ``CPPExternalPotential`` is **experimental** and subject to change in
        future minor releases.

    Attributes:
        code (str): The code of the body of the external field energy function.
            After running zero or more steps, this property cannot be modified.

            .. rubric:: Example

            .. skip: next if(llvm_not_available)

            .. code-block:: python

                code = cpp_external_potential.code

        param_array ((*N*, ) `numpy.ndarray` of ``float``): Numpy
            array containing dynamically adjustable elements in the potential
            energy function as defined by the user. After running zero or more
            steps, the array cannot be set, although individual values can still
            be changed.

            .. rubric:: Example

            .. skip: next if(llvm_not_available)

            .. code-block:: python

                cpp_external_potential.param_array[0] = 10.0

    """

    def __init__(self, code, param_array=[]):
        param_dict = ParameterDict(
            code=str,
            param_array=NDArrayValidator(dtype=np.float32, shape=(None,)),
        )
        param_dict['param_array'] = param_array
        self._param_dict.update(param_dict)
        self.code = code

        warnings.warn(
            "CPPExternalPotential is deprecated since 4.8.0. "
            "Use hpmc.external.linear or a custom component.",
            FutureWarning,
            stacklevel=2)

    def _getattr_param(self, attr):
        if attr == 'code':
            return self._param_dict._dict[attr]
        return super()._getattr_param(attr)

    def _wrap_cpu_code(self, code):
        """Helper function to wrap the provided code into a function \
            with the expected signature.

        Args:
            code (str): Body of the C++ function
        """
        cpp_function = """
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"
                        #include "hoomd/BoxDim.h"

                        // param_array is allocated by the library
                        float *param_array;

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

    def _attach_hook(self):
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

        cpu_code = self._wrap_cpu_code(self.code)
        cpu_include_options = _compile.get_cpu_compiler_arguments()

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self._simulation.device._cpp_exec_conf,
                                cpu_code, cpu_include_options, self.param_array)
        super()._attach_hook()

    @log(requires_run=True)
    def energy(self):
        """float: Total field energy of the system in the current state.

        .. math::

            U = \\sum_{i=0}^\\mathrm{N_particles-1}
            U_{\\mathrm{external},i}

        .. rubric:: Example:

        .. skip: next if(llvm_not_available)

        .. code-block:: python

            logger.add(cpp_external_potential, quantities=['energy'])

        Returns `None` when the patch object and integrator are not attached.
        """
        timestep = self._simulation.timestep
        return self._cpp_obj.computeEnergy(timestep)
