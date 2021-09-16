# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""User-defined pair potentials for HPMC simulations."""

import hoomd
from hoomd import _compile
from hoomd.hpmc import integrate
if hoomd.version.llvm_enabled:
    from hoomd.hpmc import _jit
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.logging import log
import numpy as np


class CPPPotentialBase(_HOOMDBaseObject):
    """Base class for interaction between pairs of particles given in C++.

    Pair potential energies define energetic interactions between pairs of
    shapes in :py:mod:`hpmc <hoomd.hpmc>` integrators.  Shapes within a cutoff
    distance are interact and the energy of interaction is a function the type
    and orientation of the particles and the vector pointing from the *i*
    particle to the *j* particle center.

    Classes derived from :py:class:`CPPPotentialBase` take C++ code, compilesit
    at run time and executes the code natively in the MC loop. Adjust parameters
    to the code with the `param_array` attribute without requiring a recompile.
    These arrays are **read-only** during function evaluation.

    .. rubric:: C++ code

    Classes derived from :py:class:`CPPPotentialBase` will compile the code
    provided by the user and call it to evaluate patch energies. The text
    provided in *code* is the body of a function with the following signature:

    .. code::

        float eval(const vec3<float>& r_ij,
                    unsigned int type_i,
                    const quat<float>& q_i,
                    float d_i,
                    float charge_i,
                    unsigned int type_j,
                    const quat<float>& q_j,
                    float d_j,
                    float charge_j)

    * ``r_ij`` is a vector pointing from the center of particle *i* to the
        center of particle *j*.
    * ``type_i`` is the integer type id of particle *i*
    * ``q_i`` is the quaternion orientation of particle *i*
    * ``d_i`` is the diameter of particle *i*
    * ``charge_i`` is the charge of particle *i*
    * ``type_j`` is the integer type id of particle *j*
    * ``q_j`` is the quaternion orientation of particle *j*
    * ``d_j`` is the diameter of particle *j*
    * ``charge_j`` is the charge of particle *j*
    * Your code *must* return a value.

    ``vec3`` and ``quat`` are defined in :file:`HOOMDMath.h`.

    See Also:
        `CPPPotential`

        `CPPPotentialUnion`

    Attributes:
        r_cut (float): Particle center to center distance cutoff beyond which
            all pair interactions are assumed 0.
        param_array ((N,) `numpy.ndarray` of float): Numpy array containing
            dynamically adjustable elements defined by the user. Cannot change
            size after calling `Simulation.run`. The elements are still mutable
            however.
    """

    def __init__(self, r_cut, code, param_array=None):
        param_dict = ParameterDict(r_cut=float,
                                   param_array=hoomd.data.typeconverter.Array(
                                       dtype=np.float32, ndim=1),
                                   code=str)
        param_dict['r_cut'] = r_cut
        param_dict['code'] = code
        if param_array is None:
            param_dict['param_array'] = np.array([])
        else:
            param_dict['param_array'] = param_array
        self._param_dict.update(param_dict)

    def _getattr_param(self, attr):
        if attr == 'code':
            return self._param_dict['code']
        return super()._getattr_param(attr)

    @log(requires_run=True)
    def energy(self):
        """float: Total interaction energy of the system in the current state.

        Returns `None` when the patch object and integrator are not
        attached.
        """
        integrator = self._simulation.operations.integrator
        timestep = self._simulation.timestep
        return integrator._cpp_obj.computePatchEnergy(timestep)

    def _wrap_cpu_code(self, code):
        r"""Helper function to wrap the provided code into a function
            with the expected signature.

        Args:
            code (`str`): Body of the C++ function
        """
        cpp_function = """
                        #include <stdio.h>
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"

                        // these are allocated by the library
                        float *param_array;
                        float *alpha_union;

                        extern "C"
                        {
                        float eval(const vec3<float>& r_ij,
                            unsigned int type_i,
                            const quat<float>& q_i,
                            float d_i,
                            float charge_i,
                            unsigned int type_j,
                            const quat<float>& q_j,
                            float d_j,
                            float charge_j)
                            {
                        """
        cpp_function += code
        cpp_function += """
                            }
                        }
                        """
        return cpp_function

    def _wrap_gpu_code(self, code):
        r"""Helper function to convert the provided code into a device function
            with the expected signature.

        Args:
            code (`str`): Body of the C++ function
        """
        cpp_function = """
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"
                        #include "hoomd/hpmc/IntegratorHPMCMonoGPUJIT.inc"

                        // these are allocated by the library
                        __device__ float *param_array;
                        __device__ float *alpha_union;

                        __device__ inline float eval(const vec3<float>& r_ij,
                            unsigned int type_i,
                            const quat<float>& q_i,
                            float d_i,
                            float charge_i,
                            unsigned int type_j,
                            const quat<float>& q_j,
                            float d_j,
                            float charge_j)
                            {
                        """
        cpp_function += code
        cpp_function += """
                            }
                        """
        return cpp_function


class CPPPotential(CPPPotentialBase):
    """Define an energetic interaction between pairs of particles.

    Args:
        r_cut (float): Particle center to center distance cutoff beyond which
            all pair interactions are assumed 0.
        code (str): C++ code defining the function body for pair interactions
            between particles.
        param_array (list[float]): Parameter values to pass into ``param_array``
            in the compiled code.

    See Also:
        `CPPPotentialBase`

    Examples:

        .. code-block:: python

            square_well = '''float rsq = dot(r_ij, r_ij);
                                if (rsq < 1.21f)
                                    return -1.0f;
                                else
                                    return 0.0f;
                        '''
            patch = hoomd.jit.patch.CPPPotential(r_cut=1.1, code=square_well)
            sim.operations += patch
            sim.run(1000)
    """

    def __init__(self, r_cut, code, param_array=None):
        super().__init__(r_cut=r_cut, code=code, param_array=param_array)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        device = self._simulation.device
        cpp_sys_def = self._simulation.state._cpp_sys_def

        cpu_code = self._wrap_cpu_code(self.code)
        cpu_include_options = _compile.get_cpu_include_options()

        if isinstance(device, hoomd.device.GPU):
            gpu_settings = _compile.get_gpu_compilation_settings(device)
            gpu_code = self._wrap_gpu_code(self.code)

            self._cpp_obj = _jit.PatchEnergyJITGPU(
                cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code,
                cpu_include_options,
                self.r_cut,
                self.param_array,
                gpu_code,
                "hpmc::gpu::kernel::hpmc_narrow_phase_patch",
                gpu_settings["includes"],
                gpu_settings["cuda_devrt_lib_path"],
                gpu_settings["max_arch"],
            )
        else:  # running on cpu
            self._cpp_obj = _jit.PatchEnergyJIT(
                cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code,
                cpu_include_options,
                self.r_cut,
                self.param_array,
            )
        # attach patch object to the integrator
        super()._attach()


class _CPPUnionPotential(CPPPotentialBase):
    r'''Define an arbitrary energetic interaction between unions of particles.


    Warning:
        This class does not currenlty work. Please do not attempt to use this.

    Args:
        r_cut_union (`float`): Constituent particle center to center distance
            cutoff beyond which all pair interactions are assumed 0.
        r_cut (`float`, **default** 0): Cut-off for isotropic interaction
            between centers of union particles.
        code_union (`str`): C++ code defining the custom pair interactions
            between constituent particles.
        code (`str`): C++ code for isotropic part.
        array_size_union (`int`, **default:** 1): Size of array with adjustable
            elements.
        array_size (`int`, **default:** 1): Size of array with adjustable
            elements for the isotropic part.

    Note:
        If both `code_union` and `llvm_ir_fname_union` are provided, the former
            takes precedence. The latter will be used as a fallback in case the
            compilation of `code_union` fails.

    Note:
        This class uses an internal OBB tree for fast interaction queries
            between constituents of interacting particles.
        Depending on the number of constituent particles per type in the tree,
            different values of the particles per leaf
        node may yield different optimal performance. The capacity of leaf nodes
            is configurable.

    Attributes:
        positions (`TypeParameter` [``particle type``, `list` [`tuple` [`float`, `float`, `float`]]])
            The positions of the constituent particles.
        orientations (`TypeParameter` [``particle type``, `list` [`tuple` [`float`, `float`, `float, `float`]]])
            The orientations of the constituent particles.
        diameters (`TypeParameter` [``particle type``, `list` [`float`]])
            The diameters of the constituent particles.
        charges (`TypeParameter` [``particle type``, `list` [`float`]])
            The charges of the constituent particles.
        typeids (`TypeParameter` [``particle type``, `list` [`float`]])
            The charges of the constituent particles.
        leaf_capacity (`int`, **default:** 4) : The number of particles in a leaf of the internal tree data structure
        alpha_union (``ndarray<float>``): Length array_size_union numpy array containing dynamically adjustable elements
                                          defined by the user for unions of shapes.

    Example:

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.CPPUnionPotential(r_cut_union=1.1, code_union=square_well)
        patch.positions['A'] = [(0,0,-5.),(0,0,.5)]
        patch.diameters['A'] =[0,0]
        patch.typeids['A'] =[0,0]

    Example with added isotropic interactions:

    .. code-block:: python

        # square well attraction on constituent spheres
        square_well = """float rsq = dot(r_ij, r_ij);
                              float r_cut = alpha_union[0];
                              if (rsq < r_cut*r_cut)
                                  return alpha_union[1];
                              else
                                  return 0.0f;
                           """

        # soft repulsion between centers of unions
        soft_repulsion = """float rsq = dot(r_ij, r_ij);
                                  float r_cut = param_array[0];
                                  if (rsq < r_cut*r_cut)
                                    return param_array[1];
                                  else
                                    return 0.0f;
                         """

        patch = hoomd.jit.patch.CPPUnionPotential(r_cut_union=2.5, code_union=square_well, array_size_union=2, \
                                               r_cut=5, code=soft_repulsion, array_size=2)
        patch.positions['A'] = [(0,0,-5.),(0,0,.5)]
        patch.typeids['A'] = [0,0]
        patch.diameters['A'] = [0,0]
        # [r_cut, epsilon]
        patch.param_array[:] = [2.5, 1.3];
        patch.alpha_union[:] = [2.5, -1.7];
    '''

    def __init__(self,
                 r_cut_union,
                 array_size_union=1,
                 code_union=None,
                 r_cut=0,
                 array_size=1):

        # initialize base class
        super().__init__(r_cut=r_cut, array_size=array_size, code=code)

        # add union specific params
        param_dict = ParameterDict(r_cut_union=float(r_cut_union),
                                   array_size_union=int(array_size_union),
                                   leaf_capacity=int(4))
        self._param_dict.update(param_dict)

        # add union specific per-type parameters
        typeparam_positions = TypeParameter('positions',
                                            type_kind='particle_types',
                                            param_dict=TypeParameterDict(
                                                [tuple], len_keys=1))

        typeparam_orientations = TypeParameter('orientations',
                                               type_kind='particle_types',
                                               param_dict=TypeParameterDict(
                                                   [tuple], len_keys=1))

        typeparam_diameters = TypeParameter('diameters',
                                            type_kind='particle_types',
                                            param_dict=TypeParameterDict(
                                                [float], len_keys=1))

        typeparam_charges = TypeParameter('charges',
                                          type_kind='particle_types',
                                          param_dict=TypeParameterDict(
                                              [float], len_keys=1))

        typeparam_typeids = TypeParameter('typeids',
                                          type_kind='particle_types',
                                          param_dict=TypeParameterDict(
                                              [int], len_keys=1))

        self._extend_typeparam([
            typeparam_positions, typeparam_orientations, typeparam_diameters,
            typeparam_charges, typeparam_typeids
        ])

        # these only exist on python
        self._code_union = code_union
        self.alpha_union = np.zeros(array_size_union)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        cpu_code_constituent = self._wrap_cpu_code(self._code_union)
        cpu_code_iso = self._wrap_cpu_code(self._code)
        cpu_include_options = _compile.get_cpu_include_options()

        device = self._simulation.device
        if isinstance(self._simulation.device, hoomd.device.GPU):
            gpu_settings = _compile.get_gpu_compilation_settings(device)
            # use union evaluator
            gpu_code = self._wrap_gpu_code(self._code)
            self._cpp_obj = _jit.PatchEnergyJITUnionGPU(
                self._simulation.state._cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code_iso,
                self.r_cut,
                self.param_array,
                cpu_code_constituent,
                self.r_cut_union,
                self.array_size_union,
                gpu_code,
                "hpmc::gpu::kernel::hpmc_narrow_phase_patch",
                gpu_settings["includes"] + ["-DUNION_EVAL"],
                gpu_settings["cuda_devrt_lib_path"],
                gpu_settings["max_arch"],
            )
        else:
            self._cpp_obj = _jit.PatchEnergyJITUnion(
                self._simulation.state._cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code_iso,
                cpu_include_options,
                self.r_cut,
                self.param_array,
                cpu_code_constituent,
                self.r_cut_union,
                self.array_size_union,
            )

        # Set the C++ mirror array with the cached values
        # and override the python array
        self._cpp_obj.alpha_iso[:] = self.alpha_iso[:]
        self._cpp_obj.alpha_union[:] = self.alpha_union[:]
        self.alpha_iso = self._cpp_obj.alpha_iso
        self.alpha_union = self._cpp_obj.alpha_union
        # attach patch object to the integrator
        super()._attach()

    @property
    def code_union(self):
        """str: The C++ code defining the custom pair interactions between
        constituent particles."""
        return self._code_union

    @code_union.setter
    def code_union(self, code):
        if self._attached:
            raise AttributeError(
                "This attribute can only be set when the object is not attached."
            )
        else:
            self._code_union = code
