# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""User-defined pair potentials for HPMC simulations.

Set :math:`U_{\\mathrm{pair},ij}` evaluated in
`hoomd.hpmc.integrate.HPMCIntegrator` to a user-defined expression.

See Also:
    :doc:`features` explains the compile time options needed for user defined
    pair potentials.
"""

import hoomd
from hoomd import _compile
from hoomd.hpmc import integrate
if hoomd.version.llvm_enabled:
    from hoomd.hpmc import _jit
from hoomd.operation import AutotunedObject
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.logging import log
import numpy as np
import warnings


class CPPPotentialBase(AutotunedObject):
    """Base class for interaction between pairs of particles given in C++.

    Pair potential energies define energetic interactions between pairs of
    particles in `hoomd.hpmc.integrate.HPMCIntegrator`.  Particles within a
    cutoff distance interact with an energy that is a function the
    type and orientation of the particles and the vector pointing from the *i*
    particle to the *j* particle center.

    Classes derived from `CPPPotentialBase` take C++ code, compile it at
    runtime, and execute the code natively in the MC loop.

    Warning:
        `CPPPotentialBase` is **experimental** and subject to change in future
        minor releases.

    .. rubric:: C++ code

    Classes derived from `CPPPotentialBase` will compile the code provided by
    the user and call it to evaluate the energy :math:`U_{\\mathrm{pair},ij}`.
    The text provided is the body of a function with the following signature:

    .. code::

        float eval(const vec3<float>& r_ij,
                   unsigned int type_i,
                   const quat<float>& q_i,
                   float d_i,
                   float charge_i,
                   unsigned int type_j,
                   const quat<float>& q_j,
                   float d_j,
                   float charge_j
        )

    * ``r_ij`` is a vector pointing from the center of particle *i* to the \
        center of particle *j*.
    * ``type_i`` is the integer type id of particle *i*
    * ``q_i`` is the quaternion orientation of particle *i*
    * ``d_i`` is the diameter of particle *i*
    * ``charge_i`` is the charge of particle *i*
    * ``type_j`` is the integer type id of particle *j*
    * ``q_j`` is the quaternion orientation of particle *j*
    * ``d_j`` is the diameter of particle *j*
    * ``charge_j`` is the charge of particle *j*

    Note:
        ``vec3`` and ``quat`` are defined in the file `VectorMath.h`_ in the \
                HOOMD-blue source code.

    .. _VectorMath.h: https://github.com/glotzerlab/hoomd-blue/blob/\
            v4.8.2/hoomd/VectorMath.h

    Note:
        Your code *must* return a value.

    .. rubric:: Mixed precision

    `CPPPotentialBase` uses 32-bit precision floating point arithmetic when
    computing energies in the local particle reference frame.

    """

    @log(requires_run=True)
    def energy(self):
        """float: Total interaction energy of the system in the current state.

        .. math::

            U = \\sum_{i=0}^\\mathrm{N_particles-1}
            \\sum_{j=0}^\\mathrm{N_particles-1}
            U_{\\mathrm{pair},ij}

        Returns `None` when the patch object and integrator are not
        attached.
        """
        integrator = self._simulation.operations.integrator
        timestep = self._simulation.timestep
        return integrator._cpp_obj.computeTotalPairEnergy(timestep)

    def _wrap_cpu_code(self, code):
        r"""Wrap the provided code into a function with the expected signature.

        Args:
            code (`str`): Body of the C++ function
        """
        param_array_suffix = {True: '_isotropic', False: ''}[self._is_union]
        constituent_param_array = {
            True: 'float *param_array_constituent;',
            False: ''
        }[self._is_union]
        cpp_function = f"""
                        #include <stdio.h>
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"

                        // param_array (singlet class) or param_array_isotropic
                        // and param_array_constituent (union class) are
                        // allocated by the library
                        float *param_array{param_array_suffix};
                        {constituent_param_array}

                        using namespace hoomd;

                        extern "C"
                        {{
                        float eval(const vec3<float>& r_ij,
                            unsigned int type_i,
                            const quat<float>& q_i,
                            float d_i,
                            float charge_i,
                            unsigned int type_j,
                            const quat<float>& q_j,
                            float d_j,
                            float charge_j)
                            {{
                        """
        cpp_function += code
        cpp_function += """
                            }
                        }
                        """
        return cpp_function

    def _wrap_gpu_code(self, code):
        """Convert the provided code into a device function with the expected \
                signature.

        Args:
            code (`str`): Body of the C++ function
        """
        param_array_suffix = {True: '_isotropic', False: ''}[self._is_union]
        constituent_str = '__device__ float *param_array_constituent;'
        constituent_param_array = {
            True: constituent_str,
            False: ''
        }[self._is_union]
        cpp_function = f"""
                        #include "hoomd/HOOMDMath.h"
                        #include "hoomd/VectorMath.h"
                        #include "hoomd/hpmc/IntegratorHPMCMonoGPUJIT.inc"

                        using namespace hoomd;

                        // param_array (singlet class) or param_array_isotropic
                        // and param_array_constituent (union class) are
                        // allocated by the library
                        __device__ float *param_array{param_array_suffix};
                        {constituent_param_array}

                        __device__ inline float eval(const vec3<float>& r_ij,
                            unsigned int type_i,
                            const quat<float>& q_i,
                            float d_i,
                            float charge_i,
                            unsigned int type_j,
                            const quat<float>& q_j,
                            float d_j,
                            float charge_j)
                            {{
                        """
        cpp_function += code
        cpp_function += """
                            }
                        """
        return cpp_function


class CPPPotential(CPPPotentialBase):
    r"""Define an energetic interaction between pairs of particles.

    Adjust parameters within the code with the `param_array` attribute without
    requiring a recompile. These arrays are **read-only** during function
    evaluation.

    Args:
        r_cut (float): Particle center to center distance cutoff beyond which
            all pair interactions are 0.
        code (str): C++ code defining the function body for pair interactions
            between particles.
        param_array (list[float]): Parameter values to make available in
            ``float *param_array`` in the compiled code. If no adjustable
            parameters are needed in the C++ code, pass an empty array.

    See Also:
        `CPPPotentialBase` for the documentation of the parent class.

    Warning:
        `CPPPotential` is **experimental** and subject to change in future minor
        releases.

    Attributes:
        code (str): The C++ code that defines the body of the patch energy
            function. After running zero or more steps, this property cannot be
            changed.
        param_array ((*N*, ) `numpy.ndarray` of ``float``): Numpy
            array containing dynamically adjustable elements in the potential
            energy function as defined by the user. After running zero or more
            steps, the array cannot be set, although individual values can still
            be changed.
        energy (float): The potential energy resulting from the interactions
            defind in the C++ code at the current timestep.
    """

    _is_union = False

    def __init__(self, r_cut, code, param_array):
        param_dict = ParameterDict(r_cut=float,
                                   param_array=NDArrayValidator(
                                       dtype=np.float32, shape=(None,)),
                                   code=str)
        param_dict['r_cut'] = r_cut
        param_dict['param_array'] = param_array
        self._param_dict.update(param_dict)
        self.code = code

        warnings.warn(
            "CPPPotential is deprecated since 4.6.0. "
            "Use a hpmc.pair.Pair potential.",
            FutureWarning,
            stacklevel=2)

    def _getattr_param(self, attr):
        if attr == 'code':
            return self._param_dict[attr]
        return super()._getattr_param(attr)

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        device = self._simulation.device
        cpp_sys_def = self._simulation.state._cpp_sys_def

        cpu_code = self._wrap_cpu_code(self.code)
        cpu_include_options = _compile.get_cpu_compiler_arguments()

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
        super()._attach_hook()


class CPPPotentialUnion(CPPPotentialBase):
    r"""Define an arbitrary energetic interaction between unions of particles.

    Args:
        r_cut_constituent (`float`): Constituent particle center to center
                distance cutoff beyond which all pair interactions are 0.
        code_constituent (`str`): C++ code defining the custom pair
                interactions between constituent particles.
        r_cut_isotropic (`float`): Cut-off for isotropic interaction between
                centers of union particles.
        code_isotropic (`str`): C++ code for isotropic part of the interaction.
                Must be ``''`` when executing on a GPU.
        param_array_constituent (list[float]): Parameter values to make
            available in ``float *param_array_constituent`` in the compiled
            code.  Pass an empty array if no adjustable parameters are needed
            for the constituent interactions.
        param_array_isotropic (list[float]): Parameter values to make available
            in ``float *param_array_isotropic`` in the compiled code. Pass an
            empty array if no adjustable parameters are needed for the isotropic
            interactions.

    Note:
        Code passed into ``code_isotropic`` is not used when executing on the
        GPU. A `RuntimeError` is raised on attachment if the value of this
        argument is anything besides ``''`` on the GPU.

    Note:
        This class uses an internal OBB tree for fast interaction queries
        between constituents of interacting particles.  Depending on the number
        of constituent particles per type in the tree, different values of the
        particles per leaf node may yield different optimal performance. The
        capacity of leaf nodes is configurable.

    See Also:
        `CPPPotentialBase` for the documentation of the parent class.

    Warning:
        `CPPPotentialUnion` is **experimental** and subject to change in future
        minor releases.

    .. rubric:: Threading

    CPPPotentialUnion uses threaded execution on multiple CPU cores.

    .. deprecated:: 4.8.2

        ``num_cpu_threads >= 1`` is deprecated. Set ``num_cpu_threads = 1``.

    .. py:attribute:: positions

        The positions of the constituent particles.

        Type: `TypeParameter` [``particle type``, `list` [`tuple` [`float`,
        `float`, `float`]]]

    .. py:attribute:: orientations

        The orientations of the constituent particles.

        Type: `TypeParameter` [``particle type``, `list` [`tuple` [`float`,
        `float`, `float`, `float`]]]

    Attributes:
        diameters (`TypeParameter` [``particle type``, `list` [`float`]])
            The diameters of the constituent particles.

        charges (`TypeParameter` [``particle type``, `list` [`float`]])
            The charges of the constituent particles.

        typeids (`TypeParameter` [``particle type``, `list` [`float`]])
            The integer types of the constituent particles.

        leaf_capacity (`int`) : The number of particles in a
                leaf of the internal tree data structure (**default:** 4).

        code_constituent (str): The C++ code that defines the body of the patch
            energy function between the constituent particles. This property
            cannot be modified after running for zero or more steps.

        code_isotropic (str): The C++ code that defines the body of the patch
            energy function between the centers of the particles. This property
            cannot be modified after running for zero or more steps.

        param_array_isotropic ((*N*, ) `numpy.ndarray` of ``float``): Numpy
            array containing dynamically adjustable elements in the isotropic
            part of the potential as defined by the user. After running zero or
            more steps, the array cannot be set, although individual values can
            still be changed.

        param_array_constituent ((*N*, ) `numpy.ndarray` of ``float``): Numpy
            array containing dynamically adjustable elements in the constituent
            part of the potential part of the potential as defined by the user.
            After running zero or more steps, the array cannot be set, although
            individual values can still be changed.
        energy (float): The potential energy resulting from the interactions
            defind in the C++ code at the current timestep.

    Example without isotropic interactions:

    .. code-block:: python

        square_well = '''float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      '''
        patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
            r_cut_constituent=1.1,
            r_cut_isotropic=0.0
            code_constituent=square_well,
            code_isotropic='',
            param_array_constituent=[],
            param_array_isotropic=[]
        )
        patch.positions['A'] = [
            (0, 0, -0.5),
            (0, 0, 0.5)
        ]
        patch.diameters['A'] = [0, 0]
        patch.typeids['A'] = [0, 0]
        mc.pair_potential = patch

    Example with added isotropic interactions:

    .. code-block:: python

        # square well attraction on constituent spheres
        square_well = '''float rsq = dot(r_ij, r_ij);
                              float r_cut = param_array_constituent[0];
                              if (rsq < r_cut*r_cut)
                                  return param_array_constituent[1];
                              else
                                  return 0.0f;
                        '''

        # soft repulsion between centers of unions
        soft_repulsion = '''float rsq = dot(r_ij, r_ij);
                                  float r_cut = param_array_isotropic[0];
                                  if (rsq < r_cut*r_cut)
                                    return param_array_isotropic[1];
                                  else
                                    return 0.0f;
                         '''

        patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
            r_cut_constituent=2.5,
            r_cut_isotropic=5.0,
            code_union=square_well,
            code_isotropic=soft_repulsion,
            param_array_constituent=[2.0, -5.0],
            param_array_isotropic=[2.5, 1.3],
        )
        patch.positions['A'] = [
            (0, 0, -0.5),
            (0, 0, 0.5)
        ]
        patch.typeids['A'] = [0, 0]
        patch.diameters['A'] = [0, 0]
        mc.pair_potential = patch

    """

    _is_union = True

    def __init__(self, r_cut_constituent, code_constituent, r_cut_isotropic,
                 code_isotropic, param_array_constituent,
                 param_array_isotropic):

        param_dict = ParameterDict(
            r_cut_constituent=float(r_cut_constituent),
            r_cut_isotropic=float(r_cut_isotropic),
            leaf_capacity=int(4),
            param_array_constituent=NDArrayValidator(dtype=np.float32,
                                                     shape=(None,)),
            param_array_isotropic=NDArrayValidator(dtype=np.float32,
                                                   shape=(None,)),
            code_constituent=str,
            code_isotropic=str,
        )

        param_dict['param_array_constituent'] = param_array_constituent
        param_dict['param_array_isotropic'] = param_array_isotropic
        self._param_dict.update(param_dict)

        # add union specific per-type parameters
        typeparam_positions = TypeParameter(
            'positions',
            type_kind='particle_types',
            param_dict=TypeParameterDict([(float, float, float)], len_keys=1),
        )

        typeparam_orientations = TypeParameter(
            'orientations',
            type_kind='particle_types',
            param_dict=TypeParameterDict([(float, float, float, float)],
                                         len_keys=1),
        )

        typeparam_diameters = TypeParameter(
            'diameters',
            type_kind='particle_types',
            param_dict=TypeParameterDict([float], len_keys=1),
        )

        typeparam_charges = TypeParameter(
            'charges',
            type_kind='particle_types',
            param_dict=TypeParameterDict([float], len_keys=1),
        )

        typeparam_typeids = TypeParameter(
            'typeids',
            type_kind='particle_types',
            param_dict=TypeParameterDict([int], len_keys=1),
        )

        self._extend_typeparam([
            typeparam_positions, typeparam_orientations, typeparam_diameters,
            typeparam_charges, typeparam_typeids
        ])

        self.code_constituent = code_constituent
        self.code_isotropic = code_isotropic

        warnings.warn(
            "CPPPotentialUnion is deprecated since 4.6.0. "
            "Use a hpmc.pair.Pair potential.",
            FutureWarning,
            stacklevel=2)

    def _getattr_param(self, attr):
        code_attrs = {'code_isotropic', 'code_constituent'}
        if attr in code_attrs:
            return self._param_dict[attr]
        return super()._getattr_param(attr)

    def _attach_hook(self):
        if (isinstance(self._simulation.device, hoomd.device.CPU)
                and self._simulation.device.num_cpu_threads > 1):
            warnings.warn(
                "num_cpu_threads > 1 is deprecated since 4.6.0. "
                "Use num_cpu_threads=1.",
                FutureWarning,
                stacklevel=1)

        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        if isinstance(self._simulation.device, hoomd.device.GPU):
            if self.code_isotropic != '':
                msg = 'Code passed into code_isotropic when excuting on the '
                msg += 'GPU is unused.'
                raise RuntimeError(msg)

        cpu_code_constituent = self._wrap_cpu_code(self.code_constituent)
        if self.code_isotropic != '':
            cpu_code_isotropic = self._wrap_cpu_code(self.code_isotropic)
        else:
            cpu_code_isotropic = self._wrap_cpu_code('return 0;')
        cpu_include_options = _compile.get_cpu_compiler_arguments()

        device = self._simulation.device
        if isinstance(self._simulation.device, hoomd.device.GPU):
            gpu_settings = _compile.get_gpu_compilation_settings(device)
            # use union evaluator
            gpu_code_constituent = self._wrap_gpu_code(self.code_constituent)
            self._cpp_obj = _jit.PatchEnergyJITUnionGPU(
                self._simulation.state._cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code_isotropic,
                cpu_include_options,
                self.r_cut_isotropic,
                self.param_array_isotropic,
                cpu_code_constituent,
                self.r_cut_constituent,
                self.param_array_constituent,
                gpu_code_constituent,
                "hpmc::gpu::kernel::hpmc_narrow_phase_patch",
                gpu_settings["includes"] + ["-DUNION_EVAL"],
                gpu_settings["cuda_devrt_lib_path"],
                gpu_settings["max_arch"],
            )
        else:
            self._cpp_obj = _jit.PatchEnergyJITUnion(
                self._simulation.state._cpp_sys_def,
                device._cpp_exec_conf,
                cpu_code_isotropic,
                cpu_include_options,
                self.r_cut_isotropic,
                self.param_array_isotropic,
                cpu_code_constituent,
                self.r_cut_constituent,
                self.param_array_constituent,
            )
        # attach patch object to the integrator
        super()._attach_hook()
