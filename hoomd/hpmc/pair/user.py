# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd
from hoomd.hpmc import integrate
from hoomd.hpmc import _jit
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.logging import log
import subprocess
import os
import numpy as np


def to_llvm_ir(code, clang_exec):
    r"""Helper function to compile the provided code into an executable

    Args:
        code (`str`): C++ code to compile
        clang_exec (`str`): The Clang executable to use
        fn (`str`, **optional**): If provided, the code will be written to a
        file.
    """
    hoomd_include_path = os.path.dirname(hoomd.__file__) + '/include'

    cmd = [
        clang_exec, '-O3', '--std=c++14', '-DHOOMD_LLVMJIT_BUILD', '-I',
        hoomd_include_path, '-I', hoomd.version.source_dir, '-S', '-emit-llvm',
        '-x', 'c++', '-o', '-', '-'
    ]
    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    # pass C++ function to stdin
    output = p.communicate(code.encode('utf-8'))
    llvm_ir = output[0].decode()

    if p.returncode != 0:
        raise RuntimeError(f"Error initializing potential: {output[1]}")

    return llvm_ir


def get_gpu_compilation_settings(gpu):
    """Helper function to set CUDA libraries for GPU execution. """
    includes = [
        "-I" + os.path.dirname(hoomd.__file__) + '/include',
        "-I" + hoomd.version.source_dir,
        "-I" + _jit.__cuda_include_path__
    ]
    cuda_devrt_lib_path = _jit.__cuda_devrt_library_path__

    # select maximum supported compute capability out of those we compile for
    compute_archs = _jit.__cuda_compute_archs__
    compute_capability = gpu._cpp_exec_conf.getComputeCapability(0)  # GPU 0
    compute_major, compute_minor = compute_capability.split('.')
    max_arch = 0
    for a in compute_archs.split('_'):
        if int(a) < int(compute_major) * 10 + int(compute_major):
            max_arch = max(max_arch, int(a))
    return {
        "includes": includes,
        "cuda_devrt_lib_path": cuda_devrt_lib_path,
        "compute_archs": compute_archs,
        "max_arch": max_arch
        }


class CPPPotentialBase(_HOOMDBaseObject):
    """Base class for all HOOMD JIT interaction between pairs of particles.

    Note:
        Users should not invoke `CPPPotentialBase` directly. The class can be
        used for `isinstance` or `issubclass` checks. The attributes documented
        here are available to all patch computes.

        Patch energies define energetic interactions between pairs of shapes in
        :py:mod:`hpmc <hoomd.hpmc>` integrators.  Shapes within a cutoff
        distance of *r_cut* are potentially interacting and the energy of
        interaction is a function the type and orientation of the particles and
        the vector pointing from the *i* particle to the *j* particle center.

        Classes derived from :py:class:`CPPPotentialBase` take C++ code, JIT
        compiles it at run time and executes the code natively in the MC loop
        with full performance. It enables researchers to quickly and easily
        implement custom energetic interactions without the need to modify and
        recompile HOOMD. Additionally, :py:class:`CPPPotentialBase` provides a
        mechanism, through the `alpha_iso` attribute (numpy array), to adjust
        user defined potential parameters without the need to recompile the
        patch energy code. These arrays are **read-only** during function
        evaluation.

        .. rubric:: C++ code

        Classes derived from :py:class:`CPPPotentialBase` will compile the code
        provided by the user and call it to evaluate patch energies. Compilation
        assumes that a recent ``clang`` installation is on your PATH. This is
        convenient when the energy evaluation is simple or needs to be modified
        in python. More complex code (i.e. code that requires auxiliary
        functions or initialization of static data arrays) should be compiled
        outside of HOOMD and provided via the *llvm_ir_file* input (see below).

        The text provided in *code* is the body of a function with the following
        signature:

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

        * ``vec3`` and ``quat`` are defined in HOOMDMath.h.
        * *r_ij* is a vector pointing from the center of particle *i* to the center of particle *j*.
        * *type_i* is the integer type of particle *i*
        * *q_i* is the quaternion orientation of particle *i*
        * *d_i* is the diameter of particle *i*
        * *charge_i* is the charge of particle *i*
        * *type_j* is the integer type of particle *j*
        * *q_j* is the quaternion orientation of particle *j*
        * *d_j* is the diameter of particle *j*
        * *charge_j* is the charge of particle *j*
        * Your code *must* return a value.
        * When \|r_ij\| is greater than *r_cut*, the energy *must* be 0. This
        *r_cut* is applied between the centers of the two particles: compute it
        accordingly based on the maximum range of the anisotropic interaction
        that you implement.

        .. rubric:: LLVM IR code

        You can compile outside of HOOMD and provide a direct link to the LLVM
        IR file in *llvm_ir_file*. A compatible file contains an extern "C" eval
        function with the signature mentioned above.

        Compile the file with clang: ``clang -O3 --std=c++14
        -DHOOMD_LLVMJIT_BUILD -I /path/to/hoomd/include -S -emit-llvm code.cc``
        to produce the LLVM IR in ``code.ll``.

    Atrributes:
        r_cut (`float`): Particle center to center distance cutoff beyond which
        all pair interactions are assumed 0.
        code (`str`): C++ code defining the custom pair interactions between
        particles.
        llvm_ir_fname (`str`): File name of the llvm IR file to load.
        clang_exec (`str`): The Clang executable to compile the provided code.
        array_size (`int`): Size of array with adjustable elements.
        energy (`float`): Total interaction energy of the system in the current state.
        alpha_iso (``ndarray<float>``): Length `array_size` numpy array
        containing dynamically adjustable elements defined by the user.
    """

    def __init__(self,
                 r_cut,
                 array_size=1,
                 clang_exec='clang',
                 code=None,
                 llvm_ir_file=None):
        super().__init__(clang_exec=clang_exec,
                         code=code,
                         llvm_ir_file=llvm_ir_file)
        param_dict = ParameterDict(r_cut=float(r_cut),
                                   array_size=int(array_size))
        self._param_dict.update(param_dict)
        # these only exist on python
        self.alpha_iso = np.zeros(array_size)

    def _attach(self):
        self._simulation.operations.integrator._cpp_obj.setPatchEnergy(
            self._cpp_obj)
        super()._attach()

    @log
    def energy(self):
        """float: Total interaction energy of the system in the current state.
                  Returns `None` when the patch object and integrator are not attached.
        """
        integrator = self._simulation.operations.integrator
        if self._attached and integrator._attached:
            timestep = self._simulation.timestep
            return integrator._cpp_obj.computePatchEnergy(timestep)
        else:
            return None

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
                        float *alpha_iso;
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
                        __device__ float *alpha_iso;
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
    r'''Define an arbitrary patch energetic interaction between pairs of particles.

    Args:
        r_cut (`float`): Particle center to center distance cutoff beyond which all pair interactions are assumed 0.
        code (`str`): C++ code defining the custom pair interactions between particles.
        llvm_ir_fname (`str`): File name of the llvm IR file to load.
        clang_exec (`str` **default:** `clang`): The Clang executable to compile the provided code.
        array_size (`int`, **default:** 1): Size of array with adjustable elements.

    Note:
        If both `code` and `llvm_ir_fname` are provided, the former takes
        precedence. The latter will be used as a fallback in case the
        compilation of `code` fails.

    Examples:

    Static potential parameters

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.CPPPotential(r_cut=1.1, code=square_well)
        sim.operations += patch
        sim.run(1000)

    Dynamic potential parameters

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                         float r_cut = alpha_iso[0];
                            if (rsq < r_cut*r_cut)
                                return alpha_iso[1];
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.CPPPotential(r_cut=1.1, array_size=2, code=square_well)
        patch.alpha_iso[:] = [1.1, 1.5] # [rcut, epsilon]
        sim.operations += patch
        sim.run(1000)
        patch.alpha_iso[1] = 2.0
        sim.run(1000)
    '''

    def __init__(self,
                 r_cut,
                 array_size=1,
                 clang_exec='clang',
                 code=None,
                 llvm_ir_file=None):
        super().__init__(r_cut=r_cut,
                         array_size=array_size,
                         clang_exec=clang_exec,
                         code=code,
                         llvm_ir_file=llvm_ir_file)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        # compile code if provided
        if self._code is not None:
            cpp_function = self._wrap_cpu_code(self._code)
            llvm_ir = to_llvm_ir(cpp_function, self._clang_exec)
        # fall back to LLVM IR file in case code is not provided
        elif self._llvm_ir_file is not None:
            # IR is a text file
            with open(self._llvm_ir_file, 'r') as f:
                llvm_ir = f.read()
        else:
            raise RuntimeError("Must provide code or LLVM IR file.")

        device = self._simulation.device
        cpp_sys_def = self._simulation.state._cpp_sys_def
        if isinstance(device, hoomd.device.GPU):
            gpu_settings = get_gpu_compilation_settings(device)
            gpu_code = self._wrap_gpu_code(self._code)
            self._cpp_obj = _jit.PatchEnergyJITGPU(
                cpp_sys_def, device._cpp_exec_conf, llvm_ir, self.r_cut,
                self.array_size, gpu_code,
                "hpmc::gpu::kernel::hpmc_narrow_phase_patch",
                gpu_settings["includes"],
                gpu_settings["cuda_devrt_lib_path"],
                gpu_settings["max_arch"])
        else:
            self._cpp_obj = _jit.PatchEnergyJIT(
                cpp_sys_def, device._cpp_exec_conf, llvm_ir, self.r_cut,
                self.array_size)
        # Set the C++ mirror array with the cached values
        # and override the python array
        self._cpp_obj.alpha_iso[:] = self.alpha_iso[:]
        self.alpha_iso = self._cpp_obj.alpha_iso
        # attach patch object to the integrator
        super()._attach()


class CPPUnionPotential(CPPPotentialBase):
    r'''Define an arbitrary patch energetic interaction between unions of particles.

    Args:
        r_cut_union (`float`): Constituent particle center to center distance cutoff beyond which all pair interactions are assumed 0.
        r_cut (`float`, **default** 0): Cut-off for isotropic interaction between centers of union particles.
        code_union (`str`): C++ code defining the custom pair interactions between constituent particles.
        code (`str`): C++ code for isotropic part.
        llvm_ir_fname_union (`str`): File name of the llvm IR file to load.
        llvm_ir_fname (`str`): File name of the llvm IR file to load for isotropic interaction.
        clang_exec (`str`, **default:** `clang`): The Clang executable to compile the provided code.
        array_size_union (`int`, **default:** 1): Size of array with adjustable elements.
        array_size (`int`, **default:** 1): Size of array with adjustable elements for the isotropic part.

    Note:
        If both `code_union` and `llvm_ir_fname_union` are provided, the former takes precedence. The latter will be used
        as a fallback in case the compilation of `code_union` fails.

    Note:
        This class uses an internal OBB tree for fast interaction queries bewteeen constituents of interacting particles.
        Depending on the number of constituent particles per type in the tree, different values of the particles per leaf
        node may yield different optimal performance. The capacity of leaf nodes is configurable.

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
                                  float r_cut = alpha_iso[0];
                                  if (rsq < r_cut*r_cut)
                                    return alpha_iso[1];
                                  else
                                    return 0.0f;
                         """

        patch = hoomd.jit.patch.CPPUnionPotential(r_cut_union=2.5, code_union=square_well, array_size_union=2, \
                                               r_cut=5, code=soft_repulsion, array_size=2)
        patch.positions['A'] = [(0,0,-5.),(0,0,.5)]
        patch.typeids['A'] = [0,0]
        patch.diameters['A'] = [0,0]
        # [r_cut, epsilon]
        patch.alpha_iso[:] = [2.5, 1.3];
        patch.alpha_union[:] = [2.5, -1.7];
    '''

    def __init__(self,
                 r_cut_union,
                 array_size_union=1,
                 clang_exec='clang',
                 code_union=None,
                 llvm_ir_file_union=None,
                 r_cut=0,
                 array_size=1,
                 code='return 0;',
                 llvm_ir_file=None):

        # initialize base class
        super().__init__(r_cut=r_cut,
                         array_size=array_size,
                         code=code,
                         llvm_ir_file=llvm_ir_file,
                         clang_exec=clang_exec)

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
        self._llvm_ir_file_union = llvm_ir_file_union
        self.alpha_union = np.zeros(array_size_union)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        # compile code if provided
        if self._code_union is not None:
            cpp_function_union = self._wrap_cpu_code(self._code_union)
            llvm_ir_union = to_llvm_ir(cpp_function_union, self._clang_exec)
        # fall back to LLVM IR file in case code is not provided
        elif self._llvm_ir_file_union is not None:
            # IR is a text file
            with open(self._llvm_ir_file_union, 'r') as f:
                llvm_ir_union = f.read()
        else:
            raise RuntimeError("Must provide code or LLVM IR file.")

        if self._code is not None:
            cpp_function = self._wrap_cpu_code(self._code)
            llvm_ir = to_llvm_ir(cpp_function, self._clang_exec)
        elif self._llvm_ir_file is not None:
            # IR is a text file
            with open(self._llvm_ir_file, 'r') as f:
                llvm_ir = f.read()
        else:
            # provide a dummy function
            cpp_dummy = self._wrap_cpu_code('return 0.0;')
            llvm_ir = to_llvm_ir(cpp_dummy, self._clang_exec)

        device = self._simulation.device
        if isinstance(self._simulation.device, hoomd.device.GPU):
            gpu_settings = get_gpu_compilation_settings(device)
            # use union evaluator
            gpu_code = self._wrap_gpu_code(self._code)
            self._cpp_obj = _jit.PatchEnergyJITUnionGPU(
                self._simulation.state._cpp_sys_def, device._cpp_exec_conf,
                llvm_ir, self.r_cut, self.array_size, llvm_ir_union,
                self.r_cut_union, self.array_size_union, gpu_code,
                "hpmc::gpu::kernel::hpmc_narrow_phase_patch",
                gpu_settings["includes"] + ["-DUNION_EVAL"],
                gpu_settings["cuda_devrt_lib_path"], gpu_settings["max_arch"])
        else:
            self._cpp_obj = _jit.PatchEnergyJITUnion(
                self._simulation.state._cpp_sys_def, device._cpp_exec_conf,
                llvm_ir, self.r_cut, self.array_size, llvm_ir_union,
                self.r_cut_union, self.array_size_union)

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
        """str: The C++ code defining the custom pair interactions between constituent particles."""
        return self._code_union

    @code_union.setter
    def code_union(self, code):
        if self._attached:
            raise AttributeError(
                "This attribute can only be set when the object is not attached."
            )
        else:
            self._code_union = code

    @property
    def llvm_ir_file_union(self):
        """str: File name of the llvm IR file to load."""
        return self._llvm_ir_file_union

    @llvm_ir_file_union.setter
    def llvm_ir_file_union(self, llvm_ir):
        if self._attached:
            raise AttributeError(
                "This attribute can only be set when the object is not attached."
            )
        else:
            self._llvm_ir_file_union = llvm_ir
