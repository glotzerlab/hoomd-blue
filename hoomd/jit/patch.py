# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.jit import _jit
import hoomd

import subprocess
import os

import numpy as np

class user(object):
    R''' Define an arbitrary patch energy.

    Args:
        r_cut (float): Particle center to center distance cutoff beyond which all pair interactions are assumed 0.
        code (str): C++ code to compile
        llvm_ir_fname (str): File name of the llvm IR file to load.
        clang_exec (str): The Clang executable to use
        array_size (int): Size of array with adjustable elements. (added in version 2.8)

    Attributes:
        alpha_iso (numpy.ndarray, float): Length array_size numpy array containing dynamically adjustable elements
                                          defined by the user (added in version 2.8)

    Patch energies define energetic interactions between pairs of shapes in :py:mod:`hpmc <hoomd.hpmc>` integrators.
    Shapes within a cutoff distance of *r_cut* are potentially interacting and the energy of interaction is a function
    the type and orientation of the particles and the vector pointing from the *i* particle to the *j* particle center.

    The :py:class:`user` patch energy takes C++ code, JIT compiles it at run time and executes the code natively
    in the MC loop with full performance. It enables researchers to quickly and easily implement custom energetic
    interactions without the need to modify and recompile HOOMD. Additionally, :py:class:`user` provides a mechanism,
    through the `alpha_iso` attribute (numpy array), to adjust user defined potential parameters without the need
    to recompile the patch energy code. These arrays are **read-only** during function evaluation.

    .. rubric:: C++ code

    Supply C++ code to the *code* argument and :py:class:`user` will compile the code and call it to evaluate
    patch energies. Compilation assumes that a recent ``clang`` installation is on your PATH. This is convenient
    when the energy evaluation is simple or needs to be modified in python. More complex code (i.e. code that
    requires auxiliary functions or initialization of static data arrays) should be compiled outside of HOOMD
    and provided via the *llvm_ir_file* input (see below).

    The text provided in *code* is the body of a function with the following signature:

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
    * When \|r_ij\| is greater than *r_cut*, the energy *must* be 0. This *r_cut* is applied between
      the centers of the two particles: compute it accordingly based on the maximum range of the anisotropic
      interaction that you implement.

    Examples:

    Static potential parameters

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.user(mc=mc, r_cut=1.1, code=square_well)
        hoomd.run(1000)

    Dynamic potential parameters

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                         float r_cut = alpha_iso[0];
                            if (rsq < r_cut*r_cut)
                                return alpha_iso[1];
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.user(mc=mc, r_cut=1.1, array_size=2, code=square_well)
        patch.alpha_iso[:] = [1.1, 1.5] # [rcut, epsilon]
        hoomd.run(1000)
        patch.alpha_iso[1] = 2.0
        hoomd.run(1000)

    .. rubric:: LLVM IR code

    You can compile outside of HOOMD and provide a direct link
    to the LLVM IR file in *llvm_ir_file*. A compatible file contains an extern "C" eval function with this signature:

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

    ``vec3`` and ``quat`` are defined in HOOMDMath.h.

    Compile the file with clang: ``clang -O3 --std=c++11 -DHOOMD_LLVMJIT_BUILD -I /path/to/hoomd/include -S -emit-llvm code.cc`` to produce
    the LLVM IR in ``code.ll``.

    .. versionadded:: 2.3
    '''
    def __init__(self, mc, r_cut, array_size=1, code=None, llvm_ir_file=None, clang_exec=None):

        # check if initialization has occurred
        hoomd.context._verify_init()

        self.compute_name = "patch"

        # Find a clang executable if none is provided (we need the CPU version even when running on GPU)
        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang'

        if code is not None:
            llvm_ir = self.compile_user(array_size, 1, code, clang)
        else:
            # IR is a text file
            with open(llvm_ir_file,'r') as f:
                llvm_ir = f.read()

        if hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            include_path_hoomd = os.path.dirname(hoomd.__file__) + '/include';
            include_path_source = hoomd._hoomd.__hoomd_source_dir__
            include_path_cuda = _jit.__cuda_include_path__
            options = ["-I"+include_path_hoomd, "-I"+include_path_source, "-I"+include_path_cuda]
            cuda_devrt_library_path = _jit.__cuda_devrt_library_path__

            # select maximum supported compute capability out of those we compile for
            compute_archs = _jit.__cuda_compute_archs__;
            compute_archs_vec = _hoomd.std_vector_uint()
            compute_capability = hoomd.context.current.device.cpp_exec_conf.getComputeCapability(0) # GPU 0
            compute_major, compute_minor = compute_capability.split('.')
            max_arch = 0
            for a in compute_archs.split('_'):
                if int(a) < int(compute_major)*10+int(compute_major):
                    max_arch = int(a)

            gpu_code = self.wrap_gpu_code(code)
            self.cpp_evaluator = _jit.PatchEnergyJITGPU(hoomd.context.current.device.cpp_exec_conf, llvm_ir, r_cut, array_size,
                gpu_code, "hpmc::gpu::kernel::hpmc_narrow_phase_patch", options, cuda_devrt_library_path, max_arch);
        else:
            self.cpp_evaluator = _jit.PatchEnergyJIT(hoomd.context.current.device.cpp_exec_conf, llvm_ir, r_cut, array_size);

        mc.set_PatchEnergyEvaluator(self);

        self.mc = mc
        self.enabled = True
        self.log = False
        self.cpp_evaluator.alpha_iso[:] = [0]*array_size
        self.alpha_iso = self.cpp_evaluator.alpha_iso

    def compile_user(self, array_size_iso, array_size_union, code, clang_exec, fn=None):
        R'''Helper function to compile the provided code into an executable

        Args:
            code (str): C++ code to compile
            clang_exec (str): The Clang executable to use
            fn (str): If provided, the code will be written to a file.
            array_size_iso (int): Size of array with adjustable elements for the isotropic part. (added in version 2.8)
            array_size_union (int): Size of array with adjustable elements for unions of shapes. (added in version 2.8)

        .. versionadded:: 2.3
        '''
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

        include_path = os.path.dirname(hoomd.__file__) + '/include';
        include_path_source = hoomd._hoomd.__hoomd_source_dir__;

        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang';

        if fn is not None:
            cmd = [clang, '-O3', '--std=c++11', '-DHOOMD_LLVMJIT_BUILD', '-I', include_path, '-I', include_path_source, '-S', '-emit-llvm','-x','c++', '-o',fn,'-']
        else:
            cmd = [clang, '-O3', '--std=c++11', '-DHOOMD_LLVMJIT_BUILD', '-I', include_path, '-I', include_path_source, '-S', '-emit-llvm','-x','c++', '-o','-','-']
        p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

        # pass C++ function to stdin
        output = p.communicate(cpp_function.encode('utf-8'))
        llvm_ir = output[0].decode()

        if p.returncode != 0:
            hoomd.context.current.device.cpp_msg.error("Error compiling provided code\n");
            hoomd.context.current.device.cpp_msg.error("Command "+' '.join(cmd)+"\n");
            hoomd.context.current.device.cpp_msg.error(output[1].decode()+"\n");
            raise RuntimeError("Error initializing patch energy");

        return llvm_ir

    def wrap_gpu_code(self, code):
        R'''Helper function to compile the provided code into a device function

        Args:
            code (str): C++ code to compile

        .. versionadded:: 3.0
        '''

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

        # Compile on C++ side
        return cpp_function

    R''' Disable the patch energy and optionally enable it only for logging

    Args:
        log (bool): If true, only use patch energy as a log quantity

    '''
    def disable(self,log=None):

        if log:
            # enable only for logging purposes
            self.mc.cpp_integrator.disablePatchEnergyLogOnly(log)
            self.log = True
        else:
            # disable completely
            self.mc.cpp_integrator.setPatchEnergy(None);
            self.log = False

        self.enabled = False

    R''' (Re-)Enable the patch energy

    '''
    def enable(self):
        self.mc.cpp_integrator.setPatchEnergy(self.cpp_evaluator);

class user_union(user):
    R''' Define an arbitrary patch energy on a union of particles

    Args:
        r_cut (float): Constituent particle center to center distance cutoff beyond which all pair interactions are assumed 0.
        r_cut_iso (float, **optional**): Cut-off for isotropic interaction between centers of union particles
        code (str): C++ code to compile
        code_iso (str, **optional**): C++ code for isotropic part
        llvm_ir_fname (str): File name of the llvm IR file to load.
        llvm_ir_fname_iso (str, **optional**): File name of the llvm IR file to load for isotropic interaction
        array_size (int): Size of array with adjustable elements. (added in version 2.8)
        array_size_iso (int): Size of array with adjustable elements for the isotropic part. (added in version 2.8)

    Attributes:
        alpha_union (numpy.ndarray, float): Length array_size numpy array containing dynamically adjustable elements
                                            defined by the user for unions of shapes (added in version 2.8)
        alpha_iso (numpy.ndarray, float): Length array_size_iso numpy array containing dynamically adjustable elements
                                          defined by the user for the isotropic part. (added in version 2.8)

    Example:

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.user_union(r_cut=1.1, code=square_well)
        patch.set_params('A',positions=[(0,0,-5.),(0,0,.5)], typeids=[0,0])

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

        patch = hoomd.jit.patch.user_union(r_cut=2.5, code=square_well, array_size=2, \
                                           r_cut_iso=5, code_iso=soft_repulsion, array_size_iso=2)
        patch.set_params('A',positions=[(0,0,-5.),(0,0,.5)], typeids=[0,0])
        # [r_cut, epsilon]
        patch.alpha_iso[:] = [2.5, 1.3];
        patch.alpha_union[:] = [2.5, -1.7];

    .. versionadded:: 2.3
    '''
    def __init__(self, mc, r_cut, array_size=1, code=None, llvm_ir_file=None, r_cut_iso=None, code_iso=None,
        llvm_ir_file_iso=None, array_size_iso=1, clang_exec=None):

        # check if initialization has occurred
        hoomd.context._verify_init()

        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang'

        if code is not None:
            llvm_ir = self.compile_user(array_size_iso, array_size, code, clang)
        else:
            # IR is a text file
            with open(llvm_ir_file,'r') as f:
                llvm_ir = f.read()

        if code_iso is not None:
            llvm_ir_iso = self.compile_user(array_size_iso, array_size, code_iso, clang)
        else:
            if llvm_ir_file_iso is not None:
                # IR is a text file
                with open(llvm_ir_file_iso,'r') as f:
                    llvm_ir_iso = f.read()
            else:
                # provide a dummy function
                llvm_ir_iso = self.compile_user(array_size_iso, array_size, 'return 0;', clang)

        if r_cut_iso is None:
            r_cut_iso = -1.0

        self.compute_name = "patch_union"

        if hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            include_path_hoomd = os.path.dirname(hoomd.__file__) + '/include';
            include_path_source = hoomd._hoomd.__hoomd_source_dir__
            include_path_cuda = _jit.__cuda_include_path__
            options = ["-I"+include_path_hoomd, "-I"+include_path_source, "-I"+include_path_cuda]

            # use union evaluator
            options += ["-DUNION_EVAL"]

            cuda_devrt_library_path = _jit.__cuda_devrt_library_path__

            # select maximum supported compute capability out of those we compile for
            compute_archs = _jit.__cuda_compute_archs__;
            compute_archs_vec = _hoomd.std_vector_uint()
            compute_capability = hoomd.context.current.device.cpp_exec_conf.getComputeCapability(0) # GPU 0
            compute_major, compute_minor = compute_capability.split('.')
            max_arch = 0
            for a in compute_archs.split('_'):
                if int(a) < int(compute_major)*10+int(compute_major):
                    max_arch = int(a)

            gpu_code = self.wrap_gpu_code(code)
            self.cpp_evaluator = _jit.PatchEnergyJITUnionGPU(hoomd.context.current.system_definition, hoomd.context.current.device.cpp_exec_conf,
                llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir, r_cut,  array_size,
                gpu_code, "hpmc::gpu::kernel::hpmc_narrow_phase_patch", options, cuda_devrt_library_path, max_arch);
        else:
            self.cpp_evaluator = _jit.PatchEnergyJITUnion(hoomd.context.current.system_definition, hoomd.context.current.device.cpp_exec_conf,
                llvm_ir_iso, r_cut_iso, array_size_iso, llvm_ir, r_cut,  array_size);

        mc.set_PatchEnergyEvaluator(self);

        self.mc = mc
        self.enabled = True
        self.log = False
        self.cpp_evaluator.alpha_iso[:] = [0]*array_size_iso
        self.cpp_evaluator.alpha_union[:] = [0]*array_size
        self.alpha_iso = self.cpp_evaluator.alpha_iso[:]
        self.alpha_union = self.cpp_evaluator.alpha_union[:]

    def set_params(self, type, positions, typeids, orientations=None, charges=None, diameters=None, leaf_capacity=4):
        R''' Set the union shape parameters for a given particle type

        Args:
            type (str): The type to set the interactions for
            positions (list): The positions of the constituent particles (list of vectors)
            orientations (list): The orientations of the constituent particles (list of four-vectors)
            diameters (list): The diameters of the constituent particles (list of floats)
            charges (list): The charges of the constituent particles (list of floats)
            leaf_capacity (int): The number of particles in a leaf of the internal tree data structure
        '''

        if orientations is None:
            orientations = [[1,0,0,0]]*len(positions)

        if charges is None:
            charges = [0]*len(positions)

        if diameters is None:
            diameters = [0.0]*len(positions)

        positions = np.array(positions).tolist()
        orientations = np.array(orientations).tolist()
        diameters = np.array(diameters).tolist()
        charges = np.array(charges).tolist()
        typeids = np.array(typeids).tolist()

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_names = [ hoomd.context.current.system_definition.getParticleData().getNameByType(i) for i in range(0,ntypes) ];
        if not type in type_names:
            hoomd.context.current.device.cpp_msg.error("{} is not a valid particle type.\n".format(type));
            raise RuntimeError("Error initializing patch energy.");
        typeid = type_names.index(type)

        self.cpp_evaluator.setParam(typeid, typeids, positions, orientations, diameters, charges, leaf_capacity)
