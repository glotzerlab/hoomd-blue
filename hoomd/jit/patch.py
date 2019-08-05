# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.jit import _jit
import hoomd

import tempfile
import shutil
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

    Patch energies define energetic interactions between pairs of shapes in :py:mod:`hpmc <hoomd.hpmc>` integrators.
    Shapes within a cutoff distance of *r_cut* are potentially interacting and the energy of interaction is a function
    the type and orientation of the particles and the vector pointing from the *i* particle to the *j* particle center.

    The :py:class:`user` patch energy takes C++ code, JIT compiles it at run time and executes the code natively
    in the MC loop at with full performance. It enables researchers to quickly and easily implement custom energetic
    interactions without the need to modify and recompile HOOMD.

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

    Example:

    .. code-block:: python

        square_well = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """
        patch = hoomd.jit.patch.user(mc=mc, r_cut=1.1, code=square_well)

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
    def __init__(self, mc, r_cut, code=None, llvm_ir_file=None, clang_exec=None):

        # check if initialization has occurred
        hoomd.context._verify_init()

        # raise an error if this run is on the GPU
        if hoomd.context.current.device.cpp_device.isCUDAEnabled():
            hoomd.context.msg.error("Patch energies are not supported on the GPU\n");
            raise RuntimeError("Error initializing patch energy");

        # Find a clang executable if none is provided
        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang'

        if code is not None:
            llvm_ir = self.compile_user(code, clang)
        else:
            # IR is a text file
            with open(llvm_ir_file,'r') as f:
                llvm_ir = f.read()

        self.compute_name = "patch"
        self.cpp_evaluator = _jit.PatchEnergyJIT(hoomd.context.current.device.cpp_device, llvm_ir, r_cut);
        mc.set_PatchEnergyEvaluator(self);

        self.mc = mc
        self.enabled = True
        self.log = False

    def compile_user(self, code, clang_exec, fn=None):
        R'''Helper function to compile the provided code into an executable

        Args:
            code (str): C++ code to compile
            clang_exec (str): The Clang executable to use
            fn (str): If provided, the code will be written to a file.


        .. versionadded:: 2.3
        '''
        cpp_function = """
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

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
            hoomd.context.msg.error("Error compiling provided code\n");
            hoomd.context.msg.error("Command "+' '.join(cmd)+"\n");
            hoomd.context.msg.error(output[1].decode()+"\n");
            raise RuntimeError("Error initializing patch energy");

        return llvm_ir

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
                            if (rsq < 1.21f)
                                return -1.0f;
                            else
                                return 0.0f;
                      """

        # soft repulsion between centers of unions
        soft_repulsion = """float rsq = dot(r_ij, r_ij);
                            if (rsq < 6.25f)
                                return 1.0f;
                            else
                                return 0.0f;
                      """

        patch = hoomd.jit.patch.user_union(r_cut=1.1, code=square_well, r_cut_iso=5, code_iso=soft_repulsion)
        patch.set_params('A',positions=[(0,0,-5.),(0,0,.5)], typeids=[0,0])

    .. versionadded:: 2.3
    '''
    def __init__(self, mc, r_cut, code=None, llvm_ir_file=None, r_cut_iso=None, code_iso=None,
        llvm_ir_file_iso=None, clang_exec=None):

        # check if initialization has occurred
        hoomd.context._verify_init()

        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang'

        if code is not None:
            llvm_ir = self.compile_user(code,clang)
        else:
            # IR is a text file
            with open(llvm_ir_file,'r') as f:
                llvm_ir = f.read()

        if code_iso is not None:
            llvm_ir_iso = self.compile_user(code_iso,clang)
        else:
            if llvm_ir_file_iso is not None:
                # IR is a text file
                with open(llvm_ir_file_iso,'r') as f:
                    llvm_ir_iso = f.read()
            else:
                # provide a dummy function
                llvm_ir_iso = self.compile_user('return 0;',clang)

        if r_cut_iso is None:
            r_cut_iso = -1.0

        self.compute_name = "patch_union"
        self.cpp_evaluator = _jit.PatchEnergyJITUnion(hoomd.context.current.system_definition, hoomd.context.current.device.cpp_device,
            llvm_ir_iso, r_cut_iso, llvm_ir, r_cut);
        mc.set_PatchEnergyEvaluator(self);

        self.mc = mc
        self.enabled = True
        self.log = False

    R''' Set the union shape parameters for a given particle type

    Args:
        type (string): The type to set the interactions for
        positions: The positions of the constituent particles (list of vectors)
        orientations: The orientations of the constituent particles (list of four-vectors)
        diameters: The diameters of the constituent particles (list of floats)
        charges: The charges of the constituent particles (list of floats)
        leaf_capacity: The number of particles in a leaf of the internal tree data structure
    '''
    def set_params(self, type, positions, typeids, orientations=None, charges=None, diameters=None, leaf_capacity=4):
        if orientations is None:
            orientations = [[1,0,0,0]]*len(positions)

        if charges is None:
            charges = [0]*len(positions)

        if diameters is None:
            diameters = [1.0]*len(positions)

        positions = np.array(positions).tolist()
        orientations = np.array(orientations).tolist()
        diameters = np.array(diameters).tolist()
        charges = np.array(charges).tolist()
        typeids = np.array(typeids).tolist()

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_names = [ hoomd.context.current.system_definition.getParticleData().getNameByType(i) for i in range(0,ntypes) ];
        if not type in type_names:
            hoomd.context.msg.error("{} is not a valid particle type.\n".format(type));
            raise RuntimeError("Error initializing patch energy.");
        typeid = type_names.index(type)

        self.cpp_evaluator.setParam(typeid, typeids, positions, orientations, diameters, charges, leaf_capacity)
