# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.jit import _jit
import hoomd

import tempfile
import shutil
import subprocess
import os

class user(object):
    R''' Define an arbitrary patch energy.

    Args:
        r_cut (float): Particle center to center distance cutoff beyond which all pair interactions are assumed 0.
        code (str): C++ code to compile
        llvm_ir_fname (str): File name of the llvm IR file to load.

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
                   unsigned int type_j,
                   const quat<float>& q_j)

    * ``vec3`` and ``quat`` are defined in HOOMDMath.h.
    * *r_ij* is a vector pointing from the center of particle *i* to the center of particle *j*.
    * *type_i* is the integer type of particle *i*
    * *q_i* is the quaternion orientation of particle *i*
    * *type_j* is the integer type of particle *j*
    * *q_j* is the quaternion orientation of particle *j*
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
        patch = hoomd.jit.patch.user(r_cut=1.1, code=square_well)

    .. rubric:: LLVM IR code

    You can compile outside of HOOMD and provide a direct link
    to the LLVM IR file in *llvm_ir_file*. A compatible file contains an extern "C" eval function with this signature:

    .. code::

        float eval(const vec3<float>& r_ij,
                   unsigned int type_i,
                   const quat<float>& q_i,
                   unsigned int type_j,
                   const quat<float>& q_j)

    ``vec3`` and ``quat`` are defined in HOOMDMath.h.

    Compile the file with clang: ``clang -O3 --std=c++11 -DHOOMD_NOPYTHON -I /path/to/hoomd/include -S -emit-llvm code.cc`` to produce
    the LLVM IR in ``code.ll``.
    '''
    def __init__(self, r_cut, code=None, llvm_ir_file=None, clang_exec=None):
        hoomd.util.print_status_line();

        # check if initialization has occurred
        if hoomd.context.exec_conf is None:
            hoomd.context.msg.error("Cannot create patch energy before context initialization\n");
            raise RuntimeError('Error creating patch energy');

        # raise an error if this run is on the GPU
        if hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("Patch energies are not supported on the GPU\n");
            raise RuntimeError("Error initializing patch energy");

        dirpath = None;
        if code is not None:
            # compile on the root rank only
            if hoomd.comm.get_rank() == 0:
                dirpath = tempfile.mkdtemp()
                with open(dirpath + '/code.cc', 'w') as f:
                    f.write("""
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

extern "C"
{
float eval(const vec3<float>& r_ij, unsigned int type_i, const quat<float>& q_i, unsigned int type_j, const quat<float>& q_j)
   {
""");
                    f.write(code)
                    f.write("""
   }
}
""");

                include_path = os.path.dirname(hoomd.__file__) + '/include';
                print(include_path)

                if clang_exec is not None:
                    clang = clang_exec;
                else: clang = 'clang';
                
                ret = subprocess.call([clang, '-O3', '--std=c++11', '-DHOOMD_NOPYTHON', '-I', include_path, '-S', '-emit-llvm', dirpath+'/code.cc', '-o', dirpath+'/code.ll'])
                if ret != 0:
                    hoomd.context.msg.error("Error compiling provided code\n");
                    shutil.rmtree(dirpath)
                    raise RuntimeError("Error initializing patch energy");

                llvm_ir_file = dirpath+'/code.ll';

        # TODO: add MPI support - read code.ll on the root rank and broadcast to all others, modify the C++ code
        # to take LLVM IR in a string rather than a file

        self.cpp_evaluator = _jit.PatchEnergyJIT(hoomd.context.exec_conf, llvm_ir_file, r_cut);

        if dirpath is not None:
            shutil.rmtree(dirpath)
