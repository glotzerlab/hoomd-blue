# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.jit import _jit
from hoomd.hpmc import field
from hoomd.hpmc import integrate
import hoomd

import tempfile
import shutil
import subprocess
import os

import numpy as np

class user(field._external):
    R''' Define an external field imposed on all particles in the system.

    Args:
        code (str): C++ code to compile
        llvm_ir_fname (str): File name of the llvm IR file to load.
        clang_exec (str): The Clang executable to use

    Potentials in jit.external behave similarly to external fields assigned via
    hpmc.field.callback. Potentials added using external.user are added to the total
    energy calculation in :py:mod:`hpmc <hoomd.hpmc>` integrators. The
    :py:class:`user` external field takes C++ code, JIT compiles it at run time
    and executes the code natively in the MC loop at with full performance. It
    enables researchers to quickly and easily implement custom energetic
    interactions without the need to modify and recompile HOOMD.

    .. rubric:: C++ code

    Supply C++ code to the *code* argument and :py:class:`user` will compile the code and call it to evaluate
    forces. Compilation assumes that a recent ``clang`` installation is on your PATH. This is convenient
    when the energy evaluation is simple or needs to be modified in python. More complex code (i.e. code that
    requires auxiliary functions or initialization of static data arrays) should be compiled outside of HOOMD
    and provided via the *llvm_ir_file* input (see below).

    The text provided in *code* is the body of a function with the following signature:

    .. code::

        float eval(const BoxDim& box,
        unsigned int type_i,
        const vec3<Scalar>& r_i,
        const quat<Scalar>& q_i
        Scalar diameter,
        Scalar charge
        )

    * ``vec3`` and ``quat`` are is defined in HOOMDMath.h.
    * *box* is the system box.
    * *type_i* is the particle type.
    * *r_i* is the particle position
    * *q_i* the particle orientation.
    * *diameter* the particle diameter.
    * *charge* the particle charge.
    * Your code *must* return a value.

    Once initialized, the following log quantities are provided to analyze.log:

    * **external_field_jit** -- total energy of the field

    Example:

    .. code-block:: python

        gravity = """return r_i.z + box.getL().z/2;"""
        external = hoomd.jit.external.user(mc=mc, code=gravity)

    .. rubric:: LLVM IR code

    You can compile outside of HOOMD and provide a direct link
    to the LLVM IR file in *llvm_ir_file*. A compatible file contains an extern "C" eval function with this signature:

    .. code::

        float eval(const BoxDim& box, unsigned int type_i, const vec3<Scalar>& r_i, const quat<Scalar>& q_i, Scalar diameter, Scalar charge)

    ``vec3`` and ``quat`` is defined in HOOMDMath.h.

    Compile the file with clang: ``clang -O3 --std=c++11 -DHOOMD_LLVMJIT_BUILD -I /path/to/hoomd/include -S -emit-llvm code.cc`` to produce
    the LLVM IR in ``code.ll``.

    .. versionadded:: 2.5
    '''
    def __init__(self, mc, code=None, llvm_ir_file=None, clang_exec=None):
        super(user, self).__init__()

        # raise an error if this run is on the GPU
        cls = None;
        if hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("JIT forces are not supported on the GPU\n");
            raise RuntimeError("Error initializing force energy");
        else:
            if isinstance(mc, integrate.sphere):
                cls = _jit.ExternalFieldJITSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _jit.ExternalFieldJITConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _jit.ExternalFieldJITSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _jit.ExternalFieldJITConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _jit.ExternalFieldJITSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _jit.ExternalFieldJITEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_jit.ExternalFieldJITSpheropolygon;
            elif isinstance(mc, integrate.faceted_ellipsoid):
                cls =_jit.ExternalFieldJITFacetedEllipsoid;
            elif isinstance(mc, integrate.polyhedron):
                cls =_jit.ExternalFieldJITPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_jit.ExternalFieldJITSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls = _jit.ExternalFieldJITSphereUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls = _jit.ExternalFieldJITConvexPolyhedronUnion;
            else:
                hoomd.context.msg.error("jit.field.user: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.position_lattice_field");

        hoomd.util.print_status_line();

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

        self.compute_name = "external_field_jit"
        self.cpp_compute = cls(hoomd.context.current.system_definition,
            hoomd.context.exec_conf, llvm_ir);
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name)

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

        include_path = os.path.dirname(hoomd.__file__) + '/include';
        include_patsource = hoomd._hoomd.__hoomd_source_dir__;

        if clang_exec is not None:
            clang = clang_exec;
        else:
            clang = 'clang';

        if fn is not None:
            cmd = [clang, '-O3', '--std=c++11', '-DHOOMD_LLVMJIT_BUILD', '-I', include_path, '-I', include_patsource, '-S', '-emit-llvm','-x','c++', '-o',fn,'-']
        else:
            cmd = [clang, '-O3', '--std=c++11', '-DHOOMD_LLVMJIT_BUILD', '-I', include_path, '-I', include_patsource, '-S', '-emit-llvm','-x','c++', '-o','-','-']
        p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

        # pass C++ function to stdin
        output = p.communicate(cpp_function.encode('utf-8'))
        llvm_ir = output[0].decode()

        if p.returncode != 0:
            hoomd.context.msg.error("Error compiling provided code\n");
            hoomd.context.msg.error("Command "+' '.join(cmd)+"\n");
            hoomd.context.msg.error(output[1].decode()+"\n");
            raise RuntimeError("Error initializing force.");

        return llvm_ir
