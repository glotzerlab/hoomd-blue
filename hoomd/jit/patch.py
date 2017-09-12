# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.jit import _jit
import hoomd

class llvm(object):
    R""" Define an arbitrary patch energy.

    Args:
        fname (str): File name of the ll file to load.
        r_cut (float): Particle center to center distance cutoff beyond which all pair interactions are assumed 0.

    Patch energies define enthalpic interactions between pairs of shapes in :py:mod:`hpmc <hoomd.hpmc>` integrators.
    Shapes within a cutoff distance of r_cut are potentially interacting and the energy of interaction is a function
    the type and orientation of the particles and the position of the *j* particle:
    *f(type_i, orientation_i, pos_j, type_j, orientation_j)*. The patch energy is evaluated in a coordinate system where
    particle *i* is at the origin.

    The :py:class:`llvm` patch energy takes C++ code, JIT compiles it at run time and executes the code natively
    in the MC loop at with full performance. It enables researchers to quickly and easily implement custom energetic
    interactions without the need to modify and recompile HOOMD.
    """
    def __init__(self, fname, r_cut):
        hoomd.util.print_status_line();

        # check if initialization has occurred
        if hoomd.context.exec_conf is None:
            hoomd.context.msg.error("Cannot create patch energy before context initialization\n");
            raise RuntimeError('Error creating patch energy');

        # raise an error if this run is on the GPU
        if hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.error("Patch energies are not supported on the GPU\n");
            raise RuntimeError("Error initializing patch energy");

        self.cpp_evaluator = _jit.PatchEnergyJIT(hoomd.context.exec_conf, fname, r_cut);
