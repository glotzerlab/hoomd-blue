// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

//! Helper functions used by GPU kernels

#pragma once
namespace hpmc {

namespace gpu {

namespace kernel {

//! Device function to compute the cell that a particle sits in
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
                                                   const BoxDim& box,
                                                   const Scalar3& ghost_width,
                                                   const uint3& cell_dim,
                                                   const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    return ci(ib,jb,kb);
    }

} // end namespace hpmc
} // end namespace gpu
} // end namespaec kernel
