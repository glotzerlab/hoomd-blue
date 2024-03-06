// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

//! Helper functions used by GPU kernels

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#pragma once
namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
namespace kernel
    {
//! Device function to compute the cell that a particle sits in
/*! \param p particle position
    \param box box dimensions
    \param ghost_width widht of ghost layer
    \param cell_dim dimensions of cell list
    \param ci cell indexer
    \param strict if true, return a sentinel value if particles leave the cell
 */
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
                                                   const BoxDim& box,
                                                   const Scalar3& ghost_width,
                                                   const uint3& cell_dim,
                                                   const Index3D& ci,
                                                   bool strict)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p, ghost_width);
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
    if (!strict
        || (f.x >= Scalar(0.0) && f.x < Scalar(1.0) && f.y >= Scalar(0.0) && f.y < Scalar(1.0)
            && f.z >= Scalar(0.0) && f.z < Scalar(1.0)))
        return ci(ib, jb, kb);
    else
        return 0xffffffff;
    }

    } // namespace kernel
    } // end namespace gpu
    } // namespace hpmc
    } // end namespace hoomd
