// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/GPUPartition.cuh"

namespace hpmc {

namespace gpu {

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size);

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
                int3 *d_image,
                const unsigned int N,
                const BoxDim& box,
                const Scalar3 shift,
                const unsigned int block_size);

//! Kernel to evaluate convergence
void hpmc_check_convergence(
     const unsigned int *d_trial_move_type,
     const unsigned int *d_reject_out_of_cell,
     unsigned int *d_reject_in,
     unsigned int *d_reject_out,
     unsigned int *d_condition,
     const GPUPartition& gpu_partition,
     unsigned int block_size);

} // end namespace gpu

} // end namespace hpmc
