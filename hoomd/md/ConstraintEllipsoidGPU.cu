// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "ConstraintEllipsoidGPU.cuh"
#include "EvaluatorConstraint.h"
#include "EvaluatorConstraintEllipsoid.h"

#include <assert.h>

/*! \file ConstraintEllipsoidGPU.cu
    \brief Defines GPU kernel code for calculating ellipsoid constraint forces. Used by ConstraintEllipsoidGPU.
*/

//! Kernel for calculating ellipsoid constraint forces on the GPU
/*! \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param N number of particles in system
    \param d_pos particle positions on device
    \param P Position of the ellipsoid
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param deltaT step size from the Integrator
*/
extern "C" __global__
void gpu_compute_constraint_ellipsoid_constraint_kernel(const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 Scalar4 *d_pos,
                                                 Scalar3 P,
                                                 Scalar rx,
                                                 Scalar ry,
                                                 Scalar rz)
    {
    // start by identifying which particle we are to handle
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= group_size)
        return;

    unsigned int idx = d_group_members[group_idx];

    // read in position, velocity, net force, and mass
    Scalar4 pos = d_pos[idx];

    // convert to Scalar3's for passing to the evaluators
    Scalar3 X = make_scalar3(pos.x, pos.y, pos.z);

    // evaluate the constraint position
    EvaluatorConstraintEllipsoid Ellipsoid(P, rx, ry, rz);
    Scalar3 C = Ellipsoid.evalClosest(X);

    // apply the constraint
    d_pos[idx] = make_scalar4(C.x, C.y, C.z, Scalar(0.0));
    }


/*! \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param N number of particles
    \param d_pos particle positions on the device
    \param P Position of the ellipsoid
    \param rx radius of the ellipsoid in x direction
    \param ry radius of the ellipsoid in y direction
    \param rz radius of the ellipsoid in z direction
    \param deltaT step size from the Integrator
    \param block_size Block size to execute on the GPU

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_constraint_ellipsoid_constraint(const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 Scalar4 *d_pos,
                                                 const Scalar3 P,
                                                 Scalar rx,
                                                 Scalar ry,
                                                 Scalar rz,
                                                 unsigned int block_size)
    {
    assert(d_group_members);

    // setup the grid to run the kernel
    dim3 grid( group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_compute_constraint_ellipsoid_constraint_kernel<<< grid, threads>>>(d_group_members,
                                                                    group_size,
                                                                    N,
                                                                    d_pos,
                                                                    P,
                                                                    rx,
                                                                    ry,
                                                                    rz);

    return cudaSuccess;
    }
