/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include "gpu_settings.h"
#include "ConstraintSphereGPU.cuh"
#include "EvaluatorConstraint.h"
#include "EvaluatorConstraintSphere.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ConstraintSphereGPU.cu
    \brief Defines GPU kernel code for calculating sphere constraint forces. Used by ConstraintSphereGPU.
*/

//! Kernel for caculating sphere constraint forces on the GPU
/*! \param force_data Data to write the compute forces to
    \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param pdata Particle data arrays to calculate forces on
    \param d_net_force Total unconstrained net force on the particles
    \param P Position of the sphere
    \param r radius of the sphere
    \param deltaT step size from the Integrator
*/
extern "C" __global__
void gpu_compute_constraint_sphere_forces_kernel(gpu_force_data_arrays force_data,
                                                 const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 gpu_pdata_arrays pdata,
                                                 const float4 *d_net_force,
                                                 Scalar3 P,
                                                 Scalar r,
                                                 Scalar deltaT)
    {
    // start by identifying which particle we are to handle
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= group_size)
        return;
    
    unsigned int idx = d_group_members[group_idx];
                
    // read in position, velocity, net force, and mass
    float4 pos = pdata.pos[idx];
    float4 vel = pdata.vel[idx];
    float4 net_force = d_net_force[idx];
    Scalar m = pdata.mass[idx];
    
    // convert to Scalar3's for passing to the evaluators
    Scalar3 X = make_scalar3(pos.x, pos.y, pos.z);
    Scalar3 V = make_scalar3(vel.x, vel.y, vel.z);
    Scalar3 F = make_scalar3(net_force.x, net_force.y, net_force.z);
    
    // evaluate the constraint position
    EvaluatorConstraint constraint(X, V, F, m, deltaT);
    EvaluatorConstraintSphere sphere(P, r);
    Scalar3 C = sphere.evalClosest(constraint.evalU());
    
    // evaluate the constraint force
    Scalar3 FC;
    Scalar virial;
    constraint.evalConstraintForce(FC, virial, C);

    // now that the force calculation is complete, write out the results
    force_data.force[idx] = make_float4(FC.x, FC.y, FC.z, 0.0f);
    force_data.virial[idx] = virial;
    }


/*! \param force_data Data to write the compute forces to
    \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param pdata Particle data arrays to calculate forces on
    \param d_net_force Total unconstrained net force on the particles
    \param P Position of the sphere
    \param r radius of the sphere
    \param deltaT step size from the Integrator
    \param block_size Block size to execute on the GPU
    
    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_constraint_sphere_forces(const gpu_force_data_arrays& force_data,
                                                 const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const gpu_pdata_arrays &pdata,
                                                 const float4 *d_net_force,
                                                 const Scalar3& P,
                                                 Scalar r,
                                                 Scalar deltaT,
                                                 unsigned int block_size)
    {
    assert(d_group_members);
    assert(d_net_force);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    cudaMemset(force_data.force, 0, sizeof(float4)*pdata.local_num);
    cudaMemset(force_data.virial, 0, sizeof(float)*pdata.local_num);
    gpu_compute_constraint_sphere_forces_kernel<<< grid, threads>>>(force_data,
                                                                    d_group_members,
                                                                    group_size,
                                                                    pdata,
                                                                    d_net_force,
                                                                    P,
                                                                    r,
                                                                    deltaT);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

