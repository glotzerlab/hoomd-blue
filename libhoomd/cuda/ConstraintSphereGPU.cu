/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

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
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param N number of particles in system
    \param d_pos particle positions on device
    \param d_vel particle velocities and masses on device
    \param d_net_force Total unconstrained net force on the particles
    \param P Position of the sphere
    \param r radius of the sphere
    \param deltaT step size from the Integrator
*/
extern "C" __global__
void gpu_compute_constraint_sphere_forces_kernel(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const Scalar4 *d_vel,
                                                 const Scalar4 *d_net_force,
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
    Scalar4 pos = d_pos[idx];
    Scalar4 vel = d_vel[idx];
    Scalar4 net_force = d_net_force[idx];
    Scalar m = vel.w;

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
    Scalar virial[6];
    constraint.evalConstraintForce(FC, virial, C);

    // now that the force calculation is complete, write out the results
    d_force[idx] = make_scalar4(FC.x, FC.y, FC.z, Scalar(0.0));
    for (unsigned int i = 0; i < 6; i++)
        d_virial[i*virial_pitch+idx] = virial[i];
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param d_group_members List of members in the group
    \param group_size number of members in the group
    \param N nunmber of particles
    \param d_pos particle positions on the device
    \param d_vel particle velocities on the device
    \param d_net_force Total unconstrained net force on the particles
    \param P Position of the sphere
    \param r radius of the sphere
    \param deltaT step size from the Integrator
    \param block_size Block size to execute on the GPU

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_constraint_sphere_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const Scalar4 *d_vel,
                                                 const Scalar4 *d_net_force,
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
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    cudaMemset(d_virial, 0, 6*sizeof(Scalar)*virial_pitch);
    gpu_compute_constraint_sphere_forces_kernel<<< grid, threads>>>(d_force,
                                                                    d_virial,
                                                                    virial_pitch,
                                                                    d_group_members,
                                                                    group_size,
                                                                    N,
                                                                    d_pos,
                                                                    d_vel,
                                                                    d_net_force,
                                                                    P,
                                                                    r,
                                                                    deltaT);

    return cudaSuccess;
    }
