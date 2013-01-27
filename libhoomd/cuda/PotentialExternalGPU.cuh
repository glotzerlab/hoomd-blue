/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: jglaser

#include "HOOMDMath.h"
#include "ParticleData.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file PotentialExternalGPU.cuh
    \brief Defines templated GPU kernel code for calculating the external forces.
*/

#ifndef __POTENTIAL_EXTERNAL_GPU_CUH__
#define __POTENTIAL_EXTERNAL_GPU_CUH__

//! Wraps arguments to gpu_cpef
struct external_potential_args_t
    {
    //! Construct a external_potential_args_t
    external_potential_args_t(float4 *_d_force,
              float *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const Scalar4 *_d_pos,
              const BoxDim& _box,
              const unsigned int _block_size)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  box(_box),
                  N(_N),
                  d_pos(_d_pos),
                  block_size(_block_size)
        {
        };

    float4 *d_force;                //!< Force to write out
    float *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int N;           //!< Number of particles
    const Scalar4 *d_pos;           //!< Device array of particle positions
    const unsigned int block_size;  //!< Block size to execute
    };

#ifdef NVCC
//! Kernel for calculating external forces
/*! This kernel is called to calculate the external forces on all N particles. Actual evaluation of the potentials and
    forces for each particle is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param params per-type array of parameters for the potential

*/
template< class evaluator >
__global__ void gpu_compute_external_forces_kernel(float4 *d_force,
                                               float *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const BoxDim box,
                                               const typename evaluator::param_type *params)
    {
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    float4 posi = d_pos[idx];

    // initialize the force to 0
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float virial[6];
    for (unsigned int k = 0; k < 6; k++)
        virial[k] = 0.0f;
    float energy = 0.0f;

    unsigned int typei = __float_as_int(posi.w);
    float3 Xi = make_float3(posi.x, posi.y, posi.z);
    Scalar3 L = box.getL();
    evaluator eval(Xi, box, params[typei]);

    eval.evalForceEnergyAndVirial(force, energy, virial);

    // now that the force calculation is complete, write out the result)
    d_force[idx].x = force.x;
    d_force[idx].y = force.y;
    d_force[idx].z = force.z;
    d_force[idx].w = energy;

    for (unsigned int k = 0; k < 6; k++)
        d_virial[k*virial_pitch+idx] = virial[k];
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param external_potential_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential

    This is just a driver function for gpu_compute_external_forces(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_external_forces(const external_potential_args_t& external_potential_args,
                                    const typename evaluator::param_type *d_params)
    {
    // setup the grid to run the kernel
    dim3 grid( external_potential_args.N / external_potential_args.block_size + 1, 1, 1);
    dim3 threads(external_potential_args.block_size, 1, 1);

    // bind the position texture
    gpu_compute_external_forces_kernel<evaluator>
           <<<grid, threads>>>(external_potential_args.d_force, external_potential_args.d_virial, external_potential_args.virial_pitch, external_potential_args.N, external_potential_args.d_pos, external_potential_args.box, d_params);

    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_PAIR_GPU_CUH__

