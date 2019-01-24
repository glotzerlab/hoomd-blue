// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#include <assert.h>

/*! \file PotentialExternalGPU.cuh
    \brief Defines templated GPU kernel code for calculating the external forces.
*/

#ifndef __POTENTIAL_EXTERNAL_GPU_CUH__
#define __POTENTIAL_EXTERNAL_GPU_CUH__

//! Wraps arguments to gpu_cpef
struct external_potential_args_t
    {
    //! Construct a external_potential_args_t
    external_potential_args_t(Scalar4 *_d_force,
              Scalar *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const Scalar4 *_d_pos,
              const Scalar *_d_diameter,
              const Scalar *_d_charge,
              const BoxDim& _box,
              const unsigned int _block_size)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  box(_box),
                  N(_N),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  block_size(_block_size)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    Scalar *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int N;           //!< Number of particles
    const Scalar4 *d_pos;           //!< Device array of particle positions
    const Scalar *d_diameter;       //!< particle diameters
    const Scalar *d_charge;         //!< particle charges
    const unsigned int block_size;  //!< Block size to execute
    };

//! Driver function for compute external field kernel
/*!
 * \param external_potential_args External potential parameters
 * \param d_params External evaluator parameters
 * \param d_field External field parameters
 * \tparam Evaluator functor
 */
template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field);

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
__global__ void gpu_compute_external_forces_kernel(Scalar4 *d_force,
                                               Scalar *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const Scalar *d_diameter,
                                               const Scalar *d_charge,
                                               const BoxDim box,
                                               const typename evaluator::param_type *params,
                                               const typename evaluator::field_type *d_field)
    {
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // read in field data cooperatively
    extern __shared__ char s_data[];
    typename evaluator::field_type *s_field = (typename evaluator::field_type *)(&s_data[0]);

        {
        unsigned int tidx = threadIdx.x;
        unsigned int block_size = blockDim.x;
        unsigned int field_size = sizeof(typename evaluator::field_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < field_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < field_size)
                {
                ((int *)s_field)[cur_offset + tidx] = ((int *)d_field)[cur_offset + tidx];
                }
            }
        }
    const typename evaluator::field_type& field = *s_field;

    __syncthreads();

    if (idx >= N)
        return;

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    Scalar4 posi = d_pos[idx];
    Scalar di;
    Scalar qi;
    if (evaluator::needsDiameter())
        di = d_diameter[idx];
    else
        di += Scalar(1.0); // shut up compiler warning

    if (evaluator::needsCharge())
        qi = d_charge[idx];
    else
        qi = Scalar(0.0); // shut up compiler warning


    // initialize the force to 0
    Scalar3 force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (unsigned int k = 0; k < 6; k++)
        virial[k] = Scalar(0.0);
    Scalar energy = Scalar(0.0);

    unsigned int typei = __scalar_as_int(posi.w);
    Scalar3 Xi = make_scalar3(posi.x, posi.y, posi.z);
    evaluator eval(Xi, box, params[typei], field);

    if (evaluator::needsDiameter())
        eval.setDiameter(di);
    if (evaluator::needsCharge())
        eval.setCharge(qi);

    eval.evalForceEnergyAndVirial(force, energy, virial);

    // now that the force calculation is complete, write out the result)
    d_force[idx].x = force.x;
    d_force[idx].y = force.y;
    d_force[idx].z = force.z;
    d_force[idx].w = energy;

    for (unsigned int k = 0; k < 6; k++)
        d_virial[k*virial_pitch+idx] = virial[k];
    }

/*!
 * This implements the templated kernel driver. The template must be explicitly
 * instantiated per potential in a cu file.
 */
template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field)
    {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_external_forces_kernel<evaluator>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(external_potential_args.block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( external_potential_args.N / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);
        unsigned int bytes = (sizeof(typename evaluator::field_type)/sizeof(int)+1)*sizeof(int);

        // run the kernel
        gpu_compute_external_forces_kernel<evaluator><<<grid, threads, bytes>>>(external_potential_args.d_force,
                                                                                external_potential_args.d_virial,
                                                                                external_potential_args.virial_pitch,
                                                                                external_potential_args.N,
                                                                                external_potential_args.d_pos,
                                                                                external_potential_args.d_diameter,
                                                                                external_potential_args.d_charge,
                                                                                external_potential_args.box,
                                                                                d_params,
                                                                                d_field);

        return cudaSuccess;
    };
#endif // NVCC
#endif // __POTENTIAL_PAIR_GPU_CUH__
