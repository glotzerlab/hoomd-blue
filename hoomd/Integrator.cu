// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "Integrator.cuh"

#include <assert.h>

/*! \file Integrator.cu
    \brief Defines methods and data structures used by the Integrator class on the GPU
*/

//! helper to add a given force/virial pointer pair
template< unsigned int compute_virial >
__device__ void add_force_total(Scalar4& net_force, Scalar *net_virial, Scalar4& net_torque, Scalar4* d_f, Scalar* d_v, const unsigned int virial_pitch, Scalar4* d_t, int idx)
    {
    if (d_f != NULL && d_v != NULL && d_t != NULL)
        {
        Scalar4 f = d_f[idx];
        Scalar4 t = d_t[idx];

        net_force.x += f.x;
        net_force.y += f.y;
        net_force.z += f.z;
        net_force.w += f.w;

        if (compute_virial)
            {
            for (int i=0; i < 6; i++)
                net_virial[i] += d_v[i*virial_pitch+idx];
            }

        net_torque.x += t.x;
        net_torque.y += t.y;
        net_torque.z += t.z;
        net_torque.w += t.w;
        }
    }

//! Kernel for summing forces on the GPU
/*! The specified forces and virials are summed for every particle into \a d_net_force and \a d_net_virial

    \param d_net_force Output device array to hold the computed net force
    \param d_net_virial Output device array to hold the computed net virial
    \param net_virial_pitch The pitch of the 2D net_virial array
    \param d_net_torque Output device array to hold the computed net torque
    \param force_list List of pointers to force data to sum
    \param nwork Number of particles this GPU processes
    \param clear When true, initializes the sums to 0 before adding. When false, reads in the current \a d_net_force
           and \a d_net_virial and adds to that
    \param offset of this GPU in ptls array

    \tparam compute_virial When set to 0, the virial sum is not computed
*/
template< unsigned int compute_virial >
__global__ void gpu_integrator_sum_net_force_kernel(Scalar4 *d_net_force,
                                                    Scalar *d_net_virial,
                                                    const unsigned int net_virial_pitch,
                                                    Scalar4 *d_net_torque,
                                                    const gpu_force_list force_list,
                                                    unsigned int nwork,
                                                    bool clear,
                                                    unsigned int offset)
    {
    // calculate the index we will be handling
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nwork)
        {
        idx += offset;

        // set the initial net_force and net_virial to sum into
        Scalar4 net_force;
        Scalar net_virial[6];
        Scalar4 net_torque;
        if (clear)
            {
            net_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
            if (compute_virial)
                {
                for (int i=0; i<6; i++)
                    net_virial[i] = Scalar(0.0);
                }
            net_torque = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
            }
        else
            {
            // if clear is false, initialize to the current d_net_force and d_net_virial
            net_force = d_net_force[idx];
            if (compute_virial)
                {
                for (int i=0; i<6; i++)
                    net_virial[i] = d_net_virial[i*net_virial_pitch+idx];
                }
            net_torque = d_net_torque[idx];
            }

        // sum up the totals
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f0, force_list.v0, force_list.vpitch0, force_list.t0, idx);
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f1, force_list.v1, force_list.vpitch1, force_list.t1, idx);
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f2, force_list.v2, force_list.vpitch2, force_list.t2, idx);
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f3, force_list.v3, force_list.vpitch3, force_list.t3, idx);
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f4, force_list.v4, force_list.vpitch4, force_list.t4, idx);
        add_force_total<compute_virial>(net_force, net_virial, net_torque, force_list.f5, force_list.v5, force_list.vpitch5, force_list.t5, idx);

        // write out the final result
        d_net_force[idx] = net_force;
        if (compute_virial)
            {
            for (int i=0; i < 6; i++)
                d_net_virial[i*net_virial_pitch+idx] = net_virial[i];
            }
        d_net_torque[idx] = net_torque;
        }
    }

cudaError_t gpu_integrator_sum_net_force(Scalar4 *d_net_force,
                                         Scalar *d_net_virial,
                                         const unsigned int net_virial_pitch,
                                         Scalar4 *d_net_torque,
                                         const gpu_force_list& force_list,
                                         unsigned int nparticles,
                                         bool clear,
                                         bool compute_virial,
                                         const GPUPartition& gpu_partition)
    {
    // sanity check
    assert(d_net_force);
    assert(d_net_virial);
    assert(d_net_torque);

    const int block_size = 256;

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        if (compute_virial)
            {
            gpu_integrator_sum_net_force_kernel<1><<< nwork/block_size+1, block_size >>>(d_net_force,
                                                                                              d_net_virial,
                                                                                              net_virial_pitch,
                                                                                              d_net_torque,
                                                                                              force_list,
                                                                                              nwork,
                                                                                              clear,
                                                                                              range.first);
            }
        else
            {
            gpu_integrator_sum_net_force_kernel<0><<< nwork/block_size+1, block_size >>>(d_net_force,
                                                                                              d_net_virial,
                                                                                              net_virial_pitch,
                                                                                              d_net_torque,
                                                                                              force_list,
                                                                                              nwork,
                                                                                              clear,
                                                                                              range.first);
            }
        }

    return cudaSuccess;
    }
