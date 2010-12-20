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

#include "ComputeThermoGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! Shared memory used in reducing the sums
extern __shared__ float3 compute_thermo_sdata[];

/*! \file ComputeThermoGPU.cu
    \brief Defines GPU kernel code for computing thermodynamic properties on the GPU. Used by ComputeThermoGPU.
*/

//! Perform partial sums of the thermo properties on the GPU
/*! \param d_scratch Scratch space to hold partial sums. One element is written per block
    \param d_net_force Net force / pe array from ParticleData
    \param d_net_virial Net virial array from ParticleData
    \param d_mass Particle mass array from ParticleData
    \param d_velocity Particle velocity array from ParticleData
    \param d_group_members List of group members for which to sum properties
    \param group_size Number of particles in the group
    
    All partial sums are packaged up in a float4 to keep pointer management down.
     - 2*Kinetic energy is summed in .x
     - Potential energy is summed in .y
     - W is summed in .z
    
    One thread is executed per group member. That thread reads in the values for its member into shared memory
    and then the block performs a reduction in parallel to produce a partial sum output for the block. These
    partial sums are written to d_scratch[blockIdx.x]. sizeof(float3)*block_size of dynamic shared memory are needed
    for this kernel to run.
*/

__global__ void gpu_compute_thermo_partial_sums(float4 *d_scratch,
                                                float4 *d_net_force,
                                                float *d_net_virial,
                                                float *d_mass,
                                                float4 *d_velocity,
                                                unsigned int *d_group_members,
                                                unsigned int group_size)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float3 my_element; // element of scratch space read in
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
   
        // update positions to the next timestep and update velocities to the next half step
        float4 net_force = d_net_force[idx];
        float net_virial = d_net_virial[idx];
        float4 vel = d_velocity[idx];
        float mass = d_mass[idx];
        
        // compute our contribution to the sum
        my_element.x = mass * (vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        my_element.y = net_force.w;
        my_element.z = net_virial;
        }
    else
        {
        // non-participating thread: contribute 0 to the sum
        my_element = make_float3(0.0f, 0.0f, 0.0f);
        }
        
    compute_thermo_sdata[threadIdx.x] = my_element;
    __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            compute_thermo_sdata[threadIdx.x].x += compute_thermo_sdata[threadIdx.x + offs].x;
            compute_thermo_sdata[threadIdx.x].y += compute_thermo_sdata[threadIdx.x + offs].y;
            compute_thermo_sdata[threadIdx.x].z += compute_thermo_sdata[threadIdx.x + offs].z;
            }
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        float3 res = compute_thermo_sdata[0];
        d_scratch[blockIdx.x] = make_float4(res.x, res.y, res.z, 0.0f);
        }
    }

//! Complete partial sums and compute final thermodynamic quantities
/*! \param d_properties Property array to write final values
    \param d_scratch Partial sums
    \param ndof Number of degrees of freedom this group posesses
    \param box Box the particles are in
    \param D Dimensionality of the system
    \param group_size Number of particles in the group
    \param num_partial_sums Number of partial sums in \a d_scratch
    
    Only one block is executed. In that block, the partial sums are read in and reduced to final values. From the final
    sums, the thermodynamic properties are computed and written to d_properties.
    
    sizeof(float3)*block_size bytes of shared memory are needed for this kernel to run.
*/
__global__ void gpu_compute_thermo_final_sums(float *d_properties,
                                              float4 *d_scratch,
                                              unsigned int ndof,
                                              gpu_boxsize box,
                                              unsigned int D,
                                              unsigned int group_size,
                                              unsigned int num_partial_sums)
    {
    float3 final_sum = make_float3(0.0f, 0.0f, 0.0f);
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            float4 scratch = d_scratch[start + threadIdx.x];
            compute_thermo_sdata[threadIdx.x] = make_float3(scratch.x, scratch.y, scratch.z);
            }
        else
            compute_thermo_sdata[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                compute_thermo_sdata[threadIdx.x].x += compute_thermo_sdata[threadIdx.x + offs].x;
                compute_thermo_sdata[threadIdx.x].y += compute_thermo_sdata[threadIdx.x + offs].y;
                compute_thermo_sdata[threadIdx.x].z += compute_thermo_sdata[threadIdx.x + offs].z;
                }
            offs >>= 1;
            __syncthreads();
            }
        
        if (threadIdx.x == 0)
            {
            final_sum.x += compute_thermo_sdata[0].x;
            final_sum.y += compute_thermo_sdata[0].y;
            final_sum.z += compute_thermo_sdata[0].z;
            }
        }
        
    if (threadIdx.x == 0)
        {
        // compute final quantities
        float ke_total = final_sum.x * 0.5f;
        float pe_total = final_sum.y;
        float W = final_sum.z;
        
        // compute the temperature
        Scalar temperature = Scalar(2.0) * Scalar(ke_total) / Scalar(ndof);

        // compute the pressure
        // volume/area & other 2D stuff needed

        Scalar volume;

        if (D == 2)
            {
            // "volume" is area in 2D
            volume = box.Lx * box.Ly;
            // W needs to be corrected since the 1/3 factor is built in
            W *= Scalar(3.0)/Scalar(2.0);
            }
        else
            {
            volume = box.Lx * box.Ly * box.Lz;
            }

        // pressure: P = (N * K_B * T + W)/V
        Scalar pressure =  (Scalar(2.0) * ke_total / Scalar(D) + W) / volume;

        // fill out the GPUArray
        d_properties[thermo_index::temperature] = temperature;
        d_properties[thermo_index::pressure] = pressure;
        d_properties[thermo_index::kinetic_energy] = Scalar(ke_total);
        d_properties[thermo_index::potential_energy] = Scalar(pe_total);
        }
    }


//! Compute thermodynamic properties of a group on the GPU
/*! \param d_properties Array to write computed properties
    \param pdata Particle data
    \param d_group_members List of group members
    \param group_size Number of group members
    \param box Box the particles are in
    \param args Additional arguments
    
    This function drives gpu_compute_thermo_partial_sums and gpu_compute_thermo_final_sums, see them for details.
*/

cudaError_t gpu_compute_thermo(float *d_properties,
                               const gpu_pdata_arrays &pdata,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const gpu_boxsize &box,
                               const compute_thermo_args& args
                               )
    {
    assert(d_properties);
    assert(pdata.mass);
    assert(pdata.vel);
    assert(d_group_members);
    assert(args.d_net_force);
    assert(args.d_net_virial);
    assert(args.d_scratch);
    
    dim3 grid(args.n_blocks, 1, 1);
    dim3 threads(args.block_size, 1, 1);
    unsigned int shared_bytes = sizeof(float3)*args.block_size;
    
    gpu_compute_thermo_partial_sums<<<grid,threads, shared_bytes>>>(args.d_scratch,
                                                                    args.d_net_force,
                                                                    args.d_net_virial,
                                                                    pdata.mass,
                                                                    pdata.vel,
                                                                    d_group_members,
                                                                    group_size);

        
    // setup the grid to run the final kernel
    int final_block_size = 512;
    grid = dim3(1, 1, 1);
    threads = dim3(final_block_size, 1, 1);
    shared_bytes = sizeof(float3)*final_block_size;
    
    // run the kernel
    gpu_compute_thermo_final_sums<<<grid, threads, shared_bytes>>>(d_properties,
                                                                   args.d_scratch,
                                                                   args.ndof,
                                                                   box,
                                                                   args.D,
                                                                   group_size,
                                                                   args.n_blocks);
    
    return cudaSuccess;
    }

