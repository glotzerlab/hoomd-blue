// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoHMAGPU.cuh"
#include "hoomd/VectorMath.h"

#include <assert.h>

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Shared memory used in reducing the sums
extern __shared__ Scalar3 compute_thermo_hma_sdata[];
//! Shared memory used in final reduction
extern __shared__ Scalar3 compute_thermo_hma_final_sdata[];

/*! \file ComputeThermoGPU.cu
    \brief Defines GPU kernel code for computing thermodynamic properties on the GPU. Used by
   ComputeThermoGPU.
*/

//! Perform partial sums of the thermo properties on the GPU
/*! \param d_scratch Scratch space to hold partial sums. One element is written per block
    \param box Box the particles are in
    \param d_net_force Net force / pe array from ParticleData
    \param d_net_virial Net virial array from ParticleData
    \param virial_pitch pitch of 2D virial array
    \param d_position Particle position array from ParticleData
    \param d_lattice_site Particle lattice site array
    \param d_image Image array from ParticleData
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members for which to sum properties
    \param work_size Number of particles in the group this GPU processes
    \param offset Offset of this GPU in list of group members
    \param block_offset Offset of this GPU in the array of partial sums

    All partial sums are packaged up in a Scalar3 to keep pointer management down.
     - force * dr is summed in .x
     - Potential energy is summed in .y
     - W is summed in .z

    One thread is executed per group member. That thread reads in the values for its member into
   shared memory and then the block performs a reduction in parallel to produce a partial sum output
   for the block. These partial sums are written to d_scratch[blockIdx.x].
   sizeof(Scalar3)*block_size of dynamic shared memory are needed for this kernel to run.
*/

__global__ void gpu_compute_thermo_hma_partial_sums(Scalar3* d_scratch,
                                                    BoxDim box,
                                                    Scalar4* d_net_force,
                                                    Scalar* d_net_virial,
                                                    const size_t virial_pitch,
                                                    Scalar4* d_position,
                                                    Scalar3* d_lattice_site,
                                                    int3* d_image,
                                                    unsigned int* d_body,
                                                    unsigned int* d_tag,
                                                    unsigned int* d_group_members,
                                                    unsigned int work_size,
                                                    unsigned int offset,
                                                    unsigned int block_offset)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar3 my_element; // element of scratch space read in

    // non-participating thread: contribute 0 to the sum
    my_element = make_scalar3(0, 0, 0);

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx + offset];

        // ignore rigid body constituent particles in the sum
        unsigned int body = d_body[idx];
        unsigned int tag = d_tag[idx];
        if (body >= MIN_FLOPPY || body == tag)
            {
            Scalar4 net_force = d_net_force[idx];
            Scalar net_isotropic_virial;
            // (1/3)*trace of virial tensor
            net_isotropic_virial = Scalar(1.0 / 3.0)
                                   * (d_net_virial[0 * virial_pitch + idx]     // xx
                                      + d_net_virial[3 * virial_pitch + idx]   // yy
                                      + d_net_virial[5 * virial_pitch + idx]); // zz
            Scalar4 pos4 = d_position[idx];
            Scalar3 pos3 = make_scalar3(pos4.x, pos4.y, pos4.z);
            Scalar3 lat = d_lattice_site[tag];
            Scalar3 dr = box.shift(pos3, d_image[idx]) - lat;
            double fdr = 0;
            fdr += (double)d_net_force[idx].x * dr.x;
            fdr += (double)d_net_force[idx].y * dr.y;
            fdr += (double)d_net_force[idx].z * dr.z;

            // compute our contribution to the sum
            my_element.x = Scalar(fdr);
            my_element.y = net_force.w;
            my_element.z = net_isotropic_virial;
            }
        }

    compute_thermo_hma_sdata[threadIdx.x] = my_element;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            compute_thermo_hma_sdata[threadIdx.x].x
                += compute_thermo_hma_sdata[threadIdx.x + offs].x;
            compute_thermo_hma_sdata[threadIdx.x].y
                += compute_thermo_hma_sdata[threadIdx.x + offs].y;
            compute_thermo_hma_sdata[threadIdx.x].z
                += compute_thermo_hma_sdata[threadIdx.x + offs].z;
            }
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        Scalar3 res = compute_thermo_hma_sdata[0];
        d_scratch[block_offset + blockIdx.x] = make_scalar3(res.x, res.y, res.z);
        }
    }

//! Complete partial sums and compute final thermodynamic quantities (for pressure, only isotropic
//! contribution)
/*! \param d_properties Property array to write final values
    \param d_scratch Partial sums
    \param box Box the particles are in
    \param D Dimensionality of the system
    \param group_size Number of particles in the group
    \param num_partial_sums Number of partial sums in \a d_scratch
    \param temperature The temperature that governs sampling of the integrator
    \param harmonicPressure The contribution to the pressure from harmonic fluctuations
    \param external_virial External contribution to virial (1/3 trace)
    \param external_energy External contribution to potential energy


    Only one block is executed. In that block, the partial sums are read in and reduced to final
   values. From the final sums, the thermodynamic properties are computed and written to
   d_properties.

    sizeof(Scalar3)*block_size bytes of shared memory are needed for this kernel to run.
*/
__global__ void gpu_compute_thermo_hma_final_sums(Scalar* d_properties,
                                                  Scalar3* d_scratch,
                                                  BoxDim box,
                                                  unsigned int D,
                                                  unsigned int group_size,
                                                  unsigned int num_partial_sums,
                                                  Scalar temperature,
                                                  Scalar harmonicPressure,
                                                  Scalar external_virial,
                                                  Scalar external_energy)
    {
    Scalar3 final_sum = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            Scalar3 scratch = d_scratch[start + threadIdx.x];

            compute_thermo_hma_final_sdata[threadIdx.x]
                = make_scalar3(scratch.x, scratch.y, scratch.z);
            }
        else
            compute_thermo_hma_final_sdata[threadIdx.x]
                = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                compute_thermo_hma_final_sdata[threadIdx.x].x
                    += compute_thermo_hma_final_sdata[threadIdx.x + offs].x;
                compute_thermo_hma_final_sdata[threadIdx.x].y
                    += compute_thermo_hma_final_sdata[threadIdx.x + offs].y;
                compute_thermo_hma_final_sdata[threadIdx.x].z
                    += compute_thermo_hma_final_sdata[threadIdx.x + offs].z;
                }
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            final_sum.x += compute_thermo_hma_final_sdata[0].x;
            final_sum.y += compute_thermo_hma_final_sdata[0].y;
            final_sum.z += compute_thermo_hma_final_sdata[0].z;
            }
        }

    if (threadIdx.x == 0)
        {
        // compute final quantities
        Scalar fdr = final_sum.x;
        Scalar pe_total = final_sum.y + external_energy;
        Scalar W = final_sum.z + external_virial;

        // compute the pressure
        // volume/area & other 2D stuff needed

        Scalar volume;
        Scalar3 L = box.getL();

        if (D == 2)
            {
            // "volume" is area in 2D
            volume = L.x * L.y;
            // W needs to be corrected since the 1/3 factor is built in
            W *= Scalar(3.0) / Scalar(2.0);
            }
        else
            {
            volume = L.x * L.y * L.z;
            }

        // pressure: P = (N * K_B * T + W)/V
        Scalar fV = (harmonicPressure / temperature - group_size / volume) / (D * (group_size - 1));
        Scalar pressure = harmonicPressure + W / volume + fV * fdr;

        // fill out the GPUArray
        d_properties[thermoHMA_index::potential_energyHMA]
            = pe_total + 1.5 * (group_size - 1) * temperature + 0.5 * fdr;
        d_properties[thermoHMA_index::pressureHMA] = pressure;
        }
    }

//! Compute partial sums of thermodynamic properties of a group on the GPU,
/*! \param d_pos Particle position array from ParticleData
    \param d_lattice_site Particle lattice site array
    \param d_image Image array from ParticleData
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members
    \param group_size Number of group members
    \param box Box the particles are in
    \param args Additional arguments
    \param gpu_partition Load balancing info for multi-GPU reduction

    This function drives gpu_compute_thermo_partial_sums and gpu_compute_thermo_final_sums, see them
   for details.
*/

hipError_t gpu_compute_thermo_hma_partial(Scalar4* d_pos,
                                          Scalar3* d_lattice_site,
                                          int3* d_image,
                                          unsigned int* d_body,
                                          unsigned int* d_tag,
                                          unsigned int* d_group_members,
                                          unsigned int group_size,
                                          const BoxDim& box,
                                          const compute_thermo_hma_args& args,
                                          const GPUPartition& gpu_partition)
    {
    assert(d_pos);
    assert(d_group_members);
    assert(args.d_net_force);
    assert(args.d_net_virial);
    assert(args.d_scratch);

    unsigned int block_offset = 0;

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        dim3 grid(nwork / args.block_size + 1, 1, 1);
        dim3 threads(args.block_size, 1, 1);

        const size_t shared_bytes = sizeof(Scalar3) * args.block_size;

        gpu_compute_thermo_hma_partial_sums<<<grid, threads, shared_bytes>>>(args.d_scratch,
                                                                             box,
                                                                             args.d_net_force,
                                                                             args.d_net_virial,
                                                                             args.virial_pitch,
                                                                             d_pos,
                                                                             d_lattice_site,
                                                                             d_image,
                                                                             d_body,
                                                                             d_tag,
                                                                             d_group_members,
                                                                             nwork,
                                                                             range.first,
                                                                             block_offset);

        block_offset += grid.x;
        }

    assert(block_offset <= args.n_blocks);

    return hipSuccess;
    }

//! Compute thermodynamic properties of a group on the GPU
/*! \param d_properties Array to write computed properties
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members
    \param group_size Number of group members
    \param box Box the particles are in
    \param args Additional arguments
    \param num_blocks Number of partial sums to reduce

    This function drives gpu_compute_thermo_partial_sums and gpu_compute_thermo_final_sums, see them
   for details.
*/

hipError_t gpu_compute_thermo_hma_final(Scalar* d_properties,
                                        unsigned int* d_body,
                                        unsigned int* d_tag,
                                        unsigned int* d_group_members,
                                        unsigned int group_size,
                                        const BoxDim& box,
                                        const compute_thermo_hma_args& args)
    {
    assert(d_properties);
    assert(d_group_members);
    assert(args.d_net_force);
    assert(args.d_net_virial);
    assert(args.d_scratch);

    // setup the grid to run the final kernel
    int final_block_size = 512;
    dim3 grid = dim3(1, 1, 1);
    dim3 threads = dim3(final_block_size, 1, 1);

    const size_t shared_bytes = sizeof(Scalar3) * final_block_size;

    Scalar external_virial
        = Scalar(1.0 / 3.0)
          * (args.external_virial_xx + args.external_virial_yy + args.external_virial_zz);

    // run the kernel
    gpu_compute_thermo_hma_final_sums<<<grid, threads, shared_bytes>>>(d_properties,
                                                                       args.d_scratch,
                                                                       box,
                                                                       args.D,
                                                                       group_size,
                                                                       args.n_blocks,
                                                                       args.temperature,
                                                                       args.harmonicPressure,
                                                                       external_virial,
                                                                       args.external_energy);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
