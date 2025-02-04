// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ParticleData.cuh"

/*! \file ParticleData.cu
    \brief ImplementsGPU kernel code and data structure functions used by ParticleData
*/

#ifdef ENABLE_MPI

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hipcub/hipcub.hpp>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#pragma GCC diagnostic pop

namespace hoomd
    {
namespace kernel
    {
//! Kernel to partition particle data
__global__ void gpu_scatter_particle_data_kernel(const unsigned int nwork,
                                                 const Scalar4* d_pos,
                                                 const Scalar4* d_vel,
                                                 const Scalar3* d_accel,
                                                 const Scalar* d_charge,
                                                 const Scalar* d_diameter,
                                                 const int3* d_image,
                                                 const unsigned int* d_body,
                                                 const Scalar4* d_orientation,
                                                 const Scalar4* d_angmom,
                                                 const Scalar3* d_inertia,
                                                 const Scalar4* d_net_force,
                                                 const Scalar4* d_net_torque,
                                                 const Scalar* d_net_virial,
                                                 unsigned int net_virial_pitch,
                                                 const unsigned int* d_tag,
                                                 unsigned int* d_rtag,
                                                 Scalar4* d_pos_alt,
                                                 Scalar4* d_vel_alt,
                                                 Scalar3* d_accel_alt,
                                                 Scalar* d_charge_alt,
                                                 Scalar* d_diameter_alt,
                                                 int3* d_image_alt,
                                                 unsigned int* d_body_alt,
                                                 Scalar4* d_orientation_alt,
                                                 Scalar4* d_angmom_alt,
                                                 Scalar3* d_inertia_alt,
                                                 Scalar4* d_net_force_alt,
                                                 Scalar4* d_net_torque_alt,
                                                 Scalar* d_net_virial_alt,
                                                 unsigned int* d_tag_alt,
                                                 detail::pdata_element* d_out,
                                                 unsigned int* d_comm_flags,
                                                 unsigned int* d_comm_flags_out,
                                                 const unsigned int* d_scan,
                                                 const unsigned int offset)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nwork)
        return;
    idx += offset;
    bool remove = d_comm_flags[idx];

    unsigned int scan_remove = d_scan[idx];
    unsigned int scan_keep = idx - scan_remove;

    if (remove)
        {
        detail::pdata_element p;
        p.pos = d_pos[idx];
        p.vel = d_vel[idx];
        p.accel = d_accel[idx];
        p.charge = d_charge[idx];
        p.diameter = d_diameter[idx];
        p.image = d_image[idx];
        p.body = d_body[idx];
        p.orientation = d_orientation[idx];
        p.angmom = d_angmom[idx];
        p.inertia = d_inertia[idx];
        p.net_force = d_net_force[idx];
        p.net_torque = d_net_torque[idx];
        for (unsigned int j = 0; j < 6; ++j)
            p.net_virial[j] = d_net_virial[j * net_virial_pitch + idx];
        p.tag = d_tag[idx];
        d_out[scan_remove] = p;
        d_comm_flags_out[scan_remove] = d_comm_flags[idx];

        // reset communication flags
        d_comm_flags[idx] = 0;

        // reset rtag
        d_rtag[p.tag] = NOT_LOCAL;
        }
    else
        {
        d_pos_alt[scan_keep] = d_pos[idx];
        d_vel_alt[scan_keep] = d_vel[idx];
        d_accel_alt[scan_keep] = d_accel[idx];
        d_charge_alt[scan_keep] = d_charge[idx];
        d_diameter_alt[scan_keep] = d_diameter[idx];
        d_image_alt[scan_keep] = d_image[idx];
        d_body_alt[scan_keep] = d_body[idx];
        d_orientation_alt[scan_keep] = d_orientation[idx];
        d_angmom_alt[scan_keep] = d_angmom[idx];
        d_inertia_alt[scan_keep] = d_inertia[idx];
        d_net_force_alt[scan_keep] = d_net_force[idx];
        d_net_torque_alt[scan_keep] = d_net_torque[idx];
        for (unsigned int j = 0; j < 6; ++j)
            d_net_virial_alt[j * net_virial_pitch + scan_keep]
                = d_net_virial[j * net_virial_pitch + idx];
        unsigned int tag = d_tag[idx];
        d_tag_alt[scan_keep] = tag;

        // update rtag
        d_rtag[tag] = scan_keep;
        }
    }

__global__ void
gpu_select_sent_particles(unsigned int N, unsigned int* d_comm_flags, unsigned int* d_tmp)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;
    d_tmp[idx] = d_comm_flags[idx] ? 1 : 0;
    }

/*! \param N Number of local particles
    \param d_pos Device array of particle positions
    \param d_vel Device array of particle velocities
    \param d_accel Device array of particle accelerations
    \param d_charge Device array of particle charges
    \param d_diameter Device array of particle diameters
    \param d_image Device array of particle images
    \param d_body Device array of particle body tags
    \param d_orientation Device array of particle orientations
    \param d_angmom Device array of particle angular momenta
    \param d_inertia Device array of particle moments of inertia
    \param d_net_force Net force
    \param d_net_torque Net torque
    \param d_net_virial Net virial
    \param net_virial_pitch Pitch of net virial array
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param d_pos_alt Device array of particle positions (output)
    \param d_vel_alt Device array of particle velocities (output)
    \param d_accel_alt Device array of particle accelerations (output)
    \param d_charge_alt Device array of particle charges (output)
    \param d_diameter_alt Device array of particle diameters (output)
    \param d_image_alt Device array of particle images (output)
    \param d_body_alt Device array of particle body tags (output)
    \param d_orientation_alt Device array of particle orientations (output)
    \param d_angmom_alt Device array of particle angular momenta (output)
    \param d_inertia Device array of particle moments of inertia (output)
    \param d_net_force Net force (output)
    \param d_net_torque Net torque (output)
    \param d_net_virial Net virial (output)
    \param d_out Output array for packed particle data
    \param max_n_out Maximum number of elements to write to output array

    \returns Number of elements marked for removal
 */
unsigned int gpu_pdata_remove(const unsigned int N,
                              const Scalar4* d_pos,
                              const Scalar4* d_vel,
                              const Scalar3* d_accel,
                              const Scalar* d_charge,
                              const Scalar* d_diameter,
                              const int3* d_image,
                              const unsigned int* d_body,
                              const Scalar4* d_orientation,
                              const Scalar4* d_angmom,
                              const Scalar3* d_inertia,
                              const Scalar4* d_net_force,
                              const Scalar4* d_net_torque,
                              const Scalar* d_net_virial,
                              unsigned int net_virial_pitch,
                              const unsigned int* d_tag,
                              unsigned int* d_rtag,
                              Scalar4* d_pos_alt,
                              Scalar4* d_vel_alt,
                              Scalar3* d_accel_alt,
                              Scalar* d_charge_alt,
                              Scalar* d_diameter_alt,
                              int3* d_image_alt,
                              unsigned int* d_body_alt,
                              Scalar4* d_orientation_alt,
                              Scalar4* d_angmom_alt,
                              Scalar3* d_inertia_alt,
                              Scalar4* d_net_force_alt,
                              Scalar4* d_net_torque_alt,
                              Scalar* d_net_virial_alt,
                              unsigned int* d_tag_alt,
                              detail::pdata_element* d_out,
                              unsigned int* d_comm_flags,
                              unsigned int* d_comm_flags_out,
                              unsigned int max_n_out,
                              unsigned int* d_tmp,
                              CachedAllocator& alloc,
                              GPUPartition& gpu_partition)
    {
    if (!N)
        return 0;

    assert(d_pos);
    assert(d_vel);
    assert(d_accel);
    assert(d_charge);
    assert(d_diameter);
    assert(d_image);
    assert(d_body);
    assert(d_orientation);
    assert(d_angmom);
    assert(d_inertia);
    assert(d_net_force);
    assert(d_net_torque);
    assert(d_net_virial);
    assert(d_tag);
    assert(d_rtag);
    assert(d_pos_alt);
    assert(d_vel_alt);
    assert(d_accel_alt);
    assert(d_charge_alt);
    assert(d_diameter_alt);
    assert(d_image_alt);
    assert(d_body_alt);
    assert(d_orientation_alt);
    assert(d_angmom_alt);
    assert(d_inertia_alt);
    assert(d_net_force_alt);
    assert(d_net_torque_alt);
    assert(d_net_virial_alt);
    assert(d_tag_alt);
    assert(d_out);
    assert(d_comm_flags);
    assert(d_comm_flags_out);
    assert(d_tmp);

    unsigned int n_out;

    // partition particle data into local and removed particles
    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    // select nonzero communication flags
    hipLaunchKernelGGL(gpu_select_sent_particles,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_comm_flags,
                       d_tmp);

    // perform a scan over the array of ones and zeroes
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    unsigned int* d_scan = alloc.getTemporaryBuffer<unsigned int>(N);
    assert(d_scan);

    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_tmp, d_scan, N);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_tmp, d_scan, N);
    alloc.deallocate((char*)d_temp_storage);

    // determine total number of sent particles
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    unsigned int* d_n_out = (unsigned int*)alloc.getTemporaryBuffer<unsigned int>(1);
    assert(d_n_out);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_tmp, d_n_out, N);
    d_temp_storage = alloc.allocate(temp_storage_bytes);
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_tmp, d_n_out, N);
    alloc.deallocate((char*)d_temp_storage);
    hipMemcpy(&n_out, d_n_out, sizeof(unsigned int), hipMemcpyDeviceToHost);
    alloc.deallocate((char*)d_n_out);

    // Don't write past end of buffer
    if (n_out <= max_n_out)
        {
        // partition particle data into local and removed particles
        for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = gpu_partition.getRangeAndSetGPU(idev);

            unsigned int nwork = range.second - range.first;
            unsigned int offset = range.first;

            unsigned int block_size = 256;
            unsigned int n_blocks = nwork / block_size + 1;

            hipLaunchKernelGGL(gpu_scatter_particle_data_kernel,
                               dim3(n_blocks),
                               dim3(block_size),
                               0,
                               0,
                               nwork,
                               d_pos,
                               d_vel,
                               d_accel,
                               d_charge,
                               d_diameter,
                               d_image,
                               d_body,
                               d_orientation,
                               d_angmom,
                               d_inertia,
                               d_net_force,
                               d_net_torque,
                               d_net_virial,
                               net_virial_pitch,
                               d_tag,
                               d_rtag,
                               d_pos_alt,
                               d_vel_alt,
                               d_accel_alt,
                               d_charge_alt,
                               d_diameter_alt,
                               d_image_alt,
                               d_body_alt,
                               d_orientation_alt,
                               d_angmom_alt,
                               d_inertia_alt,
                               d_net_force_alt,
                               d_net_torque_alt,
                               d_net_virial_alt,
                               d_tag_alt,
                               d_out,
                               d_comm_flags,
                               d_comm_flags_out,
                               d_scan,
                               offset);
            }
        }

    // free temp buf
    alloc.deallocate((char*)d_scan);

    // return elements written to output stream
    return n_out;
    }

__global__ void gpu_pdata_add_particles_kernel(unsigned int old_nparticles,
                                               unsigned int num_add_ptls,
                                               Scalar4* d_pos,
                                               Scalar4* d_vel,
                                               Scalar3* d_accel,
                                               Scalar* d_charge,
                                               Scalar* d_diameter,
                                               int3* d_image,
                                               unsigned int* d_body,
                                               Scalar4* d_orientation,
                                               Scalar4* d_angmom,
                                               Scalar3* d_inertia,
                                               Scalar4* d_net_force,
                                               Scalar4* d_net_torque,
                                               Scalar* d_net_virial,
                                               unsigned int net_virial_pitch,
                                               unsigned int* d_tag,
                                               unsigned int* d_rtag,
                                               const detail::pdata_element* d_in,
                                               unsigned int* d_comm_flags)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_add_ptls)
        return;

    detail::pdata_element p = d_in[idx];

    unsigned int add_idx = old_nparticles + idx;
    d_pos[add_idx] = p.pos;
    d_vel[add_idx] = p.vel;
    d_accel[add_idx] = p.accel;
    d_charge[add_idx] = p.charge;
    d_diameter[add_idx] = p.diameter;
    d_image[add_idx] = p.image;
    d_body[add_idx] = p.body;
    d_orientation[add_idx] = p.orientation;
    d_angmom[add_idx] = p.angmom;
    d_inertia[add_idx] = p.inertia;
    d_net_force[add_idx] = p.net_force;
    d_net_torque[add_idx] = p.net_torque;
    for (unsigned int j = 0; j < 6; ++j)
        d_net_virial[j * net_virial_pitch + add_idx] = p.net_virial[j];
    d_tag[add_idx] = p.tag;
    d_rtag[p.tag] = add_idx;
    d_comm_flags[add_idx] = 0;
    }

/*! \param old_nparticles old local particle count
    \param num_add_ptls Number of particles in input array
    \param d_pos Device array of particle positions
    \param d_vel Device iarray of particle velocities
    \param d_accel Device array of particle accelerations
    \param d_charge Device array of particle charges
    \param d_diameter Device array of particle diameters
    \param d_image Device array of particle images
    \param d_body Device array of particle body tags
    \param d_orientation Device array of particle orientations
    \param d_angmom Device array of particle angular momenta
    \param d_inertia Device array of particle moments of inertia
    \param d_net_force Net force
    \param d_net_torque Net torque
    \param d_net_virial Net virial
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param d_in Device array of packed input particle data
    \param d_comm_flags Device array of communication flags (pdata)
*/
void gpu_pdata_add_particles(const unsigned int old_nparticles,
                             const unsigned int num_add_ptls,
                             Scalar4* d_pos,
                             Scalar4* d_vel,
                             Scalar3* d_accel,
                             Scalar* d_charge,
                             Scalar* d_diameter,
                             int3* d_image,
                             unsigned int* d_body,
                             Scalar4* d_orientation,
                             Scalar4* d_angmom,
                             Scalar3* d_inertia,
                             Scalar4* d_net_force,
                             Scalar4* d_net_torque,
                             Scalar* d_net_virial,
                             unsigned int net_virial_pitch,
                             unsigned int* d_tag,
                             unsigned int* d_rtag,
                             const detail::pdata_element* d_in,
                             unsigned int* d_comm_flags)
    {
    assert(d_pos);
    assert(d_vel);
    assert(d_accel);
    assert(d_charge);
    assert(d_diameter);
    assert(d_image);
    assert(d_body);
    assert(d_orientation);
    assert(d_angmom);
    assert(d_inertia);
    assert(d_net_force);
    assert(d_net_torque);
    assert(d_net_virial);
    assert(d_tag);
    assert(d_rtag);
    assert(d_in);

    unsigned int block_size = 256;
    unsigned int n_blocks = num_add_ptls / block_size + 1;

    hipLaunchKernelGGL(gpu_pdata_add_particles_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       old_nparticles,
                       num_add_ptls,
                       d_pos,
                       d_vel,
                       d_accel,
                       d_charge,
                       d_diameter,
                       d_image,
                       d_body,
                       d_orientation,
                       d_angmom,
                       d_inertia,
                       d_net_force,
                       d_net_torque,
                       d_net_virial,
                       net_virial_pitch,
                       d_tag,
                       d_rtag,
                       d_in,
                       d_comm_flags);
    }

    } // end namespace kernel

    } // end namespace hoomd

#endif // ENABLE_MPI
