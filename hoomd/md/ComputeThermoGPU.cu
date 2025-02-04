// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermoGPU.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <hip/hip_runtime.h>

#include <assert.h>

/*! \file ComputeThermoGPU.cu
    \brief Defines GPU kernel code for computing thermodynamic properties on the GPU. Used by
   ComputeThermoGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Perform partial sums of the thermo properties on the GPU
/*! \param d_scratch Scratch space to hold partial sums. One element is written per block
    \param d_net_force Net force / pe array from ParticleData
    \param d_net_virial Net virial array from ParticleData
    \param virial_pitch pitch of 2D virial array
    \param d_velocity Particle velocity and mass array from ParticleData
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members for which to sum properties
    \param work_size Number of particles in the group this GPU processes
    \param offset Offset of this GPU in list of group members
    \param block_offset Offset of this GPU in the array of partial sums

    All partial sums are packaged up in a Scalar4 to keep pointer management down.
     - 2*Kinetic energy is summed in .x
     - Potential energy is summed in .y
     - W is summed in .z

    One thread is executed per group member. That thread reads in the values for its member into
   shared memory and then the block performs a reduction in parallel to produce a partial sum output
   for the block. These partial sums are written to d_scratch[blockIdx.x].
   sizeof(Scalar3)*block_size of dynamic shared memory are needed for this kernel to run.
*/

__global__ void gpu_compute_thermo_partial_sums(Scalar4* d_scratch,
                                                Scalar4* d_net_force,
                                                Scalar* d_net_virial,
                                                const size_t virial_pitch,
                                                Scalar4* d_velocity,
                                                unsigned int* d_body,
                                                unsigned int* d_tag,
                                                unsigned int* d_group_members,
                                                unsigned int work_size,
                                                unsigned int offset,
                                                unsigned int block_offset)
    {
    extern __shared__ Scalar3 compute_thermo_sdata[];

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
            // update positions to the next timestep and update velocities to the next half step
            Scalar4 net_force = d_net_force[idx];
            Scalar net_isotropic_virial;
            // (1/3)*trace of virial tensor
            net_isotropic_virial = Scalar(1.0 / 3.0)
                                   * (d_net_virial[0 * virial_pitch + idx]     // xx
                                      + d_net_virial[3 * virial_pitch + idx]   // yy
                                      + d_net_virial[5 * virial_pitch + idx]); // zz
            Scalar4 vel = d_velocity[idx];
            Scalar mass = vel.w;

            // compute our contribution to the sum
            my_element.x = mass * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            my_element.y = net_force.w;
            my_element.z = net_isotropic_virial;
            }
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
        Scalar3 res = compute_thermo_sdata[0];
        d_scratch[block_offset + blockIdx.x] = make_scalar4(res.x, res.y, res.z, 0);
        }
    }

//! Perform partial sums of the pressure tensor on the GPU
/*! \param d_scratch Scratch space to hold partial sums. One element is written per block
    \param d_net_force Net force / pe array from ParticleData
    \param d_net_virial Net virial array from ParticleData
    \param virial_pitch pitch of 2D virial array
    \param d_velocity Particle velocity and mass array from ParticleData
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members for which to sum properties
    \param work_size Number of particles in the group
    \param offset Offset of this GPU in the list of group members
    \param block_offset Offset of this GPU in the array of partial sums
    \param num_blocks Total number of partial sums by all GPUs

    One thread is executed per group member. That thread reads in the six values (components of the
   pressure tensor) for its member into shared memory and then the block performs a reduction in
   parallel to produce a partial sum output for the block. These partial sums are written to
   d_scratch[i*gridDim.x + blockIdx.x], where i=0..5 is the index of the component. For this kernel
   to run, 6*sizeof(Scalar)*block_size of dynamic shared memory are needed.
*/

__global__ void gpu_compute_pressure_tensor_partial_sums(Scalar* d_scratch,
                                                         Scalar4* d_net_force,
                                                         Scalar* d_net_virial,
                                                         const size_t virial_pitch,
                                                         Scalar4* d_velocity,
                                                         unsigned int* d_body,
                                                         unsigned int* d_tag,
                                                         unsigned int* d_group_members,
                                                         unsigned int work_size,
                                                         unsigned int offset,
                                                         unsigned int block_offset,
                                                         unsigned int num_blocks)
    {
    extern __shared__ Scalar compute_pressure_tensor_sdata[];

    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar my_element[6]; // element of scratch space read in

    // non-participating threads: contribute 0 to the sum
    my_element[0] = 0;
    my_element[1] = 0;
    my_element[2] = 0;
    my_element[3] = 0;
    my_element[4] = 0;
    my_element[5] = 0;

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx + offset];

        // ignore rigid body constituent particles in the sum
        unsigned int body = d_body[idx];
        unsigned int tag = d_tag[idx];
        if (body >= MIN_FLOPPY || body == tag)
            {
            // compute contribution to pressure tensor and store it in my_element
            Scalar4 vel = d_velocity[idx];
            Scalar mass = vel.w;
            my_element[0] = mass * vel.x * vel.x + d_net_virial[0 * virial_pitch + idx]; // xx
            my_element[1] = mass * vel.x * vel.y + d_net_virial[1 * virial_pitch + idx]; // xy
            my_element[2] = mass * vel.x * vel.z + d_net_virial[2 * virial_pitch + idx]; // xz
            my_element[3] = mass * vel.y * vel.y + d_net_virial[3 * virial_pitch + idx]; // yy
            my_element[4] = mass * vel.y * vel.z + d_net_virial[4 * virial_pitch + idx]; // yz
            my_element[5] = mass * vel.z * vel.z + d_net_virial[5 * virial_pitch + idx]; // zz
            }
        }

    for (unsigned int i = 0; i < 6; i++)
        compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x] = my_element[i];

    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            for (unsigned int i = 0; i < 6; i++)
                compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x]
                    += compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        for (unsigned int i = 0; i < 6; i++)
            d_scratch[num_blocks * i + blockIdx.x + block_offset]
                = compute_pressure_tensor_sdata[i * blockDim.x];
        }
    }

//! Perform partial sums of the rotational KE on the GPU
/*! \param d_scratch Scratch space to hold partial sums. One element is written per block
    \param d_orientation Orientation quaternions from ParticleData
    \param d_angmom Conjugate quaternions from ParticleData
    \param d_inertia Moments of inertia from ParticleData
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members for which to sum properties
    \param work_size Number of particles in the group processed by this GPU
    \param offset Offset of this GPU in the list of group members
    \param block_offset Output offset of this GPU
*/

__global__ void gpu_compute_rotational_ke_partial_sums(Scalar* d_scratch,
                                                       const Scalar4* d_orientation,
                                                       const Scalar4* d_angmom,
                                                       const Scalar3* d_inertia,
                                                       unsigned int* d_body,
                                                       unsigned int* d_tag,
                                                       unsigned int* d_group_members,
                                                       unsigned int work_size,
                                                       unsigned int offset,
                                                       unsigned int block_offset)
    {
    extern __shared__ Scalar compute_ke_rot_sdata[];

    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar my_element; // element of scratch space read in
    // non-participating thread: contribute 0 to the sum
    my_element = Scalar(0.0);

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx + offset];

        // ignore rigid body constituent particles in the sum
        unsigned int body = d_body[idx];
        unsigned int tag = d_tag[idx];
        if (body >= MIN_FLOPPY || body == tag)
            {
            quat<Scalar> q(d_orientation[idx]);
            quat<Scalar> p(d_angmom[idx]);
            vec3<Scalar> I(d_inertia[idx]);
            quat<Scalar> s(Scalar(0.5) * conj(q) * p);

            Scalar ke_rot(0.0);

            if (I.x > 0)
                {
                ke_rot += s.v.x * s.v.x / I.x;
                }
            if (I.y > 0)
                {
                ke_rot += s.v.y * s.v.y / I.y;
                }
            if (I.z > 0)
                {
                ke_rot += s.v.z * s.v.z / I.z;
                }

            // compute our contribution to the sum
            my_element = ke_rot * Scalar(1.0 / 2.0);
            }
        }

    compute_ke_rot_sdata[threadIdx.x] = my_element;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            compute_ke_rot_sdata[threadIdx.x] += compute_ke_rot_sdata[threadIdx.x + offs];

        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_scratch[blockIdx.x + block_offset] = compute_ke_rot_sdata[0];
        }
    }

//! Complete partial sums and compute final thermodynamic quantities (for pressure, only isotropic
//! contribution)
/*! \param d_properties Property array to write final values
    \param d_scratch Partial sums
    \param d_scratch_rot Partial sums of rotational kinetic energy
    \param ndof Number of degrees of freedom this group possesses
    \param box Box the particles are in
    \param D Dimensionality of the system
    \param group_size Number of particles in the group
    \param num_partial_sums Number of partial sums in \a d_scratch
    \param external_virial External contribution to virial (1/3 trace)
    \param external_energy External contribution to potential energy


    Only one block is executed. In that block, the partial sums are read in and reduced to final
   values. From the final sums, the thermodynamic properties are computed and written to
   d_properties.

    sizeof(Scalar4)*block_size bytes of shared memory are needed for this kernel to run.
*/
__global__ void gpu_compute_thermo_final_sums(Scalar* d_properties,
                                              Scalar4* d_scratch,
                                              Scalar* d_scratch_rot,
                                              Scalar ndof,
                                              BoxDim box,
                                              unsigned int D,
                                              unsigned int group_size,
                                              unsigned int num_partial_sums,
                                              Scalar external_virial,
                                              Scalar external_energy)
    {
    extern __shared__ Scalar4 compute_thermo_final_sdata[];

    Scalar4 final_sum = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            Scalar4 scratch = d_scratch[start + threadIdx.x];
            Scalar scratch_rot = d_scratch_rot[start + threadIdx.x];

            compute_thermo_final_sdata[threadIdx.x]
                = make_scalar4(scratch.x, scratch.y, scratch.z, scratch_rot);
            }
        else
            compute_thermo_final_sdata[threadIdx.x]
                = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                compute_thermo_final_sdata[threadIdx.x].x
                    += compute_thermo_final_sdata[threadIdx.x + offs].x;
                compute_thermo_final_sdata[threadIdx.x].y
                    += compute_thermo_final_sdata[threadIdx.x + offs].y;
                compute_thermo_final_sdata[threadIdx.x].z
                    += compute_thermo_final_sdata[threadIdx.x + offs].z;
                compute_thermo_final_sdata[threadIdx.x].w
                    += compute_thermo_final_sdata[threadIdx.x + offs].w;
                }
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            final_sum.x += compute_thermo_final_sdata[0].x;
            final_sum.y += compute_thermo_final_sdata[0].y;
            final_sum.z += compute_thermo_final_sdata[0].z;
            final_sum.w += compute_thermo_final_sdata[0].w;
            }
        }

    if (threadIdx.x == 0)
        {
        // compute final quantities
        Scalar ke_trans_total = final_sum.x * Scalar(0.5);
        Scalar pe_total = final_sum.y + external_energy;
        Scalar W = final_sum.z + external_virial;
        Scalar ke_rot_total = final_sum.w;

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
        Scalar pressure = (Scalar(2.0) * ke_trans_total / Scalar(D) + W) / volume;

        // fill out the GPUArray
        d_properties[thermo_index::translational_kinetic_energy] = Scalar(ke_trans_total);
        d_properties[thermo_index::rotational_kinetic_energy] = Scalar(ke_rot_total);
        d_properties[thermo_index::potential_energy] = Scalar(pe_total);
        d_properties[thermo_index::pressure] = pressure;
        }
    }

//! Complete partial sums and compute final pressure tensor
/*! \param d_properties Property array to write final values
    \param d_scratch Partial sums
    \param box Box the particles are in
    \param group_size Number of particles in the group
    \param num_partial_sums Number of partial sums in \a d_scratch

    \param external_virial_xx External contribution to virial (xx component)
    \param external_virial_xy External contribution to virial (xy component)
    \param external_virial_xz External contribution to virial (xz component)
    \param external_virial_yy External contribution to virial (yy component)
    \param external_virial_yz External contribution to virial (yz component)
    \param external_virial_zz External contribution to virial (zz component)

    Only one block is executed. In that block, the partial sums are read in and reduced to final
   values. From the final sums, the thermodynamic properties are computed and written to
   d_properties.

    6*sizeof(Scalar)*block_size bytes of shared memory are needed for this kernel to run.
*/
__global__ void gpu_compute_pressure_tensor_final_sums(Scalar* d_properties,
                                                       Scalar* d_scratch,
                                                       BoxDim box,
                                                       unsigned int group_size,
                                                       unsigned int num_partial_sums,
                                                       Scalar external_virial_xx,
                                                       Scalar external_virial_xy,
                                                       Scalar external_virial_xz,
                                                       Scalar external_virial_yy,
                                                       Scalar external_virial_yz,
                                                       Scalar external_virial_zz,
                                                       bool twod)
    {
    extern __shared__ Scalar compute_pressure_tensor_sdata[];

    Scalar final_sum[6];

    final_sum[0] = external_virial_xx;
    final_sum[1] = external_virial_xy;
    final_sum[2] = external_virial_xz;
    final_sum[3] = external_virial_yy;
    final_sum[4] = external_virial_yz;
    final_sum[5] = external_virial_zz;

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            for (unsigned int i = 0; i < 6; i++)
                compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x]
                    = d_scratch[i * num_partial_sums + start + threadIdx.x];
            }
        else
            for (unsigned int i = 0; i < 6; i++)
                compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x] = Scalar(0.0);
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                for (unsigned int i = 0; i < 6; i++)
                    compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x]
                        += compute_pressure_tensor_sdata[i * blockDim.x + threadIdx.x + offs];
                }
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            for (unsigned int i = 0; i < 6; i++)
                final_sum[i] += compute_pressure_tensor_sdata[i * blockDim.x];
            }
        }

    if (threadIdx.x == 0)
        {
        // fill out the GPUArray
        // we have thus far calculated the sum of the kinetic part of the pressure tensor
        // and the virial part, the definition includes an inverse factor of the box volume
        Scalar V = box.getVolume(twod);

        d_properties[thermo_index::pressure_xx] = final_sum[0] / V;
        d_properties[thermo_index::pressure_xy] = final_sum[1] / V;
        d_properties[thermo_index::pressure_xz] = final_sum[2] / V;
        d_properties[thermo_index::pressure_yy] = final_sum[3] / V;
        d_properties[thermo_index::pressure_yz] = final_sum[4] / V;
        d_properties[thermo_index::pressure_zz] = final_sum[5] / V;
        }
    }
//! Compute partial sums of thermodynamic properties of a group on the GPU,
/*! \param d_properties Array to write computed properties
    \param d_vel particle velocities and masses on the GPU
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members
    \param group_size Number of group members
    \param box Box the particles are in
    \param args Additional arguments
    \param compute_pressure_tensor whether to compute the full pressure tensor
    \param compute_rotational_energy whether to compute the rotational kinetic energy
    \param gpu_partition Load balancing info for multi-GPU reduction

    This function drives gpu_compute_thermo_partial_sums and gpu_compute_thermo_final_sums, see them
   for details.
*/

hipError_t gpu_compute_thermo_partial(Scalar* d_properties,
                                      Scalar4* d_vel,
                                      unsigned int* d_body,
                                      unsigned int* d_tag,
                                      unsigned int* d_group_members,
                                      unsigned int group_size,
                                      const BoxDim& box,
                                      const compute_thermo_args& args,
                                      bool compute_pressure_tensor,
                                      bool compute_rotational_energy,
                                      const GPUPartition& gpu_partition)
    {
    assert(d_properties);
    assert(d_vel);
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

        size_t shared_bytes = sizeof(Scalar3) * args.block_size;

        hipLaunchKernelGGL(gpu_compute_thermo_partial_sums,
                           dim3(grid),
                           dim3(threads),
                           shared_bytes,
                           0,
                           args.d_scratch,
                           args.d_net_force,
                           args.d_net_virial,
                           args.virial_pitch,
                           d_vel,
                           d_body,
                           d_tag,
                           d_group_members,
                           nwork,
                           range.first,
                           block_offset);

        if (compute_pressure_tensor)
            {
            assert(args.d_scratch_pressure_tensor);

            shared_bytes = 6 * sizeof(Scalar) * args.block_size;

            // run the kernel
            hipLaunchKernelGGL(gpu_compute_pressure_tensor_partial_sums,
                               dim3(grid),
                               dim3(threads),
                               shared_bytes,
                               0,
                               args.d_scratch_pressure_tensor,
                               args.d_net_force,
                               args.d_net_virial,
                               args.virial_pitch,
                               d_vel,
                               d_body,
                               d_tag,
                               d_group_members,
                               nwork,
                               range.first,
                               block_offset,
                               args.n_blocks);
            }

        if (compute_rotational_energy)
            {
            assert(args.d_scratch_pressure_tensor);

            shared_bytes = sizeof(Scalar) * args.block_size;

            // run the kernel
            hipLaunchKernelGGL(gpu_compute_rotational_ke_partial_sums,
                               dim3(grid),
                               dim3(threads),
                               shared_bytes,
                               0,
                               args.d_scratch_rot,
                               args.d_orientation,
                               args.d_angmom,
                               args.d_inertia,
                               d_body,
                               d_tag,
                               d_group_members,
                               nwork,
                               range.first,
                               block_offset);
            }

        block_offset += grid.x;
        }

    assert(block_offset <= args.n_blocks);

    return hipSuccess;
    }

//! Compute thermodynamic properties of a group on the GPU
/*! \param d_properties Array to write computed properties
    \param d_vel particle velocities and masses on the GPU
    \param d_body Particle body id
    \param d_tag Particle tag
    \param d_group_members List of group members
    \param group_size Number of group members
    \param box Box the particles are in
    \param args Additional arguments
    \param compute_pressure_tensor whether to compute the full pressure tensor
    \param compute_rotational_energy whether to compute the rotational kinetic energy
    \param num_blocks Number of partial sums to reduce

    This function drives gpu_compute_thermo_partial_sums and gpu_compute_thermo_final_sums, see them
   for details.
*/

hipError_t gpu_compute_thermo_final(Scalar* d_properties,
                                    Scalar4* d_vel,
                                    unsigned int* d_body,
                                    unsigned int* d_tag,
                                    unsigned int* d_group_members,
                                    unsigned int group_size,
                                    const BoxDim& box,
                                    const compute_thermo_args& args,
                                    bool compute_pressure_tensor,
                                    bool compute_rotational_energy)
    {
    assert(d_properties);
    assert(d_vel);
    assert(d_group_members);
    assert(args.d_net_force);
    assert(args.d_net_virial);
    assert(args.d_scratch);

    // setup the grid to run the final kernel
    int final_block_size = 256;
    dim3 grid = dim3(1, 1, 1);
    dim3 threads = dim3(final_block_size, 1, 1);

    size_t shared_bytes = sizeof(Scalar4) * final_block_size;

    Scalar external_virial
        = Scalar(1.0 / 3.0)
          * (args.external_virial_xx + args.external_virial_yy + args.external_virial_zz);

    // run the kernel
    hipLaunchKernelGGL(gpu_compute_thermo_final_sums,
                       dim3(grid),
                       dim3(threads),
                       shared_bytes,
                       0,
                       d_properties,
                       args.d_scratch,
                       args.d_scratch_rot,
                       args.ndof,
                       box,
                       args.D,
                       group_size,
                       args.n_blocks,
                       external_virial,
                       args.external_energy);

    if (compute_pressure_tensor)
        {
        shared_bytes = 6 * sizeof(Scalar) * final_block_size;
        // run the kernel
        hipLaunchKernelGGL(gpu_compute_pressure_tensor_final_sums,
                           dim3(grid),
                           dim3(threads),
                           shared_bytes,
                           0,
                           d_properties,
                           args.d_scratch_pressure_tensor,
                           box,
                           group_size,
                           args.n_blocks,
                           args.external_virial_xx,
                           args.external_virial_xy,
                           args.external_virial_xz,
                           args.external_virial_yy,
                           args.external_virial_yz,
                           args.external_virial_zz,
                           args.D == 2);
        }

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
