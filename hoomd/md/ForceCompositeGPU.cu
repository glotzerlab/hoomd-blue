// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

#include "ForceCompositeGPU.cuh"
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// Maintainer: jglaser

/*! \file ForceComposite.cu
    \brief Defines GPU kernel code for the composite particle integration on the GPU.
*/

//! Shared memory for body force and torque reduction, required allocation when the kernel is called
extern __shared__ char sum[];
extern __shared__ Scalar sum_virial[];

//! Calculates the body forces and torques by summing the constituent particle forces using a fixed sliding window size
/*  Compute the force and torque sum on all bodies in the system from their constituent particles. n_bodies_per_block
    bodies are handled within each block of execution on the GPU. The reason for this is to decrease
    over-parallelism and use the GPU cores more effectively when bodies are smaller than the block size. Otherwise,
    small bodies leave many threads in the block idle with nothing to do.

    On start, the properties common to each body are read in, computed, and stored in shared memory for all the threads
    working on that body to access. Then, the threads loop over all particles that are part of the body with
    a sliding window. Each loop of the window computes the force and torque for block_size/n_bodies_per_block particles
    in as many threads in parallel. These quantities are summed over enough windows to cover the whole body.

    The block_size/n_bodies_per_block partial sums are stored in shared memory. Then n_bodies_per_block partial
    reductions are performed in parallel using all threads to sum the total force and torque on each body. This looks
    just like a normal reduction, except that it terminates at a certain level in the tree. To make the math
    for the partial reduction work out, block_size must be a power of 2 as must n_bodies_per_block.

    Performance testing on GF100 with many different bodies of different sizes ranging from 4-256 particles per body
    has found that the optimum block size for most bodies is 64 threads. Performance increases for all body sizes
    as n_bodies_per_block is increased, but only up to 8. n_bodies_per_block=16 slows performance significantly.
    Based on these performance results, this kernel is hardcoded to handle only 1,2,4,8 n_bodies_per_block
    with a power of 2 block size (hardcoded to 64 in the kernel launch).
*/
__global__ void gpu_rigid_force_sliding_kernel(Scalar4* d_force,
                                                 Scalar4* d_torque,
                                                 const unsigned int *d_molecule_len,
                                                 const unsigned int *d_molecule_list,
                                                 const unsigned int *d_molecule_idx,
                                                 const unsigned int *d_rigid_center,
                                                 Index2D molecule_indexer,
                                                 const Scalar4 *d_postype,
                                                 const Scalar4* d_orientation,
                                                 Index2D body_indexer,
                                                 Scalar3* d_body_pos,
                                                 Scalar4* d_body_orientation,
                                                 const unsigned int *d_body_len,
                                                 const unsigned int *d_body,
                                                 const unsigned int *d_tag,
                                                 uint2 *d_flag,
                                                 Scalar4* d_net_force,
                                                 Scalar4* d_net_torque,
                                                 unsigned int n_mol,
                                                 unsigned int N,
                                                 unsigned int window_size,
                                                 unsigned int thread_mask,
                                                 unsigned int n_bodies_per_block,
                                                 bool zero_force,
                                                 unsigned int first_body,
                                                 unsigned int nwork)
    {
    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);

    // body_force and body_torque are each shared memory arrays with 1 element per threads
    Scalar4 *body_force = (Scalar4 *)sum;
    Scalar3 *body_torque = (Scalar3 *) (body_force + blockDim.x);

    // store body type, orientation and the index in molecule list in shared memory. Up to 16 bodies per block can
    // be handled.
    __shared__ unsigned int body_type[16];
    __shared__ Scalar4 body_orientation[16];
    __shared__ unsigned int mol_idx[16];
    __shared__ unsigned int central_idx[16];

    // each thread makes partial sums of force and torque of all the particles that this thread loops over
    Scalar4 sum_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0),Scalar(0.0));
    Scalar3 sum_torque = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and determines what offset
    // this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared memory
        int group_idx = blockIdx.x*n_bodies_per_block + m;
        if (group_idx < nwork)
            {
            central_idx[m] = d_rigid_center[group_idx + first_body];
            mol_idx[m] = d_molecule_idx[central_idx[m]];

            if (d_tag[central_idx[m]] != d_body[central_idx[m]])
                {
                // this is not the central ptl, molecule is incomplete - mark as such
                body_type[m] = 0xffffffff;
                body_orientation[m] = make_scalar4(1,0,0,0);
                }
            else
                {
                body_type[m] = __scalar_as_int(d_postype[central_idx[m]].w);
                body_orientation[m] = d_orientation[central_idx[m]];
                }
            }
        else
            {
            mol_idx[m] = NO_BODY;
            }
        }

    __syncthreads();

    if (mol_idx[m] < MIN_FLOPPY)
        {
        // compute the number of windows that we need to loop over
        unsigned int mol_len = d_molecule_len[mol_idx[m]];
        unsigned int n_windows = mol_len / window_size + 1;

        // slide the window throughout the block
        for (unsigned int start = 0; start < n_windows; start++)
            {
            // determine the index with this body that this particle should handle
            unsigned int k = start * window_size + (threadIdx.x & thread_mask);

            // if that index is in the body we are actually handling a real body
            if (k < mol_len)
                {
                // determine the particle idx of the particle
                unsigned int pidx = d_molecule_list[molecule_indexer(k,mol_idx[m])];

                // if this particle is not the central particle
                if (body_type[m] != 0xffffffff && pidx != central_idx[m])
                    {
                    Scalar4 fi = d_net_force[pidx];

                    //will likely need to rotate these components too
                    vec3<Scalar> ti(d_net_torque[pidx]);

                    // zero net torque on constituent particles
                    d_net_torque[pidx] = make_scalar4(0.0,0.0,0.0,0.0);

                    // zero force only if we don't need it later
                    if (zero_force)
                        {
                        // zero net energy on constituent ptls to avoid double counting
                        // also zero net force for consistency
                        d_net_force[pidx] = make_scalar4(0.0,0.0,0.0,0.0);
                        }

                    if (central_idx[m] < N)
                        {
                        // at this point, the molecule needs to be complete
                        if (mol_len != d_body_len[body_type[m]] + 1)
                            {
                            // incomplete molecule
                            atomicMax(&(d_flag->x), d_body[central_idx[m]] + 1);
                            }

                        // calculate body force and torques
                        vec3<Scalar> particle_pos(d_body_pos[body_indexer(body_type[m], k-1)]);

                        // tally the force in the per thread counter
                        sum_force.x += fi.x;
                        sum_force.y += fi.y;
                        sum_force.z += fi.z;

                        // sum up energy
                        sum_force.w += fi.w;

                        vec3<Scalar> ri = rotate(quat<Scalar>(body_orientation[m]), particle_pos);

                        // torque = r x f
                        vec3<Scalar> del_torque(cross(ri, vec3<Scalar>(fi)));

                        // tally the torque in the per thread counter
                        sum_torque.x += ti.x+del_torque.x;
                        sum_torque.y += ti.y+del_torque.y;
                        sum_torque.z += ti.z+del_torque.z;
                        }
                    }
                }
            }
        }

    __syncthreads();

    // put the partial sums into shared memory
    body_force[threadIdx.x] = sum_force;
    body_torque[threadIdx.x] = sum_torque;

    __syncthreads();

    // perform a set of partial reductions. Each block_size/n_bodies_per_block threads performs a sum reduction
    // just within its own group
    unsigned int offset = window_size >> 1;
    while (offset > 0)
        {
        if ((threadIdx.x & thread_mask) < offset)
            {
            body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
            body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
            body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
            body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;

            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            }

        offset >>= 1;

        __syncthreads();
        }

    // thread 0 within this body writes out the total force and torque for the body
    if ((threadIdx.x & thread_mask) == 0 && mol_idx[m] < MIN_FLOPPY && central_idx[m] < N)
        {
        d_force[central_idx[m]] = body_force[threadIdx.x];
        d_torque[central_idx[m]] = make_scalar4(body_torque[threadIdx.x].x, body_torque[threadIdx.x].y, body_torque[threadIdx.x].z, 0.0f);
        }
    }

__global__ void gpu_rigid_virial_sliding_kernel(Scalar* d_virial,
                                                const unsigned int *d_molecule_len,
                                                const unsigned int *d_molecule_list,
                                                const unsigned int *d_molecule_idx,
                                                const unsigned int *d_rigid_center,
                                                Index2D molecule_indexer,
                                                const Scalar4 *d_postype,
                                                const Scalar4* d_orientation,
                                                Index2D body_indexer,
                                                Scalar3* d_body_pos,
                                                Scalar4* d_body_orientation,
                                                Scalar4* d_net_force,
                                                Scalar* d_net_virial,
                                                const unsigned int *d_body,
                                                const unsigned int *d_tag,
                                                unsigned int n_mol,
                                                unsigned int N,
                                                unsigned int net_virial_pitch,
                                                unsigned int virial_pitch,
                                                unsigned int window_size,
                                                unsigned int thread_mask,
                                                unsigned int n_bodies_per_block,
                                                unsigned int first_body,
                                                unsigned int nwork)
    {
    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);

    // body_force and body_torque are each shared memory arrays with 1 element per threads
    Scalar *body_virial_xx = sum_virial;
    Scalar *body_virial_xy = &sum_virial[1*blockDim.x];
    Scalar *body_virial_xz = &sum_virial[2*blockDim.x];
    Scalar *body_virial_yy = &sum_virial[3*blockDim.x];
    Scalar *body_virial_yz = &sum_virial[4*blockDim.x];
    Scalar *body_virial_zz = &sum_virial[5*blockDim.x];

    // store body type, orientation and the index in molecule list in shared memory. Up to 16 bodies per block can
    // be handled.
    __shared__ unsigned int body_type[16];
    __shared__ Scalar4 body_orientation[16];
    __shared__ unsigned int mol_idx[16];
    __shared__ unsigned int central_idx[16];

    // each thread makes partial sums of the virial of all the particles that this thread loops over
    Scalar sum_virial_xx(0.0);
    Scalar sum_virial_xy(0.0);
    Scalar sum_virial_xz(0.0);
    Scalar sum_virial_yy(0.0);
    Scalar sum_virial_yz(0.0);
    Scalar sum_virial_zz(0.0);

    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and determines what offset
    // this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared memory
        int group_idx = blockIdx.x*n_bodies_per_block + m;
        if (group_idx < nwork)
            {
            central_idx[m] = d_rigid_center[group_idx + first_body];
            mol_idx[m] = d_molecule_idx[central_idx[m]];

            if (d_tag[central_idx[m]] != d_body[central_idx[m]])
                {
                // this is not the central ptl, molecule is incomplete - mark as such
                body_type[m] = NO_BODY;
                body_orientation[m] = make_scalar4(1,0,0,0);
                }
            else
                {
                body_type[m] = __scalar_as_int(d_postype[central_idx[m]].w);
                body_orientation[m] = d_orientation[central_idx[m]];
                }
            }
        else
            {
            mol_idx[m] = NO_BODY;
            }
        }

    __syncthreads();

    if (mol_idx[m] < MIN_FLOPPY)
        {
        // compute the number of windows that we need to loop over
        unsigned int mol_len = d_molecule_len[mol_idx[m]];
        unsigned int n_windows = mol_len / window_size + 1;

        // slide the window throughout the block
        for (unsigned int start = 0; start < n_windows; start++)
            {
            // determine the index with this body that this particle should handle
            unsigned int k = start * window_size + (threadIdx.x & thread_mask);

            // if that index is in the body we are actually handling a real body
            if (k < mol_len)
                {
                // determine the particle idx of the particle
                unsigned int pidx = d_molecule_list[molecule_indexer(k,mol_idx[m])];

                if (body_type[m] < MIN_FLOPPY && pidx != central_idx[m])
                    {
                    // calculate body force and torques
                    Scalar4 fi = d_net_force[pidx];

                    // sum up virial
                    Scalar virialxx = d_net_virial[0*net_virial_pitch+pidx];
                    Scalar virialxy = d_net_virial[1*net_virial_pitch+pidx];
                    Scalar virialxz = d_net_virial[2*net_virial_pitch+pidx];
                    Scalar virialyy = d_net_virial[3*net_virial_pitch+pidx];
                    Scalar virialyz = d_net_virial[4*net_virial_pitch+pidx];
                    Scalar virialzz = d_net_virial[5*net_virial_pitch+pidx];

                    // zero force and virial on constituent particles
                    d_net_force[pidx] = make_scalar4(0.0,0.0,0.0,0.0);

                    d_net_virial[0*net_virial_pitch+pidx] = Scalar(0.0);
                    d_net_virial[1*net_virial_pitch+pidx] = Scalar(0.0);
                    d_net_virial[2*net_virial_pitch+pidx] = Scalar(0.0);
                    d_net_virial[3*net_virial_pitch+pidx] = Scalar(0.0);
                    d_net_virial[4*net_virial_pitch+pidx] = Scalar(0.0);
                    d_net_virial[5*net_virial_pitch+pidx] = Scalar(0.0);

                    // if this particle is not the central particle (incomplete molecules can't have local members)
                    if (central_idx[m] < N)
                        {
                        vec3<Scalar> particle_pos(d_body_pos[body_indexer(body_type[m], k-1)]);
                        vec3<Scalar> ri = rotate(quat<Scalar>(body_orientation[m]), particle_pos);

                        // subtract intra-body virial prt
                        sum_virial_xx += virialxx - fi.x*ri.x;
                        sum_virial_xy += virialxy - fi.x*ri.y;
                        sum_virial_xz += virialxz - fi.x*ri.z;
                        sum_virial_yy += virialyy - fi.y*ri.y;
                        sum_virial_yz += virialyz - fi.y*ri.z;
                        sum_virial_zz += virialzz - fi.z*ri.z;
                        }
                    }
                }
            }
        }

    __syncthreads();

    // put the partial sums into shared memory
    body_virial_xx[threadIdx.x] = sum_virial_xx;
    body_virial_xy[threadIdx.x] = sum_virial_xy;
    body_virial_xz[threadIdx.x] = sum_virial_xz;
    body_virial_yy[threadIdx.x] = sum_virial_yy;
    body_virial_yz[threadIdx.x] = sum_virial_yz;
    body_virial_zz[threadIdx.x] = sum_virial_zz;

    __syncthreads();

    // perform a set of partial reductions. Each block_size/n_bodies_per_block threads performs a sum reduction
    // just within its own group
    unsigned int offset = window_size >> 1;
    while (offset > 0)
        {
        if ((threadIdx.x & thread_mask) < offset)
            {
            body_virial_xx[threadIdx.x] += body_virial_xx[threadIdx.x + offset];
            body_virial_xy[threadIdx.x] += body_virial_xy[threadIdx.x + offset];
            body_virial_xz[threadIdx.x] += body_virial_xz[threadIdx.x + offset];
            body_virial_yy[threadIdx.x] += body_virial_yy[threadIdx.x + offset];
            body_virial_yz[threadIdx.x] += body_virial_yz[threadIdx.x + offset];
            body_virial_zz[threadIdx.x] += body_virial_zz[threadIdx.x + offset];
            }

        offset >>= 1;

        __syncthreads();
        }

    // thread 0 within this body writes out the total virial for the body
    if ((threadIdx.x & thread_mask) == 0 && mol_idx[m] < MIN_FLOPPY && central_idx[m] < N)
        {
        d_virial[0*virial_pitch+central_idx[m]] = body_virial_xx[threadIdx.x];
        d_virial[1*virial_pitch+central_idx[m]] = body_virial_xy[threadIdx.x];
        d_virial[2*virial_pitch+central_idx[m]] = body_virial_xz[threadIdx.x];
        d_virial[3*virial_pitch+central_idx[m]] = body_virial_yy[threadIdx.x];
        d_virial[4*virial_pitch+central_idx[m]] = body_virial_yz[threadIdx.x];
        d_virial[5*virial_pitch+central_idx[m]] = body_virial_zz[threadIdx.x];
        }
    }


/*!
*/
cudaError_t gpu_rigid_force(Scalar4* d_force,
                 Scalar4* d_torque,
                 const unsigned int *d_molecule_len,
                 const unsigned int *d_molecule_list,
                 const unsigned int *d_molecule_idx,
                 const unsigned int *d_rigid_center,
                 Index2D molecule_indexer,
                 const Scalar4 *d_postype,
                 const Scalar4* d_orientation,
                 Index2D body_indexer,
                 Scalar3* d_body_pos,
                 Scalar4* d_body_orientation,
                 const unsigned int *d_body_len,
                 const unsigned int *d_body,
                 const unsigned int *d_tag,
                 uint2 *d_flag,
                 Scalar4* d_net_force,
                 Scalar4* d_net_torque,
                 unsigned int n_mol,
                 unsigned int N,
                 unsigned int n_bodies_per_block,
                 unsigned int block_size,
                 const cudaDeviceProp& dev_prop,
                 bool zero_force,
                 const GPUPartition &gpu_partition)
    {
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        dim3 force_grid(nwork / n_bodies_per_block + 1, 1, 1);

        static unsigned int max_block_size = UINT_MAX;
        static cudaFuncAttributes attr;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncGetAttributes(&attr, (const void *) gpu_rigid_force_sliding_kernel);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = max_block_size < block_size ? max_block_size : block_size;

        // round down to nearest power of two
        unsigned int b = 1;
        while (b * 2 <= run_block_size) { b *= 2; }
        run_block_size = b;

        unsigned int window_size = run_block_size / n_bodies_per_block;
        unsigned int thread_mask = window_size - 1;

        unsigned int shared_bytes = run_block_size * (sizeof(Scalar4) + sizeof(Scalar3));

        while (shared_bytes + attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
            {
            // block size is power of two
            run_block_size /= 2;

            shared_bytes = run_block_size * (sizeof(Scalar4) + sizeof(Scalar3));

            window_size = run_block_size / n_bodies_per_block;
            thread_mask = window_size - 1;
            }

        gpu_rigid_force_sliding_kernel<<< force_grid, run_block_size, shared_bytes >>>(
            d_force,
            d_torque,
            d_molecule_len,
            d_molecule_list,
            d_molecule_idx,
            d_rigid_center,
            molecule_indexer,
            d_postype,
            d_orientation,
            body_indexer,
            d_body_pos,
            d_body_orientation,
            d_body_len,
            d_body,
            d_tag,
            d_flag,
            d_net_force,
            d_net_torque,
            n_mol,
            N,
            window_size,
            thread_mask,
            n_bodies_per_block,
            zero_force,
            range.first,
            nwork);
        }
    return cudaSuccess;
    }

cudaError_t gpu_rigid_virial(Scalar* d_virial,
                 const unsigned int *d_molecule_len,
                 const unsigned int *d_molecule_list,
                 const unsigned int *d_molecule_idx,
                 const unsigned int *d_rigid_center,
                 Index2D molecule_indexer,
                 const Scalar4 *d_postype,
                 const Scalar4* d_orientation,
                 Index2D body_indexer,
                 Scalar3* d_body_pos,
                 Scalar4* d_body_orientation,
                 Scalar4* d_net_force,
                 Scalar* d_net_virial,
                 const unsigned int *d_body,
                 const unsigned int *d_tag,
                 unsigned int n_mol,
                 unsigned int N,
                 unsigned int n_bodies_per_block,
                 unsigned int net_virial_pitch,
                 unsigned int virial_pitch,
                 unsigned int block_size,
                 const cudaDeviceProp& dev_prop,
                 const GPUPartition &gpu_partition)
    {
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        dim3 force_grid(nwork / n_bodies_per_block + 1, 1, 1);

        static unsigned int max_block_size = UINT_MAX;
        static cudaFuncAttributes attr;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncGetAttributes(&attr, (const void *) gpu_rigid_virial_sliding_kernel);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = max_block_size < block_size ? max_block_size : block_size;

        // round down to nearest power of two
        unsigned int b = 1;
        while (b * 2 <= run_block_size) { b *= 2; }
        run_block_size = b;

        unsigned int window_size = run_block_size / n_bodies_per_block;
        unsigned int thread_mask = window_size - 1;

        unsigned int shared_bytes = 6 * run_block_size * sizeof(Scalar);

        while (shared_bytes + attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
            {
            // block size is power of two
            run_block_size /= 2;

            shared_bytes = 6 * run_block_size * sizeof(Scalar);

            window_size = run_block_size / n_bodies_per_block;
            thread_mask = window_size - 1;
            }

        gpu_rigid_virial_sliding_kernel<<< force_grid, run_block_size, shared_bytes >>>(
            d_virial,
            d_molecule_len,
            d_molecule_list,
            d_molecule_idx,
            d_rigid_center,
            molecule_indexer,
            d_postype,
            d_orientation,
            body_indexer,
            d_body_pos,
            d_body_orientation,
            d_net_force,
            d_net_virial,
            d_body,
            d_tag,
            n_mol,
            N,
            net_virial_pitch,
            virial_pitch,
            window_size,
            thread_mask,
            n_bodies_per_block,
            range.first,
            nwork);
        }

    return cudaSuccess;
    }


__global__ void gpu_update_composite_kernel(unsigned int N,
    unsigned int nwork,
    unsigned int offset,
    unsigned int n_ghost,
    const unsigned int *d_lookup_center,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    Index2D body_indexer,
    const Scalar3 *d_body_pos,
    const Scalar4 *d_body_orientation,
    const unsigned int *d_body_len,
    const unsigned int *d_molecule_order,
    const unsigned int *d_molecule_len,
    const unsigned int *d_molecule_idx,
    int3 *d_image,
    const BoxDim box,
    const BoxDim global_box,
    uint2 *d_flag)
    {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nwork)
        return;

    idx += offset;

    unsigned int central_idx = d_lookup_center[idx];
    if (central_idx == NO_BODY)
        return;

    if (central_idx >= N + n_ghost)
        {
        // if a molecule with a local member has no central particle, error out
        if (idx < N)
            {
            atomicMax(&(d_flag->x), idx+1);
            }

        // otherwise, ignore
        return;
        }

    // do not overwrite central ptl
    if (idx == central_idx) return;

    Scalar4 postype = d_postype[central_idx];
    vec3<Scalar> pos(postype);
    quat<Scalar> orientation(d_orientation[central_idx]);

    unsigned int body_type = __scalar_as_int(postype.w);

    unsigned int body_len = d_body_len[body_type];
    unsigned int mol_idx = d_molecule_idx[idx];

    if (body_len != d_molecule_len[mol_idx]-1)
        {
        // if a molecule with a local member is incomplete, this is an error
        if (idx < N)
            {
            atomicMax(&(d_flag->y), idx+1);
            }

        // otherwise, ignore
        return;
        }

    int3 img = d_image[central_idx];

    unsigned int idx_in_body = d_molecule_order[idx] - 1;

    vec3<Scalar> local_pos(d_body_pos[body_indexer(body_type, idx_in_body)]);
    vec3<Scalar> dr_space = rotate(orientation, local_pos);

    vec3<Scalar> updated_pos(pos);
    updated_pos += dr_space;

    quat<Scalar> local_orientation(d_body_orientation[body_indexer(body_type, idx_in_body)]);
    quat<Scalar> updated_orientation = orientation*local_orientation;

    // this runs before the ForceComputes,
    // wrap into box, allowing rigid bodies to span multiple images
    int3 imgi = box.getImage(vec_to_scalar3(updated_pos));
    int3 negimgi = make_int3(-imgi.x,-imgi.y,-imgi.z);
    updated_pos = global_box.shift(updated_pos, negimgi);

    unsigned int type = __scalar_as_int(d_postype[idx].w);

    d_postype[idx] = make_scalar4(updated_pos.x, updated_pos.y, updated_pos.z, __int_as_scalar(type));
    d_orientation[idx] = quat_to_scalar4(updated_orientation);
    d_image[idx] = img+imgi;
    }

void gpu_update_composite(unsigned int N,
    unsigned int n_ghost,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    Index2D body_indexer,
    const unsigned int *d_lookup_center,
    const Scalar3 *d_body_pos,
    const Scalar4 *d_body_orientation,
    const unsigned int *d_body_len,
    const unsigned int *d_molecule_order,
    const unsigned int *d_molecule_len,
    const unsigned int *d_molecule_idx,
    int3 *d_image,
    const BoxDim box,
    const BoxDim global_box,
    unsigned int block_size,
    uint2 *d_flag,
    const GPUPartition &gpu_partition)
    {
    unsigned int run_block_size = block_size;

    static unsigned int max_block_size = UINT_MAX;
    static cudaFuncAttributes attr;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncGetAttributes(&attr, (const void *) gpu_update_composite_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    if (max_block_size <= run_block_size)
        {
        run_block_size = max_block_size;
        }

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // process ghosts in final range
        if (idev == (int)gpu_partition.getNumActiveGPUs()-1)
            nwork += n_ghost;

        unsigned int n_blocks = nwork/run_block_size + 1;
        gpu_update_composite_kernel<<<n_blocks,run_block_size>>>(N,
            nwork,
            range.first,
            n_ghost,
            d_lookup_center,
            d_postype,
            d_orientation,
            body_indexer,
            d_body_pos,
            d_body_orientation,
            d_body_len,
            d_molecule_order,
            d_molecule_len,
            d_molecule_idx,
            d_image,
            box,
            global_box,
            d_flag);
        }
    }

struct is_center
    {
    __host__ __device__
    bool operator()(const HOOMD_THRUST::tuple<unsigned int, unsigned int>& t)
        {
        return t.get<0>() == t.get<1>();
        }
    };

// create a lookup table ptl idx -> center idx
struct lookup_op : HOOMD_THRUST::unary_function<unsigned int, unsigned int>
    {
    __host__ __device__ lookup_op(const unsigned int *_d_rtag)
        : d_rtag(_d_rtag) {}

    __device__ unsigned int operator()(const unsigned int& body)
        {
        return (body < MIN_FLOPPY) ? d_rtag[body] : NO_BODY;
        }

    const unsigned int *d_rtag;
    };


cudaError_t gpu_find_rigid_centers(const unsigned int *d_body,
                                const unsigned int *d_tag,
                                const unsigned int *d_rtag,
                                const unsigned int N,
                                const unsigned int nghost,
                                unsigned int *d_rigid_center,
                                unsigned int *d_lookup_center,
                                unsigned int &n_rigid)
    {
    HOOMD_THRUST::device_ptr<const unsigned int> body(d_body);
    HOOMD_THRUST::device_ptr<const unsigned int> tag(d_tag);
    HOOMD_THRUST::device_ptr<unsigned int> rigid_center(d_rigid_center);
    HOOMD_THRUST::counting_iterator<unsigned int> count(0);

    // create a contiguos list of rigid center indicies
    auto it = HOOMD_THRUST::copy_if(count,
                    count + N + nghost,
                    HOOMD_THRUST::make_zip_iterator(HOOMD_THRUST::make_tuple(body, tag)),
                    rigid_center,
                    is_center());

    n_rigid = it - rigid_center;

    HOOMD_THRUST::device_ptr<unsigned int> lookup_center(d_lookup_center);
    HOOMD_THRUST::transform(body,
        body + N + nghost,
        lookup_center,
        lookup_op(d_rtag));

    return cudaSuccess;
    }
