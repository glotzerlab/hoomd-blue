/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

#include <VectorMath.h>
#include "Index1D.h"

const unsigned int NO_BODY = 0xffffffff;

// Maintainer: jglaser

/*! \file ForceComposite.cu
    \brief Defines GPU kernel code for the composite particle integration on the GPU.
*/

//! Shared memory for body force and torque reduction, required allocation when the kernel is called
extern __shared__ Scalar3 sum[];

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

    However, there is one issue to the n_bodies_per_block parallelism reduction. If the reduction results in too few
    blocks, performance can actually be reduced. For example, if there are only 64 bodies running at the "most optimal"
    n_bodies_per_block=8 results in only 8 blocks on the GPU! That isn't even enough to heat up all 15 SMs on GF100.
    Even though n_bodies_per_block=1 is not fully optimal per block, running 64 slow blocks in parallel is faster than
    running 8 fast blocks in parallel. Testing on GF100 determines that 60 blocks is the dividing line (makes sense -
    that's 4 blocks active on each SM).
*/
__global__ void gpu_rigid_force_sliding_kernel(Scalar4* d_force,
                                                 Scalar4* d_torque,
                                                 const unsigned int *d_molecule_len,
                                                 const unsigned int *d_molecule_list,
                                                 const unsigned int *d_tag,
                                                 const unsigned int *d_rtag,
                                                 Index2D molecule_indexer,
                                                 const Scalar4 *d_postype,
                                                 const Scalar4* d_orientation,
                                                 Index2D body_indexer,
                                                 Scalar3* d_body_pos,
                                                 Scalar4* d_body_orientation,
                                                 const Scalar4* d_net_force,
                                                 const Scalar4* d_net_torque,
                                                 unsigned int n_mol,
                                                 unsigned int N,
                                                 unsigned int window_size,
                                                 unsigned int thread_mask,
                                                 unsigned int n_bodies_per_block)
    {
    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);

    // body_force and body_torque are each shared memory arrays with 1 element per threads
    Scalar3 *body_force = sum;
    Scalar3 *body_torque = &sum[blockDim.x];

    // store body type, orientation and the index in molecule list in shared memory. Up to 16 bodies per block can
    // be handled.
    __shared__ unsigned int body_type[16];
    __shared__ Scalar4 body_orientation[16];
    __shared__ unsigned int mol_idx[16];
    __shared__ unsigned int central_idx[16];

    // each thread makes partial sums of force and torque of all the particles that this thread loops over
    Scalar3 sum_force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar3 sum_torque = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and determines what offset
    // this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared memory
        int group_idx = blockIdx.x*n_bodies_per_block + m;
        if (group_idx < n_mol)
            {
            mol_idx[m] = group_idx;

            // first ptl is central ptl
            central_idx[m] = d_molecule_list[molecule_indexer(group_idx, 0)];
            body_type[m] = __scalar_as_int(d_postype[central_idx[m]].w);
            body_orientation[m] = d_orientation[central_idx[m]];
            }
        else
            {
            mol_idx[m] = NO_BODY;
            }
        }

    __syncthreads();

    // compute the number of windows that we need to loop over
    unsigned int n_windows = n_mol / window_size + 1;

    // slide the window throughout the block
    for (unsigned int start = 0; start < n_windows; start++)
        {
        Scalar4 fi = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        Scalar4 ti = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

        // determine the index with this body that this particle should handle
        unsigned int k = start * window_size + (threadIdx.x & thread_mask);

        if (mol_idx[m] == NO_BODY || central_idx[m] >= N)
            {
            // only local central ptls
            continue;
            }

        unsigned int mol_len = d_molecule_len[mol_idx[m]];
        unsigned int central_tag = d_tag[central_idx[m]];

        // if that index is in the body we are actually handling a real body
        if (k < mol_len)
            {
            // determine the particle idx of the particle
            unsigned int pidx = d_molecule_list[molecule_indexer(mol_idx[m],k)];

            unsigned int tag = d_tag[pidx];

            // indices are in tag order, and the first ptl is the central ptl
            unsigned int local_idx = tag-central_tag - 1;

            // if this particle is not the central particle
            if (pidx != central_idx[m])
                {
                // calculate body force and torques
                vec3<Scalar> particle_pos(d_body_pos[body_indexer(body_type[m], local_idx)]);
                fi = d_net_force[pidx];

                //will likely need to rotate these components too
                ti = d_net_torque[pidx];

                // tally the force in the per thread counter
                sum_force.x += fi.x;
                sum_force.y += fi.y;
                sum_force.z += fi.z;

                // This might require more calculations but more stable
                // particularly when rigid bodies are bigger than half the box
                vec3<Scalar> ri = rotate(quat<Scalar>(body_orientation[m]), particle_pos);

                // torque = r x f
                vec3<Scalar> del_torque(cross(ri, vec3<Scalar>(fi)));

                // tally the torque in the per thread counter
                sum_torque.x += del_torque.x;
                sum_torque.y += del_torque.y;
                sum_torque.z += del_torque.z;
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

            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            }

        offset >>= 1;

        __syncthreads();
        }

    // thread 0 within this body writes out the total force and torque for the body
    if ((threadIdx.x & thread_mask) == 0 && mol_idx[m] != NO_BODY)
        {
        d_force[central_idx[m]] = make_scalar4(body_force[threadIdx.x].x, body_force[threadIdx.x].y, body_force[threadIdx.x].z, 0.0f);
        d_torque[central_idx[m]] = make_scalar4(body_torque[threadIdx.x].x, body_torque[threadIdx.x].y, body_torque[threadIdx.x].z, 0.0f);
        }
    }


/*!
*/
cudaError_t gpu_rigid_force(Scalar4* d_force,
                 Scalar4* d_torque,
                 const unsigned int *d_molecule_len,
                 const unsigned int *d_molecule_list,
                 const unsigned int *d_tag,
                 const unsigned int *d_rtag,
                 Index2D molecule_indexer,
                 const Scalar4 *d_postype,
                 const Scalar4* d_orientation,
                 Index2D body_indexer,
                 Scalar3* d_body_pos,
                 Scalar4* d_body_orientation,
                 const Scalar4* d_net_force,
                 const Scalar4* d_net_torque,
                 unsigned int n_mol,
                 unsigned int N,
                 unsigned int n_bodies_per_block,
                 unsigned int block_size,
                 const cudaDeviceProp& dev_prop)
    {
    // reset force and torque
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    cudaMemset(d_torque, 0, sizeof(Scalar4)*N);

    dim3 force_grid(n_mol / n_bodies_per_block + 1, 1, 1);

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
    while (b * 2 < run_block_size) { b *= 2; }
    run_block_size = b;

    unsigned int window_size = run_block_size / n_bodies_per_block;
    unsigned int thread_mask = window_size - 1;

    unsigned int shared_bytes = 2 * run_block_size * sizeof(Scalar3);

    while (shared_bytes + attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
        {
        // block size is power of two
        run_block_size /= 2;

        shared_bytes = 2 * run_block_size * sizeof(Scalar3);

        window_size = run_block_size / n_bodies_per_block;
        thread_mask = window_size - 1;
        }

    gpu_rigid_force_sliding_kernel<<< force_grid, run_block_size, shared_bytes >>>(
        d_force,
        d_torque,
        d_molecule_len,
        d_molecule_list,
        d_tag,
        d_rtag,
        molecule_indexer,
        d_postype,
        d_orientation,
        body_indexer,
        d_body_pos,
        d_body_orientation,
        d_net_force,
        d_net_torque,
        n_mol,
        N,
        window_size,
        thread_mask,
        n_bodies_per_block);

        return cudaSuccess;
    }


