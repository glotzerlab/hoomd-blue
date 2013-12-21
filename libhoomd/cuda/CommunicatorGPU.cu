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

/*! \file CommunicatorGPU.cu
    \brief Implementation of communication algorithms on the GPU
*/

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"
#include "ParticleData.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// moderngpu
#include "moderngpu/util/mgpucontext.h"
#include "moderngpu/device/loadstore.cuh"
#include "moderngpu/device/launchbox.cuh"
#include "moderngpu/device/ctaloadbalance.cuh"
#include "moderngpu/kernels/localitysort.cuh"
#include "moderngpu/kernels/search.cuh"
#include "moderngpu/kernels/scan.cuh"
#include "moderngpu/kernels/sortedsearch.cuh"

using namespace thrust;

//! Select a particle for migration
struct select_particle_migrate_gpu : public thrust::unary_function<const Scalar4, unsigned int>
    {
    const BoxDim box;          //!< Local simulation box dimensions
    unsigned int comm_mask;    //!< Allowed communication directions

    //! Constructor
    /*!
     */
    select_particle_migrate_gpu(const BoxDim & _box, unsigned int _comm_mask)
        : box(_box), comm_mask(_comm_mask)
        { }

    //! Select a particle
    /*! t particle data to consider for sending
     * \return true if particle stays in the box
     */
    __host__ __device__ unsigned int operator()(const Scalar4 postype)
        {
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        unsigned int flags = 0;
        if (f.x >= Scalar(1.0)) flags |= send_east;
        if (f.x < Scalar(0.0)) flags |= send_west;
        if (f.y >= Scalar(1.0)) flags |= send_north;
        if (f.y < Scalar(0.0)) flags |= send_south;
        if (f.z >= Scalar(1.0)) flags |= send_up;
        if (f.z < Scalar(0.0)) flags |= send_down;

        // filter allowed directions
        flags &= comm_mask;

        return flags;
        }

     };

//! Select a particle for migration
struct get_migrate_key_gpu : public thrust::unary_function<const unsigned int, unsigned int>
    {
    const uint3 my_pos;     //!< My domain decomposition position
    const Index3D di;             //!< Domain indexer
    const unsigned int mask; //!< Mask of allowed directions

    //! Constructor
    /*!
     */
    get_migrate_key_gpu(const uint3 _my_pos, const Index3D _di, const unsigned int _mask)
        : my_pos(_my_pos), di(_di), mask(_mask)
        { }

    //! Generate key for a sent particle
    __device__ unsigned int operator()(const unsigned int flags)
        {
        int ix, iy, iz;
        ix = iy = iz = 0;

        if ((flags & send_east) && (mask & send_east))
            ix = 1;
        else if ((flags & send_west) && (mask & send_west))
            ix = -1;

        if ((flags & send_north) && (mask & send_north))
            iy = 1;
        else if ((flags & send_south) && (mask & send_south))
            iy = -1;

        if ((flags & send_up) && (mask & send_up))
            iz = 1;
        else if ((flags & send_down) && (mask & send_down))
            iz = -1;

        int i = my_pos.x;
        int j = my_pos.y;
        int k = my_pos.z;

        i += ix;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        j += iy;
        if (j == (int) di.getH())
            j = 0;
        else if (j < 0)
            j += di.getH();

        k += iz;
        if (k == (int) di.getD())
            k = 0;
        else if (k < 0)
            k += di.getD();

        return di(i,j,k);
        }

     };


/*! \param N Number of local particles
    \param d_pos Device array of particle positions
    \param d_tag Device array of particle tags
    \param d_rtag Device array for reverse-lookup table
    \param box Local box
    \param comm_mask Mask of allowed communication directions
    \param alloc Caching allocator
 */
void gpu_stage_particles(const unsigned int N,
                         const Scalar4 *d_pos,
                         unsigned int *d_comm_flag,
                         const BoxDim& box,
                         const unsigned int comm_mask)
    {
    // Wrap particle data arrays
    thrust::device_ptr<const Scalar4> pos_ptr(d_pos);
    thrust::device_ptr<unsigned int> comm_flag_ptr(d_comm_flag);

    // set flag for particles that are to be sent
    thrust::transform(pos_ptr, pos_ptr + N, comm_flag_ptr,
        select_particle_migrate_gpu(box,comm_mask));
    }

/*! \param nsend Number of particles in buffer
    \param d_in Send buf (in-place sort)
    \param d_comm_flags Buffer of communication flags
    \param di Domain indexer
    \param box Local box
    \param d_keys Output array (target domains)
    \param d_begin Output array (start indices per key in send buf)
    \param d_end Output array (end indices per key in send buf)
    \param d_neighbors List of neighbor ranks
    \param mask Mask of communicating directions
    \param alloc Caching allocator
 */
void gpu_sort_migrating_particles(const unsigned int nsend,
                   pdata_element *d_in,
                   const unsigned int *d_comm_flags,
                   const Index3D& di,
                   const uint3 my_pos,
                   unsigned int *d_keys,
                   unsigned int *d_begin,
                   unsigned int *d_end,
                   const unsigned int *d_neighbors,
                   const unsigned int nneigh,
                   const unsigned int mask,
                   mgpu::ContextPtr mgpu_context,
                   unsigned int *d_tmp,
                   pdata_element *d_in_copy)
    {
    // Wrap input & output
    thrust::device_ptr<pdata_element> in_ptr(d_in);
    thrust::device_ptr<const unsigned int> comm_flags_ptr(d_comm_flags);
    thrust::device_ptr<unsigned int> keys_ptr(d_keys);
    thrust::device_ptr<const unsigned int> neighbors_ptr(d_neighbors);

    // generate keys
    thrust::transform(comm_flags_ptr, comm_flags_ptr + nsend, keys_ptr, get_migrate_key_gpu(my_pos, di,mask));

    // allocate temp arrays
    thrust::device_ptr<unsigned int> tmp_ptr(d_tmp);
    thrust::device_ptr<pdata_element> in_copy_ptr(d_in_copy);

    // copy and fill with ascending integer sequence
    thrust::counting_iterator<unsigned int> count_it(0);
    thrust::copy(make_zip_iterator(thrust::make_tuple(count_it, in_ptr)),
        thrust::make_zip_iterator(thrust::make_tuple(count_it + nsend, in_ptr + nsend)),
        thrust::make_zip_iterator(thrust::make_tuple(tmp_ptr, in_copy_ptr)));

    // sort buffer by neighbors
    if (nsend) mgpu::LocalitySortPairs(thrust::raw_pointer_cast(keys_ptr), d_tmp, nsend, *mgpu_context);

    // reorder send buf
    thrust::gather(tmp_ptr, tmp_ptr + nsend, in_copy_ptr, in_ptr);

    mgpu::SortedSearch<mgpu::MgpuBoundsLower>(d_neighbors, nneigh,
        thrust::raw_pointer_cast(keys_ptr), nsend, d_begin, *mgpu_context);
    mgpu::SortedSearch<mgpu::MgpuBoundsUpper>(d_neighbors, nneigh,
        thrust::raw_pointer_cast(keys_ptr), nsend, d_end, *mgpu_context);
    }

//! Wrap a particle in a pdata_element
struct wrap_particle_op_gpu : public thrust::unary_function<const pdata_element, pdata_element>
    {
    const BoxDim box; //!< The box for which we are applying boundary conditions

    //! Constructor
    /*!
     */
    wrap_particle_op_gpu(const BoxDim _box)
        : box(_box)
        {
        }

    //! Wrap position information inside particle data element
    /*! \param p Particle data element
     * \returns The particle data element with wrapped coordinates
     */
    __device__ pdata_element operator()(const pdata_element p)
        {
        pdata_element ret = p;
        box.wrap(ret.pos, ret.image);
        return ret;
        }
     };


/*! \param n_recv Number of particles in buffer
    \param d_in Buffer of particle data elements
    \param box Box for which to apply boundary conditions
 */
void gpu_wrap_particles(const unsigned int n_recv,
                        pdata_element *d_in,
                        const BoxDim& box)
    {
    // Wrap device ptr
    thrust::device_ptr<pdata_element> in_ptr(d_in);

    // Apply box wrap to input buffer
    thrust::transform(in_ptr, in_ptr + n_recv, in_ptr, wrap_particle_op_gpu(box));
    }

//! Reset reverse lookup tags of particles we are removing
/* \param n_delete_ptls Number of particles to delete
 * \param d_delete_tags Array of particle tags to delete
 * \param d_rtag Array for tag->idx lookup
 */
void gpu_reset_rtags(unsigned int n_delete_ptls,
                     unsigned int *d_delete_tags,
                     unsigned int *d_rtag)
    {
    thrust::device_ptr<unsigned int> delete_tags_ptr(d_delete_tags);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::constant_iterator<unsigned int> not_local(NOT_LOCAL);
    thrust::scatter(not_local,
                    not_local + n_delete_ptls,
                    delete_tags_ptr,
                    rtag_ptr);
    }

//! Kernel to select ghost atoms due to non-bonded interactions
__global__ void gpu_make_ghost_exchange_plan_kernel(
    unsigned int N,
    const Scalar4 *d_postype,
    unsigned int *d_plan,
    const BoxDim box,
    Scalar3 ghost_fraction,
    unsigned int mask
    )
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N) return;

    Scalar4 postype = d_postype[idx];
    Scalar3 pos = make_scalar3(postype.x,postype.y,postype.z);
    Scalar3 f = box.makeFraction(pos);

    unsigned int plan = 0;

    // is particle inside ghost layer? set plan accordingly.
    if (f.x >= Scalar(1.0) - ghost_fraction.x)
        plan |= send_east;
    if (f.x < ghost_fraction.x)
        plan |= send_west;
    if (f.y >= Scalar(1.0) - ghost_fraction.y)
        plan |= send_north;
    if (f.y < ghost_fraction.y)
        plan |= send_south;
    if (f.z >= Scalar(1.0) - ghost_fraction.z)
        plan |= send_up;
    if (f.z < ghost_fraction.z)
        plan |= send_down;

    // filter out non-communiating directions
    plan &= mask;

    d_plan[idx] = plan;
    };

//! Construct plans for sending non-bonded ghost particles
/*! \param d_plan Array of ghost particle plans
 * \param N number of particles to check
 * \param d_pos Array of particle positions
 * \param box Dimensions of local simulation box
 * \param r_ghost Width of boundary layer
 */
void gpu_make_ghost_exchange_plan(unsigned int *d_plan,
                                  unsigned int N,
                                  const Scalar4 *d_pos,
                                  const BoxDim &box,
                                  Scalar3 ghost_fraction,
                                  unsigned int mask)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;

    gpu_make_ghost_exchange_plan_kernel<<<n_blocks, block_size>>>(
        N,
        d_pos,
        d_plan,
        box,
        ghost_fraction,
        mask);
    }

__device__ unsigned int get_direction_mask(unsigned int plan)
    {
    unsigned int mask = 0;
    for (int ix = -1; ix <= 1; ix++)
        for (int iy = -1; iy <= 1; iy++)
            for (int iz = -1; iz <= 1; iz++)
                {
                unsigned int flags = 0;
                if (ix == 1) flags |= send_east;
                if (ix == -1) flags |= send_west;
                if (iy == 1) flags |= send_north;
                if (iy == -1) flags |= send_south;
                if (iz == 1) flags |= send_up;
                if (iz == -1) flags |= send_down;

                unsigned int dir = ((iz+1)*3+(iy+1))*3+(ix + 1);
                if (flags && (flags & plan) == flags)
                    mask |= (1 << dir);
                }

    return mask;
    }

//! Apply adjacency masks to plan and return number of matching neighbors
__global__ void  gpu_ghost_neighbor_counts(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    unsigned int *d_counts,
    const unsigned int * d_adj,
    unsigned int nneigh)
    {
    unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned int plan = d_ghost_plan[idx];
    unsigned int count = 0;
    unsigned int mask = get_direction_mask(plan);

    for (unsigned int i = 0; i < nneigh; i++)
        {
        unsigned int adj = d_adj[i];

        if (adj & mask) count++;
        }

    d_counts[idx] = count;
    };

//! Apply adjacency masks to plan and integer and return nth matching neighbor rank
struct get_neighbor_rank_n
    {
    const unsigned int *d_adj;
    const unsigned int *d_neighbor;
    const unsigned int nneigh;

    __host__ __device__ get_neighbor_rank_n(const unsigned int *_d_adj,
        const unsigned int *_d_neighbor,
        unsigned int _nneigh)
        : d_adj(_d_adj),
          d_neighbor(_d_neighbor),
          nneigh(_nneigh)
        { }


    __device__ unsigned int operator() (unsigned int plan, unsigned int n)
        {
        unsigned int count = 0;
        unsigned int ineigh;
        unsigned int mask = get_direction_mask(plan);

        for (ineigh = 0; ineigh < nneigh; ineigh++)
            {
            unsigned int adj = d_adj[ineigh];
            if (adj & mask)
                {
                if (count == n) break;
                count++;
                }
            }
        return d_neighbor[ineigh];
        }
    };

unsigned int gpu_exchange_ghosts_count_neighbors(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_adj,
    unsigned int *d_counts,
    unsigned int nneigh,
    mgpu::ContextPtr mgpu_context)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;

    // compute neighbor counts
    gpu_ghost_neighbor_counts<<<n_blocks, block_size>>>(
        N,
        d_ghost_plan,
        d_counts,
        d_adj,
        nneigh);

    // determine output size
    unsigned int total = 0;
    if (N) mgpu::ScanExc(d_counts, N, &total, *mgpu_context);

    return total;
    }

template<typename Tuning>
__global__ void gpu_expand_neighbors_kernel(const unsigned int n_out,
    const int *d_offs,
    const unsigned int *d_tag,
    const unsigned int *d_plan,
    const unsigned int n_offs,
    const int* mp_global,
    unsigned int *d_idx_out,
    const unsigned int *d_neighbors,
    const unsigned int *d_adj,
    const unsigned int nneigh,
    unsigned int *d_neighbors_out)
    {
    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;

    union Shared
        {
        int indices[NT * (VT + 1)];
        unsigned int values[NT * VT];
        };
    __shared__ Shared shared;
    int tid = threadIdx.x;
    int block = blockIdx.x;

    // Compute the input and output intervals this CTA processes.
    int4 range = mgpu::CTALoadBalance<NT, VT>(n_out, d_offs, n_offs,
        block, tid, mp_global, shared.indices, true);

    // The interval indices are in the left part of shared memory (n_out).
    // The scan of interval counts are in the right part (n_offs)
    int destCount = range.y - range.x;

    // Copy the source indices into register.
    int sources[VT];
    mgpu::DeviceSharedToReg<NT, VT>(shared.indices, tid, sources);

    __syncthreads();

    // Now use the segmented scan to fetch nth neighbor
    get_neighbor_rank_n getn(d_adj, d_neighbors, nneigh);

    // register to hold neighbors
    unsigned int neighbors[VT];

    int *intervals = shared.indices + destCount;

    #pragma unroll
    for(int i = 0; i < VT; ++i)
        {
        int index = NT * i + tid;
        int gid = range.x + index;

        if(index < destCount)
            {
            int interval = sources[i];
            int rank = gid - intervals[interval - range.z];
            int plan = d_plan[interval];
            neighbors[i] = getn(plan,rank);
            }
        }

    // write out neighbors to global mem
    mgpu::DeviceRegToGlobal<NT, VT>(destCount, neighbors, tid, d_neighbors_out + range.x);

    // store indices to global mem
    mgpu::DeviceRegToGlobal<NT, VT>(destCount, sources, tid, d_idx_out + range.x);
    }

void gpu_expand_neighbors(unsigned int n_out,
    const unsigned int *d_offs,
    const unsigned int *d_tag,
    const unsigned int *d_plan,
    unsigned int n_offs,
    unsigned int *d_idx_out,
    const unsigned int *d_neighbors,
    const unsigned int *d_adj,
    const unsigned int nneigh,
    unsigned int *d_neighbors_out,
    mgpu::CudaContext& context)
    {
    const int NT = 128;
    const int VT = 7;
    typedef mgpu::LaunchBoxVT<NT, VT> Tuning;
    int2 launch = Tuning::GetLaunchParams(context);

    int NV = launch.x * launch.y;
    int numBlocks = MGPU_DIV_UP(n_out + n_offs, NV);

    // Partition the input and output sequences so that the load-balancing
    // search results in a CTA fit in shared memory.
    MGPU_MEM(int) partitionsDevice = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>(
        mgpu::counting_iterator<int>(0), n_out, (int *) d_offs,
        n_offs, NV, 0, mgpu::less<int>(), context);

    gpu_expand_neighbors_kernel<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
        n_out, (int *) d_offs, d_tag, d_plan, n_offs,
        partitionsDevice->get(), d_idx_out,
        d_neighbors, d_adj, nneigh, d_neighbors_out);
    }

void gpu_exchange_ghosts_make_indices(
    unsigned int N,
    const unsigned int *d_ghost_plan,
    const unsigned int *d_tag,
    const unsigned int *d_adj,
    const unsigned int *d_unique_neighbors,
    const unsigned int *d_counts,
    unsigned int *d_ghost_idx,
    unsigned int *d_ghost_neigh,
    unsigned int *d_ghost_begin,
    unsigned int *d_ghost_end,
    unsigned int n_unique_neigh,
    unsigned int n_out,
    unsigned int mask,
    mgpu::ContextPtr mgpu_context)
    {
    /*
     * expand each tag by the number of neighbors to send the corresponding ptl to
     * and assign each copy to a different neighbor
     */

    if (n_out)
        {
        // allocate temporary array
        gpu_expand_neighbors(n_out,
            d_counts,
            d_tag, d_ghost_plan, N, d_ghost_idx,
            d_unique_neighbors, d_adj, n_unique_neigh,
            d_ghost_neigh,
            *mgpu_context);

        // sort tags by neighbors
        mgpu::LocalitySortPairs(d_ghost_neigh, d_ghost_idx, n_out, *mgpu_context);

        mgpu::SortedSearch<mgpu::MgpuBoundsLower>(d_unique_neighbors, n_unique_neigh,
            d_ghost_neigh, n_out, d_ghost_begin, *mgpu_context);
        mgpu::SortedSearch<mgpu::MgpuBoundsUpper>(d_unique_neighbors, n_unique_neigh,
            d_ghost_neigh, n_out, d_ghost_end, *mgpu_context);
        }
    else
        {
        cudaMemset(d_ghost_begin, 0, sizeof(unsigned int)*n_unique_neigh);
        cudaMemset(d_ghost_end, 0, sizeof(unsigned int)*n_unique_neigh);
        }
    }

template<typename T>
__global__ void gpu_pack_kernel(
    unsigned int n_out,
    const unsigned int *d_ghost_idx,
    const T *in,
    T *out)
    {
    unsigned int buf_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (buf_idx >= n_out) return;
    unsigned int idx = d_ghost_idx[buf_idx];
    out[buf_idx] = in[idx];
    }

void gpu_exchange_ghosts_pack(
    unsigned int n_out,
    const unsigned int *d_ghost_idx,
    const unsigned int *d_tag,
    const Scalar4 *d_pos,
    const Scalar4 *d_vel,
    const Scalar *d_charge,
    const Scalar *d_diameter,
    const Scalar4 *d_orientation,
    unsigned int *d_tag_sendbuf,
    Scalar4 *d_pos_sendbuf,
    Scalar4 *d_vel_sendbuf,
    Scalar *d_charge_sendbuf,
    Scalar *d_diameter_sendbuf,
    Scalar4 *d_orientation_sendbuf,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_orientation)
    {
    unsigned int block_size = 256;
    unsigned int n_blocks = n_out/block_size + 1;
    if (send_tag) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_tag, d_tag_sendbuf);
    if (send_pos) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_pos, d_pos_sendbuf);
    if (send_vel) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_vel, d_vel_sendbuf);
    if (send_charge) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_charge, d_charge_sendbuf);
    if (send_diameter) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_diameter, d_diameter_sendbuf);
    if (send_orientation) gpu_pack_kernel<<<n_blocks, block_size>>>(n_out, d_ghost_idx, d_orientation, d_orientation_sendbuf);
    }

void gpu_communicator_initialize_cache_config()
    {
    cudaFuncSetCacheConfig(gpu_pack_kernel<Scalar>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(gpu_pack_kernel<Scalar4>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(gpu_pack_kernel<unsigned int>, cudaFuncCachePreferL1);
    }

//! Wrap particles
struct wrap_ghost_pos_gpu : public thrust::unary_function<Scalar4, Scalar4>
    {
    const BoxDim box; //!< The box for which we are applying boundary conditions

    //! Constructor
    /*!
     */
    wrap_ghost_pos_gpu(const BoxDim _box)
        : box(_box)
        {
        }

    //! Wrap position Scalar4
    /*! \param p The position
     * \returns The wrapped position
     */
    __device__ Scalar4 operator()(Scalar4 p)
        {
        int3 image;
        box.wrap(p,image);
        return p;
        }
     };


/*! \param n_recv Number of particles in buffer
    \param d_pos The particle positions array
    \param box Box for which to apply boundary conditions
 */
void gpu_wrap_ghosts(const unsigned int n_recv,
                     Scalar4 *d_pos,
                     BoxDim box)
    {
    // Wrap device ptr
    thrust::device_ptr<Scalar4> pos_ptr(d_pos);

    // Apply box wrap to input buffer
    thrust::transform(pos_ptr, pos_ptr + n_recv, pos_ptr, wrap_ghost_pos_gpu(box));
    }

void gpu_exchange_ghosts_copy_buf(
    unsigned int n_recv,
    const unsigned int *d_tag_recvbuf,
    const Scalar4 *d_pos_recvbuf,
    const Scalar4 *d_vel_recvbuf,
    const Scalar *d_charge_recvbuf,
    const Scalar *d_diameter_recvbuf,
    const Scalar4 *d_orientation_recvbuf,
    unsigned int *d_tag,
    Scalar4 *d_pos,
    Scalar4 *d_vel,
    Scalar *d_charge,
    Scalar *d_diameter,
    Scalar4 *d_orientation,
    bool send_tag,
    bool send_pos,
    bool send_vel,
    bool send_charge,
    bool send_diameter,
    bool send_orientation)
    {
    if (send_tag) cudaMemcpyAsync(d_tag, d_tag_recvbuf, n_recv*sizeof(unsigned int), cudaMemcpyDeviceToDevice,0);
    if (send_pos) cudaMemcpyAsync(d_pos, d_pos_recvbuf, n_recv*sizeof(Scalar4), cudaMemcpyDeviceToDevice,0);
    if (send_vel) cudaMemcpyAsync(d_vel, d_vel_recvbuf, n_recv*sizeof(Scalar4), cudaMemcpyDeviceToDevice,0);
    if (send_charge) cudaMemcpyAsync(d_charge, d_charge_recvbuf, n_recv*sizeof(Scalar), cudaMemcpyDeviceToDevice,0);
    if (send_diameter) cudaMemcpyAsync(d_diameter, d_diameter_recvbuf, n_recv*sizeof(Scalar), cudaMemcpyDeviceToDevice,0);
    if (send_orientation) cudaMemcpyAsync(d_orientation, d_orientation_recvbuf, n_recv*sizeof(Scalar4), cudaMemcpyDeviceToDevice,0);
    }

void gpu_compute_ghost_rtags(
     unsigned int first_idx,
     unsigned int n_ghost,
     const unsigned int *d_tag,
     unsigned int *d_rtag)
    {
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    thrust::counting_iterator<unsigned int> idx(first_idx);
    thrust::scatter(idx, idx + n_ghost, tag_ptr, rtag_ptr);
    }

/*!
 * Routines for communication of bonded groups
 */
template<unsigned int group_size, typename group_t, typename ranks_t>
__global__ void gpu_mark_groups_kernel(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_t *d_members,
    ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    unsigned int *d_scan,
    const unsigned int *d_rtag,
    const Index3D di,
    uint3 my_pos,
    bool incomplete)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_groups) return;

    // Load group
    group_t g = d_members[group_idx];

    ranks_t r = d_group_ranks[group_idx];

    // initialize bit field
    unsigned int mask = 0;
    unsigned int my_rank = di(my_pos.x, my_pos.y, my_pos.z);

    bool update = false;

    // loop through members of group
    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag = g.tag[i];
        unsigned int pidx = d_rtag[tag];

        if (pidx == NOT_LOCAL)
            {
            // if any ptl is non-local, send
            update = true;
            }
        else // local ptl
            {
            if (incomplete)
                {
                // initially, update rank information for all local ptls
                r.idx[i] = my_rank;
                mask |= (1 << i);
                }

            unsigned int flags = d_comm_flags[pidx];

            if (flags)
                {
                // the local particle is going to be sent to a different domain
                mask |= (1 << i);

                // parse communication flags
                int ix, iy, iz;
                ix = iy = iz = 0;

                if (flags & send_east)
                    ix = 1;
                else if (flags & send_west)
                    ix = -1;

                if (flags & send_north)
                    iy = 1;
                else if (flags & send_south)
                    iy = -1;

                if (flags & send_up)
                    iz = 1;
                else if (flags & send_down)
                    iz = -1;

                int ni = my_pos.x;
                int nj = my_pos.y;
                int nk = my_pos.z;

                ni += ix;
                if (ni == (int)di.getW())
                    ni = 0;
                else if (ni < 0)
                    ni += di.getW();

                nj += iy;
                if (nj == (int) di.getH())
                    nj = 0;
                else if (nj < 0)
                    nj += di.getH();

                nk += iz;
                if (nk == (int) di.getD())
                    nk = 0;
                else if (nk < 0)
                    nk += di.getD();

                // update ranks information
                r.idx[i] = di(ni,nj,nk);

                // a local ptl has changed place, send ranks information
                update = true;
                }
            }
        } // end for

    // write out ranks
    d_group_ranks[group_idx] = r;

    // if group is purely local do not send
    if (!update) mask = 0;

    // write out bitmask
    d_rank_mask[group_idx] = mask;

    // set zero-one input for scan
    d_scan[group_idx] = mask ? 1 : 0;
    }

/*! \param N Number of particles
    \param d_comm_flags Array of communication flags
    \param n_groups Number of local groups
    \param d_members Array of group member tags
    \param d_group_tag Array of group tags
    \param d_group_rtag Array of group rtags
    \param d_group_ranks Auxillary array of group member ranks
    \param d_rtag Particle data reverse-lookup table for tags
    \param di Domain decomposition indexer
    \param my_pos Integer triple of domain coordinates
    \param incomplete If true, initially update auxillary rank information
 */
template<unsigned int group_size, typename group_t, typename ranks_t>
void gpu_mark_groups(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_t *d_members,
    ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    unsigned int *d_scan,
    unsigned int &n_out,
    const Index3D di,
    uint3 my_pos,
    bool incomplete,
    mgpu::ContextPtr mgpu_context)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_groups/block_size + 1;

    gpu_mark_groups_kernel<group_size><<<n_blocks,block_size>>>(N,
        d_comm_flags,
        n_groups,
        d_members,
        d_group_ranks,
        d_rank_mask,
        d_scan,
        d_rtag,
        di,
        my_pos,
        incomplete);

    // scan over marked groups
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_scan, n_groups, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *)NULL, &n_out, d_scan, *mgpu_context);
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename rank_element_t>
__global__ void gpu_scatter_ranks_and_mark_send_groups_kernel(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_t *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    rank_element_t *d_out_ranks)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_groups) return;

    unsigned int mask = d_rank_mask[group_idx];

    // determine if rank information needs to be sent
    if (mask)
        {
        unsigned int out_idx = d_scan[group_idx];
        rank_element_t el;
        el.ranks = d_group_ranks[group_idx];
        el.mask = mask;
        el.tag = d_group_tag[group_idx];
        d_out_ranks[out_idx] = el;
        }

    // determine if whole group needs to be sent
    group_t members = d_groups[group_idx];

    mask = 0;
    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag = members.tag[i];
        unsigned int pidx = d_rtag[tag];

        // are the communication flags set for this member particle?
        if (pidx != NOT_LOCAL && d_comm_flags[pidx])
            mask |= (1 << i);
        }

    // output to 0-1 array
    d_scan[group_idx] = mask ? 1 :0;
    d_rank_mask[group_idx] = mask;
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename rank_element_t>
void gpu_scatter_ranks_and_mark_send_groups(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_t *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    unsigned int &n_send,
    rank_element_t *d_out_ranks,
    mgpu::ContextPtr mgpu_context)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_groups/block_size + 1;

    gpu_scatter_ranks_and_mark_send_groups_kernel<group_size><<<n_blocks,block_size>>>(n_groups,
        d_group_tag,
        d_group_ranks,
        d_rank_mask,
        d_groups,
        d_rtag,
        d_comm_flags,
        d_scan,
        d_out_ranks);

    // scan over groups marked for sending
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_scan, n_groups, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *)NULL, &n_send, d_scan, *mgpu_context);
    }

template<unsigned int group_size, typename ranks_t, typename rank_element_t>
__global__ void gpu_update_ranks_table_kernel(
    unsigned int n_groups,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element_t *d_ranks_recvbuf
    )
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv) return;

    rank_element_t el = d_ranks_recvbuf[recv_idx];
    unsigned int tag = el.tag;
    unsigned int gidx = d_group_rtag[tag];

    if (gidx != GROUP_NOT_LOCAL)
        {
        ranks_t new_ranks = el.ranks;
        unsigned int mask = el.mask;

        for (unsigned int i = 0; i < group_size; ++i)
            {
            bool update = mask & (1 << i);

            if (update)
                d_group_ranks[gidx].idx[i] = new_ranks.idx[i];
            }
        }
    }

template<unsigned int group_size, typename ranks_t, typename rank_element_t>
void gpu_update_ranks_table(
    unsigned int n_groups,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element_t *d_ranks_recvbuf
    )
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_recv/block_size + 1;

    gpu_update_ranks_table_kernel<group_size><<<n_blocks, block_size>>>(
        n_groups,
        d_group_ranks,
        d_group_rtag,
        n_recv,
        d_ranks_recvbuf);
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename packed_t>
__global__ void gpu_scatter_and_mark_groups_for_removal_kernel(
    unsigned int n_groups,
    const group_t *d_groups,
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_t *d_out_groups,
    unsigned int *d_out_rank_mask)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_groups) return;

    unsigned int mask = d_rank_mask[group_idx];
    unsigned int flag = 1;

    // are we sending this group?
    if (mask)
        {
        unsigned int out_idx = d_scan[group_idx];

        packed_t el;
        el.tags = d_groups[group_idx];
        el.type = d_group_type[group_idx];
        el.group_tag = d_group_tag[group_idx];
        el.ranks = d_group_ranks[group_idx];
        d_out_groups[out_idx] = el;
        d_out_rank_mask[out_idx] = mask;

        // determine if the group still has any local ptls
        bool is_local = false;
        for (unsigned int i = 0; i < group_size; ++i)
            {
            unsigned int tag = el.tags.tag[i];
            unsigned int pidx = d_rtag[tag];
            if (pidx != NOT_LOCAL && !d_comm_flags[pidx])
                is_local = true;
            }

        // if group is no longer local, flag for removal
        if (!is_local)
            {
            d_group_rtag[el.group_tag] = GROUP_NOT_LOCAL;
            flag = 0;
            }
        }

    // update zero-one array (zero == remove group, one == retain group)
    d_scan[group_idx] = flag;
    }

template<unsigned int group_size, typename group_t, typename ranks_t, typename packed_t>
void gpu_scatter_and_mark_groups_for_removal(
    unsigned int n_groups,
    const group_t *d_groups,
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const ranks_t *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_t *d_out_groups,
    unsigned int *d_out_rank_mask)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_groups/block_size + 1;

    gpu_scatter_and_mark_groups_for_removal_kernel<group_size><<<n_blocks, block_size>>>(
        n_groups,
        d_groups,
        d_group_type,
        d_group_tag,
        d_group_rtag,
        d_group_ranks,
        d_rank_mask,
        d_rtag,
        d_comm_flags,
        my_rank,
        d_scan,
        d_out_groups,
        d_out_rank_mask);
    }

template<typename group_t, typename ranks_t>
__global__ void gpu_remove_groups_kernel(
    unsigned int n_groups,
    const group_t *d_groups,
    group_t *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const ranks_t *d_group_ranks,
    ranks_t *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    const unsigned int *d_scan)
    {
    unsigned int group_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (group_idx >= n_groups) return;

    unsigned int group_tag = d_group_tag[group_idx];
    bool keep = (d_group_rtag[group_tag] != GROUP_NOT_LOCAL);

    unsigned int out_idx =  d_scan[group_idx];

    if (keep)
        {
        // scatter into output array
        d_groups_alt[out_idx] = d_groups[group_idx];
        d_group_type_alt[out_idx] = d_group_type[group_idx];
        d_group_tag_alt[out_idx] = group_tag;
        d_group_ranks_alt[out_idx] = d_group_ranks[group_idx];

        // rebuild rtags
        d_group_rtag[group_tag] = out_idx;
        }
    }

template<typename group_t, typename ranks_t>
void gpu_remove_groups(unsigned int n_groups,
    const group_t *d_groups,
    group_t *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const ranks_t *d_group_ranks,
    ranks_t *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_scan,
    mgpu::ContextPtr mgpu_context)
    {
    // scan over marked groups
    mgpu::Scan<mgpu::MgpuScanTypeExc>( d_scan, n_groups, (unsigned int) 0,
        mgpu::plus<unsigned int>(), (unsigned int *)NULL, &new_ngroups, d_scan, *mgpu_context);

    unsigned int block_size = 512;
    unsigned int n_blocks = n_groups/block_size + 1;

    gpu_remove_groups_kernel<<<n_blocks,block_size>>>(
        n_groups,
        d_groups,
        d_groups_alt,
        d_group_type,
        d_group_type_alt,
        d_group_tag,
        d_group_tag_alt,
        d_group_ranks,
        d_group_ranks_alt,
        d_group_rtag,
        d_scan);
    }

template<typename packed_t>
__global__ void gpu_count_unique_groups_kernel(
    unsigned int n_recv,
    const packed_t *d_groups_in,
    const unsigned int *d_group_rtag,
    unsigned int *d_scan)
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv) return;

    packed_t el = d_groups_in[recv_idx];

    unsigned int rtag = d_group_rtag[el.group_tag];

    // write out zero-one array
    d_scan[recv_idx] = (rtag == GROUP_NOT_LOCAL) ? 1 : 0;
    }

template<typename packed_t, typename group_t, typename ranks_t>
__global__ void gpu_add_groups_kernel(
    unsigned int n_recv,
    unsigned int n_groups,
    const packed_t *d_groups_in,
    const unsigned int *d_scan,
    group_t *d_groups,
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag)
    {
    unsigned int recv_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (recv_idx >= n_recv) return;

    packed_t el = d_groups_in[recv_idx];

    unsigned int tag = el.group_tag;
    unsigned int rtag = d_group_rtag[tag];
    if (rtag == GROUP_NOT_LOCAL)
        {
        unsigned int add_idx = n_groups + d_scan[recv_idx];

        d_groups[add_idx] = el.tags;
        d_group_type[add_idx] = el.type;
        d_group_tag[add_idx] = tag;
        d_group_ranks[add_idx] = el.ranks;

        // update reverse-lookup table
        d_group_rtag[tag] = add_idx;
        }
    }

template<typename packed_t, typename group_t, typename ranks_t>
void gpu_add_groups(unsigned int n_groups,
    unsigned int n_recv,
    const packed_t *d_groups_in,
    group_t *d_groups,
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    ranks_t *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    mgpu::ContextPtr mgpu_context)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_recv/block_size + 1;

    // update locally existing groups
    gpu_count_unique_groups_kernel<<<n_blocks, block_size>>>(
        n_recv,
        d_groups_in,
        d_group_rtag,
        d_tmp);

    unsigned int n_unique;

    // scan over input groups, select those which are not already local
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_tmp, n_recv, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *)NULL, &n_unique, d_tmp, *mgpu_context);

    new_ngroups = n_groups + n_unique;

    // add new groups at the end
    gpu_add_groups_kernel<<<n_blocks, block_size>>>(
        n_recv,
        n_groups,
        d_groups_in,
        d_tmp,
        d_groups,
        d_group_type,
        d_group_tag,
        d_group_ranks,
        d_group_rtag);
    }

template<unsigned int group_size, typename members_t, typename ranks_t>
__global__ void gpu_mark_bonded_ghosts_kernel(
    unsigned int n_groups,
    members_t *d_groups,
    ranks_t *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D di,
    uint3 my_pos,
    unsigned int mask)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_groups) return;

    // load group member tags
    members_t g = d_groups[group_idx];

    // load group member ranks
    ranks_t r = d_ranks[group_idx];

    unsigned int my_rank = di(my_pos.x,my_pos.y,my_pos.z);

    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int rank = r.idx[i];
        if (rank != my_rank)
            {
            // incomplete group

            // send group to neighbor rank stored for that member
            uint3 neigh_pos = di.getTriple(rank);

            // only neighbors are considered for communication
            unsigned int flags = 0;
            if (neigh_pos.x == my_pos.x + 1 || (my_pos.x == di.getW()-1 && neigh_pos.x == 0))
                flags |= send_east;
            if (neigh_pos.x == my_pos.x - 1 || (my_pos.x == 0 && neigh_pos.x == di.getW()-1))
                flags |= send_west;
            if (neigh_pos.y == my_pos.y + 1 || (my_pos.y == di.getH()-1 && neigh_pos.y == 0))
                flags |= send_north;
            if (neigh_pos.y == my_pos.y - 1 || (my_pos.y == 0 && neigh_pos.y == di.getH()-1))
                flags |= send_south;
            if (neigh_pos.z == my_pos.z + 1 || (my_pos.z == di.getD()-1 && neigh_pos.z == 0))
                flags |= send_up;
            if (neigh_pos.z == my_pos.z - 1 || (my_pos.z == 0 && neigh_pos.z == di.getD()-1))
                flags |= send_down;

            flags &= mask;

            // Send all local members of the group to this neighbor
            for (unsigned int j = 0; j < group_size; ++j)
                {
                unsigned int tag_j = g.tag[j];
                unsigned int rtag_j = d_rtag[tag_j];

                if (rtag_j != NOT_LOCAL)
                    atomicOr(&d_plan[rtag_j], flags);
                }
            }
        }
    }

template<unsigned int group_size, typename members_t, typename ranks_t>
void gpu_mark_bonded_ghosts(
    unsigned int n_groups,
    members_t *d_groups,
    ranks_t *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    unsigned int mask)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_groups/block_size + 1;

    gpu_mark_bonded_ghosts_kernel<group_size><<<n_blocks, block_size>>>(
        n_groups,
        d_groups,
        d_ranks,
        d_rtag,
        d_plan,
        di,
        my_pos,
        mask);
    }

void gpu_reset_exchange_plan(
    unsigned int N,
    unsigned int *d_plan)
    {
    cudaMemsetAsync(d_plan, 0, sizeof(unsigned int)*N);
    }
/*
 *! Explicit template instantiations for BondData (n=2)
 */

template void gpu_mark_groups<2, group_storage<2>, group_storage<2> >(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_storage<2> *d_members,
    group_storage<2> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    unsigned int *d_scan,
    unsigned int &n_out,
    const Index3D di,
    uint3 my_pos,
    bool incomplete,
    mgpu::ContextPtr mgpu_context);

template void gpu_scatter_ranks_and_mark_send_groups<2>(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const group_storage<2> *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_storage<2> *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    unsigned int &n_send,
    rank_element<group_storage<2> > *d_out_ranks,
    mgpu::ContextPtr mgpu_context);

template void gpu_update_ranks_table<2>(
    unsigned int n_groups,
    group_storage<2> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element<group_storage<2> > *d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<2>(
    unsigned int n_groups,
    const group_storage<2> *d_groups,
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const group_storage<2> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_storage<2> *d_out_groups,
    unsigned int *d_out_rank_mask);

template void gpu_remove_groups(unsigned int n_groups,
    const group_storage<2> *d_groups,
    group_storage<2> *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const group_storage<2> *d_group_ranks,
    group_storage<2> *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_scan,
    mgpu::ContextPtr mgpu_context);

template void gpu_add_groups(unsigned int n_groups,
    unsigned int n_recv,
    const packed_storage<2> *d_groups_in,
    group_storage<2> *d_groups,
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    group_storage<2> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    mgpu::ContextPtr mgpu_context);

template void gpu_mark_bonded_ghosts<2>(
    unsigned int n_groups,
    group_storage<2> *d_groups,
    group_storage<2> *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    unsigned int mask);

/*
 *! Explicit template instantiations for BondData (n=3)
 */

template void gpu_mark_groups<3, group_storage<3>, group_storage<3> >(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_storage<3> *d_members,
    group_storage<3> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    unsigned int *d_scan,
    unsigned int &n_out,
    const Index3D di,
    uint3 my_pos,
    bool incomplete,
    mgpu::ContextPtr mgpu_context);

template void gpu_scatter_ranks_and_mark_send_groups<3>(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const group_storage<3> *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_storage<3> *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    unsigned int &n_send,
    rank_element<group_storage<3> > *d_out_ranks,
    mgpu::ContextPtr mgpu_context);

template void gpu_update_ranks_table<3>(
    unsigned int n_groups,
    group_storage<3> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element<group_storage<3> > *d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<3>(
    unsigned int n_groups,
    const group_storage<3> *d_groups,
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const group_storage<3> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_storage<3> *d_out_groups,
    unsigned int *d_out_rank_mask);

template void gpu_remove_groups(unsigned int n_groups,
    const group_storage<3> *d_groups,
    group_storage<3> *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const group_storage<3> *d_group_ranks,
    group_storage<3> *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_scan,
    mgpu::ContextPtr mgpu_context);

template void gpu_add_groups(unsigned int n_groups,
    unsigned int n_recv,
    const packed_storage<3> *d_groups_in,
    group_storage<3> *d_groups,
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    group_storage<3> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    mgpu::ContextPtr mgpu_context);

template void gpu_mark_bonded_ghosts<3>(
    unsigned int n_groups,
    group_storage<3> *d_groups,
    group_storage<3> *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    unsigned int mask);

/*
 *! Explicit template instantiations for DihedralData and ImproperData (n=4)
 */

template void gpu_mark_groups<4, group_storage<4>, group_storage<4> >(
    unsigned int N,
    const unsigned int *d_comm_flags,
    unsigned int n_groups,
    const group_storage<4> *d_members,
    group_storage<4> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    unsigned int *d_scan,
    unsigned int &n_out,
    const Index3D di,
    uint3 my_pos,
    bool incomplete,
    mgpu::ContextPtr mgpu_context);

template void gpu_scatter_ranks_and_mark_send_groups<4>(
    unsigned int n_groups,
    const unsigned int *d_group_tag,
    const group_storage<4> *d_group_ranks,
    unsigned int *d_rank_mask,
    const group_storage<4> *d_groups,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int *d_scan,
    unsigned int &n_send,
    rank_element<group_storage<4> > *d_out_ranks,
    mgpu::ContextPtr mgpu_context);

template void gpu_update_ranks_table<4>(
    unsigned int n_groups,
    group_storage<4> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int n_recv,
    const rank_element<group_storage<4> > *d_ranks_recvbuf);

template void gpu_scatter_and_mark_groups_for_removal<4>(
    unsigned int n_groups,
    const group_storage<4> *d_groups,
    const unsigned int *d_group_type,
    const unsigned int *d_group_tag,
    unsigned int *d_group_rtag,
    const group_storage<4> *d_group_ranks,
    unsigned int *d_rank_mask,
    const unsigned int *d_rtag,
    const unsigned int *d_comm_flags,
    unsigned int my_rank,
    unsigned int *d_scan,
    packed_storage<4> *d_out_groups,
    unsigned int *d_out_rank_mask);

template void gpu_remove_groups(unsigned int n_groups,
    const group_storage<4> *d_groups,
    group_storage<4> *d_groups_alt,
    const unsigned int *d_group_type,
    unsigned int *d_group_type_alt,
    const unsigned int *d_group_tag,
    unsigned int *d_group_tag_alt,
    const group_storage<4> *d_group_ranks,
    group_storage<4> *d_group_ranks_alt,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_scan,
    mgpu::ContextPtr mgpu_context);

template void gpu_add_groups(unsigned int n_groups,
    unsigned int n_recv,
    const packed_storage<4> *d_groups_in,
    group_storage<4> *d_groups,
    unsigned int *d_group_type,
    unsigned int *d_group_tag,
    group_storage<4> *d_group_ranks,
    unsigned int *d_group_rtag,
    unsigned int &new_ngroups,
    unsigned int *d_tmp,
    mgpu::ContextPtr mgpu_context);

template void gpu_mark_bonded_ghosts<4>(
    unsigned int n_groups,
    group_storage<4> *d_groups,
    group_storage<4> *d_ranks,
    const unsigned int *d_rtag,
    unsigned int *d_plan,
    Index3D& di,
    uint3 my_pos,
    unsigned int mask);

#endif // ENABLE_MPI
