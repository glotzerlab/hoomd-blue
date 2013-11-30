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

#include <thrust/replace.h>
#include <thrust/device_ptr.h>
#include <thrust/scatter.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
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
struct select_particle_migrate_gpu : public thrust::unary_function<const Scalar4, bool>
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
    __host__ __device__ bool operator()(const Scalar4 postype)
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

        return flags > 0;
        }

     };

//! Select a particle for migration
struct get_migrate_key_gpu : public thrust::unary_function<const pdata_element, unsigned int>
    {
    const BoxDim box;       //!< Local simulation box dimensions
    const uint3 my_pos;     //!< My domain decomposition position
    const Index3D di;             //!< Domain indexer
    const unsigned int mask; //!< Mask of allowed directions

    //! Constructor
    /*!
     */
    get_migrate_key_gpu(const BoxDim & _box, const uint3 _my_pos, const Index3D _di, const unsigned int _mask)
        : box(_box), my_pos(_my_pos), di(_di), mask(_mask)
        { }

    //! Generate key for a sent particle
    __device__ unsigned int operator()(const pdata_element p)
        {
        Scalar4 postype = p.pos;
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        int ix, iy, iz;
        ix = iy = iz = 0;

        // we allow for a tolerance, large enough so we don't loose particles
        // due to numerical precision
        const Scalar tol(1e-5);
        if (f.x >= Scalar(1.0)-tol && (mask & send_east))
            ix = 1;
        else if (f.x < tol && (mask & send_west))
            ix = -1;

        if (f.y >= Scalar(1.0)-tol && (mask & send_north))
            iy = 1;
        else if (f.y < tol && (mask & send_south))
            iy = -1;

        if (f.z >= Scalar(1.0)-tol && (mask & send_up))
            iz = 1;
        else if (f.z < tol && (mask & send_down))
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
                         const unsigned int *d_tag,
                         unsigned int *d_rtag,
                         const BoxDim& box,
                         const unsigned int comm_mask,
                         cached_allocator& alloc)
    {
    // Wrap particle data arrays
    thrust::device_ptr<const Scalar4> pos_ptr(d_pos);
    thrust::device_ptr<const unsigned int> tag_ptr(d_tag);

    // Wrap rtag array
    thrust::device_ptr<unsigned int> rtag_ptr(d_rtag);

    // pointer from tag into rtag
    thrust::permutation_iterator<
        thrust::device_ptr<unsigned int>, thrust::device_ptr<const unsigned int> > rtag_prm(rtag_ptr, tag_ptr);

    // set flag for particles that are to be sent
    thrust::replace_if(thrust::cuda::par(alloc),
        rtag_prm, rtag_prm + N, pos_ptr,
        select_particle_migrate_gpu(box,comm_mask),
        NOT_LOCAL);
    }

/*! \param nsend Number of particles in buffer
    \param d_in Send buf (in-place sort)
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
                   const Index3D& di,
                   const uint3 my_pos,
                   const BoxDim& box,
                   unsigned int *d_keys,
                   unsigned int *d_begin,
                   unsigned int *d_end,
                   const unsigned int *d_neighbors,
                   const unsigned int nneigh,
                   const unsigned int mask,
                   mgpu::ContextPtr mgpu_context,
                   cached_allocator& alloc)
    {
    // Wrap input & output
    thrust::device_ptr<pdata_element> in_ptr(d_in);
    thrust::device_ptr<unsigned int> keys_ptr(d_keys);
    thrust::device_ptr<const unsigned int> neighbors_ptr(d_neighbors);

    // generate keys
    thrust::transform(in_ptr, in_ptr + nsend, keys_ptr, get_migrate_key_gpu(box, my_pos, di,mask));


    // allocate temp arrays
    unsigned int *d_tmp = (unsigned int *)alloc.allocate(nsend*sizeof(unsigned int));
    thrust::device_ptr<unsigned int> tmp_ptr(d_tmp);

    pdata_element *d_in_copy = (pdata_element *)alloc.allocate(nsend*sizeof(pdata_element));
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

    // release temporary buffers
    alloc.deallocate((char *)d_in_copy,0);
    alloc.deallocate((char *)d_tmp,0);
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
struct make_ghost_exchange_plan_gpu : thrust::unary_function<const Scalar4, unsigned int>
    {
    const BoxDim box;       //!< Local box
    Scalar3 ghost_fraction; //!< Fractional width of ghost layer
    unsigned int mask;      //!< Mask of allowed communication directions

    //! Constructor
    make_ghost_exchange_plan_gpu(const BoxDim& _box, Scalar3 _ghost_fraction, unsigned int _mask)
        : box(_box), ghost_fraction(_ghost_fraction), mask(_mask)
        { }

    __device__ unsigned int operator() (const Scalar4 postype)
        {
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

        return plan;
        }
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
                                  unsigned int mask,
                                  cached_allocator& alloc)
    {
    // wrap position array
    thrust::device_ptr<const Scalar4> pos_ptr(d_pos);

    // wrap plan (output) array
    thrust::device_ptr<unsigned int> plan_ptr(d_plan);

    // compute plans
    thrust::transform(thrust::cuda::par(alloc),
        pos_ptr, pos_ptr + N, plan_ptr,
        make_ghost_exchange_plan_gpu(box, ghost_fraction,mask));
    }

//! Apply adjacency masks to plan and return number of matching neighbors
struct num_neighbors_gpu
    {
    thrust::device_ptr<const unsigned int> adj_ptr;
    const unsigned int nneigh;

    num_neighbors_gpu(thrust::device_ptr<const unsigned int> _adj_ptr, unsigned int _nneigh)
        : adj_ptr(_adj_ptr), nneigh(_nneigh)
        { }

    __device__ unsigned int operator() (unsigned int plan)
        {
        unsigned int count = 0;
        for (unsigned int i = 0; i < nneigh; i++)
            {
            unsigned int adj = adj_ptr[i];
            if ((adj & plan) == adj) count++;
            }
        return count;
        }
    };

//! Apply adjacency masks to plan and integer and return nth matching neighbor rank
struct get_neighbor_rank_n : thrust::unary_function<
    thrust::tuple<unsigned int, unsigned int>, unsigned int >
    {
    thrust::device_ptr<const unsigned int> adj_ptr;
    thrust::device_ptr<const unsigned int> neighbor_ptr;
    const unsigned int nneigh;

    __host__ __device__ get_neighbor_rank_n(thrust::device_ptr<const unsigned int> _adj_ptr,
        thrust::device_ptr<const unsigned int> _neighbor_ptr,
        unsigned int _nneigh)
        : adj_ptr(_adj_ptr), neighbor_ptr(_neighbor_ptr), nneigh(_nneigh)
        { }

    __host__ __device__ get_neighbor_rank_n(const unsigned int *_d_adj,
        const unsigned int *_d_neighbor,
        unsigned int _nneigh)
        : adj_ptr(thrust::device_ptr<const unsigned int>(_d_adj)),
          neighbor_ptr(thrust::device_ptr<const unsigned int>(_d_neighbor)),
          nneigh(_nneigh)
        { }


    __device__ unsigned int operator() (thrust::tuple<unsigned int, unsigned int> t)
        {
        unsigned int plan = thrust::get<0>(t);
        unsigned int n = thrust::get<1>(t);
        unsigned int count = 0;
        unsigned int ineigh;
        for (ineigh = 0; ineigh < nneigh; ineigh++)
            {
            unsigned int adj = adj_ptr[ineigh];
            if ((adj & plan) == adj)
                {
                if (count == n) break;
                count++;
                }
            }
        return neighbor_ptr[ineigh];
        }

    __device__ unsigned int operator() (unsigned int plan, unsigned int n)
        {
        unsigned int count = 0;
        unsigned int ineigh;
        for (ineigh = 0; ineigh < nneigh; ineigh++)
            {
            unsigned int adj = adj_ptr[ineigh];
            if ((adj & plan) == adj)
                {
                if (count == n) break;
                count++;
                }
            }
        return neighbor_ptr[ineigh];
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
    thrust::device_ptr<const unsigned int> ghost_plan_ptr(d_ghost_plan);
    thrust::device_ptr<const unsigned int> adj_ptr(d_adj);
    thrust::device_ptr<unsigned int> counts_ptr(d_counts);

    // compute neighbor counts
    thrust::transform(ghost_plan_ptr, ghost_plan_ptr + N, counts_ptr, num_neighbors_gpu(adj_ptr, nneigh));

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
    const unsigned int *d_neighbors,
    const unsigned int *d_unique_neighbors,
    const unsigned int *d_counts,
    unsigned int *d_ghost_idx,
    unsigned int *d_ghost_begin,
    unsigned int *d_ghost_end,
    unsigned int nneigh,
    unsigned int n_unique_neigh,
    unsigned int n_out,
    unsigned int mask,
    mgpu::ContextPtr mgpu_context,
    cached_allocator& alloc)
    {
    // temporary array for output neighbor ranks
    unsigned int *d_out_neighbors = (unsigned int *)alloc.allocate(n_out*sizeof(unsigned int));

    /*
     * expand each tag by the number of neighbors to send the corresponding ptl to
     * and assign each copy to a different neighbor
     */

    // allocate temporary array
    gpu_expand_neighbors(n_out,
        d_counts,
        d_tag, d_ghost_plan, N, d_ghost_idx,
        d_neighbors, d_adj, nneigh,
        d_out_neighbors,
        *mgpu_context);

    // sort tags by neighbors
    if (n_out) mgpu::LocalitySortPairs(d_out_neighbors, d_ghost_idx, n_out, *mgpu_context);

    mgpu::SortedSearch<mgpu::MgpuBoundsLower>(d_unique_neighbors, n_unique_neigh,
        d_out_neighbors, n_out, d_ghost_begin, *mgpu_context);
    mgpu::SortedSearch<mgpu::MgpuBoundsUpper>(d_unique_neighbors, n_unique_neigh,
        d_out_neighbors, n_out, d_ghost_end, *mgpu_context);

    // deallocate temporary arrays
    alloc.deallocate((char *)d_out_neighbors,0);
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
                        const BoxDim& box)
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


#endif
