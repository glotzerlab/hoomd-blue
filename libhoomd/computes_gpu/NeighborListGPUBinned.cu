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

// Maintainer: joaander

#include "NeighborListGPUBinned.cuh"
#include "TextureTools.h"

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for O(N) neighbor list generation on the GPU
*/

//! Texture for reading d_cell_xyzf
scalar4_tex_t cell_xyzf_1d_tex;

//! Warp-centric scan
template<int NT>
struct warp_scan
    {
    #if __CUDA_ARCH__ >= 300
    enum { capacity = 0 }; // uses no shared memory
    #else
    enum { capacity = NT > 1 ? (2 * NT + 1) : 1};
    #endif

    __device__ static int Scan(int tid, unsigned char x, volatile unsigned char *shared, unsigned char* total)
        {
        #if __CUDA_ARCH__ >= 300
        // Kepler version
        unsigned int laneid;
        //This command gets the lane ID within the current warp
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

        int first = laneid - tid;

        #pragma unroll
        for(int offset = 1; offset < NT; offset += offset)
            {
            int y = __shfl(x,(first + tid - offset) &(WARP_SIZE -1));
            if(tid >= offset) x += y;
            }

        // all threads get the total from the last thread in the cta
        *total = __shfl(x,first + NT - 1);

        // shift by one (exclusive scan)
        int y = __shfl(x,(first + tid - 1) &(WARP_SIZE-1));
        x = tid ? y : 0;

        #else // __CUDA_ARCH__ >= 300

        shared[tid] = x;
        int first = 0;
        // no syncthreads here (inside warp)

        for(int offset = 1; offset < NT; offset += offset)
            {
            if(tid >= offset)
                x = shared[first + tid - offset] + x;
            first = NT - first;
            shared[first + tid] = x;
            // no syncthreads here (inside warp)
            }
        *total = shared[first + NT - 1];

        // shift by one (exclusive scan)
        x = tid ? shared[first + tid - 1] : 0;
        #endif
        // no syncthreads here (inside warp)
        return x;
        }
    };

//! Kernel call for generating neighbor list on the GPU (shared memory version)
/*! \tparam flags Set bit 1 to enable body filtering. Set bit 2 to enable diameter filtering.
    \param d_nlist Neighbor list data structure to write
    \param d_n_neigh Number of neighbors to write
    \param d_last_updated_pos Particle positions at this update are written to this array
    \param d_conditions Conditions array for writing overflow condition
    \param nli Indexer to access \a d_nlist
    \param d_pos Particle positions
    \param d_body Particle body indices
    \param d_diameter Particle diameters
    \param N Number of particles
    \param d_cell_size Number of particles in each cell
    \param d_cell_xyzf Cell contents (xyzf array from CellList with flag=type)
    \param d_cell_tdb Cell contents (tdb array from CellList with)
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer for indexing cells
    \param cli Cell list indexer for indexing into d_cell_xyzf
    \param cadji Adjacent cell indexer listing the 27 neighboring cells
    \param box Simulation box dimensions
    \param r_maxsq The maximum radius for which to include particles as neighbors, squared
    \param r_max The maximum radius for which to include particles as neighbors
    \param ghost_width Width of ghost cell layer

    \note optimized for Fermi
*/
template<unsigned char flags, int threads_per_particle>
__global__ void gpu_compute_nlist_binned_shared_kernel(unsigned int *d_nlist,
                                                    unsigned int *d_n_neigh,
                                                    Scalar4 *d_last_updated_pos,
                                                    unsigned int *d_conditions,
                                                    const Index2D nli,
                                                    const Scalar4 *d_pos,
                                                    const unsigned int *d_body,
                                                    const Scalar *d_diameter,
                                                    const unsigned int N,
                                                    const unsigned int *d_cell_size,
                                                    const Scalar4 *d_cell_xyzf,
                                                    const Scalar4 *d_cell_tdb,
                                                    const unsigned int *d_cell_adj,
                                                    const Index3D ci,
                                                    const Index2D cli,
                                                    const Index2D cadji,
                                                    const BoxDim box,
                                                    const Scalar r_maxsq,
                                                    const Scalar r_max,
                                                    const Scalar3 ghost_width)
    {
    bool filter_body = flags & 1;
    bool filter_diameter = flags & 2;

    // each set of threads_per_particle threads is going to compute the neighbor list for a single particle
    int my_pidx;
    if (gridDim.y > 1)
        {
        // fermi workaround
        my_pidx = (blockIdx.x + blockIdx.y*65535) * (blockDim.x/threads_per_particle) + threadIdx.x/threads_per_particle;
        }
    else
        {
        my_pidx = blockIdx.x * (blockDim.x/threads_per_particle) + threadIdx.x/threads_per_particle;
        }

    // return early if out of bounds
    if (my_pidx >= N) return;

    // first, determine which bin this particle belongs to
    Scalar4 my_postype = d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_body = d_body[my_pidx];
    Scalar my_diameter = d_diameter[my_pidx];

    Scalar3 f = box.makeFraction(my_pos, ghost_width);

    // find the bin each particle belongs in
    int ib = (int)(f.x * ci.getW());
    int jb = (int)(f.y * ci.getH());
    int kb = (int)(f.z * ci.getD());

    uchar3 periodic = box.getPeriodic();

    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW() && periodic.x)
        ib = 0;
    if (jb == ci.getH() && periodic.y)
        jb = 0;
    if (kb == ci.getD() && periodic.z)
        kb = 0;

    int my_cell = ci(ib,jb,kb);

    // shared memory (volatile is required, since we are doing warp-centric)
    volatile extern __shared__ unsigned char sh[];

    // index of current neighbor
    unsigned int cur_adj = 0;

    // current cell
    unsigned int neigh_cell = d_cell_adj[cadji(cur_adj, my_cell)];

    // size of current cell
    unsigned int neigh_size = d_cell_size[neigh_cell];

    // offset of cta in shared memory
    int cta_offs = (threadIdx.x/threads_per_particle)*warp_scan<threads_per_particle>::capacity;

    // current index in cell
    int cur_offset = threadIdx.x % threads_per_particle;

    bool done = false;

    // total number of neighbors
    unsigned int nneigh = 0;

    while (! done)
        {
        // initalize with default
        unsigned int neighbor;
        unsigned char has_neighbor = 0;

        // advance neighbor cell
        while (cur_offset >= neigh_size && !done )
            {
            cur_offset -= neigh_size;
            cur_adj++;
            if (cur_adj < cadji.getW())
                {
                neigh_cell = d_cell_adj[cadji(cur_adj, my_cell)];
                neigh_size = d_cell_size[neigh_cell];
                }
            else
                // we are past the end of the cell neighbors
                done = true;
            }

        // if the first thread in the cta has no work, terminate the loop
        if (done && !(threadIdx.x % threads_per_particle)) break;

        if (!done)
            {
            Scalar4 cur_xyzf = texFetchScalar4(d_cell_xyzf, cell_xyzf_1d_tex, cli(cur_offset, neigh_cell));

            Scalar4 cur_tdb = make_scalar4(0, 0, 0, 0);
            if (filter_diameter || filter_body)
                cur_tdb = d_cell_tdb[cli(cur_offset, neigh_cell)];

            // advance cur_offset
            cur_offset += threads_per_particle;

            unsigned int neigh_body = __scalar_as_int(cur_tdb.z);
            Scalar neigh_diameter = cur_tdb.y;

            Scalar3 neigh_pos = make_scalar3(cur_xyzf.x,
                                           cur_xyzf.y,
                                           cur_xyzf.z);
            int cur_neigh = __scalar_as_int(cur_xyzf.w);

            // compute the distance between the two particles
            Scalar3 dx = my_pos - neigh_pos;

            // wrap the periodic boundary conditions
            dx = box.minImage(dx);

            // compute dr squared
            Scalar drsq = dot(dx,dx);

            bool excluded = (my_pidx == cur_neigh);

            if (filter_body && my_body != 0xffffffff)
                excluded = excluded | (my_body == neigh_body);

            Scalar sqshift(0.0);
            if (filter_diameter)
                {
                // compute the shift in radius to accept neighbors based on their diameters
                Scalar delta = (my_diameter + neigh_diameter) * Scalar(0.5) - Scalar(1.0);
                // r^2 < (r_max + delta)^2
                // r^2 < r_maxsq + delta^2 + 2*r_max*delta
                sqshift = (delta + Scalar(2.0) * r_max) * delta;
                }

            // store result in shared memory
            if (drsq <= (r_maxsq + sqshift) && !excluded)
                {
                neighbor = cur_neigh;
                has_neighbor = 1;
                }
            }

        // no syncthreads here, we assume threads_per_particle < warp size

        // scan over flags
        unsigned char n;
        int k = warp_scan<threads_per_particle>::Scan(threadIdx.x % threads_per_particle,
            has_neighbor, &sh[cta_offs], &n);

        if (has_neighbor && nneigh + k < nli.getH())
            d_nlist[nli(my_pidx, nneigh + k)] = neighbor;

        // increment total neighbor count
        nneigh += n;
        } // end while

    if (threadIdx.x % threads_per_particle == 0)
        {
        // flag if we need to grow the neighbor list
        if (nneigh >= nli.getH())
            atomicMax(&d_conditions[0], nneigh);

        d_n_neigh[my_pidx] = nneigh;
        d_last_updated_pos[my_pidx] = my_postype;
        }
    }

//! determine maximum possible block size
template<typename T>
int get_max_block_size(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % max_threads_per_particle;
    return max_threads;
    }

template<typename T>
int get_compute_capability(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, func);
    return attr.binaryVersion;
    }

void gpu_nlist_binned_bind_texture(const Scalar4 *d_cell_xyzf, unsigned int n_elements)
    {
    // bind the position texture
    cell_xyzf_1d_tex.normalized = false;
    cell_xyzf_1d_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, cell_xyzf_1d_tex, d_cell_xyzf, sizeof(Scalar4)*n_elements);
    }

//! recursive template to launch neighborlist with given template parameters
/* \tparam cur_tpp Number of threads per particle (assumed to be power of two) */
template<int cur_tpp>
inline void launcher(unsigned int *d_nlist,
              unsigned int *d_n_neigh,
              Scalar4 *d_last_updated_pos,
              unsigned int *d_conditions,
              const Index2D nli,
              const Scalar4 *d_pos,
              const unsigned int *d_body,
              const Scalar *d_diameter,
              const unsigned int N,
              const unsigned int *d_cell_size,
              const Scalar4 *d_cell_xyzf,
              const Scalar4 *d_cell_tdb,
              const unsigned int *d_cell_adj,
              const Index3D ci,
              const Index2D cli,
              const Index2D cadji,
              const BoxDim box,
              const Scalar r_maxsq,
              const Scalar r_max,
              const Scalar3 ghost_width,
              unsigned int tpp,
              bool filter_diameter,
              bool filter_body,
              unsigned int block_size)
    {
    unsigned int shared_size = 0;

    if (tpp == cur_tpp && cur_tpp != 0)
        {
        if (!filter_diameter && !filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            static unsigned int sm = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size(gpu_compute_nlist_binned_shared_kernel<0,cur_tpp>);
            if (sm == UINT_MAX)
                sm = get_compute_capability(gpu_compute_nlist_binned_shared_kernel<0,cur_tpp>);
            if (sm < 35) gpu_nlist_binned_bind_texture(d_cell_xyzf, cli.getNumElements());

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size/tpp) + 1);
            if (sm < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            if (sm < 30) shared_size = warp_scan<cur_tpp>::capacity*sizeof(unsigned char)*(block_size/cur_tpp);

            gpu_compute_nlist_binned_shared_kernel<0,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                             d_n_neigh,
                                                                             d_last_updated_pos,
                                                                             d_conditions,
                                                                             nli,
                                                                             d_pos,
                                                                             d_body,
                                                                             d_diameter,
                                                                             N,
                                                                             d_cell_size,
                                                                             d_cell_xyzf,
                                                                             d_cell_tdb,
                                                                             d_cell_adj,
                                                                             ci,
                                                                             cli,
                                                                             cadji,
                                                                             box,
                                                                             r_maxsq,
                                                                             sqrtf(r_maxsq),
                                                                             ghost_width);
            }
        else if (!filter_diameter && filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            static unsigned int sm = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size(gpu_compute_nlist_binned_shared_kernel<1,cur_tpp>);
            if (sm == UINT_MAX)
                sm = get_compute_capability(gpu_compute_nlist_binned_shared_kernel<1,cur_tpp>);
            if (sm < 35) gpu_nlist_binned_bind_texture(d_cell_xyzf, cli.getNumElements());

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size/tpp) + 1);
            if (sm < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            if (sm < 30) shared_size = warp_scan<cur_tpp>::capacity*sizeof(unsigned char)*(block_size/cur_tpp);

            gpu_compute_nlist_binned_shared_kernel<1,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                             d_n_neigh,
                                                                             d_last_updated_pos,
                                                                             d_conditions,
                                                                             nli,
                                                                             d_pos,
                                                                             d_body,
                                                                             d_diameter,
                                                                             N,
                                                                             d_cell_size,
                                                                             d_cell_xyzf,
                                                                             d_cell_tdb,
                                                                             d_cell_adj,
                                                                             ci,
                                                                             cli,
                                                                             cadji,
                                                                             box,
                                                                             r_maxsq,
                                                                             sqrtf(r_maxsq),
                                                                             ghost_width);
            }
        else if (filter_diameter && !filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            static unsigned int sm = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size(gpu_compute_nlist_binned_shared_kernel<2,cur_tpp>);
            if (sm == UINT_MAX)
                sm = get_compute_capability(gpu_compute_nlist_binned_shared_kernel<2,cur_tpp>);
            if (sm < 35) gpu_nlist_binned_bind_texture(d_cell_xyzf, cli.getNumElements());

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size/tpp) + 1);
            if (sm < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            if (sm < 30) shared_size = warp_scan<cur_tpp>::capacity*sizeof(unsigned char)*(block_size/cur_tpp);

            gpu_compute_nlist_binned_shared_kernel<2,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                             d_n_neigh,
                                                                             d_last_updated_pos,
                                                                             d_conditions,
                                                                             nli,
                                                                             d_pos,
                                                                             d_body,
                                                                             d_diameter,
                                                                             N,
                                                                             d_cell_size,
                                                                             d_cell_xyzf,
                                                                             d_cell_tdb,
                                                                             d_cell_adj,
                                                                             ci,
                                                                             cli,
                                                                             cadji,
                                                                             box,
                                                                             r_maxsq,
                                                                             sqrtf(r_maxsq),
                                                                             ghost_width);
            }
        else if (filter_diameter && filter_body)
            {
            static unsigned int max_block_size = UINT_MAX;
            static unsigned int sm = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size(gpu_compute_nlist_binned_shared_kernel<3,cur_tpp>);
            if (sm == UINT_MAX)
                sm = get_compute_capability(gpu_compute_nlist_binned_shared_kernel<3,cur_tpp>);
            if (sm < 35) gpu_nlist_binned_bind_texture(d_cell_xyzf, cli.getNumElements());

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size/tpp) + 1);
            if (sm < 30 && grid.x > 65535)
                {
                grid.y = grid.x/65535 + 1;
                grid.x = 65535;
                }

            if (sm < 30) shared_size = warp_scan<cur_tpp>::capacity*sizeof(unsigned char)*(block_size/cur_tpp);

            gpu_compute_nlist_binned_shared_kernel<3,cur_tpp><<<grid, block_size,shared_size>>>(d_nlist,
                                                                             d_n_neigh,
                                                                             d_last_updated_pos,
                                                                             d_conditions,
                                                                             nli,
                                                                             d_pos,
                                                                             d_body,
                                                                             d_diameter,
                                                                             N,
                                                                             d_cell_size,
                                                                             d_cell_xyzf,
                                                                             d_cell_tdb,
                                                                             d_cell_adj,
                                                                             ci,
                                                                             cli,
                                                                             cadji,
                                                                             box,
                                                                             r_maxsq,
                                                                             sqrtf(r_maxsq),
                                                                             ghost_width);
            }
        }
    else
        {
        launcher<cur_tpp/2>(d_nlist,
                     d_n_neigh,
                     d_last_updated_pos,
                     d_conditions,
                     nli,
                     d_pos,
                     d_body,
                     d_diameter,
                     N,
                     d_cell_size,
                     d_cell_xyzf,
                     d_cell_tdb,
                     d_cell_adj,
                     ci,
                     cli,
                     cadji,
                     box,
                     r_maxsq,
                     sqrtf(r_maxsq),
                     ghost_width,
                     tpp,
                     filter_diameter,
                     filter_body,
                     block_size
                     );
        }
    }

//! template specialization to terminate recursion
template<>
inline void launcher<min_threads_per_particle/2>(unsigned int *d_nlist,
              unsigned int *d_n_neigh,
              Scalar4 *d_last_updated_pos,
              unsigned int *d_conditions,
              const Index2D nli,
              const Scalar4 *d_pos,
              const unsigned int *d_body,
              const Scalar *d_diameter,
              const unsigned int N,
              const unsigned int *d_cell_size,
              const Scalar4 *d_cell_xyzf,
              const Scalar4 *d_cell_tdb,
              const unsigned int *d_cell_adj,
              const Index3D ci,
              const Index2D cli,
              const Index2D cadji,
              const BoxDim box,
              const Scalar r_maxsq,
              const Scalar r_max,
              const Scalar3 ghost_width,
              unsigned int tpp,
              bool filter_diameter,
              bool filter_body,
              unsigned int block_size)
    { }

cudaError_t gpu_compute_nlist_binned_shared(unsigned int *d_nlist,
                                     unsigned int *d_n_neigh,
                                     Scalar4 *d_last_updated_pos,
                                     unsigned int *d_conditions,
                                     const Index2D& nli,
                                     const Scalar4 *d_pos,
                                     const unsigned int *d_body,
                                     const Scalar *d_diameter,
                                     const unsigned int N,
                                     const unsigned int *d_cell_size,
                                     const Scalar4 *d_cell_xyzf,
                                     const Scalar4 *d_cell_tdb,
                                     const unsigned int *d_cell_adj,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Index2D& cadji,
                                     const BoxDim& box,
                                     const Scalar r_maxsq,
                                     const unsigned int threads_per_particle,
                                     const unsigned int block_size,
                                     bool filter_body,
                                     bool filter_diameter,
                                     const Scalar3& ghost_width)
    {
    launcher<max_threads_per_particle>(d_nlist,
                                   d_n_neigh,
                                   d_last_updated_pos,
                                   d_conditions,
                                   nli,
                                   d_pos,
                                   d_body,
                                   d_diameter,
                                   N,
                                   d_cell_size,
                                   d_cell_xyzf,
                                   d_cell_tdb,
                                   d_cell_adj,
                                   ci,
                                   cli,
                                   cadji,
                                   box,
                                   r_maxsq,
                                   sqrtf(r_maxsq),
                                   ghost_width,
                                   threads_per_particle,
                                   filter_diameter,
                                   filter_body,
                                   block_size
                                   );

    return cudaSuccess;
    }

// don't compile the 1x nlist kernel in double precision builds
#ifdef SINGLE_PRECISION
//! Texture for reading d_cell_adj
texture<unsigned int, 2, cudaReadModeElementType> cell_adj_tex;
//! Texture for reading d_cell_size
texture<unsigned int, 1, cudaReadModeElementType> cell_size_tex;
//! Texture for reading d_cell_xyzf
texture<Scalar4, 2, cudaReadModeElementType> cell_xyzf_tex;
//! Texture for reading d_cell_tdb
texture<Scalar4, 2, cudaReadModeElementType> cell_tdb_tex;

//! Kernel call for generating neighbor list on the GPU
/*! \tparam filter_flags Set bit 1 to enable body filtering. Set bit 2 to enable diameter filtering.
    \param d_nlist Neighbor list data structure to write
    \param d_n_neigh Number of neighbors to write
    \param d_last_updated_pos Particle positions at this update are written to this array
    \param d_conditions Conditions array for writing overflow condition
    \param nli Indexer to access \a d_nlist
    \param d_pos Particle positions
    \param d_body Particle body indices
    \param d_diameter Particle diameters
    \param N Number of particles
    \param ci Cell indexer for indexing cells
    \param box Simulation box dimensions
    \param r_maxsq The maximum radius for which to include particles as neighbors, squared
    \param r_max The maximum radius for which to include particles as neighbors
    \param ghost_width Width of ghost cell layer

    \note optimized for compute 1.x devices
*/
template<unsigned char filter_flags>
__global__ void gpu_compute_nlist_binned_1x_kernel(unsigned int *d_nlist,
                                                   unsigned int *d_n_neigh,
                                                   Scalar4 *d_last_updated_pos,
                                                   unsigned int *d_conditions,
                                                   const Index2D nli,
                                                   const Scalar4 *d_pos,
                                                   const unsigned int *d_body,
                                                   const Scalar *d_diameter,
                                                   const unsigned int N,
                                                   const Index3D ci,
                                                   const BoxDim box,
                                                   const float r_maxsq,
                                                   const float r_max,
                                                   const Scalar3 ghost_width)
    {
    bool filter_body = filter_flags & 1;
    bool filter_diameter = filter_flags & 2;

    // each thread is going to compute the neighbor list for a single particle
    int my_pidx = blockDim.x * blockIdx.x + threadIdx.x;

    // count the number of neighbors needed
    unsigned int n_neigh_needed = 0;

    // quit early if we are past the end of the array
    if (my_pidx >= N)
        return;

    // first, determine which bin this particle belongs to
    Scalar4 my_postype = d_pos[my_pidx];
    Scalar3 my_pos = make_scalar3(my_postype.x, my_postype.y, my_postype.z);

    unsigned int my_body = d_body[my_pidx];
    Scalar my_diameter = d_diameter[my_pidx];

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(my_pos,ghost_width);
    unsigned int ib = (unsigned int)(f.x * ci.getW());
    unsigned int jb = (unsigned int)(f.y * ci.getH());
    unsigned int kb = (unsigned int)(f.z * ci.getD());

    // need to handle the case where the particle is exactly at the box hi
    if (ib == ci.getW() && periodic.x)
        ib = 0;
    if (jb == ci.getH() && periodic.y)
        jb = 0;
    if (kb == ci.getD() && periodic.z)
        kb = 0;

    int my_cell = ci(ib,jb,kb);

    // each thread will determine the neighborlist of a single particle
    // count number of neighbors found so far in n_neigh
    int n_neigh = 0;

    // loop over all adjacent bins
    for (unsigned int cur_adj = 0; cur_adj < 27; cur_adj++)
        {
        int neigh_cell = tex2D(cell_adj_tex, cur_adj, my_cell);
        unsigned int size = tex1Dfetch(cell_size_tex, neigh_cell);

        Scalar4 next_xyzf = tex2D(cell_xyzf_tex, 0, neigh_cell);

        // now, we are set to loop through the array
        for (int cur_offset = 0; cur_offset < size; cur_offset++)
            {
            Scalar4 cur_xyzf = next_xyzf;
            next_xyzf = tex2D(cell_xyzf_tex, cur_offset+1, neigh_cell);
            Scalar4 cur_tdb = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
            if (filter_diameter || filter_body)
                cur_tdb = tex2D(cell_tdb_tex, cur_offset, neigh_cell);
            unsigned int neigh_body = __scalar_as_int(cur_tdb.z);
            Scalar neigh_diameter = cur_tdb.y;

            Scalar3 neigh_pos = make_scalar3(cur_xyzf.x,
                                           cur_xyzf.y,
                                           cur_xyzf.z);
            int cur_neigh = __scalar_as_int(cur_xyzf.w);

            // compute the distance between the two particles
            Scalar3 dx = my_pos - neigh_pos;

            // wrap the periodic boundary conditions
            dx = box.minImage(dx);
            // compute dr squared
            Scalar drsq = dot(dx,dx);


            bool excluded = (my_pidx == cur_neigh);

            if (filter_body && my_body != 0xffffffff)
                excluded = excluded | (my_body == neigh_body);

            Scalar sqshift = Scalar(0.0);
            if (filter_diameter)
                {
                // compute the shift in radius to accept neighbors based on their diameters
                Scalar delta = (my_diameter + neigh_diameter) * Scalar(0.5) - Scalar(1.0);
                // r^2 < (r_max + delta)^2
                // r^2 < r_maxsq + delta^2 + 2*r_max*delta
                sqshift = (delta + Scalar(2.0) * r_max) * delta;
                }

            if (drsq <= (r_maxsq + sqshift) && !excluded)
                {
                if (n_neigh < nli.getH())
                    d_nlist[nli(my_pidx, n_neigh)] = cur_neigh;
                else
                    n_neigh_needed = n_neigh+1;

                n_neigh++;
                }
            }
        }

    d_n_neigh[my_pidx] = n_neigh;
    d_last_updated_pos[my_pidx] = my_postype;

    if (n_neigh_needed > 0)
        atomicMax(&d_conditions[0], n_neigh_needed);
    }
#endif  // #ifdef SINGLE_PRECISION

cudaError_t gpu_compute_nlist_binned_1x(unsigned int *d_nlist,
                                        unsigned int *d_n_neigh,
                                        Scalar4 *d_last_updated_pos,
                                        unsigned int *d_conditions,
                                        const Index2D& nli,
                                        const Scalar4 *d_pos,
                                        const unsigned int *d_body,
                                        const Scalar *d_diameter,
                                        const unsigned int N,
                                        const unsigned int *d_cell_size,
                                        const cudaArray *dca_cell_xyzf,
                                        const cudaArray *dca_cell_tdb,
                                        const cudaArray *dca_cell_adj,
                                        const Index3D& ci,
                                        const BoxDim& box,
                                        const Scalar r_maxsq,
                                        const unsigned int block_size,
                                        bool filter_body,
                                        bool filter_diameter,
                                        const Scalar3& ghost_width)
    {
    // don't compile the 1x nlist kernel in double precision builds
    #ifdef SINGLE_PRECISION
    int n_blocks = (int)ceil(double(N)/double(block_size));

    cudaError_t err = cudaBindTextureToArray(cell_adj_tex, dca_cell_adj);
    if (err != cudaSuccess)
        return err;

    err = cudaBindTextureToArray(cell_xyzf_tex, dca_cell_xyzf);
    if (err != cudaSuccess)
        return err;

    err = cudaBindTextureToArray(cell_tdb_tex, dca_cell_tdb);
    if (err != cudaSuccess)
        return err;

    err = cudaBindTexture(0, cell_size_tex, d_cell_size, sizeof(unsigned int)*ci.getNumElements());
    if (err != cudaSuccess)
        return err;

    if (!filter_diameter && !filter_body)
        {
        gpu_compute_nlist_binned_1x_kernel<0><<<n_blocks, block_size>>>(d_nlist,
                                                                        d_n_neigh,
                                                                        d_last_updated_pos,
                                                                        d_conditions,
                                                                        nli,
                                                                        d_pos,
                                                                        d_body,
                                                                        d_diameter,
                                                                        N,
                                                                        ci,
                                                                        box,
                                                                        r_maxsq,
                                                                        sqrtf(r_maxsq),
                                                                        ghost_width);
        }
    if (!filter_diameter && filter_body)
        {
        gpu_compute_nlist_binned_1x_kernel<1><<<n_blocks, block_size>>>(d_nlist,
                                                                        d_n_neigh,
                                                                        d_last_updated_pos,
                                                                        d_conditions,
                                                                        nli,
                                                                        d_pos,
                                                                        d_body,
                                                                        d_diameter,
                                                                        N,
                                                                        ci,
                                                                        box,
                                                                        r_maxsq,
                                                                        sqrtf(r_maxsq),
                                                                        ghost_width);
        }
    if (filter_diameter && !filter_body)
        {
        gpu_compute_nlist_binned_1x_kernel<2><<<n_blocks, block_size>>>(d_nlist,
                                                                        d_n_neigh,
                                                                        d_last_updated_pos,
                                                                        d_conditions,
                                                                        nli,
                                                                        d_pos,
                                                                        d_body,
                                                                        d_diameter,
                                                                        N,
                                                                        ci,
                                                                        box,
                                                                        r_maxsq,
                                                                        sqrtf(r_maxsq),
                                                                        ghost_width);
        }
    if (filter_diameter && filter_body)
        {
        gpu_compute_nlist_binned_1x_kernel<3><<<n_blocks, block_size>>>(d_nlist,
                                                                        d_n_neigh,
                                                                        d_last_updated_pos,
                                                                        d_conditions,
                                                                        nli,
                                                                        d_pos,
                                                                        d_body,
                                                                        d_diameter,
                                                                        N,
                                                                        ci,
                                                                        box,
                                                                        r_maxsq,
                                                                        sqrtf(r_maxsq),
                                                                        ghost_width );
        }
    #endif // #ifdef SINGLE_PRECISION

    return cudaSuccess;
    }
