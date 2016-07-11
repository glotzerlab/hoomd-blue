// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPU.cuh"

#include "Moves.h"
#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeSpheropolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"
#include "hoomd/TextureTools.h"

#include <stdio.h>

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoGPU.cu
    \brief Definition of CUDA kernels and drivers for IntegratorHPMCMono
*/

//! Texture for reading postype
scalar4_tex_t postype_tex;
//! Texture for reading orientation
scalar4_tex_t orientation_tex;
//! Texture for reading cell index data
texture<unsigned int, 1, cudaReadModeElementType> cell_idx_tex;

//! Compute the cell that a particle sits in
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
                                                   const BoxDim& box,
                                                   const Scalar3& ghost_width,
                                                   const uint3& cell_dim,
                                                   const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    if (f.x >= Scalar(0.0) && f.x < Scalar(1.0) && f.y >= Scalar(0.0) && f.y < Scalar(1.0) && f.z >= Scalar(0.0) && f.z < Scalar(1.0))
        return ci(ib,jb,kb);
    else
        return 0xffffffff;
    }

//! Kernel to generate expanded cells
/*! \param d_excell_idx Output array to list the particle indices in the expanded cells
    \param d_excell_size Output array to list the number of particles in each expanded cell
    \param excli Indexer for the expanded cells
    \param d_cell_idx Particle indices in the normal cells
    \param d_cell_size Number of particles in each cell
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer
    \param cli Cell list indexer
    \param cadji Cell adjacency indexer

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void gpu_hpmc_excell_kernel(unsigned int *d_excell_idx,
                                       unsigned int *d_excell_size,
                                       const Index2D excli,
                                       const unsigned int *d_cell_idx,
                                       const unsigned int *d_cell_size,
                                       const unsigned int *d_cell_adj,
                                       const Index3D ci,
                                       const Index2D cli,
                                       const Index2D cadji)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    if (gridDim.y > 1)
        {
        // if gridDim.y > 1, then the fermi workaround is in place, index blocks on a 2D grid
        my_cell = (blockIdx.x + blockIdx.y * 65535) * blockDim.x + threadIdx.x;
        }
    else
        {
        my_cell = blockDim.x * blockIdx.x + threadIdx.x;
        }

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];
        unsigned int neigh_cell_size = d_cell_size[neigh_cell];

        for (unsigned int k = 0; k < neigh_cell_size; k++)
            {
            // read in the index of the new particle to add to our cell
            unsigned int new_idx = tex1Dfetch(cell_idx_tex, cli(k, neigh_cell));
            d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
            my_cell_size++;
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
    }

//! Kernel driver for gpu_hpmc_excell_kernel()
cudaError_t gpu_hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static int sm = -1;
    if (max_block_size == -1)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_hpmc_excell_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        sm = attr.binaryVersion;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid(ci.getNumElements() / block_size + 1, 1, 1);

    // hack to enable grids of more than 65k blocks
    if (sm < 30 && grid.x > 65535)
        {
        grid.y = grid.x / 65535 + 1;
        grid.x = 65535;
        }

    // bind the textures
    cell_idx_tex.normalized = false;
    cell_idx_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, cell_idx_tex, d_cell_idx, sizeof(unsigned int)*cli.getNumElements());
    if (error != cudaSuccess)
        return error;

    gpu_hpmc_excell_kernel<<<grid, threads>>>(d_excell_idx,
                                              d_excell_size,
                                              excli,
                                              d_cell_idx,
                                              d_cell_size,
                                              d_cell_adj,
                                              ci,
                                              cli,
                                              cadji);

    return cudaSuccess;
    }


//! HPMC  update kernel
/*! \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_counters Acceptance counters to increment
    \param d_cell_idx Particle index stored in the cell list
    \param d_cell_size The size of each cell
    \param d_excell_idx Indices of particles in extended cells
    \param d_excell_size Number of particles in each extended cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param excli Extended cell list indexer
    \param cell_dim Dimensions of the cell list
    \param ghost_width Width of the ghost layer
    \param d_cell_set List of active cells
    \param n_active_cells Number of active cells
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param d_d Array of maximum move displacements
    \param d_a Array of rotation move sizes
    \param d_check_overlaps Interaction matrix
    \parma overlap_idx Indexer into interaction matrix
    \param move_ratio Ratio of translation moves to rotation moves
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param select Current index within the loop over nselect selections (for RNG generation)
    \param ghost_fraction Width of the inactive layer in MPI domain decomposition simulations
    \param domain_decomposition True if executing with domain decomposition
    \param d_params Per-type shape parameters

    MPMC in its published form has a severe limit on the number of parallel threads in 3D. This implementation launches
    group_size threads per cell (1,2,4,8,16,32). Each thread in the group performs the same trial move on the same
    particle, and then checks for overlaps against different particles from the extended cell list. The entire extended
    cell list is covered in a batched loop. The group_size is autotuned to find the fastest performance. Smaller systems
    tend to run fastest with a large group_size due to the increased parallelism. Larger systems tend to run faster
    at smaller group_sizes because they already have the parallelism from the system size - however, even the largest
    systems benefit from group_size > 1 on K20. Shared memory is used to set an overlap flag to 1 if any of the threads
    in the group detect an overlap. After all checks are complete, the master thread in the group applies the trial move
    update if accepted.

    No __synchtreads is needed after the overlap checks because the group_size is always chosen to be a power of 2 and
    smaller than the warp size. Only a __threadfence_block() is needed to ensure memory consistency.

    Move stats are tallied in local memory, then totaled in shared memory at the end and finally a single thread in the
    block runs an atomicAdd on global memory to get the system wide total. This isn't as good as a reduction, but it
    is only a tiny fraction of the compute time.

    In order to simplify indexing and boundary checks, a list of active cells is determined on the host and passed into
    the kernel. That way, only a linear indexing of threads is needed to handle any geometry of active cells.

    Heavily divergent warps are avoided by pre-building a list of all particles in the neighboring region of any given
    cell. Otherwise, extremely non-uniform cell lengths (i.e. avg 1, max 4) don't cause massive performance degradation.

    **Indexing**
        - threadIdx.z indexes the current group in the block
        - threadIdx.x is the offset within the current group
        - blockIdx.x runs enough blocks so that all active cells are covered

    **Possible enhancements**
        - Use __ldg and not tex1Dfetch on sm35

    \ingroup hpmc_kernels
*/
template< class Shape >
__global__ void gpu_hpmc_mpmc_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int n_active_cells,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const Scalar* d_d,
                                     const Scalar* d_a,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     unsigned int *d_active_cell_ptl_idx,
                                     unsigned int *d_active_cell_accept,
                                     unsigned int *d_active_cell_move_type_translate,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size)
    {
    // flags to tell what type of thread we are
    bool active = true;
    unsigned int group;
    unsigned int offset;
    unsigned int group_size;
    bool master;
    unsigned int n_groups;

    if (Shape::isParallel())
        {
        // use 3d thread block layout
        group = threadIdx.z;
        offset = threadIdx.y;
        group_size = blockDim.y;
        master = (offset == 0 && threadIdx.x == 0);
        n_groups = blockDim.z;
        }
    else
        {
        group = threadIdx.y;
        offset = threadIdx.x;
        group_size = blockDim.x;
        master = (offset == 0);
        n_groups = blockDim.y;
        }

    unsigned int err_count = 0;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    Scalar *s_d = (Scalar *)(s_pos_group + n_groups);
    Scalar *s_a = (Scalar *)(s_d + num_types);
    unsigned int *s_check_overlaps = (unsigned int *) (s_a + num_types);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_overlap =   (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }
    if (master)
        {
        s_overlap[group] = 0;
        }

    // identify the active cell that this thread handles
    unsigned int active_cell_idx = 0;
    if (gridDim.y > 1)
        {
        // if gridDim.y > 1, then the fermi workaround is in place, index blocks on a 2D grid
        active_cell_idx = (blockIdx.x + blockIdx.y * 65535) * n_groups + group;
        }
    else
        {
        active_cell_idx = blockIdx.x * n_groups + group;
        }


    // this thread is inactive if it indexes past the end of the active cell list
    if (active_cell_idx >= n_active_cells)
        active = false;

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;
    if (active)
        {
        my_cell = d_cell_set[active_cell_idx];
        my_cell_size = d_cell_size[my_cell];
        }

    // need to deactivate if there are no particles in this cell
    if (my_cell_size == 0)
        active = false;

    __syncthreads();

    // initial implementation just moves one particle per cell (nselect=1).
    // these variables are ugly, but needed to get the updated quantities outside of the scope
    unsigned int i;
    unsigned int overlap_checks = 0;
    bool move_type_translate = false;
    bool move_active = true;
    int ignore_stats = 0;

    if (active)
        {
        // one RNG per cell
        SaruGPU rng(my_cell, seed+select, timestep);

        // select one of the particles randomly from the cell
        unsigned int my_cell_offset = rand_select(rng, my_cell_size-1);
        i = tex1Dfetch(cell_idx_tex, cli(my_cell_offset, my_cell));

        // read in the position and orientation of our particle.
        Scalar4 postype_i = texFetchScalar4(d_postype, postype_tex, i);
        Scalar4 orientation_i = make_scalar4(1,0,0,0);

        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

        if (shape_i.hasOrientation())
            orientation_i = texFetchScalar4(d_orientation, orientation_tex, i);

        shape_i.orientation = quat<Scalar>(orientation_i);

        // if this looks funny, that is because it is. Using ignore_stats as a bool setting ignore_stats = ...
        // causes a compiler bug.
        if (shape_i.ignoreStatistics())
            ignore_stats = 1;

        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // for domain decomposition simulations, we need to leave all particles in the inactive region alone
        // in order to avoid even more divergence, this is done by setting the move_active flag
        // overlap checks are still processed, but the final move acceptance will be skipped
        if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
            move_active = false;

        // make the move
        unsigned int move_type_select = rng.u32() & 0xffff;
        move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

        if (move_type_translate)
            {
            move_translate(pos_i, rng, s_d[typ_i], dim);

            // need to reject any move that puts the particle in the inactive region
            if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                move_active = false;
            }
        else
            {
            move_rotate(shape_i.orientation, rng, s_a[typ_i], dim);
            }

        // stash the trial move in shared memory so that other threads in this block can process overlap checks
        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = typ_i;
            s_orientation_group[group] = quat_to_scalar4(shape_i.orientation);
            }
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    // loop while still searching
    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < excell_size)
                {
                #if (__CUDA_ARCH__ > 300)
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                #else
                next_j = d_excell_idx[excli(k, my_cell)];
                #endif
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found
            while (!s_overlap[group] && s_queue_size < max_queue_size && k < excell_size)
                {
                if (k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j;
                    vec3<Scalar> r_ij;

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        {
                        #if (__CUDA_ARCH__ > 300)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                        #else
                        next_j = d_excell_idx[excli(k, my_cell)];
                        #endif
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = texFetchScalar4(d_postype, postype_tex, j);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[__scalar_as_int(postype_j.w)]);

                    // put particle j into the coordinate system of particle i
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        }

                    } // end if k < excell_size
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j = s_queue_j[tidx_1d];
            Scalar4 postype_j;
            Scalar4 orientation_j;
            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            postype_j = texFetchScalar4(d_postype, postype_tex, check_j);
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, orientation_tex, check_j));

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)] && test_overlap(r_ij, shape_i, shape_j, err_count))
                {
                atomicAdd(&s_overlap[check_group], 1);
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

    // update the data if accepted
    if (master)
        {
        if (active && move_active)
            {
            // first need to check if the particle remains in its cell
            Scalar3 xnew_i = s_pos_group[group];
            unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width, cell_dim, ci);
            bool accepted=true;
            if (s_overlap[group])
                accepted=false;
            if (new_cell != my_cell)
                accepted=false;

            if (accepted)
                {
                // write out the updated position and orientation
                d_postype[i] = make_scalar4(xnew_i.x, xnew_i.y, xnew_i.z, __int_as_scalar(s_type_group[group]));
                d_orientation[i] = s_orientation_group[group];
                }

            if (d_active_cell_accept)
                {
                // store particle index
                d_active_cell_ptl_idx[active_cell_idx] = i;
                }

            if (d_active_cell_accept)
                {
                // store accept flag
                d_active_cell_accept[active_cell_idx] = accepted ? 1 : 0;
                }

            if (d_active_cell_move_type_translate)
                {
                // store move type
                d_active_cell_move_type_translate[active_cell_idx] = move_type_translate ? 1 : 0;
                }

            // if an auxillary array was provided, defer writing out statistics
            if (d_active_cell_ptl_idx)
                {
                ignore_stats = 1;
                }

            if (!ignore_stats && accepted && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accepted && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accepted && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accepted && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        else // active && move_active
            {
            if (d_active_cell_ptl_idx && active_cell_idx < n_active_cells)
                {
                // indicate that no particle was selected
                d_active_cell_ptl_idx[active_cell_idx] = UINT_MAX;
                }
            }

        // count the overlap checks
        atomicAdd(&s_overlap_checks, overlap_checks);
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        }
    }

//! Kernel for grid shift
/*! \param d_postype postype of each particle
    \param d_image Image flags for each particle
    \param N number of particles
    \param box Simulation box
    \param shift Vector by which to translate the particles

    Shift all the particles by a given vector.

    \ingroup hpmc_kernels
*/
__global__ void gpu_hpmc_shift_kernel(Scalar4 *d_postype,
                                      int3 *d_image,
                                      const unsigned int N,
                                      const BoxDim box,
                                      const Scalar3 shift)
    {
    // identify the active cell that this thread handles
    unsigned int my_pidx = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread is inactive if it indexes past the end of the particle list
    if (my_pidx >= N)
        return;

    // pull in the current position
    Scalar4 postype = d_postype[my_pidx];

    // shift the position
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    pos += shift;

    // wrap the particle back into the box
    int3 image = d_image[my_pidx];
    box.wrap(pos, image);

    // write out the new position and orientation
    d_postype[my_pidx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    d_image[my_pidx] = image;
    }


//! Kernel driver for gpu_update_hpmc_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);
    assert(args.block_size%(args.stride*args.group_size)==0);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static int sm = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_mpmc_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        sm = attr.binaryVersion;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // the new block size might not be a multiple of group size, decrease group size until it is
    group_size = args.group_size;

    while ((block_size%(args.stride*group_size)) != 0)
        {
        group_size--;
        }

    unsigned int n_groups = block_size / group_size / args.stride;
    unsigned int shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                                block_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                                args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
                                args.overlap_idx.getNumElements() * sizeof(unsigned int);

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;

        // the new block size might not be a multiple of group size, decrease group size until it is
        group_size = args.group_size;

        while ((block_size%(args.stride*group_size)) != 0)
            {
            group_size--;
            }

        n_groups = block_size / group_size / args.stride;
        shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                       block_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                       args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
                       args.overlap_idx.getNumElements() * sizeof(unsigned int);
        }

    // setup the grid to run the kernel
    dim3 threads;
    if (Shape::isParallel())
        {
        // use three-dimensional thread-layout with blockDim.z < 64
        threads = dim3(args.stride, group_size, n_groups);
        }
    else
        {
        threads = dim3(group_size, n_groups,1);
        }

    dim3 grid( args.n_active_cells / n_groups + 1, 1, 1);

    // hack to enable grids of more than 65k blocks
    if (sm < 30 && grid.x > 65535)
        {
        grid.y = grid.x / 65535 + 1;
        grid.x = 65535;
        }

    // bind the textures
    postype_tex.normalized = false;
    postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    if (args.has_orientation)
        {
        orientation_tex.normalized = false;
        orientation_tex.filterMode = cudaFilterModePoint;
        error = cudaBindTexture(0, orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
        if (error != cudaSuccess)
            return error;
        }

    cell_idx_tex.normalized = false;
    cell_idx_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, cell_idx_tex, args.d_cell_idx, sizeof(Scalar4)*args.cli.getNumElements());
    if (error != cudaSuccess)
        return error;

    gpu_hpmc_mpmc_kernel<Shape><<<grid, threads, shared_bytes>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.d_excell_idx,
                                                                 args.d_excell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.excli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.n_active_cells,
                                                                 args.N,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.d_d,
                                                                 args.d_a,
                                                                 args.d_check_overlaps,
                                                                 args.overlap_idx,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 args.d_active_cell_ptl_idx,
                                                                 args.d_active_cell_accept,
                                                                 args.d_active_cell_move_type_translate,
                                                                 d_params,
                                                                 block_size);

    return cudaSuccess;
    }

//! Kernel driver for gpu_hpmc_shift_kernel()
cudaError_t gpu_hpmc_shift(Scalar4 *d_postype,
                           int3 *d_image,
                           const unsigned int N,
                           const BoxDim& box,
                           const Scalar3 shift,
                           const unsigned int block_size)
    {
    assert(d_postype);
    assert(d_image);

    // setup the grid to run the kernel
    dim3 threads_shift(block_size, 1, 1);
    dim3 grid_shift(N / block_size + 1, 1, 1);

    gpu_hpmc_shift_kernel<<<grid_shift, threads_shift>>>(d_postype,
                                                         d_image,
                                                         N,
                                                         box,
                                                         shift);

    return cudaSuccess;
    }

// Instantiate shape templates

//! HPMC update for ShapeSphere
template cudaError_t gpu_hpmc_update<ShapeSphere>(const hpmc_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);

//! HPMC update for ShapeConvexPolygon
template cudaError_t gpu_hpmc_update<ShapeConvexPolygon>(const hpmc_args_t& args,
                                                         const typename ShapeConvexPolygon::param_type *d_params);

//! HPMC update for ShapePolyhedron
template cudaError_t gpu_hpmc_update<ShapePolyhedron>(const hpmc_args_t& args,
                                                      const typename ShapePolyhedron::param_type *d_params);

//! HPMC update for ShapeConvexPolyhedron
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron<8> >(const hpmc_args_t& args,
                                                            const typename ShapeConvexPolyhedron<8> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron<16> >(const hpmc_args_t& args,
                                                            const typename ShapeConvexPolyhedron<16> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron<32> >(const hpmc_args_t& args,
                                                            const typename ShapeConvexPolyhedron<32> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron<64> >(const hpmc_args_t& args,
                                                            const typename ShapeConvexPolyhedron<64> ::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeConvexPolyhedron<128> >(const hpmc_args_t& args,
                                                            const typename ShapeConvexPolyhedron<128> ::param_type *d_params);

//! HPMC update for ShapeSpheropolyhedron
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<8> >(const hpmc_args_t& args,
                                                            const typename ShapeSpheropolyhedron<8>::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<16> >(const hpmc_args_t& args,
                                                            const typename ShapeSpheropolyhedron<16>::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<32> >(const hpmc_args_t& args,
                                                            const typename ShapeSpheropolyhedron<32>::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<64> >(const hpmc_args_t& args,
                                                            const typename ShapeSpheropolyhedron<64>::param_type *d_params);
template cudaError_t gpu_hpmc_update<ShapeSpheropolyhedron<128> >(const hpmc_args_t& args,
                                                            const typename ShapeSpheropolyhedron<128>::param_type *d_params);

//! HPMC update for ShapeSimplePolygon
template cudaError_t gpu_hpmc_update<ShapeSimplePolygon>(const hpmc_args_t& args,
                                                         const typename ShapeSimplePolygon::param_type *d_params);

//! HPMC update for ShapeEllipsoid
template cudaError_t gpu_hpmc_update<ShapeEllipsoid>(const hpmc_args_t& args,
                                                     const typename ShapeEllipsoid::param_type *d_params);

//! HPMC update for ShapeSpheropolygon
template cudaError_t gpu_hpmc_update<ShapeSpheropolygon>(const hpmc_args_t& args,
                                                         const typename ShapeSpheropolygon::param_type *d_params);

//! HPMC update for ShapeFacetedSphere
template cudaError_t gpu_hpmc_update<ShapeFacetedSphere>(const hpmc_args_t& args,
                                                        const typename ShapeFacetedSphere::param_type *d_params);

#ifdef ENABLE_SPHINX_GPU
//! HPMC update for ShapeSphinx
template cudaError_t gpu_hpmc_update<ShapeSphinx>(const hpmc_args_t& args,
                                                  const typename ShapeSphinx::param_type *d_params);
#endif

//! HPMC update for ShapeUnion<ShapeSphere>
template cudaError_t gpu_hpmc_update< ShapeUnion<ShapeSphere> >(const hpmc_args_t& args,
                                                  const typename ShapeUnion<ShapeSphere>::param_type *d_params);

}; // end namespace detail

} // end namespace hpmc
