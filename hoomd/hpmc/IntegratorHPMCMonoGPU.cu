#include "IntegratorHPMCMonoGPU.cuh"
#include "hoomd/RandomNumbers.h"

namespace hpmc
{
namespace gpu
{
namespace kernel
{

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
    \param ngpu Number of active devices

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D ci,
                            const Index2D cli,
                            const Index2D cadji,
                            const unsigned int ngpu)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    my_cell = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];

        // iterate over per-device cell lists
        for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
            {
            unsigned int neigh_cell_size = d_cell_size[neigh_cell+igpu*ci.getNumElements()];

            for (unsigned int k = 0; k < neigh_cell_size; k++)
                {
                // read in the index of the new particle to add to our cell
                unsigned int new_idx = d_cell_idx[cli(k, neigh_cell)+igpu*cli.getNumElements()];
                d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
                my_cell_size++;
                }
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
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
__global__ void hpmc_shift(Scalar4 *d_postype,
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

//!< Kernel to accept/reject
__global__ void hpmc_accept(const unsigned int *d_ptl_by_update_order,
                 const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject,
                 unsigned int *d_reject_out,
                 const unsigned int *d_nneigh,
                 const unsigned int *d_nlist,
                 const unsigned int N_old,
                 const unsigned int N,
                 const unsigned int maxn,
                 bool patch,
                 const unsigned int *d_nlist_patch_old,
                 const unsigned int *d_nlist_patch_new,
                 const unsigned int *d_nneigh_patch_old,
                 const unsigned int *d_nneigh_patch_new,
                 const float *d_energy_old,
                 const float *d_energy_new,
                 const unsigned int maxn_patch,
                 unsigned int *d_condition,
                 const unsigned int seed,
                 const unsigned int select,
                 const unsigned int timestep)
    {
    unsigned int update_order_i = blockIdx.x*blockDim.x + threadIdx.x;

    if (update_order_i >= N)
        return;

    unsigned int i = d_ptl_by_update_order[update_order_i];

    bool move_active = d_trial_move_type[i] > 0;

    // has the particle move not been rejected yet?
    if (move_active)
        {
        // iterate over overlapping neighbors in old configuration
        unsigned int nneigh = d_nneigh[i];
        bool accept = !d_reject_out_of_cell[i];
        for (unsigned int cur_neigh = 0; cur_neigh < nneigh; cur_neigh++)
            {
            unsigned int primitive = d_nlist[cur_neigh+maxn*i];

            unsigned int j = primitive;
            bool old = true;
            if (j >= N_old)
                {
                j -= N_old;
                old = false;
                }

            // has j been updated? ghost particles are not updated
            bool j_has_been_updated = j < N && d_trial_move_type[j]
                && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

            // acceptance, reject if current configuration of particle overlaps
            if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                {
                accept = false;
                break;
                }

            } // end loop over neighbors

        float delta_U = 0.0;
        if (patch)
            {
            // iterate over overlapping neighbors in old configuration
            float energy_old = 0.0f;
            unsigned int nneigh = d_nneigh_patch_old[i];
            for (unsigned int cur_neigh = 0; cur_neigh < nneigh; cur_neigh++)
                {
                unsigned int primitive = d_nlist_patch_old[cur_neigh+maxn_patch*i];

                unsigned int j = primitive;
                bool old = true;
                if (j >= N_old)
                    {
                    j -= N_old;
                    old = false;
                    }

                // has j been updated? ghost particles are not updated
                bool j_has_been_updated = j < N && d_trial_move_type[j]
                    && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                // acceptance, reject if current configuration of particle overlaps
                if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                    {
                    energy_old += d_energy_old[cur_neigh+maxn_patch*i];
                    }

                } // end loop over neighbors

            // iterate over overlapping neighbors in new configuration
            float energy_new = 0.0f;
            nneigh = d_nneigh_patch_new[i];
            for (unsigned int cur_neigh = 0; cur_neigh < nneigh; cur_neigh++)
                {
                unsigned int primitive = d_nlist_patch_new[cur_neigh+maxn_patch*i];

                unsigned int j = primitive;
                bool old = true;
                if (j >= N_old)
                    {
                    j -= N_old;
                    old = false;
                    }

                // has j been updated? ghost particles are not updated
                bool j_has_been_updated = j < N && d_trial_move_type[j]
                    && d_update_order_by_ptl[j] < update_order_i && !d_reject[j];

                // acceptance, reject if current configuration of particle overlaps
                if ((old && !j_has_been_updated) || (!old && j_has_been_updated))
                    {
                    energy_new += d_energy_new[cur_neigh+maxn_patch*i];
                    }

                } // end loop over neighbors

            delta_U = energy_new - energy_old;
            }

        // Metropolis-Hastings
        hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::HPMCMonoAccept, seed, i, select, timestep);
        accept = accept && (!patch || (hoomd::detail::generate_canonical<double>(rng_i) < slow::exp(-delta_U)));

        if ((accept && d_reject[i]) || (!accept && !d_reject[i]))
            {
            // flag that we're not done yet
            atomicAdd(d_condition,1);
            }

        // write out to device memory
        d_reject_out[i] = accept ? 0 : 1;
        }  // end if move_active
    }

//! Compute energy in old and new configuration of every particle
__global__ void hpmc_narrow_phase_patch(Scalar4 *d_postype,
                           Scalar4 *d_orientation,
                           Scalar4 *d_trial_postype,
                           Scalar4 *d_trial_orientation,
                           const Scalar *d_charge,
                           const Scalar *d_diameter,
                           const unsigned int *d_excell_idx,
                           const unsigned int *d_excell_size,
                           const Index2D excli,
                           unsigned int *d_nlist,
                           float *d_energy,
                           unsigned int *d_nneigh,
                           const unsigned int maxn,
                           const unsigned int num_types,
                           const BoxDim box,
                           const Scalar3 ghost_width,
                           const uint3 cell_dim,
                           const Index3D ci,
                           const unsigned int N_old,
                           const unsigned int N_new,
                           bool old_config,
                           Scalar r_cut_patch,
                           const Scalar *d_additive_cutoff,
                           unsigned int *d_overflow,
                           const unsigned int max_extra_bytes,
                           const unsigned int max_queue_size,
                           const unsigned int work_offset,
                           const unsigned int nwork,
                           eval_func evaluator)
    {
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.x;
    unsigned int group_size = blockDim.x;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    Scalar4 *s_orientation_group = (Scalar4*)(&s_data[0]);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    Scalar *s_diameter_group = (Scalar *)(s_pos_group + n_groups);
    Scalar *s_charge_group = (Scalar *)(s_diameter_group + n_groups);
    Scalar *s_additive_cutoff = (Scalar *)(s_charge_group + n_groups);
    unsigned int *s_queue_j =   (unsigned int*)(s_additive_cutoff + num_types);
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int *s_idx_group = (unsigned int*)(s_type_group + n_groups);
    unsigned int *s_nneigh_group = (unsigned int *)(s_idx_group + n_groups);

        {
        // copy over parameters one value per thread for fast loads
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_additive_cutoff[cur_offset + tidx] = d_additive_cutoff[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    #if 0
    // initialize extra shared mem
    char *s_extra = (char *)(s_nneigh_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();
    #endif

    if (master && group == 0)
        {
        s_queue_size = 0;
        s_still_searching = 1;
        }

    bool active = true;
    unsigned int idx = blockIdx.x*n_groups+group;
    if (idx >= nwork)
        active = false;
    idx += work_offset;

    if (master && active)
        {
        // reset number of neighbors
        s_nneigh_group[group] = 0;
        }

    __syncthreads();

    unsigned int my_cell;

    if (active)
        {
        // load particle i
        Scalar4 postype_i(old_config ? d_postype[idx] : d_trial_postype[idx]);
        vec3<Scalar> pos_i(postype_i);
        unsigned int type_i = __scalar_as_int(postype_i.w);

        // find the cell this particle should be in
        vec3<Scalar> pos_i_old(d_postype[idx]);
        my_cell = computeParticleCell(vec_to_scalar3(pos_i_old), box, ghost_width, cell_dim, ci);

        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = type_i;
            s_orientation_group[group] = old_config ? d_orientation[idx] : d_trial_orientation[idx];
            s_diameter_group[group] = d_diameter[idx];
            s_charge_group[group] = d_charge[idx];
            s_idx_group[group] = idx;
            }
        }

     // sync so that s_postype_group and s_orientation are available before other threads might process energy evaluations
     __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;

    // true if we are checking against the old configuration
    if (active)
        {
        excell_size = d_excell_size[my_cell];
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
            if ((k >> 1) < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // every thread can add at most one element to the neighbor list
            while (s_queue_size < max_queue_size && (k >> 1) < excell_size)
                {
                // build some shapes, but we only need them to get diameters, so don't load orientations
                // build shape i from shared memory
                vec3<Scalar> pos_i(s_pos_group[group]);
                unsigned int type_i = s_type_group[group];

                bool old = k & 1;

                // prefetch next j
                j = next_j;
                k += group_size;
                if ((k >> 1) < excell_size)
                    {
                    next_j = __ldg(&d_excell_idx[excli(k >> 1, my_cell)]);
                    }

                // check particle circumspheres

                // load particle j (always load ghosts from particle data)
                const Scalar4 postype_j = (old || j >= N_new) ? d_postype[j] : d_trial_postype[j];
                unsigned int type_j = __scalar_as_int(postype_j.w);
                vec3<Scalar> pos_j(postype_j);

                // place ourselves into the minimum image
                vec3<Scalar> r_ij = pos_j - pos_i;
                r_ij = box.minImage(r_ij);

                OverlapReal rcut = r_cut_patch + Scalar(0.5)*(s_additive_cutoff[type_i] + s_additive_cutoff[type_j]);
                OverlapReal rsq = dot(r_ij,r_ij);

                if (idx != j && (old || j < N_new)
                    && (rsq <= rcut*rcut))
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                    if (insert_point < max_queue_size)
                        {
                        s_queue_gid[insert_point] = group;
                        s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
                        }
                    else
                        {
                        // or back up if the queue is already full
                        // we will recheck and insert this on the next time through
                        k -= group_size;
                        }
                    }
                } // end while (s_queue_size < max_queue_size && (k>>1) < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process energy evaluations
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the energy evaluation to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j_flag = s_queue_j[tidx_1d];
            bool check_old = check_j_flag & 1;
            unsigned int check_j  = check_j_flag >> 1;

            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Scalar4 orientation_i  = s_orientation_group[check_group];
            Scalar d_i = s_diameter_group[check_group];
            Scalar q_i = s_charge_group[check_group];

            // build shape j from global memory
            Scalar4 postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
            Scalar4 orientation_j = check_old ? d_orientation[check_j] : d_trial_orientation[check_j];
            Scalar d_j = d_diameter[check_j];
            Scalar q_j = d_charge[check_j];
            unsigned int type_j = __scalar_as_int(postype_j.w);

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            // evaluate energy
            float energy = (*evaluator)(r_ij, type_i, quat<float>(orientation_i), d_i, q_i, type_j, quat<float>(orientation_j), d_j, q_j);

            // store energy term in global memory
            unsigned int n = atomicAdd(&s_nneigh_group[check_group], 1);
            if (n < maxn)
                {
                unsigned int pidx = s_idx_group[check_group];
                d_nlist[n+maxn*pidx] = check_old ? check_j : (check_j + N_old);
                d_energy[n+maxn*pidx] = energy;
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && (k >> 1) < excell_size)
            atomicAdd(&s_still_searching, 1);

        __syncthreads();
        } // end while (s_still_searching)

    if (active && master)
        {
        // overflowed?
        unsigned int nneigh = s_nneigh_group[group];
        if (nneigh > maxn)
            {
            #if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
            #else
            atomicMax(d_overflow, nneigh);
            #endif
            }

        // write out number of neighbors to global mem
        d_nneigh[idx] = nneigh;
        }
    }
} // end namespace kernel

//! Driver for kernel::hpmc_excell()
void hpmc_excell(unsigned int *d_excell_idx,
                 unsigned int *d_excell_size,
                 const Index2D& excli,
                 const unsigned int *d_cell_idx,
                 const unsigned int *d_cell_size,
                 const unsigned int *d_cell_adj,
                 const Index3D& ci,
                 const Index2D& cli,
                 const Index2D& cadji,
                 const unsigned int ngpu,
                 const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_excell));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid(ci.getNumElements() / block_size + 1, 1, 1);

    hipLaunchKernelGGL(kernel::hpmc_excell, dim3(grid), dim3(threads), 0, 0, d_excell_idx,
                                           d_excell_size,
                                           excli,
                                           d_cell_idx,
                                           d_cell_size,
                                           d_cell_adj,
                                           ci,
                                           cli,
                                           cadji,
                                           ngpu);

    }

//! Kernel driver for kernel::hpmc_shift()
void hpmc_shift(Scalar4 *d_postype,
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

    hipLaunchKernelGGL(kernel::hpmc_shift, dim3(grid_shift), dim3(threads_shift), 0, 0, d_postype,
                                                      d_image,
                                                      N,
                                                      box,
                                                      shift);

    // after this kernel we return control of cuda managed memory to the host
    hipDeviceSynchronize();
    }


void hpmc_accept(const unsigned int *d_ptl_by_update_order,
                 const unsigned int *d_update_order_by_ptl,
                 const unsigned int *d_trial_move_type,
                 const unsigned int *d_reject_out_of_cell,
                 unsigned int *d_reject,
                 unsigned int *d_reject_out,
                 const unsigned int *d_nneigh,
                 const unsigned int *d_nlist,
                 const unsigned int N_old,
                 const unsigned int N,
                 const unsigned int maxn,
                 bool patch,
                 const unsigned int *d_nlist_patch_old,
                 const unsigned int *d_nlist_patch_new,
                 const unsigned int *d_nneigh_patch_old,
                 const unsigned int *d_nneigh_patch_new,
                 const float *d_energy_old,
                 const float *d_energy_new,
                 const unsigned int maxn_patch,
                 unsigned int *d_condition,
                 const unsigned int seed,
                 const unsigned int select,
                 const unsigned int timestep,
                 const unsigned int block_size)
    {
    // launch kernel in a single thread
    hipMemset(d_condition, 0, sizeof(unsigned int));

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_accept));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid((N+block_size-1)/block_size,1,1);

    hipLaunchKernelGGL(kernel::hpmc_accept, dim3(grid), dim3(threads), 0, 0, d_ptl_by_update_order,
        d_update_order_by_ptl,
        d_trial_move_type,
        d_reject_out_of_cell,
        d_reject,
        d_reject_out,
        d_nneigh,
        d_nlist,
        N_old,
        N,
        maxn,
        patch,
        d_nlist_patch_old,
        d_nlist_patch_new,
        d_nneigh_patch_old,
        d_nneigh_patch_new,
        d_energy_old,
        d_energy_new,
        maxn_patch,
        d_condition,
        seed,
        select,
        timestep);

    // update reject flags
    hipMemcpyAsync(d_reject, d_reject_out, sizeof(unsigned int)*N, hipMemcpyDeviceToDevice);
    }

//! Kernel driver for kernel::hpmc_narrow_phase_patch
void hpmc_narrow_phase_patch(const hpmc_args_t& args, const hpmc_patch_args_t& patch_args)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel::hpmc_narrow_phase_patch));
        max_block_size = attr.maxThreadsPerBlock;
        }

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = min(args.block_size, (unsigned int)max_block_size);

    unsigned int tpp = min(args.tpp,run_block_size);
    unsigned int n_groups = run_block_size/tpp;
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar);

    unsigned int shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        run_block_size -= args.devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = min(tpp, run_block_size);
        n_groups = run_block_size / tpp;
        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (3*sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    unsigned int max_extra_bytes = 0;
    #if 0
    // determine dynamically allocated shared memory size
    static unsigned int base_shared_bytes = UINT_MAX;
    bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
    base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
    static unsigned int extra_bytes = UINT_MAX;
    if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
        {
        // required for memory coherency
        cudaDeviceSynchronize();

        // determine dynamically requested shared memory
        char *ptr = (char *)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;
        }

    shared_bytes += extra_bytes;
    #endif

    dim3 thread(tpp, n_groups, 1);

    for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = args.gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        hipLaunchKernelGGL((kernel::hpmc_narrow_phase_patch), grid, thread, shared_bytes, 0,
            args.d_postype, args.d_orientation, args.d_trial_postype, args.d_trial_orientation,
            patch_args.d_charge, patch_args.d_diameter, args.d_excell_idx, args.d_excell_size, args.excli,
            patch_args.d_nlist, patch_args.d_energy, patch_args.d_nneigh, patch_args.maxn, args.num_types,
            args.box, args.ghost_width, args.cell_dim, args.ci, args.N + args.N_ghost, args.N,
            patch_args.old_config, patch_args.r_cut_patch, patch_args.d_additive_cutoff,
            patch_args.d_overflow, max_extra_bytes, max_queue_size, range.first, nwork,
            patch_args.evaluators[idev]);
        }
    }

} // end namespace gpu
} // end namespace hpmc

