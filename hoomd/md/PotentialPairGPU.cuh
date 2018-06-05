// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

#include <assert.h>

/*! \file PotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifndef __POTENTIAL_PAIR_GPU_CUH__
#define __POTENTIAL_PAIR_GPU_CUH__

//! Maximum number of threads (width of a warp)
const unsigned int gpu_pair_force_max_tpp = 32;


//! Wrapps arguments to gpu_cgpf
struct pair_args_t
    {
    //! Construct a pair_args_t
    pair_args_t(Scalar4 *_d_force,
              Scalar *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const unsigned int _n_max,
              const Scalar4 *_d_pos,
              const Scalar *_d_diameter,
              const Scalar *_d_charge,
              const BoxDim& _box,
              const unsigned int *_d_n_neigh,
              const unsigned int *_d_nlist,
              const unsigned int *_d_head_list,
              const Scalar *_d_rcutsq,
              const Scalar *_d_ronsq,
              const unsigned int _size_neigh_list,
              const unsigned int _ntypes,
              const unsigned int _block_size,
              const unsigned int _shift_mode,
              const unsigned int _compute_virial,
              const unsigned int _threads_per_particle,
              const unsigned int _compute_capability,
              const unsigned int _max_tex1d_width)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  N(_N),
                  n_max(_n_max),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  box(_box),
                  d_n_neigh(_d_n_neigh),
                  d_nlist(_d_nlist),
                  d_head_list(_d_head_list),
                  d_rcutsq(_d_rcutsq),
                  d_ronsq(_d_ronsq),
                  size_neigh_list(_size_neigh_list),
                  ntypes(_ntypes),
                  block_size(_block_size),
                  shift_mode(_shift_mode),
                  compute_virial(_compute_virial),
                  threads_per_particle(_threads_per_particle),
                  compute_capability(_compute_capability),
                  max_tex1d_width(_max_tex1d_width)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    Scalar *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;           //!< number of particles
    const unsigned int n_max;       //!< Max size of pdata arrays
    const Scalar4 *d_pos;           //!< particle positions
    const Scalar *d_diameter;       //!< particle diameters
    const Scalar *d_charge;         //!< particle charges
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const unsigned int *d_head_list;//!< Head list indexes for accessing d_nlist
    const Scalar *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const Scalar *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int size_neigh_list; //!< Size of the neighbor list for texture binding
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int shift_mode;  //!< The potential energy shift mode
    const unsigned int compute_virial;  //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads per particle (maximum: 1 warp)
    const unsigned int compute_capability;  //!< Compute capability (20 30 35, ...)
    const unsigned int max_tex1d_width;     //!< Maximum width of a linear 1D texture
    };

#ifdef NVCC

#if (__CUDA_ARCH__ >= 300)
// need this wrapper here for CUDA toolkit versions (<6.5) which do not provide a
// double specialization
__device__ inline
double __my_shfl_down(double var, unsigned int srcLane, int width=32)
    {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
    }

__device__ inline
float __my_shfl_down(float var, unsigned int srcLane, int width=32)
    {
    return __shfl_down(var, srcLane, width);
    }
#endif

//! CTA reduce, returns result in first thread
template<typename T>
__device__ static T warp_reduce(unsigned int NT, int tid, T x, volatile T* shared)
    {
    #if (__CUDA_ARCH__ < 300)
    shared[tid] = x;
    __syncthreads();
    #endif

    for (int dest_count = NT/2; dest_count >= 1; dest_count /= 2)
        {
        #if (__CUDA_ARCH__ < 300)
        if (tid < dest_count)
            {
            shared[tid] += shared[dest_count + tid];
            }
        __syncthreads();
        #else
        x += __my_shfl_down(x, dest_count, NT);
        #endif
        }

    #if (__CUDA_ARCH__ < 300)
    T total;
    if (tid == 0)
        {
        total = shared[0];
        }
    __syncthreads();
    return total;
    #else
    return x;
    #endif
    }

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;

//! Texture for reading particle diameters
scalar_tex_t pdata_diam_tex;

//! Texture for reading particle charges
scalar_tex_t pdata_charge_tex;

// there is some naming conflict between the DPD pair force and PotentialPair because
// the DPD does not extend PotentialPair, and so we need to choose a different name for this texture
//! Texture for reading neighbor list
texture<unsigned int, 1, cudaReadModeElementType> pair_nlist_tex;

//! Kernel for calculating pair forces (shared memory version)
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the potentials and
    forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles in system
    \param d_pos particle positions
    \param d_diameter particle diameters
    \param d_charge particle charges
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for reading \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_ronsq ron squared, stored per type pair
    \param ntypes Number of types in the simulation

    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocatd for this kernel launch. The amount is
    (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
    \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
                       (See PotentialPair for a discussion on what that entails)
    \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
    \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory. When zero, textures or __ldg
                           is used depending on architecture.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial, unsigned int use_gmem_nlist>
__global__ void gpu_compute_pair_forces_shared_kernel(Scalar4 *d_force,
                                               Scalar *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const Scalar *d_diameter,
                                               const Scalar *d_charge,
                                               const BoxDim box,
                                               const unsigned int *d_n_neigh,
                                               const unsigned int *d_nlist,
                                               const unsigned int *d_head_list,
                                               const typename evaluator::param_type *d_params,
                                               const Scalar *d_rcutsq,
                                               const Scalar *d_ronsq,
                                               const unsigned int ntypes,
                                               const unsigned int tpp)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params =
        (typename evaluator::param_type *)(&s_data[0]);
    Scalar *s_rcutsq = (Scalar *)(&s_data[num_typ_parameters*sizeof(evaluator::param_type)]);
    Scalar *s_ronsq = (Scalar *)(&s_data[num_typ_parameters*(sizeof(evaluator::param_type) + sizeof(Scalar))]);

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            if (shift_mode == 2)
                s_ronsq[cur_offset + threadIdx.x] = d_ronsq[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // start by identifying which particle we are to handle
    unsigned int idx;
    if (gridDim.y > 1)
        {
        // if we have blocks in the y-direction, the fermi-workaround is in place
        idx = (blockIdx.x + blockIdx.y * 65535) * (blockDim.x/tpp) + threadIdx.x/tpp;
        }
    else
        {
        idx = blockIdx.x * (blockDim.x/tpp) + threadIdx.x/tpp;
        }

    bool active = true;

    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction (because of __syncthreads())
        active = false;
        }

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virialxx = Scalar(0.0);
    Scalar virialxy = Scalar(0.0);
    Scalar virialxz = Scalar(0.0);
    Scalar virialyy = Scalar(0.0);
    Scalar virialyz = Scalar(0.0);
    Scalar virialzz = Scalar(0.0);

    if (active)
        {
        // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
        unsigned int n_neigh = d_n_neigh[idx];

        // read in the position of our particle.
        // (MEM TRANSFER: 16 bytes)
        Scalar4 postypei = texFetchScalar4(d_pos, pdata_pos_tex, idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        Scalar di;
        if (evaluator::needsDiameter())
            di = texFetchScalar(d_diameter, pdata_diam_tex, idx);
        else
            di += Scalar(1.0); // shutup compiler warning
        Scalar qi;
        if (evaluator::needsCharge())
            qi = texFetchScalar(d_charge, pdata_charge_tex, idx);
        else
            qi += Scalar(1.0); // shutup compiler warning

        unsigned int my_head = d_head_list[idx];
        unsigned int cur_j = 0;

        unsigned int next_j(0);
        if (use_gmem_nlist)
            {
            next_j = (threadIdx.x%tpp < n_neigh) ? d_nlist[my_head + threadIdx.x%tpp] : 0;
            }
        else
            {
            next_j = threadIdx.x%tpp < n_neigh ? texFetchUint(d_nlist, pair_nlist_tex, my_head + threadIdx.x%tpp) : 0;
            }

        // loop over neighbors
        for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
            {
                {
                // read the current neighbor index (MEM TRANSFER: 4 bytes)
                cur_j = next_j;
                if (neigh_idx+tpp < n_neigh)
                    {
                    if (use_gmem_nlist)
                        {
                        next_j = d_nlist[my_head + neigh_idx + tpp];
                        }
                    else
                        {
                        next_j = texFetchUint(d_nlist, pair_nlist_tex, my_head + neigh_idx+tpp);
                        }
                    }
                // get the neighbor's position (MEM TRANSFER: 16 bytes)
                Scalar4 postypej = texFetchScalar4(d_pos, pdata_pos_tex, cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                Scalar dj = Scalar(0.0);
                if (evaluator::needsDiameter())
                    dj = texFetchScalar(d_diameter, pdata_diam_tex, cur_j);
                else
                    dj += Scalar(1.0); // shutup compiler warning

                Scalar qj = Scalar(0.0);
                if (evaluator::needsCharge())
                    qj = texFetchScalar(d_charge, pdata_charge_tex, cur_j);
                else
                    qj += Scalar(1.0); // shutup compiler warning

                // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions: (FLOPS 12)
                dx = box.minImage(dx);

                // calculate r squard (FLOPS: 5)
                Scalar rsq = dot(dx, dx);

                // access the per type pair parameters
                unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
                Scalar rcutsq = s_rcutsq[typpair];
                typename evaluator::param_type param = s_params[typpair];
                Scalar ronsq = Scalar(0.0);
                if (shift_mode == 2)
                    ronsq = s_ronsq[typpair];

                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                // or 2) shift mode is explor and ron > rcut
                bool energy_shift = false;
                if (shift_mode == 1)
                    energy_shift = true;
                else if (shift_mode == 2)
                    {
                    if (ronsq > rcutsq)
                        energy_shift = true;
                    }

                // evaluate the potential
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);

                evaluator eval(rsq, rcutsq, param);
                if (evaluator::needsDiameter())
                    eval.setDiameter(di, dj);
                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);

                eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);

                if (shift_mode == 2)
                    {
                    if (rsq >= ronsq && rsq < rcutsq)
                        {
                        // Implement XPLOR smoothing (FLOPS: 16)
                        Scalar old_pair_eng = pair_eng;
                        Scalar old_force_divr = force_divr;

                        // calculate 1.0 / (xplor denominator)
                        Scalar xplor_denom_inv =
                            Scalar(1.0) / ((rcutsq - ronsq) * (rcutsq - ronsq) * (rcutsq - ronsq));

                        Scalar rsq_minus_r_cut_sq = rsq - rcutsq;
                        Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq *
                                   (rcutsq + Scalar(2.0) * rsq - Scalar(3.0) * ronsq) * xplor_denom_inv;
                        Scalar ds_dr_divr = Scalar(12.0) * (rsq - ronsq) * rsq_minus_r_cut_sq * xplor_denom_inv;

                        // make modifications to the old pair energy and force
                        pair_eng = old_pair_eng * s;
                        force_divr = s * old_force_divr - ds_dr_divr * old_pair_eng;
                        }
                    }
                // calculate the virial
                if (compute_virial)
                    {
                    Scalar force_div2r = Scalar(0.5) * force_divr;
                    virialxx +=  dx.x * dx.x * force_div2r;
                    virialxy +=  dx.x * dx.y * force_div2r;
                    virialxz +=  dx.x * dx.z * force_div2r;
                    virialyy +=  dx.y * dx.y * force_div2r;
                    virialyz +=  dx.y * dx.z * force_div2r;
                    virialzz +=  dx.z * dx.z * force_div2r;
                    }

                // add up the force vector components (FLOPS: 7)
                force.x += dx.x * force_divr;
                force.y += dx.y * force_divr;
                force.z += dx.z * force_divr;

                force.w += pair_eng;
                }
            }

        // potential energy per particle must be halved
        force.w *= Scalar(0.5);
        }

    // we need to access a separate portion of shared memory to avoid race conditions
    const unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements();

    // need to declare as volatile, because we are using warp-synchronous programming
    volatile Scalar *sh = (Scalar *) &s_data[shared_bytes];

    unsigned int cta_offs = (threadIdx.x/tpp)*tpp;

    // reduce force over threads in cta
    force.x = warp_reduce(tpp, threadIdx.x % tpp, force.x, &sh[cta_offs]);
    force.y = warp_reduce(tpp, threadIdx.x % tpp, force.y, &sh[cta_offs]);
    force.z = warp_reduce(tpp, threadIdx.x % tpp, force.z, &sh[cta_offs]);
    force.w = warp_reduce(tpp, threadIdx.x % tpp, force.w, &sh[cta_offs]);

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    if (active && threadIdx.x % tpp == 0)
        d_force[idx] = force;

    if (compute_virial)
        {
        virialxx = warp_reduce(tpp, threadIdx.x % tpp, virialxx, &sh[cta_offs]);
        virialxy = warp_reduce(tpp, threadIdx.x % tpp, virialxy, &sh[cta_offs]);
        virialxz = warp_reduce(tpp, threadIdx.x % tpp, virialxz, &sh[cta_offs]);
        virialyy = warp_reduce(tpp, threadIdx.x % tpp, virialyy, &sh[cta_offs]);
        virialyz = warp_reduce(tpp, threadIdx.x % tpp, virialyz, &sh[cta_offs]);
        virialzz = warp_reduce(tpp, threadIdx.x % tpp, virialzz, &sh[cta_offs]);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x %tpp == 0)
            {
            d_virial[0*virial_pitch+idx] = virialxx;
            d_virial[1*virial_pitch+idx] = virialxy;
            d_virial[2*virial_pitch+idx] = virialxz;
            d_virial[3*virial_pitch+idx] = virialyy;
            d_virial[4*virial_pitch+idx] = virialyz;
            d_virial[5*virial_pitch+idx] = virialzz;
            }
        }
    }

template<typename T>
int get_max_block_size(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void *)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_pair_force_max_tpp;
    return max_threads;
    }

inline void gpu_pair_force_bind_textures(const pair_args_t pair_args)
    {
    // bind the position texture
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, pdata_pos_tex, pair_args.d_pos, sizeof(Scalar4)*pair_args.n_max);

    // bind the diamter texture
    pdata_diam_tex.normalized = false;
    pdata_diam_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, pdata_diam_tex, pair_args.d_diameter, sizeof(Scalar) * pair_args.n_max);

    pdata_charge_tex.normalized = false;
    pdata_charge_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, pdata_charge_tex, pair_args.d_charge, sizeof(Scalar) * pair_args.n_max);

    // bind the neighborlist texture if it will fit
    if (pair_args.size_neigh_list <= pair_args.max_tex1d_width)
        {
        pair_nlist_tex.normalized = false;
        pair_nlist_tex.filterMode = cudaFilterModePoint;
        cudaBindTexture(0, pair_nlist_tex, pair_args.d_nlist, sizeof(unsigned int) * pair_args.size_neigh_list);
        }
    }

inline void gpu_pair_force_unbind_textures(const pair_args_t pair_args)
    {
    cudaUnbindTexture(pdata_pos_tex);
    cudaUnbindTexture(pdata_diam_tex);
    cudaUnbindTexture(pdata_charge_tex);

    if (pair_args.size_neigh_list <= pair_args.max_tex1d_width)
        {
        cudaUnbindTexture(pair_nlist_tex);
        }
    }

//! Kernel launcher to compact templated kernel launches
/*!
 * \param pair_args Other arugments to pass onto the kernel
 * \param d_params Parameters for the potential, stored per type pair
 *
 * \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
 * \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
 *                       (See PotentialPair for a discussion on what that entails)
 * \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
 * \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory. When zero, textures or __ldg
 *                        is used depending on architecture.
 */
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial, unsigned int use_gmem_nlist>
inline void launch_compute_pair_force_kernel(const pair_args_t& pair_args,
                                             const typename evaluator::param_type *d_params)
    {
    unsigned int block_size = pair_args.block_size;
    unsigned int tpp = pair_args.threads_per_particle;

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, shift_mode, compute_virial, use_gmem_nlist>);

    if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

    block_size = block_size < max_block_size ? block_size : max_block_size;
    dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
    if (pair_args.compute_capability < 30 && grid.x > 65535)
        {
        grid.y = grid.x/65535 + 1;
        grid.x = 65535;
        }

    if (pair_args.compute_capability < 30)
        {
        shared_bytes += sizeof(Scalar)*block_size;
        }

    gpu_compute_pair_forces_shared_kernel<evaluator, shift_mode, compute_virial, use_gmem_nlist>
      <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
      pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
      pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
      pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
      tpp);

    if (pair_args.compute_capability < 35) gpu_pair_force_unbind_textures(pair_args);
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_pair_forces_shared_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_pair_forces(const pair_args_t& pair_args,
                                    const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    // Launch kernel
    if (pair_args.compute_capability < 35 && pair_args.size_neigh_list > pair_args.max_tex1d_width)
        { // fall back to slow global loads when the neighbor list is too big for texture memory
        if (pair_args.compute_virial)
            {
            switch (pair_args.shift_mode)
                {
                case 0:
                    {
                    launch_compute_pair_force_kernel<evaluator, 0, 1, 1>(pair_args, d_params);
                    break;
                    }
                case 1:
                    {
                    launch_compute_pair_force_kernel<evaluator, 1, 1, 1>(pair_args, d_params);
                    break;
                    }
                case 2:
                    {
                    launch_compute_pair_force_kernel<evaluator, 2, 1, 1>(pair_args, d_params);
                    break;
                    }
                default:
                    break;
                }
            }
        else
            {
            switch (pair_args.shift_mode)
                {
                case 0:
                    {
                    launch_compute_pair_force_kernel<evaluator, 0, 0, 1>(pair_args, d_params);
                    break;
                    }
                case 1:
                    {
                    launch_compute_pair_force_kernel<evaluator, 1, 0, 1>(pair_args, d_params);
                    break;
                    }
                case 2:
                    {
                    launch_compute_pair_force_kernel<evaluator, 2, 0, 1>(pair_args, d_params);
                    break;
                    }
                default:
                    break;
                }
            }
        }
    else
        {
        if (pair_args.compute_virial)
            {
            switch (pair_args.shift_mode)
                {
                case 0:
                    {
                    launch_compute_pair_force_kernel<evaluator, 0, 1, 0>(pair_args, d_params);
                    break;
                    }
                case 1:
                    {
                    launch_compute_pair_force_kernel<evaluator, 1, 1, 0>(pair_args, d_params);
                    break;
                    }
                case 2:
                    {
                    launch_compute_pair_force_kernel<evaluator, 2, 1, 0>(pair_args, d_params);
                    break;
                    }
                default:
                    break;
                }
            }
        else
            {
            switch (pair_args.shift_mode)
                {
                case 0:
                    {
                    launch_compute_pair_force_kernel<evaluator, 0, 0, 0>(pair_args, d_params);
                    break;
                    }
                case 1:
                    {
                    launch_compute_pair_force_kernel<evaluator, 1, 0, 0>(pair_args, d_params);
                    break;
                    }
                case 2:
                    {
                    launch_compute_pair_force_kernel<evaluator, 2, 0, 0>(pair_args, d_params);
                    break;
                    }
                default:
                    break;
                }
            }
        }

    return cudaSuccess;
    }
#endif
#endif // __POTENTIAL_PAIR_GPU_CUH__
