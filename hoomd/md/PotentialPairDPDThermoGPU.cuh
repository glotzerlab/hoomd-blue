// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file PotentialPairDPDThermoGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __POTENTIAL_PAIR_DPDTHERMO_CUH__
#define __POTENTIAL_PAIR_DPDTHERMO_CUH__

#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"
#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__
#include <cassert>

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_dpd_pair_force_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_dpd_pair_force_max_tpp = 64;
#endif

//! args struct for passing additional options to gpu_compute_dpd_forces
struct dpd_pair_args_t
    {
    //! Construct a dpd_pair_args_t
    dpd_pair_args_t(Scalar4* _d_force,
                    Scalar* _d_virial,
                    const size_t _virial_pitch,
                    const unsigned int _N,
                    const unsigned int _n_max,
                    const Scalar4* _d_pos,
                    const Scalar4* _d_vel,
                    const unsigned int* _d_tag,
                    const BoxDim& _box,
                    const unsigned int* _d_n_neigh,
                    const unsigned int* _d_nlist,
                    const size_t* _d_head_list,
                    const Scalar* _d_rcutsq,
                    const size_t _size_nlist,
                    const unsigned int _ntypes,
                    const unsigned int _block_size,
                    const uint16_t _seed,
                    const uint64_t _timestep,
                    const Scalar _deltaT,
                    const Scalar _T,
                    const unsigned int _shift_mode,
                    const unsigned int _compute_virial,
                    const unsigned int _threads_per_particle,
                    const hipDeviceProp_t& _devprop)
        : d_force(_d_force), d_virial(_d_virial), virial_pitch(_virial_pitch), N(_N), n_max(_n_max),
          d_pos(_d_pos), d_vel(_d_vel), d_tag(_d_tag), box(_box), d_n_neigh(_d_n_neigh),
          d_nlist(_d_nlist), d_head_list(_d_head_list), d_rcutsq(_d_rcutsq),
          size_nlist(_size_nlist), ntypes(_ntypes), block_size(_block_size), seed(_seed),
          timestep(_timestep), deltaT(_deltaT), T(_T), shift_mode(_shift_mode),
          compute_virial(_compute_virial), threads_per_particle(_threads_per_particle),
          devprop(_devprop) { };

    Scalar4* d_force;          //!< Force to write out
    Scalar* d_virial;          //!< Virial to write out
    const size_t virial_pitch; //!< Pitch of 2D virial array
    const unsigned int N;      //!< number of particles
    const unsigned int n_max;  //!< Maximum size of particle data arrays
    const Scalar4* d_pos;      //!< particle positions
    const Scalar4* d_vel;      //!< particle velocities
    const unsigned int* d_tag; //!< particle tags
    const BoxDim box;          //!< Simulation box in GPU format
    const unsigned int*
        d_n_neigh;               //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist; //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;   //!< Indexes for accessing d_nlist
    const Scalar* d_rcutsq;      //!< Device array listing r_cut squared per particle type pair
    const size_t size_nlist;     //!< Total length of the neighbor list
    const unsigned int ntypes;   //!< Number of particle types in the simulation
    const unsigned int block_size;     //!< Block size to execute
    const uint16_t seed;               //!< user provided seed for PRNG
    const uint64_t timestep;           //!< timestep of simulation
    const Scalar deltaT;               //!< timestep size
    const Scalar T;                    //!< temperature
    const unsigned int shift_mode;     //!< The potential energy shift mode
    const unsigned int compute_virial; //!< Flag to indicate if virials should be computed
    const unsigned int
        threads_per_particle; //!< Number of threads per particle (maximum: 32==1 warp)

    const hipDeviceProp_t& devprop; //!< Device properties
    };

#ifdef __HIPCC__

//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the
   potentials and forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param d_vel particle velocities on the GPU
    \param d_tag particle tags on the GPU
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for indexing \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_seed user defined seed for PRNG
    \param d_timestep timestep of simulation
    \param d_deltaT timestep size
    \param d_T temperature
    \param ntypes Number of types in the simulation
    \param tpp Number of threads per particle

    \a d_params, and \a d_rcutsq must be indexed with an Index2DUpperTriangular(typei, typej) to
   access the unique value for that type pair. These values are all cached into shared memory for
   quick access, so a dynamic amount of shared memory must be allocated for this kernel launch. The
   amount is (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) *
   typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they
   are not enabled. \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r \tparam
   shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. \tparam
   compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not
   computed.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
*/
template<class evaluator,
         unsigned int shift_mode,
         unsigned int compute_virial,
         unsigned char use_gmem_nlist,
         int tpp>
__global__ void gpu_compute_dpd_forces_kernel(Scalar4* d_force,
                                              Scalar* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const Scalar4* d_pos,
                                              const Scalar4* d_vel,
                                              const unsigned int* d_tag,
                                              BoxDim box,
                                              const unsigned int* d_n_neigh,
                                              const unsigned int* d_nlist,
                                              const size_t* d_head_list,
                                              const typename evaluator::param_type* d_params,
                                              const Scalar* d_rcutsq,
                                              const uint16_t d_seed,
                                              const uint64_t d_timestep,
                                              const Scalar d_deltaT,
                                              const Scalar d_T,
                                              const int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);
    Scalar* s_rcutsq
        = (Scalar*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // start by identifying which particle we are to handle
    const unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction (because of
        // __syncthreads())
        active = false;
        }

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    if (active)
        {
        // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
        unsigned int n_neigh = d_n_neigh[idx];

        // read in the position of our particle.
        // (MEM TRANSFER: 16 bytes)
        Scalar4 postypei = __ldg(d_pos + idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        // read in the velocity of our particle.
        // (MEM TRANSFER: 16 bytes)
        Scalar4 velmassi = __ldg(d_vel + idx);
        Scalar3 veli = make_scalar3(velmassi.x, velmassi.y, velmassi.z);

        // prefetch neighbor index
        const size_t head_idx = d_head_list[idx];
        unsigned int cur_j = 0;
        unsigned int next_j(0);
        next_j = (threadIdx.x % tpp < n_neigh) ? __ldg(d_nlist + head_idx + threadIdx.x % tpp) : 0;

        // this particle's tag
        unsigned int tagi = d_tag[idx];

        // loop over neighbors
        for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
                {
                // read the current neighbor index (MEM TRANSFER: 4 bytes)
                // prefetch the next value and set the current one
                cur_j = next_j;
                if (neigh_idx + tpp < n_neigh)
                    {
                    next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
                    }

                // get the neighbor's position (MEM TRANSFER: 16 bytes)
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                // get the neighbor's position (MEM TRANSFER: 16 bytes)
                Scalar4 velmassj = __ldg(d_vel + cur_j);
                Scalar3 velj = make_scalar3(velmassj.x, velmassj.y, velmassj.z);

                // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions: (FLOPS 12)
                dx = box.minImage(dx);

                // calculate r squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);

                // calculate dv (FLOPS: 3)
                Scalar3 dv = veli - velj;

                Scalar rdotv = dot(dx, dv);

                // access the per type pair parameters
                unsigned int typpair
                    = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
                Scalar rcutsq = s_rcutsq[typpair];
                typename evaluator::param_type& param = s_params[typpair];

                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                // or 2) shift mode is explor and ron > rcut
                bool energy_shift = false;
                if (shift_mode == 1)
                    energy_shift = true;

                evaluator eval(rsq, rcutsq, param);

                // evaluate the potential
                Scalar force_divr = Scalar(0.0);
                Scalar force_divr_cons = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);

                // Special Potential Pair DPD Requirements
                // use particle i's and j's tags
                unsigned int tagj = __ldg(d_tag + cur_j);
                eval.set_seed_ij_timestep(d_seed, tagi, tagj, d_timestep);
                eval.setDeltaT(d_deltaT);
                eval.setRDotV(rdotv);
                eval.setT(d_T);

                eval.evalForceEnergyThermo(force_divr, force_divr_cons, pair_eng, energy_shift);

                // calculate the virial (FLOPS: 3)
                if (compute_virial)
                    {
                    Scalar force_div2r_cons = Scalar(0.5) * force_divr_cons;
                    virial[0] += dx.x * dx.x * force_div2r_cons;
                    virial[1] += dx.x * dx.y * force_div2r_cons;
                    virial[2] += dx.x * dx.z * force_div2r_cons;
                    virial[3] += dx.y * dx.y * force_div2r_cons;
                    virial[4] += dx.y * dx.z * force_div2r_cons;
                    virial[5] += dx.z * dx.z * force_div2r_cons;
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

    // reduce force over threads in cta
    hoomd::detail::WarpReduce<Scalar, tpp> reducer;
    force.x = reducer.Sum(force.x);
    force.y = reducer.Sum(force.y);
    force.z = reducer.Sum(force.z);
    force.w = reducer.Sum(force.w);

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    if (active && threadIdx.x % tpp == 0)
        d_force[idx] = force;

    if (compute_virial)
        {
        for (unsigned int i = 0; i < 6; ++i)
            virial[i] = reducer.Sum(virial[i]);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x % tpp == 0)
            for (unsigned int i = 0; i < 6; i++)
                d_virial[i * virial_pitch + idx] = virial[i];
        }
    }

template<typename T> int dpd_get_max_block_size(T func)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_dpd_pair_force_max_tpp;
    return max_threads;
    }

//! DPD force compute kernel launcher
/*!
 * \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
 * \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR
 * switching is enabled (See PotentialPair for a discussion on what that entails) \tparam
 * compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not
 * computed. \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory.
 * When zero, textures or __ldg is used depending on architecture. \tparam tpp Number of threads to
 * use per particle, must be power of 2 and smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this
 * with a struct that we are allowed to partially specialize.
 */
template<class evaluator,
         unsigned int shift_mode,
         unsigned int compute_virial,
         unsigned int use_gmem_nlist,
         int tpp>
struct DPDForceComputeKernel
    {
    //! Launcher for the DPD force kernel
    /*!
     * \param args Other arguments to pass onto the kernel
     * \param d_params Parameters for the potential, stored per type pair
     */
    static void launch(const dpd_pair_args_t& args, const typename evaluator::param_type* d_params)
        {
        if (tpp == args.threads_per_particle)
            {
            // setup the grid to run the kernel
            unsigned int block_size = args.block_size;

            Index2D typpair_idx(args.ntypes);
            const size_t shared_bytes = (sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                        * typpair_idx.getNumElements();

            if (shared_bytes > args.devprop.sharedMemPerBlock)
                {
                throw std::runtime_error("Pair potential parameters exceed the available shared "
                                         "memory per block.");
                }

            unsigned int max_block_size;
            max_block_size = dpd_get_max_block_size(gpu_compute_dpd_forces_kernel<evaluator,
                                                                                  shift_mode,
                                                                                  compute_virial,
                                                                                  use_gmem_nlist,
                                                                                  tpp>);

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(args.N / (block_size / tpp) + 1, 1, 1);

            hipLaunchKernelGGL((gpu_compute_dpd_forces_kernel<evaluator,
                                                              shift_mode,
                                                              compute_virial,
                                                              use_gmem_nlist,
                                                              tpp>),
                               dim3(grid),
                               dim3(block_size),
                               shared_bytes,
                               0,
                               args.d_force,
                               args.d_virial,
                               args.virial_pitch,
                               args.N,
                               args.d_pos,
                               args.d_vel,
                               args.d_tag,
                               args.box,
                               args.d_n_neigh,
                               args.d_nlist,
                               args.d_head_list,
                               d_params,
                               args.d_rcutsq,
                               args.seed,
                               args.timestep,
                               args.deltaT,
                               args.T,
                               args.ntypes);
            }
        else
            {
            DPDForceComputeKernel<evaluator, shift_mode, compute_virial, use_gmem_nlist, tpp / 2>::
                launch(args, d_params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator,
         unsigned int shift_mode,
         unsigned int compute_virial,
         unsigned int use_gmem_nlist>
struct DPDForceComputeKernel<evaluator, shift_mode, compute_virial, use_gmem_nlist, 0>
    {
    static void launch(const dpd_pair_args_t& args, const typename evaluator::param_type* d_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes pair DPD thermo forces on the GPU
/*! \param args Additional options
    \param d_params Per type-pair parameters for the evaluator

    This is just a driver function for gpu_compute_dpd_forces_kernel(), see it for details.
*/
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_dpd_forces(const dpd_pair_args_t& args, const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(args.d_rcutsq);
    assert(args.ntypes > 0);

    // run the kernel
    if (args.compute_virial)
        {
        switch (args.shift_mode)
            {
        case 0:
            {
            DPDForceComputeKernel<evaluator, 0, 1, 0, gpu_dpd_pair_force_max_tpp>::launch(args,
                                                                                          d_params);
            break;
            }
        case 1:
            {
            DPDForceComputeKernel<evaluator, 1, 1, 0, gpu_dpd_pair_force_max_tpp>::launch(args,
                                                                                          d_params);
            break;
            }
        default:
            return hipErrorUnknown;
            }
        }
    else
        {
        switch (args.shift_mode)
            {
        case 0:
            {
            DPDForceComputeKernel<evaluator, 0, 0, 0, gpu_dpd_pair_force_max_tpp>::launch(args,
                                                                                          d_params);
            break;
            }
        case 1:
            {
            DPDForceComputeKernel<evaluator, 1, 0, 0, gpu_dpd_pair_force_max_tpp>::launch(args,
                                                                                          d_params);
            break;
            }
        default:
            return hipErrorUnknown;
            }
        }

    return hipSuccess;
    }
#else
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_dpd_forces(const dpd_pair_args_t& args, const typename evaluator::param_type* d_params);
#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIAL_PAIR_DPDTHERMO_CUH__
