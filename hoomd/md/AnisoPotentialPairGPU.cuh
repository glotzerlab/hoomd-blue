// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

/*! \file AnisoPotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the anisotropic ptl pair forces and
   torques
*/

#ifndef __ANISO_POTENTIAL_PAIR_GPU_CUH__
#define __ANISO_POTENTIAL_PAIR_GPU_CUH__

//! Maximum number of threads (width of a warp)
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_aniso_pair_force_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_aniso_pair_force_max_tpp = 64;
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Wraps arguments to gpu_cgpf
struct a_pair_args_t
    {
    //! Construct a pair_args_t
    a_pair_args_t(Scalar4* _d_force,
                  Scalar4* _d_torque,
                  Scalar* _d_virial,
                  size_t _virial_pitch,
                  const unsigned int _N,
                  const unsigned int _n_max,
                  const Scalar4* _d_pos,
                  const Scalar* _d_charge,
                  const Scalar4* _d_orientation,
                  const unsigned int* _d_tag,
                  const BoxDim& _box,
                  const unsigned int* _d_n_neigh,
                  const unsigned int* _d_nlist,
                  const size_t* _d_head_list,
                  const Scalar* _d_rcutsq,
                  const unsigned int _ntypes,
                  const unsigned int _block_size,
                  const unsigned int _shift_mode,
                  const unsigned int _compute_virial,
                  const unsigned int _threads_per_particle,
                  const GPUPartition& _gpu_partition,
                  const hipDeviceProp_t& _devprop,
                  bool _update_shape_param)
        : d_force(_d_force), d_torque(_d_torque), d_virial(_d_virial), virial_pitch(_virial_pitch),
          N(_N), n_max(_n_max), d_pos(_d_pos), d_charge(_d_charge), d_orientation(_d_orientation),
          d_tag(_d_tag), box(_box), d_n_neigh(_d_n_neigh), d_nlist(_d_nlist),
          d_head_list(_d_head_list), d_rcutsq(_d_rcutsq), ntypes(_ntypes), block_size(_block_size),
          shift_mode(_shift_mode), compute_virial(_compute_virial),
          threads_per_particle(_threads_per_particle), gpu_partition(_gpu_partition),
          devprop(_devprop), update_shape_param(_update_shape_param) { };

    Scalar4* d_force;             //!< Force to write out
    Scalar4* d_torque;            //!< Torque to write out
    Scalar* d_virial;             //!< Virial to write out
    const size_t virial_pitch;    //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;         //!< number of particles
    const unsigned int n_max;     //!< maximum size of particle data arrays
    const Scalar4* d_pos;         //!< particle positions
    const Scalar* d_charge;       //!< particle charges
    const Scalar4* d_orientation; //!< particle orientation to compute forces over
    const unsigned int* d_tag;    //!< particle tags to compute forces over
    const BoxDim box;             //!< Simulation box in GPU format
    const unsigned int*
        d_n_neigh;               //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist; //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;   //!< Device array listing beginning of each particle's neighbors
    const Scalar* d_rcutsq;      //!< Device array listing r_cut squared per particle type pair
    const unsigned int ntypes;   //!< Number of particle types in the simulation
    const unsigned int block_size;           //!< Block size to execute
    const unsigned int shift_mode;           //!< The potential energy shift mode
    const unsigned int compute_virial;       //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads to launch per particle
    const GPUPartition& gpu_partition; //!< The load balancing partition of particles between GPUs
    const hipDeviceProp_t& devprop;    //!< CUDA device properties
    bool update_shape_param; //!< If true, update size of shape param and synchronize GPU execution
                             //!< stream
    };

#ifdef __HIPCC__

//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the
   potentials and forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_torque Device memory to write computed torques
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles in system
    \param d_pos particle positions
    \param d_charge particle charges
    \param d_orientation Quaternion data on the GPU to calculate forces on
    \param d_tag Tag data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Device memory array listing beginning of each particle's neighbors
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param ntypes Number of types in the simulation
    \param tpp Number of threads per particle

    \a d_params and \a d_rcutsq must be indexed with an Index2DUpperTriangular(typei, typej) to
   access the unique value for that type pair. These values are all cached into shared memory for
   quick access, so a dynamic amount of shared memory must be allocated for this kernel launch. The
   amount is (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) *
   typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they
   are not enabled. \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r \tparam
   shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching
   is enabled (See PotentialPair for a discussion on what that entails) \tparam compute_virial When
   non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial, int tpp>
__global__ void
gpu_compute_pair_aniso_forces_kernel(Scalar4* d_force,
                                     Scalar4* d_torque,
                                     Scalar* d_virial,
                                     const size_t virial_pitch,
                                     const unsigned int N,
                                     const Scalar4* d_pos,
                                     const Scalar* d_charge,
                                     const Scalar4* d_orientation,
                                     const unsigned int* d_tag,
                                     const BoxDim box,
                                     const unsigned int* d_n_neigh,
                                     const unsigned int* d_nlist,
                                     const size_t* d_head_list,
                                     const typename evaluator::param_type* d_params,
                                     const typename evaluator::shape_type* d_shape_params,
                                     const Scalar* d_rcutsq,
                                     const unsigned int ntypes,
                                     const unsigned int offset,
                                     unsigned int max_extra_bytes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);
    Scalar* s_rcutsq
        = (Scalar*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);
    typename evaluator::shape_type* s_shape_params
        = (typename evaluator::shape_type*)(&s_rcutsq[num_typ_parameters]);

    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            }
        }

    unsigned int param_size
        = num_typ_parameters * sizeof(typename evaluator::param_type) / sizeof(int);
    for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < param_size)
            {
            ((int*)s_params)[cur_offset + threadIdx.x] = ((int*)d_params)[cur_offset + threadIdx.x];
            }
        }

    unsigned int shape_param_size = sizeof(typename evaluator::shape_type) * ntypes / sizeof(int);
    for (unsigned int cur_offset = 0; cur_offset < shape_param_size; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < shape_param_size)
            {
            ((int*)s_shape_params)[cur_offset + threadIdx.x]
                = ((int*)d_shape_params)[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // initialize extra shared mem
    char* s_extra = (char*)(s_shape_params + ntypes);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_pair = 0; cur_pair < typpair_idx.getNumElements(); ++cur_pair)
        s_params[cur_pair].load_shared(s_extra, available_bytes);

    for (unsigned int cur_type = 0; cur_type < ntypes; ++cur_type)
        s_shape_params[cur_type].load_shared(s_extra, available_bytes);

    // start by identifying which particle we are to handle
    unsigned int idx;
    idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction
        active = false;
        }

    // particle index
    idx += offset;

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
    Scalar4 torque = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
    Scalar virialxx = Scalar(0);
    Scalar virialxy = Scalar(0);
    Scalar virialxz = Scalar(0);
    Scalar virialyy = Scalar(0);
    Scalar virialyz = Scalar(0);
    Scalar virialzz = Scalar(0);

    if (active)
        {
        // load in the length of the neighbor list
        unsigned int n_neigh = d_n_neigh[idx];

        // read in the position of our particle
        Scalar4 postypei = __ldg(d_pos + idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
        Scalar4 quati = __ldg(d_orientation + idx);

        Scalar qi = Scalar(0);
        if (evaluator::needsCharge())
            qi = __ldg(d_charge + idx);

        size_t my_head = d_head_list[idx];
        unsigned int cur_j = 0;

        unsigned int next_j(0);
        next_j = threadIdx.x % tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x % tpp) : 0;

        // loop over neighbors
        for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
            {
                {
                // read the current neighbor index
                // prefetch the next value and set the current one
                cur_j = next_j;
                if (neigh_idx + tpp < n_neigh)
                    {
                    next_j = __ldg(d_nlist + my_head + neigh_idx + tpp);
                    }

                // get the neighbor's position
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);
                Scalar4 quatj = __ldg(d_orientation + cur_j);

                Scalar qj = Scalar(0);
                if (evaluator::needsCharge())
                    qj = __ldg(d_charge + cur_j);

                // calculate dr (with periodic boundary conditions)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions
                dx = box.minImage(dx);

                // calculate r squared
                Scalar rsq = dot(dx, dx);

                // access the per type pair parameters
                unsigned int typpair
                    = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
                Scalar rcutsq = s_rcutsq[typpair];
                const typename evaluator::param_type& param = s_params[typpair];

                // design specifies that energies are shifted if
                // 1) shift mode is set to shift
                bool energy_shift = false;
                if (shift_mode == 1)
                    energy_shift = true;

                // evaluate the potential
                Scalar3 jforce = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar3 torquei = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar3 torquej = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar pair_eng = Scalar(0);

                // constructor call
                evaluator eval(dx, quati, quatj, rcutsq, param);
                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);
                if (evaluator::needsShape())
                    eval.setShape(&(s_shape_params[__scalar_as_int(postypei.w)]),
                                  &(s_shape_params[__scalar_as_int(postypej.w)]));
                if (evaluator::needsTags())
                    eval.setTags(__ldg(d_tag + idx), __ldg(d_tag + cur_j));

                // call evaluator
                eval.evaluate(jforce, pair_eng, energy_shift, torquei, torquej);

                // calculate the virial
                if (compute_virial)
                    {
                    Scalar3 jforce2 = Scalar(0.5) * jforce;
                    virialxx += dx.x * jforce2.x;
                    virialxy += dx.y * jforce2.x;
                    virialxz += dx.z * jforce2.x;
                    virialyy += dx.y * jforce2.y;
                    virialyz += dx.z * jforce2.y;
                    virialzz += dx.z * jforce2.z;
                    }

                // add up the force vector components
                force.x += jforce.x;
                force.y += jforce.y;
                force.z += jforce.z;
                torque.x += torquei.x;
                torque.y += torquei.y;
                torque.z += torquei.z;

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

    torque.x = reducer.Sum(torque.x);
    torque.y = reducer.Sum(torque.y);
    torque.z = reducer.Sum(torque.z);

    // now that the force calculation is complete, write out the result
    if (active && threadIdx.x % tpp == 0)
        {
        d_force[idx] = force;
        d_torque[idx] = torque;
        }

    if (compute_virial)
        {
        virialxx = reducer.Sum(virialxx);
        virialxy = reducer.Sum(virialxy);
        virialxz = reducer.Sum(virialxz);
        virialyy = reducer.Sum(virialyy);
        virialyz = reducer.Sum(virialyz);
        virialzz = reducer.Sum(virialzz);

        // if we are the first thread in the cta, write out virial to global mem
        if (active && threadIdx.x % tpp == 0)
            {
            d_virial[0 * virial_pitch + idx] = virialxx;
            d_virial[1 * virial_pitch + idx] = virialxy;
            d_virial[2 * virial_pitch + idx] = virialxz;
            d_virial[3 * virial_pitch + idx] = virialyy;
            d_virial[4 * virial_pitch + idx] = virialyz;
            d_virial[5 * virial_pitch + idx] = virialzz;
            }
        }
    }

//! Aniso pair force compute kernel launcher
/*!
 * \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r
 * \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut.
 * \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor
 * is not computed. \tparam tpp Number of threads to use per particle, must be power of 2 and
 * smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this
 * with a struct that we are allowed to partially specialize.
 */
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial, int tpp>
struct AnisoPairForceComputeKernel
    {
    //! Launcher for the pair force kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param range Range of particle indices this GPU operates on
     * \param params Parameters for the potential, stored per type pair
     * \param shape_params Parameters for the potential, stored per type pair
     */

    static void launch(const a_pair_args_t& pair_args,
                       std::pair<unsigned int, unsigned int> range,
                       const typename evaluator::param_type* params,
                       const typename evaluator::shape_type* shape_params)
        {
        unsigned int N = range.second - range.first;
        unsigned int offset = range.first;

        if (tpp == pair_args.threads_per_particle)
            {
            unsigned int block_size = pair_args.block_size;

            Index2D typpair_idx(pair_args.ntypes);
            size_t shared_bytes = (2 * sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                      * typpair_idx.getNumElements()
                                  + sizeof(typename evaluator::shape_type) * pair_args.ntypes;

            unsigned int max_block_size;
            hipFuncAttributes attr;
            hipFuncGetAttributes(
                &attr,
                reinterpret_cast<const void*>(&gpu_compute_pair_aniso_forces_kernel<evaluator,
                                                                                    shift_mode,
                                                                                    compute_virial,
                                                                                    tpp>));
            int max_threads = attr.maxThreadsPerBlock;
            // number of threads has to be multiple of warp size
            max_block_size = max_threads - max_threads % gpu_aniso_pair_force_max_tpp;

            unsigned int base_shared_bytes;
            base_shared_bytes = (unsigned int)(shared_bytes + attr.sharedSizeBytes);

            if (base_shared_bytes > pair_args.devprop.sharedMemPerBlock)
                {
                throw std::runtime_error("Pair potential parameters exceed the available shared "
                                         "memory per block.");
                }

            unsigned int max_extra_bytes
                = (unsigned int)(pair_args.devprop.sharedMemPerBlock - base_shared_bytes);
            unsigned int extra_bytes;
            // determine dynamically requested shared memory
            char* ptr = (char*)nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < typpair_idx.getNumElements(); ++i)
                {
                params[i].allocate_shared(ptr, available_bytes);
                }
            for (unsigned int i = 0; i < pair_args.ntypes; ++i)
                {
                shape_params[i].allocate_shared(ptr, available_bytes);
                }
            extra_bytes = max_extra_bytes - available_bytes;

            shared_bytes += extra_bytes;

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size / tpp) + 1, 1, 1);

            hipLaunchKernelGGL(
                (gpu_compute_pair_aniso_forces_kernel<evaluator, shift_mode, compute_virial, tpp>),
                dim3(grid),
                dim3(block_size),
                shared_bytes,
                0,
                pair_args.d_force,
                pair_args.d_torque,
                pair_args.d_virial,
                pair_args.virial_pitch,
                N,
                pair_args.d_pos,
                pair_args.d_charge,
                pair_args.d_orientation,
                pair_args.d_tag,
                pair_args.box,
                pair_args.d_n_neigh,
                pair_args.d_nlist,
                pair_args.d_head_list,
                params,
                shape_params,
                pair_args.d_rcutsq,
                pair_args.ntypes,
                offset,
                max_extra_bytes);
            }
        else
            {
            AnisoPairForceComputeKernel<evaluator, shift_mode, compute_virial, tpp / 2>::launch(
                pair_args,
                range,
                params,
                shape_params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial>
struct AnisoPairForceComputeKernel<evaluator, shift_mode, compute_virial, 0>
    {
    static void launch(const a_pair_args_t& pair_args,
                       std::pair<unsigned int, unsigned int> range,
                       const typename evaluator::param_type* d_params,
                       const typename evaluator::shape_type* shape_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes lj forces on the GPU for AnisoPotentialPairGPU
/*! \param pair_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_pair_aniso_forces_kernel(), see it for details.
*/
template<class evaluator>
hipError_t gpu_compute_pair_aniso_forces(const a_pair_args_t& pair_args,
                                         const typename evaluator::param_type* d_params,
                                         const typename evaluator::shape_type* d_shape_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = pair_args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = pair_args.gpu_partition.getRangeAndSetGPU(idev);

        // run the kernel
        if (pair_args.compute_virial)
            {
            switch (pair_args.shift_mode)
                {
            case 0:
                {
                AnisoPairForceComputeKernel<evaluator, 0, 1, gpu_aniso_pair_force_max_tpp>::launch(
                    pair_args,
                    range,
                    d_params,
                    d_shape_params);
                break;
                }
            case 1:
                {
                AnisoPairForceComputeKernel<evaluator,
                                            1 && evaluator::implementsEnergyShift(),
                                            1,
                                            gpu_aniso_pair_force_max_tpp>::launch(pair_args,
                                                                                  range,
                                                                                  d_params,
                                                                                  d_shape_params);
                break;
                }
            default:
                return hipErrorUnknown;
                }
            }
        else
            {
            switch (pair_args.shift_mode)
                {
            case 0:
                {
                AnisoPairForceComputeKernel<evaluator, 0, 0, gpu_aniso_pair_force_max_tpp>::launch(
                    pair_args,
                    range,
                    d_params,
                    d_shape_params);
                break;
                }
            case 1:
                {
                AnisoPairForceComputeKernel<evaluator,
                                            1 && evaluator::implementsEnergyShift(),
                                            0,
                                            gpu_aniso_pair_force_max_tpp>::launch(pair_args,
                                                                                  range,
                                                                                  d_params,
                                                                                  d_shape_params);
                break;
                }
            default:
                return hipErrorUnknown;
                }
            }
        }
    return hipSuccess;
    }
#else
template<class evaluator>
hipError_t gpu_compute_pair_aniso_forces(const a_pair_args_t& pair_args,
                                         const typename evaluator::param_type* d_params,
                                         const typename evaluator::shape_type* d_shape_params);
#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __ANISO_POTENTIAL_PAIR_GPU_CUH__
