// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

#include "hoomd/GPUPartition.cuh"

#ifdef NVCC
#include "hoomd/WarpTools.cuh"
#endif // NVCC

#include <assert.h>
#include <type_traits>

/*! \file PotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifndef __POTENTIAL_PAIR_GPU_CUH__
#define __POTENTIAL_PAIR_GPU_CUH__

//! Maximum number of threads (width of a warp)
const int gpu_pair_force_max_tpp = 32;


//! Wraps arguments to gpu_cgpf
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
              const GPUPartition& _gpu_partition)
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
                  gpu_partition(_gpu_partition)
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
    const GPUPartition& gpu_partition;      //!< The load balancing partition of particles between GPUs
    };

#ifdef NVCC

//! Kernel for calculating pair forces
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
    \param offset Offset of first particle

    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangular(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocated for this kernel launch. The amount is
    (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r
    \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
                       (See PotentialPair for a discussion on what that entails)
    \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
    \tparam tpp Number of threads to use per particle, must be power of 2 and smaller than warp size

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each group of \a tpp threads will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial, int tpp>
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
                                               const unsigned int offset)
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
    unsigned int idx = blockIdx.x * (blockDim.x/tpp) + threadIdx.x/tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction
        active = false;
        }

    // add offset to get actual particle index
    idx += offset;

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
        // load in the length of the neighbor list
        unsigned int n_neigh = d_n_neigh[idx];

        // read in the position of our particle.
        Scalar4 postypei = __ldg(d_pos + idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        Scalar di = Scalar(0);
        if (evaluator::needsDiameter())
            di = __ldg(d_diameter + idx);

        Scalar qi = Scalar(0);
        if (evaluator::needsCharge())
            qi = __ldg(d_charge + idx);

        unsigned int my_head = d_head_list[idx];
        unsigned int cur_j = 0;

        unsigned int next_j(0);
        next_j = threadIdx.x%tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x%tpp) : 0;

        // loop over neighbors
        for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
            {
                {
                // read the current neighbor index
                cur_j = next_j;
                if (neigh_idx+tpp < n_neigh)
                    {
                    next_j = __ldg(d_nlist + my_head + neigh_idx+tpp);
                    }
                // get the neighbor's position
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                Scalar dj = Scalar(0.0);
                if (evaluator::needsDiameter())
                    dj = __ldg(d_diameter + cur_j);

                Scalar qj = Scalar(0.0);
                if (evaluator::needsCharge())
                    qj = __ldg(d_charge + cur_j);

                // calculate dr (with periodic boundary conditions)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions
                dx = box.minImage(dx);

                // calculate r squared
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
                        // Implement XPLOR smoothing
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

                // add up the force vector components
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

    // now that the force calculation is complete, write out the result
    if (active && threadIdx.x % tpp == 0)
        d_force[idx] = force;

    if (compute_virial)
        {
        virialxx = reducer.Sum(virialxx);
        virialxy = reducer.Sum(virialxy);
        virialxz = reducer.Sum(virialxz);
        virialyy = reducer.Sum(virialyy);
        virialyz = reducer.Sum(virialyz);
        virialzz = reducer.Sum(virialzz);

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

//! Pair force compute kernel launcher
/*!
 * \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r
 * \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
 *                       (See PotentialPair for a discussion on what that entails)
 * \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
 * \tparam tpp Number of threads to use per particle, must be power of 2 and smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this with a struct that
 * we are allowed to partially specialize.
 */
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial, int tpp>
struct PairForceComputeKernel
    {
    //! Launcher for the pair force kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param range Range of particle indices this GPU operates on
     * \param d_params Parameters for the potential, stored per type pair
     */

    static void launch(const pair_args_t& pair_args,
        std::pair<unsigned int, unsigned int> range,
        const typename evaluator::param_type *d_params)
        {
        unsigned int N = range.second - range.first;
        unsigned int offset = range.first;


        if (tpp == pair_args.threads_per_particle)
            {
            unsigned int block_size = pair_args.block_size;

            Index2D typpair_idx(pair_args.ntypes);
            unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                        * typpair_idx.getNumElements();

            static unsigned int max_block_size = UINT_MAX;
            if (max_block_size == UINT_MAX)
                max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, shift_mode, compute_virial, tpp>);

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size/tpp) + 1, 1, 1);

            gpu_compute_pair_forces_shared_kernel<evaluator, shift_mode, compute_virial, tpp>
              <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
              pair_args.virial_pitch, N, pair_args.d_pos, pair_args.d_diameter,
              pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
              pair_args.d_head_list, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, offset);
            }
        else
            {
            PairForceComputeKernel<evaluator, shift_mode, compute_virial, tpp/2>::launch(pair_args, range, d_params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int shift_mode, unsigned int compute_virial>
struct PairForceComputeKernel<evaluator, shift_mode, compute_virial, 0>
    {
    static void launch(const pair_args_t& pair_args, std::pair<unsigned int, unsigned int> range, const typename evaluator::param_type *d_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arguments to pass onto the kernel
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

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = pair_args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = pair_args.gpu_partition.getRangeAndSetGPU(idev);

        // Launch kernel
        if (pair_args.compute_virial)
            {
            switch (pair_args.shift_mode)
                {
                case 0:
                    {
                    PairForceComputeKernel<evaluator, 0, 1, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
                    break;
                    }
                case 1:
                    {
                    PairForceComputeKernel<evaluator, 1, 1, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
                    break;
                    }
                case 2:
                    {
                    PairForceComputeKernel<evaluator, 2, 1, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
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
                    PairForceComputeKernel<evaluator, 0, 0, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
                    break;
                    }
                case 1:
                    {
                    PairForceComputeKernel<evaluator, 1, 0, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
                    break;
                    }
                case 2:
                    {
                    PairForceComputeKernel<evaluator, 2, 0, gpu_pair_force_max_tpp>::launch(pair_args, range, d_params);
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
