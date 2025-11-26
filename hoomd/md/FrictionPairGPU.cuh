// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

/*! \file FrictionPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the friction ptl pair forces and
   torques
*/

#ifndef __FRICTION_PAIR_GPU_CUH__
#define __FRICTION_PAIR_GPU_CUH__

//! Maximum number of threads (width of a warp)
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_friction_pair_force_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_friction_pair_force_max_tpp = 64;
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
                  const Scalar4* _d_vel,
                  const Scalar* _d_charge,
                  const Scalar4* _d_orientation,
                  const Scalar4* _d_angmom,
                  const Scalar* _d_diameter,
                  const Scalar3* _d_moment_inertia,
                  const unsigned int* _d_tag,
                  const BoxDim& _box,
                  const bool _third_law,
                  const unsigned int _dim,
                  uint16_t _seed,
                  uint64_t _timestep,
                  const Scalar _deltaT,
                  const unsigned int* _d_n_neigh,
                  const unsigned int* _d_nlist,
                  const size_t* _d_head_list,
                  const Scalar* _d_rcutsq,
                  const unsigned int _ntypes,
                  const unsigned int _block_size,
                  const unsigned int _compute_virial,
                  const unsigned int _threads_per_particle,
                  const hipDeviceProp_t& _devprop)
        : d_force(_d_force), d_torque(_d_torque), d_virial(_d_virial), virial_pitch(_virial_pitch),
          N(_N), n_max(_n_max), d_pos(_d_pos), d_vel(_d_vel), d_charge(_d_charge),
          d_orientation(_d_orientation), d_angmom(_d_angmom), d_diameter(_d_diameter),
          d_moment_inertia(_d_moment_inertia), d_tag(_d_tag), box(_box), third_law(_third_law),
          dim(_dim), seed(_seed), timestep(_timestep), deltaT(_deltaT), d_n_neigh(_d_n_neigh),
          d_nlist(_d_nlist), d_head_list(_d_head_list), d_rcutsq(_d_rcutsq), ntypes(_ntypes),
          block_size(_block_size), compute_virial(_compute_virial),
          threads_per_particle(_threads_per_particle), devprop(_devprop) { };

    Scalar4* d_force;                //!< Force to write out
    Scalar4* d_torque;               //!< Torque to write out
    Scalar* d_virial;                //!< Virial to write out
    const size_t virial_pitch;       //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;            //!< number of particles
    const unsigned int n_max;        //!< maximum size of particle data arrays
    const Scalar4* d_pos;            //!< particle positions
    const Scalar4* d_vel;            //!< particle velocity
    const Scalar* d_charge;          //!< particle charges
    const Scalar4* d_orientation;    //!< particle orientation to compute forces over
    const Scalar4* d_angmom;         //!< particle angular momentum
    const Scalar* d_diameter;        //!< particle diameter
    const Scalar3* d_moment_inertia; //!< particle moment of inertia
    const unsigned int* d_tag;       //!< particle tags to compute forces over
    const BoxDim box;                //!< Simulation box in GPU format
    const bool third_law;            //!< Boolean storing if only a half neighborlist is used
    const unsigned int dim;          //!< Dimension of the simulation
    uint16_t seed;                   //!< Seed of the simulation
    uint64_t timestep;               //!< Current timestep
    const Scalar deltaT;             //!< Timestep dt size of the simulation
    const unsigned int*
        d_n_neigh;               //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist; //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;   //!< Device array listing beginning of each particle's neighbors
    const Scalar* d_rcutsq;      //!< Device array listing r_cut squared per particle type pair
    const unsigned int ntypes;   //!< Number of particle types in the simulation
    const unsigned int block_size;           //!< Block size to execute
    const unsigned int compute_virial;       //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads to launch per particle
    const hipDeviceProp_t& devprop;          //!< CUDA device properties
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
    \param d_vel particle velocity
    \param d_charge particle charges
    \param d_orientation Quaternion data on the GPU to calculate forces on
    \param d_angmom particle angular momentum
    \param d_diameter particle diameter
    \param d_moment_inertia particle moment of inertia
    \param d_tag Tag data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param third_law Boolean storing if only a half neighborlist is used
    \param dim Dimension of the simulation
    \param d_seed Seed of the simulation
    \param d_timestep Current timestep
    \param d_deltaT Timestep size of the simulation
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
   No energy shifting is done. \tparam compute_virial When
   non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template<class evaluator, unsigned int compute_virial, int tpp>
__global__ void
gpu_compute_pair_friction_forces_kernel(Scalar4* d_force,
                                        Scalar4* d_torque,
                                        Scalar* d_virial,
                                        const size_t virial_pitch,
                                        const unsigned int N,
                                        const Scalar4* d_pos,
                                        const Scalar4* d_vel,
                                        const Scalar* d_charge,
                                        const Scalar4* d_orientation,
                                        const Scalar4* d_angmom,
                                        const Scalar* d_diameter,
                                        const Scalar3* d_moment_inertia,
                                        const unsigned int* d_tag,
                                        const BoxDim box,
                                        const bool third_law,
                                        const unsigned int dim,
                                        const uint16_t d_seed,
                                        const uint64_t d_timestep,
                                        const Scalar d_deltaT,
                                        const unsigned int* d_n_neigh,
                                        const unsigned int* d_nlist,
                                        const size_t* d_head_list,
                                        const typename evaluator::param_type* d_params,
                                        const Scalar* d_rcutsq,
                                        const unsigned int ntypes,
                                        unsigned int max_extra_bytes)
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

    __syncthreads();

    // initialize extra shared mem
    char* s_extra = (char*)(s_params + num_typ_parameters);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_pair = 0; cur_pair < typpair_idx.getNumElements(); ++cur_pair)
        s_params[cur_pair].load_shared(s_extra, available_bytes);

    __syncthreads();

    // start by identifying which particle we are to handle
    unsigned int idx;
    idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction
        active = false;
        }

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

        // read in the particle data
        Scalar4 postypei = __ldg(d_pos + idx);
        Scalar4 veltypei = __ldg(d_vel + idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
        Scalar3 veli = make_scalar3(veltypei.x, veltypei.y, veltypei.z);
        quat<Scalar> quati(__ldg(d_orientation + idx));
        quat<Scalar> angmomi(__ldg(d_angmom + idx));
        Scalar diai = __ldg(d_diameter + idx);
        Scalar3 moment_inertia_i = d_moment_inertia[idx];

        Scalar qi = Scalar(0);
        Scalar massi = Scalar(0);
        if (evaluator::needsCharge())
            qi = __ldg(d_charge + idx);
        if (evaluator::needsNu())
            massi = veltypei.w;

        // calculate angular momentum of i-th particle in the body frame
        quat<Scalar> qxP_i = conj(quati) * angmomi;
        vec3<Scalar> bf_vel_i = Scalar(0.5)
                                * vec3<Scalar>(qxP_i.v.x / moment_inertia_i.x,
                                               qxP_i.v.y / moment_inertia_i.y,
                                               qxP_i.v.z / moment_inertia_i.z);

        // Rotate angular velocity into global frame
        vec3<Scalar> gf_angvel_i = rotate(quati, bf_vel_i);
        Scalar3 angvel_i = make_scalar3(gf_angvel_i.x, gf_angvel_i.y, gf_angvel_i.z);

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

                // get the neighbor's particle data
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar4 veltypej = __ldg(d_vel + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);
                Scalar3 velj = make_scalar3(veltypej.x, veltypej.y, veltypej.z);

                quat<Scalar> quatj(__ldg(d_orientation + cur_j));
                quat<Scalar> angmomj(__ldg(d_angmom + cur_j));
                Scalar diaj = __ldg(d_diameter + cur_j);
                Scalar3 moment_inertia_j = d_moment_inertia[cur_j];

                Scalar qj = Scalar(0);
                Scalar massj = Scalar(0);
                if (evaluator::needsCharge())
                    qj = __ldg(d_charge + cur_j);
                if (evaluator::needsNu())
                    massj = veltypej.w;

                // calculate dv_ij
                Scalar3 dv = velj - veli;

                // calculate angular momentum of i-th particle in the body frame
                quat<Scalar> qxP_j = conj(quatj) * angmomj;
                vec3<Scalar> bf_vel_j = Scalar(0.5)
                                        * vec3<Scalar>(qxP_j.v.x / moment_inertia_j.x,
                                                       qxP_j.v.y / moment_inertia_j.y,
                                                       qxP_j.v.z / moment_inertia_j.z);

                // Rotate angular velocity into global frame
                vec3<Scalar> gf_angvel_j = rotate(quatj, bf_vel_j);
                Scalar3 angvel_j = make_scalar3(gf_angvel_j.x, gf_angvel_j.y, gf_angvel_j.z);

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

                // evaluate the potential
                Scalar3 jforce = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar3 torquei = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar3 torquej = {Scalar(0), Scalar(0), Scalar(0)};
                Scalar pair_eng = Scalar(0);

                // constructor call
                evaluator eval(dx, angvel_i, angvel_j, dv, diai, diaj, rcutsq, param);

                // Special Potential Pair DPD like Requirements

                // set seed using global tags

                unsigned int tagi = __ldg(d_tag + idx);
                unsigned int tagj = __ldg(d_tag + cur_j);

                eval.set_seed_ij_timestep(d_seed, tagi, tagj, d_timestep);
                eval.setDeltaT(d_deltaT);
                eval.setThirdLaw(third_law);

                if (evaluator::needsCharge())
                    eval.setCharge(qi, qj);
                if (evaluator::needsTags())
                    eval.setTags(__ldg(d_tag + idx), __ldg(d_tag + cur_j));

                if (evaluator::needsNu())
                    {
                    // Calculate nu for the Ito formalism
                    Scalar nu_ito = ((Scalar(1.0) / massi) + (Scalar(1.0) / massj))
                                    + (((diaj * diaj / Scalar(4.0)) / moment_inertia_j.x)
                                       + ((diai * diai / Scalar(4.0)) / moment_inertia_i.x));
                    eval.setNu(nu_ito);
                    }

                // call evaluator
                eval.evaluate(jforce, pair_eng, torquei, torquej);

                if (dim == 2)
                    {
                    jforce.z = 0.0;
                    torquei.x = 0.0;
                    torquei.y = 0.0;
                    torquej.x = 0.0;
                    torquej.y = 0.0;
                    }

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

//! Friction pair force compute kernel launcher
/*!
 * \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r
 * \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor
 * is not computed. \tparam tpp Number of threads to use per particle, must be power of 2 and
 * smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this
 * with a struct that we are allowed to partially specialize.
 */
template<class evaluator, unsigned int compute_virial, int tpp>
struct FrictionPairForceComputeKernel
    {
    //! Launcher for the pair force kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param N Number of particles to compute.
     * \param params Parameters for the potential, stored per type pair
     */

    static void launch(const a_pair_args_t& pair_args,
                       unsigned int N,
                       const typename evaluator::param_type* params)
        {
        if (tpp == pair_args.threads_per_particle)
            {
            unsigned int block_size = pair_args.block_size;

            Index2D typpair_idx(pair_args.ntypes);
            size_t shared_bytes = (2 * sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                  * typpair_idx.getNumElements();

            unsigned int max_block_size;
            hipFuncAttributes attr;
            hipFuncGetAttributes(
                &attr,
                reinterpret_cast<const void*>(
                    &gpu_compute_pair_friction_forces_kernel<evaluator, compute_virial, tpp>));
            int max_threads = attr.maxThreadsPerBlock;
            // number of threads has to be multiple of warp size
            max_block_size = max_threads - max_threads % gpu_friction_pair_force_max_tpp;

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
            extra_bytes = max_extra_bytes - available_bytes;

            shared_bytes += extra_bytes;

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(N / (block_size / tpp) + 1, 1, 1);

            hipLaunchKernelGGL(
                (gpu_compute_pair_friction_forces_kernel<evaluator, compute_virial, tpp>),
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
                pair_args.d_vel,
                pair_args.d_charge,
                pair_args.d_orientation,
                pair_args.d_angmom,
                pair_args.d_diameter,
                pair_args.d_moment_inertia,
                pair_args.d_tag,
                pair_args.box,
                pair_args.third_law,
                pair_args.dim,
                pair_args.seed,
                pair_args.timestep,
                pair_args.deltaT,
                pair_args.d_n_neigh,
                pair_args.d_nlist,
                pair_args.d_head_list,
                params,
                pair_args.d_rcutsq,
                pair_args.ntypes,
                max_extra_bytes);
            }
        else
            {
            FrictionPairForceComputeKernel<evaluator, compute_virial, tpp / 2>::launch(pair_args,
                                                                                       N,
                                                                                       params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int compute_virial>
struct FrictionPairForceComputeKernel<evaluator, compute_virial, 0>
    {
    static void launch(const a_pair_args_t& pair_args,
                       unsigned int N,
                       const typename evaluator::param_type* d_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes lj forces on the GPU for FrictionPairGPU
/*! \param pair_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_pair_friction_forces_kernel(), see it for
   details.
*/
template<class evaluator>
hipError_t gpu_compute_pair_friction_forces(const a_pair_args_t& pair_args,
                                            const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    // run the kernel
    if (pair_args.compute_virial)
        {
        FrictionPairForceComputeKernel<evaluator, 1, gpu_friction_pair_force_max_tpp>::launch(
            pair_args,
            pair_args.N,
            d_params);
        }
    else
        {
        FrictionPairForceComputeKernel<evaluator, 0, gpu_friction_pair_force_max_tpp>::launch(
            pair_args,
            pair_args.N,
            d_params);
        }
    return hipSuccess;
    }
#else
template<class evaluator>
hipError_t gpu_compute_pair_friction_forces(const a_pair_args_t& pair_args,
                                            const typename evaluator::param_type* d_params);
#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __FRICTION_PAIR_GPU_CUH__
