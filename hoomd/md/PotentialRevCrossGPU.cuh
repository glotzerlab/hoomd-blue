#include "hip/hip_runtime.h"
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

#include "hoomd/HOOMDMath.h"
#include "hoomd/TextureTools.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__
#include <assert.h>

/*! \file PotentialRevCrossGPU.cuh
    \brief Defines templated GPU kernel code for calculating certain three-body forces
*/

#ifndef __POTENTIAL_REVCROSS_GPU_CUH__
#define __POTENTIAL_REVCROSS_GPU_CUH__

//! Maximum number of threads (width of a warp)
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_revcross_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_revcross_max_tpp = 64;
#endif

//! Wraps arguments to gpu_cgpf
struct revcross_args_t
    {
    //! Construct a revcross_args_t
    revcross_args_t(Scalar4 *_d_force,
                   const unsigned int _N,
                   const unsigned int _Nghosts,
                   Scalar * _d_virial,
                   unsigned int _virial_pitch,
                   bool _compute_virial,
                   const Scalar4 *_d_pos,
                   const BoxDim& _box,
                   const unsigned int *_d_n_neigh,
                   const unsigned int *_d_nlist,
                   const unsigned int *_d_head_list,
                   const Scalar *_d_rcutsq,
                   const Scalar *_d_ronsq,
                   const unsigned int _size_nlist,
                   const unsigned int _ntypes,
                   const unsigned int _block_size,
                   const unsigned int _tpp,
                   const hipDeviceProp_t& _devprop)
                   : d_force(_d_force),
                     N(_N),
                     Nghosts(_Nghosts),
                     d_virial(_d_virial),
                     virial_pitch(_virial_pitch),
                     compute_virial(_compute_virial),
                     d_pos(_d_pos),
                     box(_box),
                     d_n_neigh(_d_n_neigh),
                     d_nlist(_d_nlist),
                     d_head_list(_d_head_list),
                     d_rcutsq(_d_rcutsq),
                     d_ronsq(_d_ronsq),
                     size_nlist(_size_nlist),
                     ntypes(_ntypes),
                     block_size(_block_size),
                     tpp(_tpp),
                     devprop(_devprop)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    const unsigned int N;            //!< Number of particles
    const unsigned int Nghosts;      //!< Number of ghost particles
    Scalar *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< Pitch for N*6 virial array
    bool compute_virial;             //!< True if we are supposed to compute the virial
    const Scalar4 *d_pos;            //!< particle positions
    const BoxDim& box;                //!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const unsigned int *d_head_list;//!< Indexes for accessing d_nlist
    const Scalar *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const Scalar *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int size_nlist;  //!< Number of elements in the neighborlist
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int tpp;         //!< Threads per particle
    const hipDeviceProp_t& devprop;   //!< CUDA device properties
    };


#ifdef __HIPCC__

#if !defined(SINGLE_PRECISION)

#if (__CUDA_ARCH__ < 600)
//! atomicAdd function for double-precision floating point numbers
/*! This function is only used when hoomd is compiled for double precision on the GPU.

    \param address Address to write the double to
    \param val Value to add to address
*/
__device__ double RevCrossAtomicAdd(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
    }
#else // CUDA_ARCH > 600)
__device__ double RevCrossAtomicAdd(double* address, double val)
    {
    return atomicAdd(address, val);
    }
#endif
#endif

// workaround for HIP bug
#ifdef __HIP_PLATFORM_HCC__
inline __device__ float RevCrossAtomicAdd(float* address, float val)
    {
    unsigned  int* address_as_uint = (unsigned  int*)address;
    unsigned  int old = *address_as_uint, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_uint,
                        assumed,
                        __float_as_uint(val + __uint_as_float(assumed)));
    } while (assumed != old);

    return __uint_as_float(old);
    }
#else
inline __device__ float RevCrossAtomicAdd(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif

//! Kernel for calculating the RevCross forces
/*! This kernel is called to calculate the forces on all N particles. Actual evaluation of the potentials and
    forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param N Number of particles in the system
    \param d_pos Positions of all the particles
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for reading \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_ronsq ron squared, stored per type pair
    \param ntypes Number of types in the simulation

    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangular(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocated for this kernel launch. The amount is
    (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evaluate V(r) and -delta V(r)/r

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned char compute_virial, int tpp>
__global__ void gpu_compute_revcross_forces_kernel(Scalar4 *d_force,
                                                  const unsigned int N,
                                                  Scalar *d_virial,
                                                  unsigned int virial_pitch,
                                                  const Scalar4 *d_pos,
                                                  const BoxDim box,
                                                  const unsigned int *d_n_neigh,
                                                  const unsigned int *d_nlist,
                                                  const unsigned int *d_head_list,
                                                  const typename evaluator::param_type *d_params,
                                                  const Scalar *d_rcutsq,
                                                  const Scalar *d_ronsq,
                                                  const unsigned int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    HIP_DYNAMIC_SHARED( char, s_data)
    typename evaluator::param_type *s_params =
        (typename evaluator::param_type *)(&s_data[0]);
    Scalar *s_rcutsq = (Scalar *)(&s_data[num_typ_parameters*sizeof(typename evaluator::param_type)]);

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
    unsigned int idx = blockIdx.x * (blockDim.x/tpp) + threadIdx.x/tpp;

    if (idx >= N)
        return;

    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of the particle
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

    // initialize the force to 0
    Scalar4 forcei = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    Scalar virialxx(0.0);
    Scalar virialxy(0.0);
    Scalar virialxz(0.0);
    Scalar virialyy(0.0);
    Scalar virialyz(0.0);
    Scalar virialzz(0.0);

    // prefetch neighbor index
    const unsigned int head_idx = d_head_list[idx];
    unsigned int cur_j = 0;
    unsigned int next_j(0);
    unsigned int my_head = d_head_list[idx];

    next_j = threadIdx.x%tpp < n_neigh ? __ldg(d_nlist + my_head + threadIdx.x%tpp) : 0;

    // loop over neighbors in strided way
    for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
        {
        // read the current neighbor index (MEM TRANSFER: 4 bytes)
        // prefetch the next value and set the current one
        cur_j = next_j;
        if (neigh_idx+tpp < n_neigh)
            {
            next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
            }

        // read the position of j (MEM TRANSFER: 16 bytes)
        Scalar4 postypej = __ldg(d_pos + cur_j);
        Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

        // initialize the force on j
        Scalar4 forcej = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

        // compute r_ij (FLOPS: 3)
        Scalar3 dxij = posi - posj;

        // apply periodic boundary conditions (FLOPS: 12)
        dxij = box.minImage(dxij);

        // compute rij_sq (FLOPS: 5)
        Scalar rij_sq = dot(dxij, dxij);

        // access the per type-pair parameters
        unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
        Scalar rcutsq = s_rcutsq[typpair];
        typename evaluator::param_type param = s_params[typpair];

        // compute the base repulsive and attractive terms of the potential
        Scalar invratio = 0.0;        
        Scalar invratio2 = 0.0;
        evaluator eval(rij_sq, rcutsq, param);
        bool evaluatedij = eval.evalRepulsiveAndAttractive(invratio, invratio2);

        if (evaluatedij)
            {
            // evaluate the force and energy from the ij interaction
            Scalar force_divr = Scalar(0.0);
            Scalar potential_eng = Scalar(0.0);
            eval.evalForceij(invratio, invratio2, force_divr, potential_eng);
    
            // add the forces and energies to their respective particles
            forcei.x += dxij.x * force_divr;
            forcei.y += dxij.y * force_divr;
            forcei.z += dxij.z * force_divr;
    
            forcej.x -= dxij.x * force_divr;
            forcej.y -= dxij.y * force_divr;
            forcej.z -= dxij.z * force_divr;
    
            forcej.w += potential_eng;
            forcei.w += potential_eng;
            
            // calculate the virial
            if (compute_virial)
               {
               virialxx +=  dxij.x * dxij.x * force_divr;
               virialxy +=  dxij.x * dxij.y * force_divr;
               virialxz +=  dxij.x * dxij.z * force_divr;
               virialyy +=  dxij.y * dxij.y * force_divr;
               virialyz +=  dxij.y * dxij.z * force_divr;
               virialzz +=  dxij.z * dxij.z * force_divr;
               }
    
            // now evaluate the force from the ik interactions
            unsigned int cur_k = 0;
            unsigned int next_k(0);  
    
            // loop over k neighbors one by one
            for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                {
                // read the current neighbor index and prefetch the next one
                cur_k = next_k;
                next_k = __ldg(d_nlist + head_idx + neigh_idy+1);
    
    	        // I continue only if k is not the same as j
    	        if((cur_k>cur_j)&&(cur_j>idx))
    	            {
                    // get the position of neighbor k
                    Scalar4 postypek = __ldg(d_pos + cur_k);
                    Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);
    
                    // get the type pair parameters for i and k
                    typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                    Scalar temp_rcutsq = s_rcutsq[typpair];
                    typename evaluator::param_type temp_param = s_params[typpair];
                    
                     // compute rik
                    Scalar3 dxik = posi - posk;
                    // apply the periodic boundary conditions
                    dxik = box.minImage(dxik);
                    // compute rik_sq
                    Scalar rik_sq = dot(dxik, dxik);
    
                    evaluator temp_eval(rij_sq, temp_rcutsq, temp_param);
                    temp_eval.setRik(rik_sq);
                    bool temp_evaluated = temp_eval.areInteractive();
    
                    if (temp_evaluated)
                        {
                        Scalar4 forcek = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    
                        // set up the evaluator
                        eval.setRik(rik_sq);
    
                        // compute the force
                        Scalar force_divr_ij = 0.0;
                        Scalar force_divr_ik = 0.0;                    
                        
                        bool evaluatedjk = eval.evalForceik(invratio, invratio2, force_divr_ij, force_divr_ik);
    
                        if (evaluatedjk)
                            {
                            // add the forces to their respective particles
                            forcei.x += force_divr_ij * dxij.x + force_divr_ik * dxik.x;
                            forcei.y += force_divr_ij * dxij.y + force_divr_ik * dxik.y;
                            forcei.z += force_divr_ij * dxij.z + force_divr_ik * dxik.z;
    
                            forcej.x -= force_divr_ij * dxij.x;
                            forcej.y -= force_divr_ij * dxij.y;
                            forcej.z -= force_divr_ij * dxij.z;
    
                            forcek.x -= force_divr_ik * dxik.x;
                            forcek.y -= force_divr_ik * dxik.y;
                            forcek.z -= force_divr_ik * dxik.z;
    
                            RevCrossAtomicAdd(&d_force[cur_k].x, forcek.x);
                            RevCrossAtomicAdd(&d_force[cur_k].y, forcek.y);
                            RevCrossAtomicAdd(&d_force[cur_k].z, forcek.z);
                        
    	        	       //evaluate the virial contribute of this 3 body interaction    
    	        	       if (compute_virial)
    	        	           {
    	        	           //a single term is needed to account for all of the 3 body virial that is stored in the 'i' particle data
    	        	           //look at S. Ciarella and W.G. Ellenbroek 2019 https://arxiv.org/abs/1912.08569 for the definition of the virial tensor
    	        	           virialxx += (force_divr_ij*dxij.x*dxij.x + force_divr_ik*dxik.x*dxik.x);	
    	        	           virialyy += (force_divr_ij*dxij.y*dxij.y + force_divr_ik*dxik.y*dxik.y);	
    	        	           virialzz += (force_divr_ij*dxij.z*dxij.z + force_divr_ik*dxik.z*dxik.z);	
    	        	           virialxy += (force_divr_ij*dxij.x*dxij.y + force_divr_ik*dxik.x*dxik.y);
    	        	           virialxz += (force_divr_ij*dxij.x*dxij.z + force_divr_ik*dxik.x*dxik.z);
    	        	           virialyz += (force_divr_ij*dxij.y*dxij.z + force_divr_ik*dxik.y*dxik.z);
    	        	           }
                            }    
                        }
                    }
	        }

            // potential energy of j must be halved
            // write out the result for particle j
            RevCrossAtomicAdd(&d_force[cur_j].x, forcej.x);
            RevCrossAtomicAdd(&d_force[cur_j].y, forcej.y);
            RevCrossAtomicAdd(&d_force[cur_j].z, forcej.z);
            RevCrossAtomicAdd(&d_force[cur_j].w, forcej.w);
            }
	}
    // potential energy per particle must be halved
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    RevCrossAtomicAdd(&d_force[idx].x, forcei.x);
    RevCrossAtomicAdd(&d_force[idx].y, forcei.y);
    RevCrossAtomicAdd(&d_force[idx].z, forcei.z);
    RevCrossAtomicAdd(&d_force[idx].w, forcei.w);
    
    if (compute_virial)
        {
	RevCrossAtomicAdd(&d_virial[0*virial_pitch+idx], virialxx);
	RevCrossAtomicAdd(&d_virial[1*virial_pitch+idx], virialxy);
	RevCrossAtomicAdd(&d_virial[2*virial_pitch+idx], virialxz);
	RevCrossAtomicAdd(&d_virial[3*virial_pitch+idx], virialyy);
	RevCrossAtomicAdd(&d_virial[4*virial_pitch+idx], virialyz);
	RevCrossAtomicAdd(&d_virial[5*virial_pitch+idx], virialzz);     
	}        
    
    
    }


//! Kernel for zeroing forces and virial before computation with atomic additions.
/*! \param d_force Device memory to write forces to
    \param N Number of particles in the system

*/
__global__ void gpu_zero_forces_kernel_revcross(Scalar4 *d_force,
                                       Scalar *d_virial,
                                       unsigned int virial_pitch,
                                       const unsigned int N)
    {
    // identify the particle we are supposed to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // zero the force
    d_force[idx] = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // zero the virial
    d_virial[0*virial_pitch+idx] = Scalar(0.0);
    d_virial[1*virial_pitch+idx] = Scalar(0.0);
    d_virial[2*virial_pitch+idx] = Scalar(0.0);
    d_virial[3*virial_pitch+idx] = Scalar(0.0);
    d_virial[4*virial_pitch+idx] = Scalar(0.0);
    d_virial[5*virial_pitch+idx] = Scalar(0.0);
    }

template<typename T>
void get_max_block_size(T func, const revcross_args_t& pair_args, unsigned int& max_block_size, unsigned int& kernel_shared_bytes)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void *)func);

    max_block_size = attr.maxThreadsPerBlock;
    max_block_size &= ~(pair_args.devprop.warpSize - 1);

    kernel_shared_bytes = attr.sharedSizeBytes;
    }

//! RevCross compute kernel launcher
/*!
 * \tparam evaluator Evaluator class
 * \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory. When zero, textures or __ldg
 *                        is used depending on architecture.
 * \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
 * \tparam tpp Number of threads to use per particle, must be power of 2 and smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this with a struct that
 * we are allowed to partially specialize.
 */
template<class evaluator, unsigned int compute_virial, int tpp>
struct RevCrossComputeKernel
    {
    //! Launcher for the revcross triplet kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param d_params Parameters for the potential, stored per type pair
     */
    static void launch(const revcross_args_t& pair_args, const typename evaluator::param_type *d_params)
        {
        if (tpp == pair_args.tpp)
            {
            static unsigned int max_block_size = UINT_MAX;
            static unsigned int kernel_shared_bytes = 0;
            if (max_block_size == UINT_MAX)
                get_max_block_size(gpu_compute_revcross_forces_kernel<evaluator, compute_virial, tpp>, pair_args, max_block_size, kernel_shared_bytes);
            int run_block_size = min(pair_args.block_size, max_block_size);

            // size shared bytes
            Index2D typpair_idx(pair_args.ntypes);
            unsigned int shared_bytes = (sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                        * typpair_idx.getNumElements() + pair_args.ntypes*run_block_size*sizeof(Scalar);

            while (shared_bytes + kernel_shared_bytes >= pair_args.devprop.sharedMemPerBlock)
                {
                run_block_size -= pair_args.devprop.warpSize;

                shared_bytes = (sizeof(Scalar) + sizeof(typename evaluator::param_type))
                               * typpair_idx.getNumElements() + pair_args.ntypes*run_block_size*sizeof(Scalar);
                }

            // zero the forces
            hipLaunchKernelGGL((gpu_zero_forces_kernel_revcross), dim3((pair_args.N + pair_args.Nghosts)/run_block_size + 1), dim3(run_block_size), 0, 0, pair_args.d_force,
                                                    pair_args.d_virial,
                                                    pair_args.virial_pitch,
                                                    pair_args.N + pair_args.Nghosts);

            // setup the grid to run the kernel
            dim3 grid( pair_args.N / (run_block_size/pair_args.tpp) + 1, 1, 1);
            dim3 threads(run_block_size, 1, 1);

            hipLaunchKernelGGL((gpu_compute_revcross_forces_kernel<evaluator, compute_virial, tpp>), dim3(grid), dim3(threads), shared_bytes, 0, pair_args.d_force,
                                                pair_args.N,
                                                pair_args.d_virial,
                                                pair_args.virial_pitch,
                                                pair_args.d_pos,
                                                pair_args.box,
                                                pair_args.d_n_neigh,
                                                pair_args.d_nlist,
                                                pair_args.d_head_list,
                                                d_params,
                                                pair_args.d_rcutsq,
                                                pair_args.d_ronsq,
                                                pair_args.ntypes);
            }
        else
            {
            RevCrossComputeKernel<evaluator, compute_virial, tpp/2>::launch(pair_args, d_params);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int compute_virial>
struct RevCrossComputeKernel<evaluator, compute_virial, 0>
    {
    static void launch(const revcross_args_t& pair_args, const typename evaluator::param_type *d_params)
        {
        // do nothing
        }
    };

//! Kernel driver that computes the three-body forces
/*! \param pair_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_revcross_forces_kernel(), see it for details.
*/
template< class evaluator >
hipError_t gpu_compute_tripletrevcross_forces(const revcross_args_t& pair_args,
                                       const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    // compute the new forces
    if (!pair_args.compute_virial)
        {
        RevCrossComputeKernel<evaluator, 0, gpu_revcross_max_tpp>::launch(pair_args, d_params);
        }
    else
        {
        RevCrossComputeKernel<evaluator, 1, gpu_revcross_max_tpp>::launch(pair_args, d_params);
        }
    return hipSuccess;
    }
#endif

#endif // __POTENTIAL_REVCROSS_GPU_CUH__
