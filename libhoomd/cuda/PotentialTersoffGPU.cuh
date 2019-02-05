/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

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

#include "HOOMDMath.h"
#include "TextureTools.h"
#include "ParticleData.cuh"
#include "Index1D.h"

#include <assert.h>

/*! \file PotentialTersoffGPU.cuh
    \brief Defines templated GPU kernel code for calculating certain three-body forces
*/

#ifndef __POTENTIAL_TERSOFF_GPU_CUH__
#define __POTENTIAL_TERSOFF_GPU_CUH__

//! Wraps arguments to gpu_cgpf
struct tersoff_args_t
    {
    //! Construct a tersoff_args_t
    tersoff_args_t(Scalar4 *_d_force,
                   const unsigned int _N,
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
                   const unsigned int _compute_capability,
                   const unsigned int _max_tex1d_width)
                   : d_force(_d_force),
                     N(_N),
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
                     compute_capability(_compute_capability),
                     max_tex1d_width(_max_tex1d_width)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    const unsigned int N;            //!< Number of particles
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
    const unsigned int compute_capability; //!< GPU compute capability (20, 30, 35, ...)
    const unsigned int max_tex1d_width;     //!< Maximum width of a linear 1D texture
    };


#ifdef NVCC
//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;
//! Texture for reading neighbor list
texture<unsigned int, 1, cudaReadModeElementType> nlist_tex;

#if !defined(SINGLE_PRECISION)

#if (__CUDA_ARCH__ < 600)
//! atomicAdd function for double-precision floating point numbers
/*! This function is only used when hoomd is compiled for double precision on the GPU.

    \param address Address to write the double to
    \param val Value to add to address
*/
__device__ double myAtomicAdd(double* address, double val)
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
__device__ double myAtomicAdd(double* address, double val)
    {
    return atomicAdd(address, val);
    }
#endif
#endif

__device__ float myAtomicAdd(float* address, float val)
    {
    return atomicAdd(address, val);
    }

//! Kernel for calculating the Tersoff forces
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

    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocatd for this kernel launch. The amount is
    (2*sizeof(Scalar) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
    \tparam use_gmem_nlist When non-zero, the neighbor list is read out of global memory. When zero, textures or __ldg
                           is used depending on architecture.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator , unsigned char use_gmem_nlist>
__global__ void gpu_compute_triplet_forces_kernel(Scalar4 *d_force,
                                                  const unsigned int N,
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
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params =
        (typename evaluator::param_type *)(&s_data[0]);
    Scalar *s_rcutsq = (Scalar *)(&s_data[num_typ_parameters*sizeof(evaluator::param_type)]);

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
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of the particle
    Scalar4 postypei = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

    // initialize the force to 0
    Scalar4 forcei = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // prefetch neighbor index
    const unsigned int head_idx = d_head_list[idx];
    unsigned int cur_j = 0;
    unsigned int next_j(0);
    if (use_gmem_nlist)
        {
        next_j = d_nlist[head_idx];
        }
    else
        {
        next_j = texFetchUint(d_nlist, nlist_tex, head_idx);
        }

    // loop over neighbors
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
    {
        // read the current neighbor index (MEM TRANSFER: 4 bytes)
        // prefetch the next value and set the current one
        cur_j = next_j;
        if (use_gmem_nlist)
            {
            next_j = d_nlist[head_idx + neigh_idx + 1];
            }
        else
            {
            next_j = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idx + 1);
            }

        // read the position of j (MEM TRANSFER: 16 bytes)
        Scalar4 postypej = texFetchScalar4(d_pos, pdata_pos_tex, cur_j);
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
        Scalar fR = Scalar(0.0);
        Scalar fA = Scalar(0.0);
        evaluator eval(rij_sq, rcutsq, param);
        bool evaluatedij = eval.evalRepulsiveAndAttractive(fR, fA);

        if (evaluatedij)
            {
            // compute chi
            Scalar chi = Scalar(0.0);
            unsigned int cur_k = 0;
            unsigned int next_k(0);
            if (use_gmem_nlist)
                {
                next_k = d_nlist[head_idx];
                }
            else
                {
                next_k = texFetchUint(d_nlist, nlist_tex, head_idx);
                }
            for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                {
                // read the current index of k and prefetch the next one
                cur_k = next_k;
                if (use_gmem_nlist)
                    {
                    next_k = d_nlist[head_idx + neigh_idy + 1];
                    }
                else
                    {
                    next_k = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idy+1);
                    }

                // get the position of neighbor k
                Scalar4 postypek = texFetchScalar4(d_pos, pdata_pos_tex, cur_k);
                Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                // get the type pair parameters for i and k
                typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                Scalar temp_rcutsq = s_rcutsq[typpair];
                typename evaluator::param_type temp_param = s_params[typpair];

                evaluator temp_eval(rij_sq, temp_rcutsq, temp_param);
                bool temp_evaluated = temp_eval.areInteractive();

                if (cur_k != cur_j && temp_evaluated)
                    {
                    // compute rik
                    Scalar3 dxik = posi - posk;

                    // apply the periodic boundary conditions
                    dxik = box.minImage(dxik);

                    // compute rik_sq
                    Scalar rik_sq = dot(dxik, dxik);

                    // compute the bond angle (if needed)
                    Scalar cos_th = Scalar(0.0);
                    if (evaluator::needsAngle())
                        cos_th = dot(dxij, dxik) * fast::rsqrt(rij_sq * rik_sq);
                    else cos_th += Scalar(1.0); // shuts up the compiler warning

                    // set up the evaluator
                    eval.setRik(rik_sq);
                    if (evaluator::needsAngle())
                        eval.setAngle(cos_th);

                    // compute the partial chi term
                    eval.evalChi(chi);
                    }
                }
            // evaluate the force and energy from the ij interaction
            Scalar force_divr = Scalar(0.0);
            Scalar potential_eng = Scalar(0.0);
            Scalar bij = Scalar(0.0);
            eval.evalForceij(fR, fA, chi, bij, force_divr, potential_eng);

            // add the forces and energies to their respective particles
            Scalar2 v_coeffs = make_scalar2(Scalar(1.0 / 6.0) * rij_sq, Scalar(0.0));
            forcei.x += dxij.x * force_divr;
            forcei.y += dxij.y * force_divr;
            forcei.z += dxij.z * force_divr;

            forcej.x -= dxij.x * force_divr;
            forcej.y -= dxij.y * force_divr;
            forcej.z -= dxij.z * force_divr;
            forcej.w += potential_eng;

            forcei.w += potential_eng;

            // now evaluate the force from the ik interactions
            cur_k = 0;
            if (use_gmem_nlist)
                {
                next_k = d_nlist[head_idx];
                }
            else
                {
                next_k = texFetchUint(d_nlist, nlist_tex, head_idx);
                }
            for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                {
                // read the current neighbor index and prefetch the next one
                cur_k = next_k;
                if (use_gmem_nlist)
                    {
                    next_k = d_nlist[head_idx + neigh_idy + 1];
                    }
                else
                    {
                    next_k = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idy+1);
                    }

                // get the position of neighbor k
                Scalar4 postypek = texFetchScalar4(d_pos, pdata_pos_tex, cur_k);
                Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                // get the type pair parameters for i and k
                typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypek.w));
                Scalar temp_rcutsq = s_rcutsq[typpair];
                typename evaluator::param_type temp_param = s_params[typpair];

                evaluator temp_eval(rij_sq, temp_rcutsq, temp_param);
                bool temp_evaluated = temp_eval.areInteractive();

                if (cur_k != cur_j && temp_evaluated)
                    {
                    Scalar4 forcek = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

                    // compute rik
                    Scalar3 dxik = posi - posk;

                    // apply the periodic boundary conditions
                    dxik = box.minImage(dxik);

                    // compute rik_sq
                    Scalar rik_sq = dot(dxik, dxik);

                    // compute the bond angle (if needed)
                    Scalar cos_th = Scalar(0.0);
                    if (evaluator::needsAngle())
                        cos_th = dot(dxij, dxik) * fast::rsqrt(rij_sq * rik_sq);
                    else cos_th += Scalar(1.0); // shuts up the compiler warning

                    // set up the evaluator
                    eval.setRik(rik_sq);
                    if (evaluator::needsAngle())
                        eval.setAngle(cos_th);

                    // compute the force
                    Scalar3 force_divr_ij = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
                    Scalar3 force_divr_ik = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
                    bool evaluatedjk = eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                    if (evaluatedjk)
                        {
                        // add the forces to their respective particles
                        v_coeffs.y = Scalar(1.0 / 6.0) * rik_sq;
                        forcei.x += force_divr_ij.x * dxij.x + force_divr_ik.x * dxik.x;
                        forcei.y += force_divr_ij.x * dxij.y + force_divr_ik.x * dxik.y;
                        forcei.z += force_divr_ij.x * dxij.z + force_divr_ik.x * dxik.z;

                        forcej.x += force_divr_ij.y * dxij.x + force_divr_ik.y * dxik.x;
                        forcej.y += force_divr_ij.y * dxij.y + force_divr_ik.y * dxik.y;
                        forcej.z += force_divr_ij.y * dxij.z + force_divr_ik.y * dxik.z;

                        forcek.x += force_divr_ij.z * dxij.x + force_divr_ik.z * dxik.x;
                        forcek.y += force_divr_ij.z * dxij.y + force_divr_ik.z * dxik.y;
                        forcek.z += force_divr_ij.z * dxij.z + force_divr_ik.z * dxik.z;

                        myAtomicAdd(&d_force[cur_k].x, forcek.x);
                        myAtomicAdd(&d_force[cur_k].y, forcek.y);
                        myAtomicAdd(&d_force[cur_k].z, forcek.z);
                        }
                    }
                }

            // potential energy of j must be halved
            forcej.w *= Scalar(0.5);
            // write out the result for particle j
            myAtomicAdd(&d_force[cur_j].x, forcej.x);
            myAtomicAdd(&d_force[cur_j].y, forcej.y);
            myAtomicAdd(&d_force[cur_j].z, forcej.z);
            myAtomicAdd(&d_force[cur_j].w, forcej.w);
            }
        }
    // potential energy per particle must be halved
    forcei.w *= Scalar(0.5);
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    myAtomicAdd(&d_force[idx].x, forcei.x);
    myAtomicAdd(&d_force[idx].y, forcei.y);
    myAtomicAdd(&d_force[idx].z, forcei.z);
    myAtomicAdd(&d_force[idx].w, forcei.w);
    }

//! Kernel for zeroing forces before computation with atomic additions.
/*! \param d_force Device memory to write forces to
    \param N Number of particles in the system

*/
__global__ void gpu_zero_forces_kernel(Scalar4 *d_force,
                                       const unsigned int N)
    {
    // identify the particle we are supposed to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // zero the force
    d_force[idx] = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    }

//! Kernel driver that computes the three-body forces
/*! \param pair_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_triplet_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_triplet_forces(const tersoff_args_t& pair_args,
                                       const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        if (pair_args.compute_capability < 35 && pair_args.size_nlist > pair_args.max_tex1d_width)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_triplet_forces_kernel<evaluator, 1>);
            max_block_size = attr.maxThreadsPerBlock;
            }
        else
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_triplet_forces_kernel<evaluator, 0>);
            max_block_size = attr.maxThreadsPerBlock;
            }
        }

    unsigned int run_block_size = min(pair_args.block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( pair_args.N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind to texture
    if (pair_args.compute_capability < 35)
        {
        pdata_pos_tex.normalized = false;
        pdata_pos_tex.filterMode = cudaFilterModePoint;
        cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pair_args.d_pos, sizeof(Scalar4) * pair_args.N);
        if (error != cudaSuccess)
            return error;
        
        if (pair_args.size_nlist <= pair_args.max_tex1d_width)
            {
            nlist_tex.normalized = false;
            nlist_tex.filterMode = cudaFilterModePoint;
            error = cudaBindTexture(0, nlist_tex, pair_args.d_nlist, sizeof(unsigned int) * pair_args.size_nlist);
            if (error != cudaSuccess)
                return error;
            }
        }

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();

    // zero the forces
    gpu_zero_forces_kernel<<<grid, threads, shared_bytes>>>(pair_args.d_force,
                                                            pair_args.N);

    // compute the new forces
    if (pair_args.compute_capability < 35 && pair_args.size_nlist > pair_args.max_tex1d_width)
        {
        gpu_compute_triplet_forces_kernel<evaluator, 1>
          <<<grid, threads, shared_bytes>>>(pair_args.d_force,
                                            pair_args.N,
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
        gpu_compute_triplet_forces_kernel<evaluator, 0>
          <<<grid, threads, shared_bytes>>>(pair_args.d_force,
                                            pair_args.N,
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
    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_TERSOFF_GPU_CUH__
