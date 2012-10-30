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

#include "HOOMDMath.h"
#include "ParticleData.cuh"
#include "Index1D.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file PotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifndef __POTENTIAL_PAIR_GPU_CUH__
#define __POTENTIAL_PAIR_GPU_CUH__

//! Wrapps arguments to gpu_cgpf
struct pair_args_t
    {
    //! Construct a pair_args_t
    pair_args_t(float4 *_d_force,
              float *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const Scalar4 *_d_pos,
              const Scalar *_d_diameter,
              const Scalar *_d_charge,
              const BoxDim& _box,
              const unsigned int *_d_n_neigh,
              const unsigned int *_d_nlist,
              const Index2D& _nli,
              const float *_d_rcutsq, 
              const float *_d_ronsq,
              const unsigned int _ntypes,
              const unsigned int _block_size,
              const unsigned int _shift_mode,
              const unsigned int _compute_virial,
              const unsigned int _add_to_force)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  N(_N),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  box(_box),
                  d_n_neigh(_d_n_neigh),
                  d_nlist(_d_nlist),
                  nli(_nli),
                  d_rcutsq(_d_rcutsq),
                  d_ronsq(_d_ronsq),
                  ntypes(_ntypes),
                  block_size(_block_size),
                  shift_mode(_shift_mode),
                  compute_virial(_compute_virial),
                  add_to_force(_add_to_force)
        {
        };

    float4 *d_force;                //!< Force to write out
    float *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;           //!< number of particles
    const Scalar4 *d_pos;           //!< particle positions
    const Scalar *d_diameter;       //!< particle diameters
    const Scalar *d_charge;         //!< particle charges
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const Index2D& nli;             //!< Indexer for accessing d_nlist
    const float *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const float *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int shift_mode;  //!< The potential energy shift mode
    const unsigned int compute_virial;  //!< Flag to indicate if virials should be computed
    const bool add_to_force;        //!< True if we should add to the force array (instead of overwriting it)
    };

#ifdef NVCC
//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading particle diameters
texture<float, 1, cudaReadModeElementType> pdata_diam_tex;

//! Texture for reading particle charges
texture<float, 1, cudaReadModeElementType> pdata_charge_tex;

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
    \param nli Indexer for indexing \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_ronsq ron squared, stored per type pair
    \param ntypes Number of types in the simulation
    \param add_to_force True if we should add to the force array
    
    \a d_params, \a d_rcutsq, and \a d_ronsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocatd for this kernel launch. The amount is
    (2*sizeof(float) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()
    
    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
    \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
                       (See PotentialPair for a discussion on what that entails)
    \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.
    
    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial >
__global__ void gpu_compute_pair_forces_kernel(float4 *d_force,
                                               float *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const Scalar *d_diameter,
                                               const Scalar *d_charge,
                                               const BoxDim box,
                                               const unsigned int *d_n_neigh,
                                               const unsigned int *d_nlist,
                                               const Index2D nli,
                                               const typename evaluator::param_type *d_params,
                                               const float *d_rcutsq,
                                               const float *d_ronsq,
                                               const unsigned int ntypes,
                                               const bool add_to_force)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params = 
        (typename evaluator::param_type *)(&s_data[0]);
    float *s_rcutsq = (float *)(&s_data[num_typ_parameters*sizeof(evaluator::param_type)]);
    float *s_ronsq = (float *)(&s_data[num_typ_parameters*(sizeof(evaluator::param_type) + sizeof(float))]);

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
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    float4 postypei = d_pos[idx];
    float3 posi = make_float3(postypei.x, postypei.y, postypei.z);

    float di;
    if (evaluator::needsDiameter())
        di = tex1Dfetch(pdata_diam_tex, idx);
    else
        di += 1.0f; // shutup compiler warning
    float qi;
    if (evaluator::needsCharge())
        qi = tex1Dfetch(pdata_charge_tex, idx);
    else
        qi += 1.0f; // shutup compiler warning


    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virialxx = 0.0f;
    float virialxy = 0.0f;
    float virialxz = 0.0f;
    float virialyy = 0.0f;
    float virialyz = 0.0f;
    float virialzz = 0.0f;

    // prefetch neighbor index
    unsigned int cur_j = 0;
    unsigned int next_j = d_nlist[nli(idx, 0)];

    // loop over neighbors
    // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    #if (__CUDA_ARCH__ < 200)
    for (int neigh_idx = 0; neigh_idx < nli.getH(); neigh_idx++)
    #else
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
    #endif
        {
        #if (__CUDA_ARCH__ < 200)
        if (neigh_idx < n_neigh)
        #endif
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_j = next_j;
            next_j = d_nlist[nli(idx, neigh_idx+1)];

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 postypej = d_pos[cur_j];
            float3 posj = make_float3(postypej.x, postypej.y, postypej.z);

            float dj = 0.0f;
            if (evaluator::needsDiameter())
                dj = tex1Dfetch(pdata_diam_tex, cur_j);
            else
                dj += 1.0f; // shutup compiler warning

            float qj = 0.0f;
            if (evaluator::needsCharge())
                qj = tex1Dfetch(pdata_charge_tex, cur_j);
            else
                qj += 1.0f; // shutup compiler warning

            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float3 dx = posi - posj;

            // apply periodic boundary conditions: (FLOPS 12)
            dx = box.minImage(dx);

            // calculate r squard (FLOPS: 5)
            float rsq = dot(dx, dx);

            // access the per type pair parameters
            unsigned int typpair = typpair_idx(__float_as_int(postypei.w), __float_as_int(postypej.w));
            float rcutsq = s_rcutsq[typpair];
            typename evaluator::param_type param = s_params[typpair];
            float ronsq = 0.0f;
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
            float force_divr = 0.0f;
            float pair_eng = 0.0f;

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
                float force_div2r = 0.5f * force_divr;
                virialxx +=  dx.x * dx.x * force_div2r;
                virialxy +=  dx.x * dx.y * force_div2r;
                virialxz +=  dx.x * dx.z * force_div2r;
                virialyy +=  dx.y * dx.y * force_div2r;
                virialyz +=  dx.y * dx.z * force_div2r;
                virialzz +=  dx.z * dx.z * force_div2r;
                }

            // add up the force vector components (FLOPS: 7)
            #if (__CUDA_ARCH__ >= 200)
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;
            #else
            // fmad causes momentum drift here, prevent it from being used
            force.x += __fmul_rn(dx.x, force_divr);
            force.y += __fmul_rn(dx.y, force_divr);
            force.z += __fmul_rn(dx.z, force_divr);
            #endif

            force.w += pair_eng;
            }
        }

    // potential energy per particle must be halved
    force.w *= 0.5f;
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    if (add_to_force)
        {
        Scalar4 old_force = d_force[idx];
        force.x += old_force.x;
        force.y += old_force.y;
        force.z += old_force.z;
        force.w += old_force.w;
        }

    d_force[idx] = force;

    if (compute_virial)
        {
        if (add_to_force)
            {
            d_virial[0*virial_pitch+idx] += virialxx;
            d_virial[1*virial_pitch+idx] += virialxy;
            d_virial[2*virial_pitch+idx] += virialxz;
            d_virial[3*virial_pitch+idx] += virialyy;
            d_virial[4*virial_pitch+idx] += virialyz;
            d_virial[5*virial_pitch+idx] += virialzz;
            } 
        else
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

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair
    
    This is just a driver function for gpu_compute_pair_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_pair_forces(const pair_args_t& pair_args,
                                    const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);
    
    // setup the grid to run the kernel
    dim3 grid( pair_args.N / pair_args.block_size + 1, 1, 1);
    dim3 threads(pair_args.block_size, 1, 1);

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(float) + sizeof(typename evaluator::param_type)) 
                                * typpair_idx.getNumElements();
    
    // run the kernel
    if (pair_args.compute_virial)
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                gpu_compute_pair_forces_kernel<evaluator, 0, 1>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            case 1:
                gpu_compute_pair_forces_kernel<evaluator, 1, 1>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            case 2:
                gpu_compute_pair_forces_kernel<evaluator, 2, 1>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            default:
                return cudaErrorUnknown;
            }
        }
    else
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                gpu_compute_pair_forces_kernel<evaluator, 0, 0>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            case 1:
                gpu_compute_pair_forces_kernel<evaluator, 1, 0>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            case 2:
                gpu_compute_pair_forces_kernel<evaluator, 2, 0>
                  <<<grid, threads, shared_bytes>>>(pair_args.d_force, pair_args.d_virial, pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter, pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist, pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes, pair_args.add_to_force);
                break;
            default:
                return cudaErrorUnknown;
            }
        }

        
    return cudaSuccess;
    }

template<class evaluator>
cudaError_t gpu_compute_pair_forces_set_cache_config()
    {
    cudaError_t error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 0, 0>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 1, 0>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 2, 0>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 0, 1>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 1, 1>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;
    error = cudaFuncSetCacheConfig(gpu_compute_pair_forces_kernel<evaluator, 2, 1>, cudaFuncCachePreferL1);
    if (error != cudaSuccess)
        return error;

    return cudaSuccess;
    }
#endif
#endif // __POTENTIAL_PAIR_GPU_CUH__

