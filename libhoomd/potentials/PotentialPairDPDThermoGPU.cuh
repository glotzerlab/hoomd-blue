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

// Maintainer: phillicl

/*! \file PotentialPairDPDThermoGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __POTENTIAL_PAIR_DPDTHERMO_CUH__
#define __POTENTIAL_PAIR_DPDTHERMO_CUH__

#include "ParticleData.cuh"
#include "EvaluatorPairDPDThermo.h"
#include "Index1D.h"
#include <cassert>

//! args struct for passing additional options to gpu_compute_dpd_forces
struct dpd_pair_args_t
    {
    //! Construct a dpd_pair_args_t
    dpd_pair_args_t(float4 *_d_force,
                    float *_d_virial,
                    const unsigned int _virial_pitch,
                    const unsigned int _N,
                    const Scalar4 *_d_pos,
                    const Scalar4 *_d_vel,
                    const BoxDim& _box,
                    const unsigned int *_d_n_neigh,
                    const unsigned int *_d_nlist,
                    const Index2D& _nli,
                    const float *_d_rcutsq,
                    const float *_d_ronsq,
                    const unsigned int _ntypes,
                    const unsigned int _block_size,
                    const unsigned int _seed,
                    const unsigned int _timestep,
                    const float _deltaT,
                    const float _T,
                    const unsigned int _shift_mode,
                    const unsigned int _compute_virial)
                        : d_force(_d_force),
                        d_virial(_d_virial),
                        virial_pitch(_virial_pitch),
                        N(_N),
                        d_pos(_d_pos),
                        d_vel(_d_vel),
                        box(_box),
                        d_n_neigh(_d_n_neigh),
                        d_nlist(_d_nlist),
                        nli(_nli),
                        d_rcutsq(_d_rcutsq),
                        d_ronsq(_d_ronsq),
                        ntypes(_ntypes),
                        block_size(_block_size),
                        seed(_seed),
                        timestep(_timestep),
                        deltaT(_deltaT),
                        T(_T),
                        shift_mode(_shift_mode),
                        compute_virial(_compute_virial)
        {
        };

    float4 *d_force;                //!< Force to write out
    float *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< Pitch of 2D virial array
    const unsigned int N;           //!< number of particles
    const Scalar4 *d_pos;           //!< particle positions
    const Scalar4 *d_vel;           //!< particle velocities
    const BoxDim& box;         //!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const Index2D& nli;             //!< Indexer for accessing d_nlist
    const float *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const float *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int seed;        //!< user provided seed for PRNG
    const unsigned int timestep;    //!< timestep of simulation
    const float deltaT;             //!< timestep size
    const float T;                  //!< temperature
    const unsigned int shift_mode;  //!< The potential energy shift mode
    const unsigned int compute_virial;  //!< Flag to indicate if virials should be computed
    };

#ifdef NVCC
//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_dpd_pos_tex;

//! Texture for reading particle velocities
texture<float4, 1, cudaReadModeElementType> pdata_dpd_vel_tex;


//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the potentials and
    forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param d_vel particle velocities on the GPU
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param nli Indexer for indexing \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param d_ronsq ron squared, stored per type pair
    \param d_seed user defined seed for PRNG
    \param d_timestep timestep of simulation
    \param d_deltaT timestep size
    \param d_T temperature
    \param ntypes Number of types in the simulation

    \a d_params, and \a d_rcutsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocatd for this kernel launch. The amount is
    (2*sizeof(float) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()

    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
    \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut.
    \tparam compute_virial When non-zero, the virial tensor is computed. When zero, the virial tensor is not computed.

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial >
__global__ void gpu_compute_dpd_forces_kernel(float4 *d_force,
                                              float *d_virial,
                                              const unsigned int virial_pitch,
                                              const unsigned int N,
                                              const Scalar4 *d_pos,
                                              const Scalar4 *d_vel,
                                              BoxDim box,
                                              const unsigned int *d_n_neigh,
                                              const unsigned int *d_nlist,
                                              const Index2D nli,
                                              const typename evaluator::param_type *d_params,
                                              const float *d_rcutsq,
                                              const float *d_ronsq,
                                              const unsigned int d_seed,
                                              const unsigned int d_timestep,
                                              const float d_deltaT,
                                              const float d_T,
                                              const int ntypes)
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
    float4 postypei = tex1Dfetch(pdata_dpd_pos_tex, idx);
    float3 posi = make_float3(postypei.x, postypei.y, postypei.z);

    // read in the velocity of our particle.
    // (MEM TRANSFER: 16 bytes)
    float4 velmassi = tex1Dfetch(pdata_dpd_vel_tex, idx);
    float3 veli = make_float3(velmassi.x, velmassi.y, velmassi.z);

    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0.0f;

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
            float4 postypej = tex1Dfetch(pdata_dpd_pos_tex, cur_j);
            float3 posj = make_float3(postypej.x, postypej.y, postypej.z);

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 velmassj = tex1Dfetch(pdata_dpd_vel_tex, cur_j);
            float3 velj = make_float3(velmassj.x, velmassj.y, velmassj.z);;

            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float3 dx = posi - posj;

            // apply periodic boundary conditions: (FLOPS 12)
            dx = box.minImage(dx);

            // calculate r squard (FLOPS: 5)
            float rsq = dot(dx,dx);

            // calculate dv (FLOPS: 3)
            float3 dv = veli - velj;

            float rdotv = dot(dx, dv);

            // access the per type pair parameters
            unsigned int typpair = typpair_idx(__float_as_int(postypei.w), __float_as_int(postypej.w));
            float rcutsq = s_rcutsq[typpair];
            typename evaluator::param_type param = s_params[typpair];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (shift_mode == 1)
                energy_shift = true;

            evaluator eval(rsq, rcutsq, param);

            // evaluate the potential
            float force_divr = 0.0f;
            float force_divr_cons = 0.0f;
            float pair_eng = 0.0f;

            // Special Potential Pair DPD Requirements
            eval.set_seed_ij_timestep(d_seed,idx,cur_j,d_timestep);
            eval.setDeltaT(d_deltaT);
            eval.setRDotV(rdotv);
            eval.setT(d_T);

            eval.evalForceEnergyThermo(force_divr, force_divr_cons, pair_eng, energy_shift);

            // calculate the virial (FLOPS: 3)
            if (compute_virial)
                {
                Scalar force_div2r_cons = Scalar(0.5) * force_divr_cons;
                virial[0] = dx.x * dx.x * force_div2r_cons;
                virial[1] = dx.x * dx.y * force_div2r_cons;
                virial[2] = dx.x * dx.z * force_div2r_cons;
                virial[3] = dx.y * dx.y * force_div2r_cons;
                virial[4] = dx.y * dx.z * force_div2r_cons;
                virial[5] = dx.z * dx.z * force_div2r_cons;
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
    d_force[idx] = force;

    if (compute_virial)
        {
        for (unsigned int i = 0; i < 6; i++)
            d_virial[i*virial_pitch+idx] = virial[i];
        }
    }

//! Kernel driver that computes pair DPD thermo forces on the GPU
/*! \param args Additional options
    \param d_params Per type-pair parameters for the evaluator

    This is just a driver function for gpu_compute_dpd_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_dpd_forces(const dpd_pair_args_t& args,
                                   const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(args.d_rcutsq);
    assert(args.d_ronsq);
    assert(args.ntypes > 0);

    // setup the grid to run the kernel
    dim3 grid( args.N / args.block_size + 1, 1, 1);
    dim3 threads(args.block_size, 1, 1);

    // bind the position texture
    pdata_dpd_pos_tex.normalized = false;
    pdata_dpd_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_dpd_pos_tex, args.d_pos, sizeof(float4)*args.N);
    if (error != cudaSuccess)
        return error;

    // bind the velocity texture
    pdata_dpd_vel_tex.normalized = false;
    pdata_dpd_vel_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, pdata_dpd_vel_tex, args.d_vel,  sizeof(float4)*args.N);
    if (error != cudaSuccess)
        return error;

    Index2D typpair_idx(args.ntypes);
    unsigned int shared_bytes = (2*sizeof(float) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();

    // run the kernel
    if (args.compute_virial)
        {
        switch (args.shift_mode)
            {
            case 0:
                gpu_compute_dpd_forces_kernel<evaluator, 0, 1>
                                    <<<grid, threads, shared_bytes>>>
                                    (args.d_force,
                                    args.d_virial,
                                    args.virial_pitch,
                                    args.N,
                                    args.d_pos,
                                    args.d_vel,
                                    args.box,
                                    args.d_n_neigh,
                                    args.d_nlist,
                                    args.nli,
                                    d_params,
                                    args.d_rcutsq,
                                    args.d_ronsq,
                                    args.seed,
                                    args.timestep,
                                    args.deltaT,
                                    args.T,
                                    args.ntypes);
                break;
            case 1:
                gpu_compute_dpd_forces_kernel<evaluator, 1, 1>
                                    <<<grid, threads, shared_bytes>>>
                                    (args.d_force,
                                    args.d_virial,
                                    args.virial_pitch,
                                    args.N,
                                    args.d_pos,
                                    args.d_vel,
                                    args.box,
                                    args.d_n_neigh,
                                    args.d_nlist,
                                    args.nli,
                                    d_params,
                                    args.d_rcutsq,
                                    args.d_ronsq,
                                    args.seed,
                                    args.timestep,
                                    args.deltaT,
                                    args.T,
                                    args.ntypes);
                break;
            default:
                return cudaErrorUnknown;
            }
        }
    else
        {
        switch (args.shift_mode)
            {
            case 0:
                gpu_compute_dpd_forces_kernel<evaluator, 0, 0>
                                    <<<grid, threads, shared_bytes>>>
                                    (args.d_force,
                                    args.d_virial,
                                    args.virial_pitch,
                                    args.N,
                                    args.d_pos,
                                    args.d_vel,
                                    args.box,
                                    args.d_n_neigh,
                                    args.d_nlist,
                                    args.nli,
                                    d_params,
                                    args.d_rcutsq,
                                    args.d_ronsq,
                                    args.seed,
                                    args.timestep,
                                    args.deltaT,
                                    args.T,
                                    args.ntypes);
                break;
            case 1:
                gpu_compute_dpd_forces_kernel<evaluator, 1, 0>
                                    <<<grid, threads, shared_bytes>>>
                                    (args.d_force,
                                    args.d_virial,
                                    args.virial_pitch,
                                    args.N,
                                    args.d_pos,
                                    args.d_vel,
                                    args.box,
                                    args.d_n_neigh,
                                    args.d_nlist,
                                    args.nli,
                                    d_params,
                                    args.d_rcutsq,
                                    args.d_ronsq,
                                    args.seed,
                                    args.timestep,
                                    args.deltaT,
                                    args.T,
                                    args.ntypes);
                break;
            default:
                return cudaErrorUnknown;
            }
        }

    return cudaSuccess;
    }

#endif

#endif // __POTENTIAL_PAIR_DPDTHERMO_CUH__

