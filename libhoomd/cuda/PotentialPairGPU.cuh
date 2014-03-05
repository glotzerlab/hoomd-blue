/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

// Maintainer: joaander

#include "HOOMDMath.h"
#include "TextureTools.h"
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

//! Maximum number of threads (width of a warp)
const unsigned int gpu_pair_force_max_tpp = 32;

//! CTA reduce
template<typename T>
__device__ static T warp_reduce(unsigned int NT, int tid, T x, volatile T* shared)
    {
    shared[tid] = x;

    for (int dest_count = NT/2; dest_count >= 1; dest_count /= 2)
        {
        if (tid < dest_count)
            {
            x += shared[dest_count + tid];
            shared[tid] = x;
            }
         }
    T total = shared[0];
    return total;
    }

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
              const Index2D& _nli,
              const Scalar *_d_rcutsq,
              const Scalar *_d_ronsq,
              const unsigned int _ntypes,
              const unsigned int _block_size,
              const unsigned int _shift_mode,
              const unsigned int _compute_virial,
              const unsigned int _threads_per_particle,
              const unsigned int _compute_capability)
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
                  nli(_nli),
                  d_rcutsq(_d_rcutsq),
                  d_ronsq(_d_ronsq),
                  ntypes(_ntypes),
                  block_size(_block_size),
                  shift_mode(_shift_mode),
                  compute_virial(_compute_virial),
                  threads_per_particle(_threads_per_particle),
                  compute_capability(_compute_capability)
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
    const Index2D& nli;             //!< Indexer for accessing d_nlist
    const Scalar *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const Scalar *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int shift_mode;  //!< The potential energy shift mode
    const unsigned int compute_virial;  //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads per particle (maximum: 1 warp)
    const unsigned int compute_capability;  //!< Compute capability (20 30 35, ...)
    };

#ifdef NVCC
//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;

//! Texture for reading particle diameters
scalar_tex_t pdata_diam_tex;

//! Texture for reading particle charges
scalar_tex_t pdata_charge_tex;

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
    \param nli Indexer for indexing \a d_nlist
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

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial>
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
                                               const Index2D nli,
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

    if (idx >= N)
        return;

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


    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virialxx = Scalar(0.0);
    Scalar virialxy = Scalar(0.0);
    Scalar virialxz = Scalar(0.0);
    Scalar virialyy = Scalar(0.0);
    Scalar virialyz = Scalar(0.0);
    Scalar virialzz = Scalar(0.0);

    unsigned int cur_j = 0;
    unsigned int next_j = threadIdx.x%tpp < n_neigh ?
        d_nlist[nli(idx, threadIdx.x%tpp)] : 0;
    // loop over neighbors
    // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    #if (__CUDA_ARCH__ < 200)
    for (int neigh_idx = threadIdx.x%tpp; neigh_idx < nli.getH(); neigh_idx+=tpp)
    #else
    for (int neigh_idx = threadIdx.x%tpp; neigh_idx < n_neigh; neigh_idx+=tpp)
    #endif
        {
        #if (__CUDA_ARCH__ < 200)
        if (neigh_idx < n_neigh)
        #endif
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            cur_j = next_j;
            if (neigh_idx+tpp < n_neigh)
                next_j = d_nlist[nli(idx, neigh_idx+tpp)];

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
    force.w *= Scalar(0.5);

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
    if (threadIdx.x % tpp == 0)
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
        if (threadIdx.x %tpp == 0)
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
    cudaFuncGetAttributes(&attr, func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_pair_force_max_tpp;
    return max_threads;
    }

void gpu_pair_force_bind_textures(const pair_args_t pair_args)
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

    // threads per particle

    // setup the grid to run the kernel
    unsigned int block_size = pair_args.block_size;
    unsigned int tpp = pair_args.threads_per_particle;

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();


    // Launch kernel
    if (pair_args.compute_virial)
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 0, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 0, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 1, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 1, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
                break;
                }
            case 2:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 2, 1>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 2, 1>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
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
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 0, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 0, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 1, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 1, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
                break;
                }
            case 2:
                {
                static unsigned int max_block_size = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = get_max_block_size(gpu_compute_pair_forces_shared_kernel<evaluator, 2, 0>);

                if (pair_args.compute_capability < 35) gpu_pair_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (pair_args.compute_capability < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_forces_shared_kernel<evaluator, 2, 0>
                  <<<grid, block_size, shared_bytes>>>(pair_args.d_force, pair_args.d_virial,
                  pair_args.virial_pitch, pair_args.N, pair_args.d_pos, pair_args.d_diameter,
                  pair_args.d_charge, pair_args.box, pair_args.d_n_neigh, pair_args.d_nlist,
                  pair_args.nli, d_params, pair_args.d_rcutsq, pair_args.d_ronsq, pair_args.ntypes,
                  tpp);
                break;
                }
            default:
                break;
            }
        }

    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_PAIR_GPU_CUH__
