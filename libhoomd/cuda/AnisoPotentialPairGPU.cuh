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

// Maintainer:  jglaser

#include "HOOMDMath.h"
#include "ParticleData.cuh"
#include "Index1D.h"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file AnisoPotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the anisotropic ptl pair forces and torques
*/

#ifndef __ANISO_POTENTIAL_PAIR_GPU_CUH__
#define __ANISO_POTENTIAL_PAIR_GPU_CUH__

//! Maximum number of threads (width of a dpd_warp)
const unsigned int gpu_aniso_pair_force_max_tpp = 32;

//! CTA reduce
template<typename T>
__device__ static T aniso_warp_reduce(unsigned int NT, int tid, T x, volatile T* shared)
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


//! Wraps arguments to gpu_cgpf
struct a_pair_args_t
    {
    //! Construct a pair_args_t
    a_pair_args_t(Scalar4 *_d_force,
              Scalar4 *_d_torque,
              Scalar *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const unsigned int _n_max,
              const Scalar4 *_d_pos,
              const Scalar *_d_diameter,
              const Scalar *_d_charge,
              const Scalar4 *_d_orientation,
              const BoxDim& _box,
              const unsigned int *_d_n_neigh,
              const unsigned int *_d_nlist,
              const Index2D& _nli,
              const Scalar *_d_rcutsq,
              const unsigned int _ntypes,
              const unsigned int _block_size,
              const unsigned int _shift_mode,
              const unsigned int _compute_virial,
              const unsigned int _threads_per_particle)
                : d_force(_d_force),
                  d_torque(_d_torque),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  N(_N),
                  n_max(_n_max),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  d_orientation(_d_orientation),
                  box(_box),
                  d_n_neigh(_d_n_neigh),
                  d_nlist(_d_nlist),
                  nli(_nli),
                  d_rcutsq(_d_rcutsq),
                  ntypes(_ntypes),
                  block_size(_block_size),
                  shift_mode(_shift_mode),
                  compute_virial(_compute_virial),
                  threads_per_particle(_threads_per_particle)
        {
        };

    Scalar4 *d_force;                //!< Force to write out
    Scalar4 *d_torque;               //!< Torque to write out
    Scalar *d_virial;                //!< Virial to write out
    const unsigned int virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;           //!< number of particles
    const unsigned int n_max;       //!< maximum size of particle data arrays
    const Scalar4 *d_pos;           //!< particle positions
    const Scalar *d_diameter;       //!< particle diameters
    const Scalar *d_charge;         //!< particle charges
    const Scalar4 *d_orientation;    //!< particle orientation to compute forces over
    const BoxDim& box;              //!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const Index2D& nli;             //!< Indexer for accessing d_nlist
    const Scalar *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int shift_mode;  //!< The potential energy shift mode
    const unsigned int compute_virial;  //!< Flag to indicate if virials should be computed
    const unsigned int threads_per_particle; //!< Number of threads to launch per particle
    };

#ifdef NVCC
//! Texture for reading particle positions
scalar4_tex_t aniso_pdata_pos_tex;

//! Texture for reading particle quaternions
scalar4_tex_t aniso_pdata_quat_tex;

//! Texture for reading particle diameters
scalar_tex_t aniso_pdata_diam_tex;

//! Texture for reading particle charges
scalar_tex_t aniso_pdata_charge_tex;

//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the potentials and 
    forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_torque Device memory to write computed torques
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles in system
    \param d_pos particle positions
    \param d_diameter particle diameters
    \param d_charge particle charges
    \param d_orientation Quaternion data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param nli Indexer for indexing \a d_nlist
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param ntypes Number of types in the simulation
    \param tpp Number of threads per particle
    
    \a d_params and \a d_rcutsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
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
template< class evaluator, unsigned int shift_mode, unsigned int compute_virial >
__global__ void gpu_compute_pair_aniso_forces_kernel(Scalar4 *d_force,
                                                     Scalar4 *d_torque,
                                                     Scalar *d_virial,
                                                     const unsigned int virial_pitch,
                                                     const unsigned int N,
                                                     const Scalar4 *d_pos,
                                                     const Scalar *d_diameter,
                                                     const Scalar *d_charge,
                                                     const Scalar4 *d_orientation,
                                                     const BoxDim box,
                                                     const unsigned int *d_n_neigh,
                                                     const unsigned int *d_nlist,
                                                     const Index2D nli,
                                                     const typename evaluator::param_type *d_params,
                                                     const Scalar *d_rcutsq,
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

    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    Scalar4 postypei = texFetchScalar4(d_pos, aniso_pdata_pos_tex, idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
    Scalar4 quati = texFetchScalar4(d_orientation,aniso_pdata_quat_tex, idx);

    Scalar di;
    if (evaluator::needsDiameter())
        di = texFetchScalar(d_diameter, aniso_pdata_diam_tex, idx);
    else
        di += 1.0f; // shutup compiler warning
    Scalar qi;
    if (evaluator::needsCharge())
        qi = texFetchScalar(d_charge, aniso_pdata_charge_tex, idx);
    else
        qi += 1.0f; // shutup compiler warning

    // initialize the force to 0
    Scalar4 force = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
    Scalar4 torque = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
    Scalar virialxx = 0.0f;
    Scalar virialxy = 0.0f;
    Scalar virialxz = 0.0f;
    Scalar virialyy = 0.0f;
    Scalar virialyz = 0.0f;
    Scalar virialzz = 0.0f;

    // prefetch neighbor index
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
            // prefetch the next value and set the current one
            cur_j = next_j;
            if (neigh_idx+tpp < n_neigh)
                next_j = d_nlist[nli(idx, neigh_idx+tpp)];

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            Scalar4 postypej = texFetchScalar4(d_pos, aniso_pdata_pos_tex, cur_j);
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);
            Scalar4 quatj = texFetchScalar4(d_orientation, aniso_pdata_quat_tex, cur_j);

            Scalar dj = 0.0f;
            if (evaluator::needsDiameter())
                dj = texFetchScalar(d_diameter, aniso_pdata_diam_tex, cur_j);
            else
                dj += 1.0f; // shutup compiler warning

            Scalar qj = 0.0f;
            if (evaluator::needsCharge())
                qj = texFetchScalar(d_charge, aniso_pdata_charge_tex, cur_j);
            else
                qj += 1.0f; // shutup compiler warning

            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            Scalar3 dx = posi - posj;

            // apply periodic boundary conditions: (FLOPS 12)
            dx = box.minImage(dx);

            // calculate r squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // access the per type pair parameters
            unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w), __scalar_as_int(postypej.w));
            Scalar rcutsq = s_rcutsq[typpair];
            typename evaluator::param_type param = s_params[typpair];

            // design specifies that energies are shifted if
            // 1) shift mode is set to shift
            // or 2) shift mode is explor and ron > rcut
            bool energy_shift = false;
            if (shift_mode == 1)
                energy_shift = true;

            // evaluate the potential
            Scalar3 jforce = { 0.0f, 0.0f, 0.0f };
            Scalar3 torquei = { 0.0f, 0.0f, 0.0f };
            Scalar3 torquej = { 0.0f, 0.0f, 0.0f };
            Scalar pair_eng = 0.0f;

            // constructor call
            evaluator eval(dx, quati, quatj, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);

            // call evaluator
            eval.evaluate(jforce, pair_eng, energy_shift, torquei, torquej);

            // calculate the virial (FLOPS: ?)
            if (compute_virial)
                {
                Scalar3 jforce2 = Scalar(0.5)*jforce;
                virialxx +=  dx.x * jforce2.x;
                virialxy +=  dx.y * jforce2.y;
                virialxz +=  dx.z * jforce2.z;
                virialyy +=  dx.x * jforce2.y;
                virialyz +=  dx.x * jforce2.z;
                virialzz +=  dx.y * jforce2.z;
                }


            // add up the force vector components (FLOPS: 14)
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
    force.w *= 0.5f;

    // we need to access a separate portion of shared memory to avoid race conditions
    const unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
        * num_typ_parameters;

    // need to declare as volatile, because we are using warp-synchronous programming
    volatile Scalar *sh = (Scalar *) &s_data[shared_bytes];

    unsigned int cta_offs = (threadIdx.x/tpp)*tpp;

    // reduce force over threads in cta
    force.x = aniso_warp_reduce(tpp, threadIdx.x % tpp, force.x, &sh[cta_offs]);
    force.y = aniso_warp_reduce(tpp, threadIdx.x % tpp, force.y, &sh[cta_offs]);
    force.z = aniso_warp_reduce(tpp, threadIdx.x % tpp, force.z, &sh[cta_offs]);
    force.w = aniso_warp_reduce(tpp, threadIdx.x % tpp, force.w, &sh[cta_offs]);

    torque.x = aniso_warp_reduce(tpp, threadIdx.x % tpp, torque.x, &sh[cta_offs]);
    torque.y = aniso_warp_reduce(tpp, threadIdx.x % tpp, torque.y, &sh[cta_offs]);
    torque.z = aniso_warp_reduce(tpp, threadIdx.x % tpp, torque.z, &sh[cta_offs]);

    // now that the force calculation is complete, write out the result (MEM TRANSFER: ? bytes)
    if (threadIdx.x % tpp == 0)
        {
        d_force[idx] = force;
        d_torque[idx] = torque;
        }

    if (compute_virial)
        {
        virialxx = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialxx, &sh[cta_offs]);
        virialxy = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialxy, &sh[cta_offs]);
        virialxz = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialxz, &sh[cta_offs]);
        virialyy = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialyy, &sh[cta_offs]);
        virialyz = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialyz, &sh[cta_offs]);
        virialzz = aniso_warp_reduce(tpp, threadIdx.x % tpp, virialzz, &sh[cta_offs]);

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
int aniso_get_max_block_size(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_aniso_pair_force_max_tpp;
    return max_threads;
    }

template<typename T>
int aniso_get_compute_capability(T func)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, func);
    return attr.binaryVersion;
    }

void gpu_pair_aniso_force_bind_textures(const a_pair_args_t pair_args)
    {
    // bind the position texture
    aniso_pdata_pos_tex.normalized = false;
    aniso_pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, aniso_pdata_pos_tex, pair_args.d_pos, sizeof(Scalar4)*pair_args.n_max);

    // bind the position texture
    aniso_pdata_quat_tex.normalized = false;
    aniso_pdata_quat_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, aniso_pdata_quat_tex, pair_args.d_orientation, sizeof(Scalar4)*pair_args.n_max);

    // bind the diamter texture
    aniso_pdata_diam_tex.normalized = false;
    aniso_pdata_diam_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, aniso_pdata_diam_tex, pair_args.d_diameter, sizeof(Scalar) * pair_args.n_max);

    aniso_pdata_charge_tex.normalized = false;
    aniso_pdata_charge_tex.filterMode = cudaFilterModePoint;
    cudaBindTexture(0, aniso_pdata_charge_tex, pair_args.d_charge, sizeof(Scalar) * pair_args.n_max);
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param pair_args Other arugments to pass onto the kernel
    \param d_params Parameters for the potential, stored per type pair

    This is just a driver function for gpu_compute_pair_aniso_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_pair_aniso_forces(const a_pair_args_t& pair_args,
                                          const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();

    unsigned int block_size = pair_args.block_size;
    unsigned int tpp = pair_args.threads_per_particle;

    // run the kernel
    if (pair_args.compute_virial)
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                {
                static unsigned int max_block_size = UINT_MAX;
                static unsigned int sm = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = aniso_get_max_block_size(gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 1>);
                if (sm == UINT_MAX)
                    sm = aniso_get_compute_capability(gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 1>);

                if (sm < 35) gpu_pair_aniso_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (sm < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 1>
                    <<<grid, block_size, shared_bytes>>>(pair_args.d_force,
                                                  pair_args.d_torque,
                                                  pair_args.d_virial,
                                                  pair_args.virial_pitch,
                                                  pair_args.N,
                                                  pair_args.d_pos,
                                                  pair_args.d_diameter,
                                                  pair_args.d_charge,
                                                  pair_args.d_orientation,
                                                  pair_args.box,
                                                  pair_args.d_n_neigh,
                                                  pair_args.d_nlist,
                                                  pair_args.nli,
                                                  d_params,
                                                  pair_args.d_rcutsq,
                                                  pair_args.ntypes,
                                                  tpp);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                static unsigned int sm = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = aniso_get_max_block_size(gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 1>);
                if (sm == UINT_MAX)
                    sm = aniso_get_compute_capability(gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 1>);

                if (sm < 35) gpu_pair_aniso_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (sm < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 1>
                    <<<grid, block_size, shared_bytes>>>(pair_args.d_force,
                                                  pair_args.d_torque,
                                                  pair_args.d_virial,
                                                  pair_args.virial_pitch,
                                                  pair_args.N,
                                                  pair_args.d_pos,
                                                  pair_args.d_diameter,
                                                  pair_args.d_charge,
                                                  pair_args.d_orientation, 
                                                  pair_args.box, 
                                                  pair_args.d_n_neigh, 
                                                  pair_args.d_nlist, 
                                                  pair_args.nli, 
                                                  d_params, 
                                                  pair_args.d_rcutsq, 
                                                  pair_args.ntypes,
                                                  tpp);
                break;
                }
            default:
                return cudaErrorUnknown;
            }
        }
    else
        {
        switch (pair_args.shift_mode)
            {
            case 0:
                {
                static unsigned int max_block_size = UINT_MAX;
                static unsigned int sm = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = aniso_get_max_block_size(gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 0>);
                if (sm == UINT_MAX)
                    sm = aniso_get_compute_capability(gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 0>);

                if (sm < 35) gpu_pair_aniso_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (sm < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_aniso_forces_kernel<evaluator, 0, 0>
                    <<<grid, block_size, shared_bytes>>>(pair_args.d_force,
                                                  pair_args.d_torque,
                                                  pair_args.d_virial,
                                                  pair_args.virial_pitch,
                                                  pair_args.N,
                                                  pair_args.d_pos,
                                                  pair_args.d_diameter,
                                                  pair_args.d_charge,
                                                  pair_args.d_orientation,
                                                  pair_args.box,
                                                  pair_args.d_n_neigh,
                                                  pair_args.d_nlist,
                                                  pair_args.nli,
                                                  d_params,
                                                  pair_args.d_rcutsq,
                                                  pair_args.ntypes,
                                                  tpp);
                break;
                }
            case 1:
                {
                static unsigned int max_block_size = UINT_MAX;
                static unsigned int sm = UINT_MAX;
                if (max_block_size == UINT_MAX)
                    max_block_size = aniso_get_max_block_size(gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 0>);
                if (sm == UINT_MAX)
                    sm = aniso_get_compute_capability(gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 0>);

                if (sm < 35) gpu_pair_aniso_force_bind_textures(pair_args);

                block_size = block_size < max_block_size ? block_size : max_block_size;
                dim3 grid(pair_args.N / (block_size/tpp) + 1, 1, 1);
                if (sm < 30 && grid.x > 65535)
                    {
                    grid.y = grid.x/65535 + 1;
                    grid.x = 65535;
                    }

                shared_bytes += sizeof(Scalar)*block_size;

                gpu_compute_pair_aniso_forces_kernel<evaluator, 1, 0>
                    <<<grid, block_size, shared_bytes>>>(pair_args.d_force,
                                                  pair_args.d_torque,
                                                  pair_args.d_virial,
                                                  pair_args.virial_pitch,
                                                  pair_args.N,
                                                  pair_args.d_pos,
                                                  pair_args.d_diameter,
                                                  pair_args.d_charge,
                                                  pair_args.d_orientation,
                                                  pair_args.box,
                                                  pair_args.d_n_neigh,
                                                  pair_args.d_nlist,
                                                  pair_args.nli,
                                                  d_params,
                                                  pair_args.d_rcutsq,
                                                  pair_args.ntypes,
                                                  tpp);
                break;
                }
            default:
                return cudaErrorUnknown;
            }
        }
    return cudaSuccess;
    }
#endif

#endif // __ANISO_POTENTIAL_PAIR_GPU_CUH__
