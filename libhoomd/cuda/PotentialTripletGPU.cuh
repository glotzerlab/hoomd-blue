/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "HOOMDMath.h"
#include "ParticleData.cuh"
#include "Index1D.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file PotentialTripletGPU.cuh
    \brief Defines templated GPU kernel code for calculating certain three-body forces
*/

#ifndef __POTENTIAL_TRIPLET_GPU_CUH__
#define __POTENTIAL_TRIPLET_GPU_CUH__

//! Wrapps arguments to gpu_cgpf
struct triplet_args_t
    {
    //! Construct a triplet_args_t
    triplet_args_t(Scalar4 *_d_force,
				const unsigned int _N,
				const Scalar4 *_d_pos,
				const BoxDim& _box,
				const unsigned int *_d_n_neigh,
				const unsigned int *_d_nlist,
				const Index2D& _nli,
				const Scalar *_d_rcutsq,
				const Scalar *_d_ronsq,
				const unsigned int _ntypes,
				const unsigned int _block_size,
				const unsigned int _shift_mode)
                : d_force(_d_force),
				  N(_N),
				  d_pos(_d_pos),
				  box(_box),
                  d_n_neigh(_d_n_neigh),
                  d_nlist(_d_nlist),
                  nli(_nli),
                  d_rcutsq(_d_rcutsq),
                  d_ronsq(_d_ronsq),
                  ntypes(_ntypes),
                  block_size(_block_size),
                  shift_mode(_shift_mode)
        {
        };

	Scalar4 *d_force;                //!< Force to write out
	const unsigned int N;			//!< Number of particles
	const Scalar4 *d_pos;			//!< particle positions
	const BoxDim& box;				//!< Simulation box in GPU format
    const unsigned int *d_n_neigh;  //!< Device array listing the number of neighbors on each particle
    const unsigned int *d_nlist;    //!< Device array listing the neighbors of each particle
    const Index2D& nli;             //!< Indexer for accessing d_nlist
    const Scalar *d_rcutsq;          //!< Device array listing r_cut squared per particle type pair
    const Scalar *d_ronsq;           //!< Device array listing r_on squared per particle type pair
    const unsigned int ntypes;      //!< Number of particle types in the simulation
    const unsigned int block_size;  //!< Block size to execute
    const unsigned int shift_mode;  //!< The potential energy shift mode
    };


#ifdef NVCC
#ifdef SINGLE_PRECISION
//! Texture for reading particle positions
texture<Scalar4, 1, cudaReadModeElementType> pdata_pos_tex;

#elif defined ENABLE_TEXTURES
//! Texture for reading particle positions
texture<int4, 1, cudaReadModeElementType> pdata_pos_tex;

#endif

#ifndef SINGLE_PRECISION
//! atomicAdd function for double-precision floating point numbers
/*! This function is only used when hoomd is compiled for double precision on the GPU.
	
	\param address Address to write the double to
	\param val Value to add to address
*/
static __device__ inline double atomicAdd(double* address, double val)
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
#endif

//! Kernel for calculating the Tersoff forces
/*! This kernel is called to calculate the forces on all N particles. Actual evaluation of the potentials and
    forces for each pair is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
	\param N Number of particles in the system
	\param d_pos Positions of all the particles
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
    \tparam shift_mode This parameter is not used by the triplet potentials

    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator, unsigned int shift_mode >
__global__ void gpu_compute_triplet_forces_kernel(Scalar4 *d_force,
												  const unsigned int N,
												  const Scalar4 *d_pos,
												  const BoxDim box,
                                                  const unsigned int *d_n_neigh,
                                                  const unsigned int *d_nlist,
                                                  const Index2D nli,
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
	// in single precision we will do a texture read, in double a global memory read
	#ifdef ENABLE_TEXTURES
	Scalar4 postypei = fetchScalar4Tex(pdata_pos_tex, idx);
	#else
	Scalar4 postypei = d_pos[idx];
	#endif
	Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

    // initialize the force to 0
    Scalar4 forcei = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

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
            next_j = d_nlist[nli(idx, neigh_idx + 1)];

            // read the position of j (MEM TRANSFER: 16 bytes)
			#ifdef ENABLE_TEXTURES
			Scalar4 postypej = fetchScalar4Tex(pdata_pos_tex, cur_j);
			#else
			Scalar4 postypej = d_pos[cur_j];
			#endif
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
                unsigned int next_k = d_nlist[nli(idx, 0)];
                #if (__CUDA_ARCH__ < 200)
                for (int neigh_idy = 0; neigh_idy < nli.getH(); neigh_idy++)
                #else
                for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                #endif
                {
                    #if (__CUDA_ARCH__ < 200)
                    if (neigh_idy < n_neigh)
                    #endif
                    {
                        // read the current index of k and prefetch the next one
                        cur_k = next_k;
                        next_k = d_nlist[nli(idx, neigh_idy+1)];

						// get the position of neighbor k
						#ifdef ENABLE_TEXTURES
						Scalar4 postypek = fetchScalar4Tex(pdata_pos_tex, cur_k);
						#else
						Scalar4 postypek = d_pos[cur_k];
						#endif
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
                                cos_th = dot(dxij, dxik) * rsqrtf(rij_sq * rik_sq);
                            else cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the evaluator
                            eval.setRik(rik_sq);
                            if (evaluator::needsAngle())
                                eval.setAngle(cos_th);

                            // compute the partial chi term
                            eval.evalChi(chi);
                        }
                    }
                }
                // evaluate the force and energy from the ij interaction
                Scalar force_divr = Scalar(0.0);
                Scalar potential_eng = Scalar(0.0);
                Scalar bij = Scalar(0.0);
                eval.evalForceij(fR, fA, chi, bij, force_divr, potential_eng);

                // add the forces and energies to their respective particles
                Scalar2 v_coeffs = make_scalar2(Scalar(1.0 / 6.0) * rij_sq, Scalar(0.0));
                #if (__CUDA_ARCH__ >= 200)
                forcei.x += dxij.x * force_divr;
                forcei.y += dxij.y * force_divr;
                forcei.z += dxij.z * force_divr;

                forcej.x -= dxij.x * force_divr;
                forcej.y -= dxij.y * force_divr;
                forcej.z -= dxij.z * force_divr;
                forcej.w += potential_eng;
                #else
                forcei.x += __fmul_rn(dxij.x, force_divr);
                forcei.y += __fmul_rn(dxij.y, force_divr);
                forcei.z += __fmul_rn(dxij.z, force_divr);

                forcej.x += Scalar(1.0); // shuts up the compiler warning
                #endif
                forcei.w += potential_eng;

                // now evaluate the force from the ik interactions
                cur_k = 0;
                next_k = d_nlist[nli(idx, 0)];
                #if (__CUDA_ARCH__ < 200)
                for (int neigh_idy = 0; neigh_idy < nli.getH(); neigh_idy++)
                #else
                for (int neigh_idy = 0; neigh_idy < n_neigh; neigh_idy++)
                #endif
                {
                    #if (__CUDA_ARCH__ < 200)
                    if (neigh_idy < n_neigh)
                    #endif
                    {
                        // read the current neighbor index and prefetch the next one
                        cur_k = next_k;
                        next_k = d_nlist[nli(idx, neigh_idy+1)];

						// get the position of neighbor k
						#ifdef ENABLE_TEXTURES
						Scalar4 postypek = fetchScalar4Tex(pdata_pos_tex, cur_k);
						#else
						Scalar4 postypek = d_pos[cur_k];
						#endif
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
                                cos_th = dot(dxij, dxik) * rsqrtf(rij_sq * rik_sq);
                            else cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the evaluator
                            eval.setRik(rik_sq);
                            if (evaluator::needsAngle())
                                eval.setAngle(cos_th);

                            // compute the force
                            Scalar4 force_divr_ij = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            Scalar4 force_divr_ik = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            bool evaluatedjk = eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_ik);

                            if (evaluatedjk)
                            {
                                // add the forces to their respective particles
                                v_coeffs.y = Scalar(1.0 / 6.0) * rik_sq;
                                #if (__CUDA_ARCH__ >= 200)
                                forcei.x += force_divr_ij.x * dxij.x + force_divr_ik.x * dxik.x;
                                forcei.y += force_divr_ij.x * dxij.y + force_divr_ik.x * dxik.y;
                                forcei.z += force_divr_ij.x * dxij.z + force_divr_ik.x * dxik.z;

                                forcej.x += force_divr_ij.y * dxij.x + force_divr_ik.y * dxik.x;
                                forcej.y += force_divr_ij.y * dxij.y + force_divr_ik.y * dxik.y;
                                forcej.z += force_divr_ij.y * dxij.z + force_divr_ik.y * dxik.z;

                                forcek.x += force_divr_ij.z * dxij.x + force_divr_ik.z * dxik.x;
                                forcek.y += force_divr_ij.z * dxij.y + force_divr_ik.z * dxik.y;
                                forcek.z += force_divr_ij.z * dxij.z + force_divr_ik.z * dxik.z;

                                atomicAdd(&d_force[cur_k].x, forcek.x);
                                atomicAdd(&d_force[cur_k].y, forcek.y);
                                atomicAdd(&d_force[cur_k].z, forcek.z);
                                #else
                                forcei.x += __fmul_rn(dxij.x, force_divr_ij.x) + __fmul_rn(dxik.x, force_divr_ik.x);
                                forcei.y += __fmul_rn(dxij.y, force_divr_ij.x) + __fmul_rn(dxik.y, force_divr_ik.x);
                                forcei.z += __fmul_rn(dxij.z, force_divr_ij.x) + __fmul_rn(dxik.z, force_divr_ik.x);

                                forcek.x += Scalar(1.0); // shuts up the compiler warning
                                #endif
                            }
                        }
                    }
                }
                // on Fermi hardware we can use atomicAdd to gain some speed
                // otherwise we need to loop over neighbors of j
                #if (__CUDA_ARCH__ >= 200)
                // potential energy of j must be halved
                forcej.w *= Scalar(0.5);
                // write out the result for particle j
                atomicAdd(&d_force[cur_j].x, forcej.x);
                atomicAdd(&d_force[cur_j].y, forcej.y);
                atomicAdd(&d_force[cur_j].z, forcej.z);
                atomicAdd(&d_force[cur_j].w, forcej.w);
                #else
                // now we have to compute the force with i as a secondary/tertiary particle
                // first consider i as the secondary particle and j as the primary particle
                // rji is -rij
                dxij.x *= -Scalar(1.0);
                dxij.y *= -Scalar(1.0);
                dxij.z *= -Scalar(1.0);

                // the fR and fA already computed are still valid, so there is no point in recomputing them

                // recompute chi by looping over neighbors of j
                unsigned int n_neighj = d_n_neigh[cur_j];
                chi = Scalar(0.0);
                cur_k = 0;
                next_k = d_nlist[nli(cur_j, 0)];
                for (int neigh_idy = 0; neigh_idy < nli.getH(); neigh_idy++)
                {
                    if (neigh_idy < n_neighj)
                    {
                        // read the index of k and prefetch the next one
                        cur_k = next_k;
                        next_k = d_nlist[nli(cur_j, neigh_idy+1)];

                        if (cur_k != idx)
                        {
                            // get the position of neighbor k
							#ifdef ENABLE_TEXTURES
							Scalar4 postypek = fetchScalar4Tex(pdata_pos_tex, cur_k);
							#else
							Scalar4 postypek = d_pos[cur_k];
							#endif
							Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                            // compute rjk
							Scalar3 dxjk = posj - posk;

                            // apply the periodic boundary conditions
							dxjk = box.minImage(dxjk);

                            // compute rjk_sq
							Scalar rjk_sq = dot(dxjk, dxjk);

                            // compute the bond angle (if needed)
                            Scalar cos_th = Scalar(0.0);
                            if (evaluator::needsAngle())
                                cos_th = dot(dxij, dxjk) * rsqrtf(rij_sq * rjk_sq);
                            else cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the evaluator
                            eval.setRik(rjk_sq);
                            if (evaluator::needsAngle())
                                eval.setAngle(cos_th);

                            // evaluate chi
                            eval.evalChi(chi);
                        }
                    }
                }
                // now compute the ji force
                eval.evalForceij(fR, fA, chi, bij, force_divr, potential_eng);

                // add the force and energy to particle i
                forcei.x -= __fmul_rn(dxij.x, force_divr);
                forcei.y -= __fmul_rn(dxij.y, force_divr);
                forcei.z -= __fmul_rn(dxij.z, force_divr);
                forcei.w += potential_eng;

                // now compute the jk force
                cur_k = 0;
                next_k = d_nlist[nli(cur_j, 0)];
                for (int neigh_idy = 0; neigh_idy < nli.getH(); neigh_idy++)
                {
                    if (neigh_idy < n_neighj)
                    {
                        // get the index of k and prefecth the next one
                        cur_k = next_k;
                        next_k = d_nlist[nli(cur_j, neigh_idy+1)];

                        if (cur_k != idx)
                        {
                            // get the position of k
							#ifdef ENABLE_TEXTURES
							Scalar4 postypek = fetchScalar4Tex(pdata_pos_tex, cur_k);
							#else
							Scalar4 postypek = d_pos[cur_k];
							#endif
							Scalar3 posk = make_scalar3(postypek.x, postypek.y, postypek.z);

                            // compute rjk
							Scalar3 dxjk = posj - posk;

                            // apply periodic boundary conditions
							dxjk = box.minImage(dxjk);

                            // compute rjk_sq
							Scalar rjk_sq = dot(dxjk, dxjk);

                            // compute the bond angle (if needed)
                            Scalar cos_th = Scalar(0.0);
                            if (evaluator::needsAngle())
                                cos_th = dot(dxij, dxjk) * rsqrtf(rij_sq * rjk_sq);
                            else cos_th += Scalar(1.0); // shuts up the compiler warning

                            // set up the evaluator
                            eval.setRik(rjk_sq);
                            if (evaluator::needsAngle())
                                eval.setAngle(cos_th);

                            // evaluate the force
                            Scalar4 force_divr_ij = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            Scalar4 force_divr_jk = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
                            eval.evalForceik(fR, fA, chi, bij, force_divr_ij, force_divr_jk);

                            // add the force to particle i
                            forcei.x += __fmul_rn(dxjk.x, force_divr_jk.y) + __fmul_rn(dxij.x, force_divr_ij.y);
                            forcei.y += __fmul_rn(dxjk.y, force_divr_jk.y) + __fmul_rn(dxij.y, force_divr_ij.y);
                            forcei.z += __fmul_rn(dxjk.z, force_divr_jk.y) + __fmul_rn(dxij.z, force_divr_ij.y);
                        }
                    }
                }
                #endif
            }
        }
    }
    // potential energy per particle must be halved
    forcei.w *= Scalar(0.5);
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    #if (__CUDA_ARCH__ >= 200)
    atomicAdd(&d_force[idx].x, forcei.x);
    atomicAdd(&d_force[idx].y, forcei.y);
    atomicAdd(&d_force[idx].z, forcei.z);
    atomicAdd(&d_force[idx].w, forcei.w);
    #else
    d_force[idx] = forcei;
    #endif
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
cudaError_t gpu_compute_triplet_forces(const triplet_args_t& pair_args,
                                       const typename evaluator::param_type *d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.d_ronsq);
    assert(pair_args.ntypes > 0);

    // setup the grid to run the kernel
    dim3 grid( pair_args.N / pair_args.block_size + 1, 1, 1);
    dim3 threads(pair_args.block_size, 1, 1);

    // bind the position texture
	#ifdef ENABLE_TEXTURES
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pair_args.d_pos, sizeof(Scalar4) * pair_args.N);
    if (error != cudaSuccess)
        return error;
	#endif

    Index2D typpair_idx(pair_args.ntypes);
    unsigned int shared_bytes = (2*sizeof(Scalar) + sizeof(typename evaluator::param_type))
                                * typpair_idx.getNumElements();

    // zero the forces
    gpu_zero_forces_kernel<<<grid, threads, shared_bytes>>>(pair_args.d_force,
                                                            pair_args.N);

    // compute the new forces
    gpu_compute_triplet_forces_kernel<evaluator, 0>
      <<<grid, threads, shared_bytes>>>(pair_args.d_force,
										pair_args.N,
										pair_args.d_pos,
                                        pair_args.box,
                                        pair_args.d_n_neigh,
                                        pair_args.d_nlist,
                                        pair_args.nli,
                                        d_params,
                                        pair_args.d_rcutsq,
                                        pair_args.d_ronsq,
                                        pair_args.ntypes);
    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_TRIPLET_GPU_CUH__

