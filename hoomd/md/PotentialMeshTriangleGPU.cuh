// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/BondedGroupData.cuh"

#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif // __HIPCC__

#include <assert.h>

/*! \file PotentialPairGPU.cuh
    \brief Defines templated GPU kernel code for calculating the pair forces.
*/

#ifndef __POTENTIALMESHTRIANGLEPAIR_GPU_CUH__
#define __POTENTIALMESHTRIANGLEPAIR_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Maximum number of threads (width of a warp)
// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_pair_force_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_pair_force_max_tpp = 64;
#endif

//! Wraps arguments to gpu_cgpf
struct meshtriangle_args_t
    {
    //! Construct a pair_args_t
    meshtriangle_args_t(Scalar4* _d_force,
                Scalar* _d_virial,
                const size_t _virial_pitch,
                const unsigned int _N,
                const unsigned int _n_max,
                const Scalar4* _d_pos,
                const BoxDim& _box,
                const unsigned int* _d_n_neigh,
                const unsigned int* _d_nlist,
                const size_t* _d_head_list,
                const Scalar* _d_rcutsq,
                const size_t _size_neigh_list,
                const unsigned int _ntypes,
                const unsigned int _block_size,
                const unsigned int _compute_virial,
                const hipDeviceProp_t& _devprop)
        : d_force(_d_force), d_virial(_d_virial), virial_pitch(_virial_pitch), N(_N), n_max(_n_max),
          d_pos(_d_pos), box(_box), d_n_neigh(_d_n_neigh), d_nlist(_d_nlist),
          d_head_list(_d_head_list), d_rcutsq(_d_rcutsq),
          size_neigh_list(_size_neigh_list), ntypes(_ntypes), block_size(_block_size),
	   compute_virial(_compute_virial), devprop(_devprop) { };

    Scalar4* d_force;          //!< Force to write out
    Scalar* d_virial;          //!< Virial to write out
    const size_t virial_pitch; //!< The pitch of the 2D array of virial matrix elements
    const unsigned int N;      //!< number of particles
    const unsigned int n_max;  //!< Max size of pdata arrays
    const Scalar4* d_pos;      //!< particle positions
    const BoxDim box;          //!< Simulation box in GPU format
    const unsigned int*
        d_n_neigh;                //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist;  //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;    //!< Head list indexes for accessing d_nlist
    const Scalar* d_rcutsq;       //!< Device array listing r_cut squared per particle type pair
    const size_t size_neigh_list; //!< Size of the neighbor list for texture binding
    const unsigned int ntypes;    //!< Number of particle types in the simulation
    const unsigned int block_size;           //!< Block size to execute
    const unsigned int compute_virial;       //!< Flag to indicate if virials should be computed
    const hipDeviceProp_t& devprop;          //!< CUDA device properties
    };

#ifdef __HIPCC__

//! Kernel for calculating bond forces
/*! This kernel is called to calculate the bond forces on all N particles. Actual evaluation of the
   potentials and forces for each bond is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N Number of particles in the system
    \param d_pos particle positions on the GPU
    \param d_charge particle charges
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param pitch Pitch of 2D bond list
    \param bpos_list List of positions in bonds stored on the GPU
    \param n_bonds_list List of numbers of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_params Parameters for the potential, stored per bond type
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
   evaluated


    Certain options are controlled via template parameters to avoid the performance hit when they
   are not enabled. \tparam evaluator EvaluatorBond class to evaluate V(r) and -delta V(r)/r

*/

template<class evaluator, unsigned int compute_virial, int tpp, bool enable_shared_cache>
__global__ void gpu_compute_mesh_triangle_triangles_force_kernel(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const Scalar4* d_pos,
                                               const BoxDim box,
                                      	       const unsigned int* d_n_neigh,
                                      	       const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                      	       const Scalar* d_rcutsq,
					       const unsigned int ntypes,
                                               const group_storage<3>* tlist,
                                               const Index2D tlist_idx,
                                               const unsigned int* tpos_list,
                                               const unsigned int* n_triangles_list,
                                               const typename evaluator::param_type* d_params,
					       unsigned int max_extra_bytes)

    {
    // shared array for per bond type parameters
    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);

    Scalar* s_rcutsq
        = (Scalar*)(&s_data[ntypes * sizeof(typename evaluator::param_type)]);

    auto s_extra = reinterpret_cast<char*>(s_rcutsq + ntypes);

    if (enable_shared_cache)
        {
        // load in per bond type parameters
        for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < ntypes)
                {
                s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
                }
            }

        unsigned int param_size
            = ntypes * sizeof(typename evaluator::param_type) / sizeof(int);
        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < param_size)
                {
                ((int*)s_params)[cur_offset + threadIdx.x]
                    = ((int*)d_params)[cur_offset + threadIdx.x];
                }
            }

        __syncthreads();

        // initialize extra shared mem
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int cur_pair = 0; cur_pair < ntypes; ++cur_pair)
            s_params[cur_pair].load_shared(s_extra, available_bytes);

        __syncthreads();
        }

    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    bool active = true;
    if (idx >= N)
        {
        // need to mask this thread, but still participate in warp-level reduction
        active = false;
        }

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));;
    Scalar virialxx = 0;
    Scalar virialxy = 0;
    Scalar virialxz = 0;
    Scalar virialyy = 0;
    Scalar virialyz = 0;
    Scalar virialzz = 0;

    if(active)
    {

    force = d_force[idx];
    virialxx = d_virial[idx];
    virialxy = d_virial[1 * virial_pitch + idx];
    virialxz = d_virial[2 * virial_pitch + idx];
    virialyy = d_virial[3 * virial_pitch + idx];
    virialyz = d_virial[4 * virial_pitch + idx];
    virialzz = d_virial[5 * virial_pitch + idx];

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_triangles = n_triangles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

    unsigned int Nsize0 = (unsigned int)d_n_neigh[idx];
    size_t myHead0 = d_head_list[idx];

    unsigned int cur_j = 0;

    unsigned int next_j(0);
    next_j = threadIdx.x % tpp < Nsize0 ? __ldg(d_nlist + myHead0 + threadIdx.x % tpp) : 0;

    // loop over neighbors
    for (int neigh_idx = threadIdx.x % tpp; neigh_idx < Nsize0; neigh_idx += tpp)
        {
            // read the current neighbor index
            cur_j = next_j;

            if (neigh_idx + tpp < Nsize0)
                {
                next_j = __ldg(d_nlist + myHead0 + neigh_idx + tpp);
                }
            // get the neighbor's position
            Scalar4 postypej = __ldg(d_pos + cur_j);
            Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

	    unsigned int typej = __scalar_as_int(postypej.w);
	    const typename evaluator::param_type* param;
	    Scalar rcutsq;
	    if (enable_shared_cache)
	        {
	        rcutsq = s_rcutsq[typej];  
	        param = s_params + typej;
	        }
	    else
	        {
	        rcutsq = d_rcutsq[typej];
	        param = d_params + typej;
	        }

	    if( rcutsq == 0) continue;

    // loop over all triangles
	    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
		{
		group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

		int cur_triangle_b = cur_triangle.idx[0];
		const unsigned int Nsize1 = (unsigned int)d_n_neigh[cur_triangle_b];
		if (Nsize1 == 0)
		       continue;	
		int cur_triangle_c = cur_triangle.idx[1];
		const unsigned int Nsize2 = (unsigned int)d_n_neigh[cur_triangle_c];
		if (Nsize2 == 0)
		       continue;	

		Scalar4 bb_postype = d_pos[cur_triangle_b];
		Scalar4 cc_postype = d_pos[cur_triangle_c];
		Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
		Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);

		Scalar3 dab;
		dab.x = pos_a.x - pos_b.x;
		dab.y = pos_a.y - pos_b.y;
		dab.z = pos_a.z - pos_b.z;

		Scalar3 dac;
		dac.x = pos_a.x - pos_c.x;
		dac.y = pos_a.y - pos_c.y;
		dac.z = pos_a.z - pos_c.z;

		dab = box.minImage(dab);
		dac = box.minImage(dac);

		Scalar3 normal_dir;
		normal_dir.x = dab.y * dac.z - dab.z * dac.y;
		normal_dir.y = dab.z * dac.x - dab.x * dac.z;
		normal_dir.z = dab.x * dac.y - dab.y * dac.x;

		Scalar normal_norm = fast::rsqrt(normal_dir.x*normal_dir.x+normal_dir.y*normal_dir.y+normal_dir.z*normal_dir.z);

		normal_dir.x = normal_dir.x*normal_norm;
		normal_dir.y = normal_dir.y*normal_norm;
		normal_dir.z = normal_dir.z*normal_norm;

		Scalar3 dbj;
		dbj.x = pos_b.x - posj.x;
		dbj.y = pos_b.y - posj.y;
		dbj.z = pos_b.z - posj.z;
		dbj = box.minImage(dbj);

		Scalar dbj_norm = dot(dbj,normal_dir);

		Scalar rsq = dbj_norm*dbj_norm;

		if( rcutsq < rsq) continue;

		Scalar3 dx = normal_dir*dbj_norm;

		Scalar Area_a;

		Scalar3 dcj;
		dcj.x = pos_c.x - posj.x;
		dcj.y = pos_c.y - posj.y;
		dcj.z = pos_c.z - posj.z;
		dcj = box.minImage(dcj);

		Scalar3 dbjcj; 
		dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
		dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
		dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;

		Area_a = dot(dbjcj,normal_dir);

		if( Area_a < 0)
			continue;
		else
			{
			Scalar3 daj;
			daj.x = pos_a.x - posj.x;
			daj.y = pos_a.y - posj.y;
			daj.z = pos_a.z - posj.z;
			daj = box.minImage(daj);

			Scalar3 dcjaj; 
			dcjaj.x = dcj.y*daj.z - dcj.z*daj.y;
			dcjaj.y = dcj.z*daj.x - dcj.x*daj.z;
			dcjaj.z = dcj.x*daj.y - dcj.y*daj.x;
			Scalar Area_b = dot(dcjaj,normal_dir);

			Scalar3 dajbj; 
			dajbj.x = daj.y*dbj.z - daj.z*dbj.y;
			dajbj.y = daj.z*dbj.x - daj.x*dbj.z;
			dajbj.z = daj.x*dbj.y - daj.y*dbj.x;

			if( dot(dcjaj,normal_dir) <0)
				{
				if( dot(dajbj,normal_dir) <0)
					{
					dx = daj;
					Area_a = 1;
					}
				else
					{
					Scalar normal_ac = fast::rsqrt(dot(dac,dac));
					Scalar3 nac = dac*normal_ac;

					Scalar length_ac = dot(daj,nac);
					Scalar ratio_ac = length_ac*normal_ac;

					if(ratio_ac > 1)
						continue;
					else
						{
						if(ratio_ac < 0)
							{
							dx = daj;
							Area_a = 1;
							}
						else
							{
							dx = daj - length_ac*nac;
							Area_a = ratio_ac;
							}
						}
					}

				}
			else
				{
				if( dot(dajbj,normal_dir) <0)
					{
					Scalar normal_ab = fast::rsqrt(dot(dab,dab));
					Scalar3 nab = dab*normal_ab;
					Scalar length_ab = dot(daj,nab);
					Scalar ratio_ab = length_ab*normal_ab;

					if(ratio_ab > 1)
						continue;
					else
						{
						if(ratio_ab < 0)
							{
							dx = daj;
							Area_a = 1;
							}
						else
							{
							dx = daj - length_ab*nab;
							Area_a = ratio_ab;
							}
						}
					}
				else
					{
					Area_a *= normal_norm;
					}

				}

			}
		rsq = dot(dx,dx);

		Scalar force_divr = Scalar(0.0);
		Scalar bond_eng = Scalar(0.0);

		evaluator eval(rsq, rcutsq, *param);

		bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng, false);

		if (evaluated)
		    {
		    force_divr *= Area_a;
		    // add up the virial (double counting, multiply by 0.5)
		    if (compute_virial)
		    {
		    virialxx += pos_a.x * dx.x * force_divr; // xx
		    virialxy += pos_a.x * dx.y * force_divr; // xy
		    virialxz += pos_a.x * dx.z * force_divr; // xz
		    virialyy += pos_a.y * dx.y * force_divr; // yy
		    virialyz += pos_a.y * dx.z * force_divr; // yz
		    virialzz += pos_a.z * dx.z * force_divr; // zz
		    }

		    // add up the forces
		    force.x += dx.x * force_divr;
		    force.y += dx.y * force_divr;
		    force.z += dx.z * force_divr;
		    // energy is double counted: multiply by 0.5
		    force.w += bond_eng * Area_a * Scalar(0.5);


		    }
		}
    	}

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

template<typename T> int get_max_block_size(T func)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)func);
    int max_threads = attr.maxThreadsPerBlock;
    // number of threads has to be multiple of warp size
    max_threads -= max_threads % gpu_pair_force_max_tpp;
    return max_threads;
    }



template<class evaluator, unsigned int compute_virial, int tpp>
struct MeshTriangleForceComputeKernel
    {
    //! Launcher for the pair force kernel
    /*!
     * \param pair_args Other arguments to pass onto the kernel
     * \param N Number of particles
     * \param d_params Parameters for the potential, stored per type pair
     */

    static void launch(const meshtriangle_args_t& pair_args,
                       const unsigned int threads_per_particle,
                       const typename evaluator::param_type* d_params,
		       const group_storage<3>* tlist,
                       const unsigned int* tpos_list,
                       const Index2D tlist_idx,
                       const unsigned int* n_triangles_list
		       )
        {
        if (tpp == threads_per_particle)
            {
            unsigned int block_size = pair_args.block_size;
            bool enable_shared_cache = true;

            size_t param_shared_bytes
                = (sizeof(Scalar) + sizeof(typename evaluator::param_type))
                  * pair_args.ntypes;

            unsigned int max_block_size;
            max_block_size
                = get_max_block_size(gpu_compute_mesh_triangle_triangles_force_kernel<evaluator,
                                                                           compute_virial,
                                                                           tpp,
                                                                           true>);

            hipFuncAttributes attr;
            hipFuncGetAttributes(
                &attr,
                reinterpret_cast<const void*>(&gpu_compute_mesh_triangle_triangles_force_kernel<evaluator,
                                                                                     compute_virial,
                                                                                     tpp,
                                                                                     true>));

            if (param_shared_bytes + attr.sharedSizeBytes > pair_args.devprop.sharedMemPerBlock)
                {
                param_shared_bytes = 0;
                enable_shared_cache = false;
                }

            unsigned int max_extra_bytes = static_cast<unsigned int>(
                pair_args.devprop.sharedMemPerBlock - param_shared_bytes - attr.sharedSizeBytes);

            // determine dynamically requested shared memory in nested managed arrays
            char* ptr = nullptr;
            unsigned int available_bytes = max_extra_bytes;
            for (unsigned int i = 0; i < pair_args.ntypes; ++i)
                {
                d_params[i].allocate_shared(ptr, available_bytes);
                }

            unsigned int extra_shared_bytes = max_extra_bytes - available_bytes;

            block_size = block_size < max_block_size ? block_size : max_block_size;
            dim3 grid(pair_args.N / (block_size / tpp) + 1, 1, 1);

            if (enable_shared_cache)
                {
                hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator,
                                                                          compute_virial,
                                                                          tpp,
                                                                          true>),
                                   dim3(grid),
                                   dim3(block_size),
                                   param_shared_bytes + extra_shared_bytes,
                                   0,
                                   pair_args.d_force,
                                   pair_args.d_virial,
                                   pair_args.virial_pitch,
                                   pair_args.N,
                                   pair_args.d_pos,
                                   pair_args.box,
                                   pair_args.d_n_neigh,
                                   pair_args.d_nlist,
                                   pair_args.d_head_list,
                                   pair_args.d_rcutsq,
                                   pair_args.ntypes,
				   tlist,
				   tlist_idx,
				   tpos_list,
				   n_triangles_list,
                                   d_params,
                                   max_extra_bytes);
                }
            else
                {
                hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator,
                                                                          compute_virial,
                                                                          tpp,
                                                                          false>),
                                   dim3(grid),
                                   dim3(block_size),
                                   param_shared_bytes + extra_shared_bytes,
                                   0,
                                   pair_args.d_force,
                                   pair_args.d_virial,
                                   pair_args.virial_pitch,
                                   pair_args.N,
                                   pair_args.d_pos,
                                   pair_args.box,
                                   pair_args.d_n_neigh,
                                   pair_args.d_nlist,
                                   pair_args.d_head_list,
                                   pair_args.d_rcutsq,
                                   pair_args.ntypes,
				   tlist,
				   tlist_idx,
				   tpos_list,
				   n_triangles_list,
                                   d_params,
                                   max_extra_bytes);
                }
            }
        else
            {
            MeshTriangleForceComputeKernel<evaluator, compute_virial, tpp / 2>::launch(
                pair_args,
		threads_per_particle,
                d_params,
                tlist,
                tpos_list,
                tlist_idx,
                n_triangles_list);
            }
        }
    };

//! Template specialization to do nothing for the tpp = 0 case
template<class evaluator, unsigned int compute_virial>
struct MeshTriangleForceComputeKernel<evaluator, compute_virial, 0>
    {
    static void launch(const meshtriangle_args_t& pair_args,
                       const unsigned int threads_per_particle,
                       const typename evaluator::param_type* d_params,
		       const group_storage<3>* tlist,
                       const unsigned int* tpos_list,
                       const Index2D tlist_idx,
                       const unsigned int* n_triangles_list)
        {
        // do nothing
        }
    };





//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param meshtriangle_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per bond type
    \param d_flags flags on the device - a 1 will be written if evaluation
                   of forces failed for any bond

    This is just a driver function for gpu_compute_bond_forces_kernel(), see it for details.
*/
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_triangles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
			const unsigned int threads_per_particle,
                        const typename evaluator::param_type* d_params,
			const group_storage<3>* tlist,                                             
                        const unsigned int* tpos_list,
                        const Index2D tlist_idx,
                        const unsigned int* n_triangles_list)
    {
    assert(d_params);
    assert(meshtriangle_args.d_rcutsq);
    assert(meshtriangle_args.ntypes > 0);

    // check that block_size is valid
    assert(meshtriangle_args.block_size != 0);


    //run the kernel
    if(meshtriangle_args.compute_virial)
            MeshTriangleForceComputeKernel<evaluator, 1, gpu_pair_force_max_tpp>::launch(meshtriangle_args,
                                                                                    threads_per_particle,
                                                                                    d_params,
										    tlist,
										    tpos_list,
										    tlist_idx,
										    n_triangles_list);
    else
            MeshTriangleForceComputeKernel<evaluator, 0, gpu_pair_force_max_tpp>::launch(meshtriangle_args,
                                                                                    threads_per_particle,
                                                                                    d_params,
										    tlist,
										    tpos_list,
										    tlist_idx,
										    n_triangles_list);

    return hipSuccess;
    }

//! Kernel for calculating bond forces
/*! This kernel is called to calculate the bond forces on all N particles. Actual evaluation of the
   potentials and forces for each bond is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N Number of particles in the system
    \param d_pos particle positions on the GPU
    \param d_charge particle charges
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param pitch Pitch of 2D bond list
    \param bpos_list List of positions in bonds stored on the GPU
    \param n_bonds_list List of numbers of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_params Parameters for the potential, stored per bond type
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
   evaluated


    Certain options are controlled via template parameters to avoid the performance hit when they
   are not enabled. \tparam evaluator EvaluatorBond class to evaluate V(r) and -delta V(r)/r

*/

template<class evaluator, unsigned int compute_virial, bool enable_shared_cache>
__global__ void gpu_compute_mesh_triangle_particles_force_kernel(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const Scalar4* d_pos,
                                               const BoxDim box,
                                      	       const unsigned int* d_n_neigh,
                                      	       const unsigned int* d_nlist,
                                               const size_t* d_head_list,
                                      	       const Scalar* d_rcutsq,
					       const unsigned int ntypes,
                                               const unsigned int* d_tag,
                                               const unsigned int* d_n_triang,
                                               const unsigned int* d_trianglist,
                                               const unsigned int* d_head_triang,
                                               const typename evaluator::param_type* d_params)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    // shared array for per bond type parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);

    Scalar* s_rcutsq
        = (Scalar*)(&s_data[ntypes * sizeof(typename evaluator::param_type)]);

    if (enable_shared_cache)
        {
        // load in per bond type parameters
        for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < ntypes)
                {
                s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
                }
            }

        __syncthreads();
        }

    if (idx >= N)
        return;


    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pi = make_scalar3(postype.x, postype.y, postype.z);

    unsigned int typei = __scalar_as_int(postype.w);
    const typename evaluator::param_type* param;
    Scalar rcutsq;
    if (enable_shared_cache)
    {
	rcutsq = s_rcutsq[typei];
	param = s_params + typei;
    }
    else
    {
   	rcutsq = d_rcutsq[typei];
	param = d_params + typei;
    }

    const unsigned int Nsize = (unsigned int)d_n_neigh[idx];
    const size_t myHead = d_head_list[idx];

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = 0;

   unsigned int counter = 0;
   uint2 highest_n[1024];
   for (unsigned int k = 0; k < Nsize; k++)
   	{
       uint2 nk;
       nk.x = d_nlist[myHead + k];
       nk.y = d_tag[nk.x];
       if (d_n_triang[nk.y] > 0)
        {
        	highest_n[counter] = nk;
        	counter++;
        }
       }

    if( counter > 2)
    {
    //unsigned int zahl = 0;
    for (unsigned int k = 0; k < counter-2; k++)
        {
	unsigned int aj = highest_n[k].x;
	unsigned int trianglesx = highest_n[k].y;

        unsigned int Nj_tri = d_n_triang[trianglesx];
    
        unsigned int headj = d_head_triang[trianglesx];
    
        for (unsigned int kk = k+1; kk < counter-1; kk++)
            {

	    unsigned int bj = highest_n[kk].x;
	    unsigned int trianglesy = highest_n[kk].y;

            unsigned int Njj_tri = d_n_triang[trianglesy];

            unsigned int headjj = d_head_triang[trianglesy];
    
            for(unsigned int j_tri=0; j_tri < Nj_tri; j_tri++)
        	{
        	unsigned int tri_idx = d_trianglist[headj+j_tri];
        	for(unsigned int jj_tri=0; jj_tri < Njj_tri; jj_tri++)
        	    {
			
        	    unsigned int tri_idy = d_trianglist[headjj+jj_tri];
        	    if( tri_idx == tri_idy )
        	        {
        		for (unsigned int kkk = kk+1; kkk < counter; kkk++)
        		   {
			   unsigned int cj = highest_n[kkk].x;
			   unsigned int trianglesz = highest_n[kkk].y;
        		   unsigned int Njjj_tri = d_n_triang[trianglesz];

        		   unsigned int headjjj = d_head_triang[trianglesz];
            
        		   for(unsigned int jjj_tri=0; jjj_tri < Njjj_tri; jjj_tri++)
        		      {
        	              unsigned int tri_idz = d_trianglist[headjjj+jjj_tri];
        		      if(tri_idx == tri_idz)
        			 {

        			 Scalar4 postypea = __ldg(d_pos + aj);		    
        			 Scalar3 pos_a = make_scalar3(postypea.x, postypea.y, postypea.z);

        			 Scalar3 daj;
        			 daj.x = pos_a.x - pi.x;
        			 daj.y = pos_a.y - pi.y;
        			 daj.z = pos_a.z - pi.z;
        			 daj = box.minImage(daj);

        			 Scalar4 postypeb = __ldg(d_pos + bj);		    
        			 Scalar3 pos_b = make_scalar3(postypeb.x, postypeb.y, postypeb.z);

        			 Scalar4 postypec = __ldg(d_pos + cj);		    
        			 Scalar3 pos_c = make_scalar3(postypec.x, postypec.y, postypec.z);


        			 Scalar3 dab;
        			 dab.x = pos_a.x - pos_b.x;
        			 dab.y = pos_a.y - pos_b.y;
        			 dab.z = pos_a.z - pos_b.z;

        			 Scalar3 dac;
        			 dac.x = pos_a.x - pos_c.x;
        			 dac.y = pos_a.y - pos_c.y;
        			 dac.z = pos_a.z - pos_c.z;

        			 dab = box.minImage(dab);
        			 dac = box.minImage(dac);

        			 Scalar3 normal_dir;
        			 normal_dir.x = dab.y * dac.z - dab.z * dac.y;
        			 normal_dir.y = dab.z * dac.x - dab.x * dac.z;
        			 normal_dir.z = dab.x * dac.y - dab.y * dac.x;

        			 Scalar normal_norm = fast::rsqrt(dot(normal_dir,normal_dir));

        			 normal_dir.x = normal_dir.x*normal_norm;
        			 normal_dir.y = normal_dir.y*normal_norm;
        			 normal_dir.z = normal_dir.z*normal_norm;

        			 Scalar daj_norm = dot(daj,normal_dir);

        			 Scalar rsq = daj_norm*daj_norm;

        			 if( rcutsq < rsq) continue;

        			 Scalar3 dx = normal_dir*daj_norm;

        			 Scalar3 dbj = daj - dab;

        			 Scalar3 dajbj; 
        			 dajbj.x = daj.y*dbj.z - daj.z*dbj.y;
        			 dajbj.y = daj.z*dbj.x - daj.x*dbj.z;
        			 dajbj.z = daj.x*dbj.y - daj.y*dbj.x;

        			 Scalar Area_c = dot(dajbj,normal_dir);

				 Scalar Area_a = 0;
        			 Scalar3 dcj = daj - dac;

        			 Scalar3 dcjaj; 
        			 dcjaj.x = dcj.y*daj.z - dcj.z*daj.y;
        			 dcjaj.y = dcj.z*daj.x - dcj.x*daj.z;
        			 dcjaj.z = dcj.x*daj.y - dcj.y*daj.x;
        			 Scalar Area_b = dot(dcjaj,normal_dir);

        			 if(Area_c <0)
        			 {
        			 	if(Area_b <0)
        			 		dx = daj;
        			 	else
        			 	{	
        			 		Scalar3 dbjcj; 
        			 		dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
        			 		dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
        			 		dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;

        			 		if(dot(dbjcj,normal_dir) < 0)
        			 			dx = dbj;
        			 		else
        			 		{
        			 			Scalar normal_ab = fast::rsqrt(dot(dab,dab));
        			 			Scalar3 nab = dab*normal_ab;
        			 			Scalar length_ab = dot(daj,nab);
        			 			Scalar ratio_ab = length_ab*normal_ab;

        			 			if(ratio_ab < 0)
							       dx = daj;
							else{
							       if(ratio_ab > 1) 
								       dx = dbj;
							       else
								       dx = daj - length_ab*nab;
							}
        			 		}
        			 	}
        			 }
        			 else{
        			 	Scalar3 dbjcj; 
        			 	dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
        			 	dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
        			 	dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;
        			 	Area_a = dot(dbjcj,normal_dir);
        			 	if(Area_b <0)
        			 	{
        			 		if(Area_a < 0)
        			 			dx = dcj;
        			 		else
        			 		{
							Scalar normal_ac = fast::rsqrt(dot(dac,dac));
							Scalar3 nac = dac*normal_ac;
        			 			Scalar length_ac = dot(daj,nac);
        			 			Scalar ratio_ac = length_ac*normal_ac;

        			 			if(ratio_ac < 0)
							       dx = daj;
							else{
							       if(ratio_ac > 1) 
								       dx = dcj;
							       else
								       dx = daj - length_ac*nac;
							}
        			 		}
        			 	}
        			 	else
        			 	{
        			 		if(Area_a <0)
        			 		{
							Scalar3 dbc = dac-dab;
							Scalar normal_bc = fast::rsqrt(dot(dbc,dbc));
							Scalar3 nbc = dbc*normal_bc;
        			 			Scalar length_bc = dot(dbj,nbc);
        			 			Scalar ratio_bc = length_bc*normal_bc;

        			 			if(ratio_bc < 0)
							       dx = dbj;
							else{
							       if(ratio_bc > 1) 
								       dx = dcj;
							       else
								       dx = dbj - length_bc*nbc;
							}
        			 		}
        			 	}
        			 }

        			rsq = dot(dx,dx);

        			Scalar force_divr = Scalar(0.0);
        			Scalar bond_eng = Scalar(0.0);

        			evaluator eval(rsq, rcutsq, *param);

        			bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng, false);

        			if (evaluated)
        			    {
        			    // add up the virial (double counting, multiply by 0.5)
        			    virial[0] -= pi.x * dx.x * force_divr; // xx
        			    virial[1] -= pi.x * dx.y * force_divr; // xy
        			    virial[2] -= pi.x * dx.z * force_divr; // xz
        			    virial[3] -= pi.y * dx.y * force_divr; // yy
        			    virial[4] -= pi.y * dx.z * force_divr; // yz
        			    virial[5] -= pi.z * dx.z * force_divr; // zz

        			    // add up the forces
        			    force.x -= dx.x * force_divr;
        			    force.y -= dx.y * force_divr;
        			    force.z -= dx.z * force_divr;
        			    // energy is double counted: multiply by 0.5
        			    force.w += bond_eng * Scalar(0.5);
        			    }
        			 }
        		      }
        		   }
        	        }
        	    }
                 }
             }
        }

    }


    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];

    }

template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_particles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
                        const typename evaluator::param_type* d_params,
                        const unsigned int* d_tag,
                        const unsigned int* d_n_triang,
                        const unsigned int* d_trianglist,
                        const unsigned int* d_head_triang)
    {
    assert(d_params);
    assert(meshtriangle_args.d_rcutsq);
    assert(meshtriangle_args.ntypes > 0);

    // check that block_size is valid
    assert(meshtriangle_args.block_size != 0);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    if(meshtriangle_args.compute_virial)
	    hipFuncGetAttributes(&attr,
				 reinterpret_cast<const void*>(
				     &gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 1, true>));
   else
	    hipFuncGetAttributes(&attr,
				 reinterpret_cast<const void*>(
				     &gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 0, true>));
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(meshtriangle_args.block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(meshtriangle_args.N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    size_t shared_bytes = sizeof(typename evaluator::param_type) * meshtriangle_args.ntypes;

    bool enable_shared_cache = true;

    if (shared_bytes > meshtriangle_args.devprop.sharedMemPerBlock)
        {
        enable_shared_cache = false;
        shared_bytes = 0;
        }

    // run the kernel
    if (enable_shared_cache)
        {
    	if(meshtriangle_args.compute_virial)
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 1, true>),
                           grid,
                           threads,
                           shared_bytes,
                           0,
                           meshtriangle_args.d_force,
                           meshtriangle_args.d_virial,
                           meshtriangle_args.virial_pitch,
                           meshtriangle_args.N,
                           meshtriangle_args.d_pos,
                           meshtriangle_args.box,
                           meshtriangle_args.d_n_neigh,
                           meshtriangle_args.d_nlist,
                           meshtriangle_args.d_head_list,
                           meshtriangle_args.d_rcutsq,
                           meshtriangle_args.ntypes,
                           d_tag,
                           d_n_triang,
                           d_trianglist,
                           d_head_triang,
                           d_params);
        else
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 0, true>),
                           grid,
                           threads,
                           shared_bytes,
                           0,
                           meshtriangle_args.d_force,
                           meshtriangle_args.d_virial,
                           meshtriangle_args.virial_pitch,
                           meshtriangle_args.N,
                           meshtriangle_args.d_pos,
                           meshtriangle_args.box,
                           meshtriangle_args.d_n_neigh,
                           meshtriangle_args.d_nlist,
                           meshtriangle_args.d_head_list,
                           meshtriangle_args.d_rcutsq,
                           meshtriangle_args.ntypes,
                           d_tag,
                           d_n_triang,
                           d_trianglist,
                           d_head_triang,
                           d_params);
        }
    else
        {
    	if(meshtriangle_args.compute_virial)
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 1, false>),
                           grid,
                           threads,
                           shared_bytes,
        		   0,
                           meshtriangle_args.d_force,
                           meshtriangle_args.d_virial,
                           meshtriangle_args.virial_pitch,
                           meshtriangle_args.N,
                           meshtriangle_args.d_pos,
                           meshtriangle_args.box,
                           meshtriangle_args.d_n_neigh,
                           meshtriangle_args.d_nlist,
                           meshtriangle_args.d_head_list,
                           meshtriangle_args.d_rcutsq,
                           meshtriangle_args.ntypes,
                           d_tag,
                           d_n_triang,
                           d_trianglist,
                           d_head_triang,
                           d_params);
        else
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, 0, false>),
                           grid,
                           threads,
                           shared_bytes,
        		   0,
                           meshtriangle_args.d_force,
                           meshtriangle_args.d_virial,
                           meshtriangle_args.virial_pitch,
                           meshtriangle_args.N,
                           meshtriangle_args.d_pos,
                           meshtriangle_args.box,
                           meshtriangle_args.d_n_neigh,
                           meshtriangle_args.d_nlist,
                           meshtriangle_args.d_head_list,
                           meshtriangle_args.d_rcutsq,
                           meshtriangle_args.ntypes,
                           d_tag,
                           d_n_triang,
                           d_trianglist,
                           d_head_triang,
                           d_params);
        }

    return hipSuccess;
    }
#else
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_triangles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
		        const unsigned int threads_per_particle,
                        const typename evaluator::param_type* d_params,
                        const group_storage<3>* tlist,
                        const unsigned int* tpos_list,
                        const Index2D tlist_idx,
                        const unsigned int* n_triangles_list);

template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_particles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
                        const typename evaluator::param_type* d_params,
                        const unsigned int* d_tag,
                        const unsigned int* d_n_triang,
                        const unsigned int* d_trianglist,
                        const unsigned int* d_head_triang);
#endif


    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIALMESHTRIANGLEPAIR_GPU_CUH__
