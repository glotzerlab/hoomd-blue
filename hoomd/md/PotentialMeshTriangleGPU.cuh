// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "hoomd/BondedGroupData.cuh"


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

template<class evaluator, unsigned int compute_virial, bool enable_shared_cache>
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

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_triangles = n_triangles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

    const unsigned int Nsize0 = (unsigned int)d_n_neigh[idx];
    const size_t myHead0 = d_head_list[idx];

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);


    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles && Nsize0 >0; triangle_idx++)
        {
        group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_b = cur_triangle.idx[0];
        int cur_triangle_c = cur_triangle.idx[1];

        const unsigned int Nsize1 = (unsigned int)d_n_neigh[cur_triangle_b];
	if (Nsize1 == 0)
	       continue;	
        const unsigned int Nsize2 = (unsigned int)d_n_neigh[cur_triangle_c];
	if (Nsize2 == 0)
	       continue;	
        const size_t myHead1 = d_head_list[cur_triangle_b];
        const size_t myHead2 = d_head_list[cur_triangle_c];

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

        Scalar3 dbc = dac-dab;

	Scalar normal_ab = fast::rsqrt(dot(dab,dab));
	Scalar normal_ac = fast::rsqrt(dot(dac,dac));
	Scalar normal_bc = fast::rsqrt(dot(dbc,dbc));

        Scalar3 nab = dab*normal_ab;
        Scalar3 nac = dac*normal_ac;
        Scalar3 nbc = dbc*normal_bc;

        Scalar3 normal_dir;
        normal_dir.x = dab.y * dac.z - dab.z * dac.y;
        normal_dir.y = dab.z * dac.x - dab.x * dac.z;
        normal_dir.z = dab.x * dac.y - dab.y * dac.x;

	Scalar normal_norm = fast::rsqrt(dot(normal_dir,normal_dir));

	normal_dir.x = normal_dir.x*normal_norm;
	normal_dir.y = normal_dir.y*normal_norm;
	normal_dir.z = normal_dir.z*normal_norm;

	for (unsigned int idx0 = 0; idx0 < Nsize0; idx0++)
		{
		unsigned int cur_j = d_nlist[myHead0 + idx0];

		bool insert = false;

		for (unsigned int idx1 = 0; idx1 < Nsize1 && !insert; idx1++)
			if( cur_j ==  d_nlist[myHead1 + idx1] ) insert = true;
		if(insert)
			{
			insert = false;
			for (unsigned int idx2 = 0; idx2 < Nsize2 && !insert; idx2++)
				if( cur_j ==  d_nlist[myHead2 + idx2] ) insert = true;
			if(insert)
				{
				insert = false;
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
							Scalar length_ac = dot(daj,nac);
							Scalar ratio_ac = length_ac*normal_ac;

							if(ratio_ac < 1 && ratio_ac > 0 )
								{
								dx = daj - length_ac*nac;
								Area_a = ratio_ac;
								}
							else continue;
							}

						}
					else
						{
						if( dot(dajbj,normal_dir) <0)
							{
							Scalar length_ab = dot(daj,nab);
							Scalar ratio_ab = length_ab*normal_ab;

							if(ratio_ab < 1 && ratio_ab > 0 )
								{
								dx = daj - length_ab*nab;
								Area_a = ratio_ab;
								}
							else continue;
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

				evaluator eval(rsq, 0, *param);

				bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng, false);

				if (evaluated)
				    {
				    force_divr *= Area_a;
				    // add up the virial (double counting, multiply by 0.5)
				    Scalar force_div2r = force_divr / Scalar(2.0);
				    virial[0] += pos_a.x * dx.x * force_div2r; // xx
				    virial[1] += pos_a.x * dx.y * force_div2r; // xy
				    virial[2] += pos_a.x * dx.z * force_div2r; // xz
				    virial[3] += pos_a.y * dx.y * force_div2r; // yy
				    virial[4] += pos_a.y * dx.z * force_div2r; // yz
				    virial[5] += pos_a.z * dx.z * force_div2r; // zz

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
    }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

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

    unsigned int max_block_size;
    hipFuncAttributes attr;
    if(meshtriangle_args.compute_virial)
	    hipFuncGetAttributes(&attr,
				 reinterpret_cast<const void*>(
				     &gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 1, true>));
   else
	    hipFuncGetAttributes(&attr,
				 reinterpret_cast<const void*>(
				     &gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 0, true>));
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

    //run the kernel
    if (enable_shared_cache)
        {
    	if(meshtriangle_args.compute_virial)
		hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 1, true>),
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
				   tlist,
				   tlist_idx,
				   tpos_list,
				   n_triangles_list,
				   d_params);
	else
		hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 0, true>),
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
				   tlist,
				   tlist_idx,
				   tpos_list,
				   n_triangles_list,
				   d_params);
        }
    else
        {
    	if(meshtriangle_args.compute_virial)
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 1, false>),
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
                           tlist,
			   tlist_idx,
			   tpos_list,
			   n_triangles_list,
                           d_params);
	else
        	hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, 0, false>),
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
                           tlist,
			   tlist_idx,
			   tpos_list,
			   n_triangles_list,
                           d_params);
        }

    return hipSuccess;
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
    //hipFuncGetAttributes(&attr,
    //                     reinterpret_cast<const void*>(
    //                         &gpu_compute_mesh_triangle_particles_force_kernel<evaluator, true>));
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
   // if (enable_shared_cache)
   //     {
   //     hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, compute_virial, true>),
   //                        grid,
   //                        threads,
   //                        shared_bytes,
   //                        0,
   //                        meshtriangle_args.d_force,
   //                        meshtriangle_args.d_virial,
   //                        meshtriangle_args.virial_pitch,
   //                        meshtriangle_args.N,
   //                        meshtriangle_args.d_pos,
   //                        meshtriangle_args.box,
   //                        meshtriangle_args.d_n_neigh,
   //                        meshtriangle_args.d_nlist,
   //                        meshtriangle_args.d_head_list,
   //                        meshtriangle_args.d_rcutsq,
   //                        meshtriangle_args.d_gpu_meshtrianglelist,
   //                        meshtriangle_args.gpu_table_indexer,
   //                        meshtriangle_args.d_gpu_meshtriangle_pos,
   //                        meshtriangle_args.d_gpu_n_meshparticles,
   //                        meshtriangle_args.n_meshtriangle_types,
   //                        d_params)
   //     }
   // else
   //     {
   //     hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, compute_virial, false>),
   //                        grid,
   //                        threads,
   //                        shared_bytes,
   //     		   0,
   //                        meshtriangle_args.d_force,
   //                        meshtriangle_args.d_virial,
   //                        meshtriangle_args.virial_pitch,
   //                        meshtriangle_args.N,
   //                        meshtriangle_args.d_pos,
   //                        meshtriangle_args.box,
   //                        meshtriangle_args.d_n_neigh,
   //                        meshtriangle_args.d_nlist,
   //                        meshtriangle_args.d_head_list,
   //                        meshtriangle_args.d_rcutsq,
   //                        meshtriangle_args.d_gpu_meshtrianglelist,
   //                        meshtriangle_args.gpu_table_indexer,
   //                        meshtriangle_args.d_gpu_meshtriangle_pos,
   //                        meshtriangle_args.d_gpu_n_meshparticles,
   //                        meshtriangle_args.n_meshtriangle_types,
   //                        d_params)
   //     }

    return hipSuccess;
    }
#else
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_triangles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
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
