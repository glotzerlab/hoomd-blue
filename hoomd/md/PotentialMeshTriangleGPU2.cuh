// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"


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
	   compute_virial(_compute_virial) { };

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
                                               const group_storage<3>* tlist,
                                               const Index2D tlist_idx,
                                               const unsigned int* tpos_list,
                                               const unsigned int* n_triangles_list,
                                               const typename evaluator::param_type* d_params,
                                               unsigned int* d_flags)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared array for per bond type parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);

    if (enable_shared_cache)
        {
        // load in per bond type parameters
        for (unsigned int cur_offset = 0; cur_offset < n_bond_type; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_bond_type)
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
    calar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

    const unsigned int Nsize0 = (unsigned int)d_n_neigh[idx];
    const size_t myHead0 = d_head_list[idx];

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_b = cur_triangle.idx[0];
        int cur_triangle_c = cur_triangle.idx[1];

	std::vector<unsigned int> combined_nlist;


        const size_t myHead1 = d_head_list[cur_triangle_b];
        const unsigned int Nsize1 = (unsigned int)d_n_neigh[cur_triangle_b];

        const size_t myHead2 = d_head_list[cur_triangle_c];
        const unsigned int Nsize2 = (unsigned int)d_n_neigh[cur_triangle_c];

	for (unsigned int idx0 = 0; idx0 < Nsize0; idx0++)
	{
		unsigned int j = d_nlist[myHead0 + idx0];

		bool insert = false;

		for (unsigned int idx1 = 0; idx1 < Nsize1 && !insert; idx1++)
			if( j ==  d_nlist[myHead1 + idx1] ) insert = true;
		if(insert)
		{
			insert = false;
			for (unsigned int idx2 = 0; idx2 < Nsize2 && !insert; idx2++)
				if( j ==  d_nlist[myHead2 + idx2] ) insert = true;
			if(insert)
				combined_nlist.push_back(j);
		}
	}
	if( combined_nlist.size() == 0) 
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


         for (unsigned int k = 0; k < combined_nlist.size(); k++)
         	{
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int cur_j = combined_nlist[k];

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
        	dbj.x = pos_b.x - pj.x;
        	dbj.y = pos_b.y - pj.y;
        	dbj.z = pos_b.z - pj.z;
                dbj = box.minImage(dbj);


		Scalar dbj_norm = dot(dbj,normal_dir);

		Scalar rsq = dbj_norm*dbj_norm;

		if( rcutsq < rsq) continue;

		Scalar3 dx = normal_dir*dbj_norm;

		Scalar Area_a;

		Scalar3 dcj;
		dcj.x = pos_c.x - pj.x;
		dcj.y = pos_c.y - pj.y;
		dcj.z = pos_c.z - pj.z;
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

			scalar Area_c = dot(dajbj,normal_dir);
			if(Area_b <0)
				{
				if(Area_c <0)
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
				if(Area_c <0)
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

        	evaluator eval(rsq, *param);

        	bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

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
        	else
        	    {
        	    *d_flags = 1;
        	    return;
        	    }
		}

        // evaluate the potential
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
                        const unsigned int* n_triangles_list,
                        unsigned int* d_flags)
    {
    assert(d_params);
    assert(meshtriangle_args.d_rcutsq);
    assert(meshtriangle_args.ntypes > 0);

    // check that block_size is valid
    assert(meshtriangle_args.block_size != 0);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,
                         reinterpret_cast<const void*>(
                             &gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, true>));
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
        hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, compute_virial, true>),
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
                           meshtriangle_args.d_gpu_meshtrianglelist,
                           meshtriangle_args.gpu_table_indexer,
                           meshtriangle_args.d_gpu_meshtriangle_pos,
                           meshtriangle_args.d_gpu_n_meshtriangles,
                           meshtriangle_args.n_meshtriangle_types,
                           d_params,
                           d_flags);
        }
    else
        {
        hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, compute_virial, false>),
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
                           meshtriangle_args.d_gpu_meshtrianglelist,
                           meshtriangle_args.gpu_table_indexer,
                           meshtriangle_args.d_gpu_meshtriangle_pos,
                           meshtriangle_args.d_gpu_n_meshtriangles,
                           meshtriangle_args.n_meshtriangle_types,
                           d_params,
                           d_flags);
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
                        const unsigned int* d_head_triang,
                        unsigned int* d_flags)
    {
    assert(d_params);
    assert(meshtriangle_args.d_rcutsq);
    assert(meshtriangle_args.ntypes > 0);

    // check that block_size is valid
    assert(meshtriangle_args.block_size != 0);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,
                         reinterpret_cast<const void*>(
                             &gpu_compute_mesh_triangle_particles_force_kernel<evaluator, true>));
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
        hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, compute_virial, true>),
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
                           meshtriangle_args.d_gpu_meshtrianglelist,
                           meshtriangle_args.gpu_table_indexer,
                           meshtriangle_args.d_gpu_meshtriangle_pos,
                           meshtriangle_args.d_gpu_n_meshparticles,
                           meshtriangle_args.n_meshtriangle_types,
                           d_params,
                           d_flags);
        }
    else
        {
        hipLaunchKernelGGL((gpu_compute_mesh_triangle_particles_force_kernel<evaluator, compute_virial, false>),
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
                           meshtriangle_args.d_gpu_meshtrianglelist,
                           meshtriangle_args.gpu_table_indexer,
                           meshtriangle_args.d_gpu_meshtriangle_pos,
                           meshtriangle_args.d_gpu_n_meshparticles,
                           meshtriangle_args.n_meshtriangle_types,
                           d_params,
                           d_flags);
        }

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
                        const unsigned int* n_triangles_list,
                        unsigned int* d_flags);

template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_mesh_triangle_particles_force(const kernel::meshtriangle_args_t& meshtriangle_args,
                        const typename evaluator::param_type* d_params,
                        const unsigned int* d_tag,
                        const unsigned int* d_n_triang,
                        const unsigned int* d_trianglist,
                        const unsigned int* d_head_triang,
                        unsigned int* d_flags);
#endif


    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __POTENTIALMESHTRIANGLEPAIR_GPU_CUH__

























// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborList.h"
#include "MeshForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/managed_allocator.h"
#include <memory>

#include <vector>

/*! \file PotentialMeshTriangle.h
    \brief Declares PotentialMeshTriangle
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __POTENTIALMESHTRIANGLEPAIR_H__
#define __POTENTIALMESHTRIANGLEPAIR_H__

namespace hoomd
    {
namespace md
    {
/*! Bond potential with evaluator support

    \ingroup computes
*/
template<class evaluator> class PotentialMeshTriangle : public MeshForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Constructs the compute
    PotentialMeshTriangle(std::shared_ptr<SystemDefinition> sysdef, 
		  std::shared_ptr<NeighborList> nlist,
                  std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~PotentialMeshTriangle();

    /// Set the parameters
    virtual void setParams(unsigned int type, const param_type& param);
    virtual void setParamsPython(std::string type, pybind11::dict param);

    /// Get the parameters
    pybind11::dict getParams(std::string type);

    virtual void setRcut(unsigned int type, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getRCut(std::string type);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(std::string type, Scalar r_cut);
    //! Set ron for a single type pair

    virtual void setNListRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getNListRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setNListRCutPython(pybind11::tuple types, Scalar r_cut);
    //! Set ron for a single type pair

    /// Validate bond type
    void validateType(unsigned int type, std::string action);

    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
        {
        no_shift = 0,
        shift,
        xplor
        };

    //! Set the mode to use for shifting the energy
    void setShiftMode(energyShiftMode mode)
        {
        m_shift_mode = mode;
        }

    void setShiftModePython(std::string mode)
        {
        if (mode == "none")
            {
            m_shift_mode = no_shift;
            }
        else if (mode == "shift")
            {
            m_shift_mode = shift;
            }
        else if (mode == "xplor")
            {
            m_shift_mode = xplor;
            }
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    /// Get the mode used for the energy shifting
    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        case shift:
            return "shift";
        case xplor:
            return "xplor";
        default:
            throw std::runtime_error("Error setting shift mode.");
            }
        }

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    GPUArray<Scalar> m_rcutsq;    //!< Cutoff radius squared for the potential list per type pair
    GPUArray<Scalar> m_nlistrcutsq;    //!< Cutoff radius squared for the neighbor list per type pair
    /// Per type pair potential parameters
    GPUArray<param_type> m_params;      //!< Bond parameters per type

    /// Track whether we have attached to the Simulation object
    bool m_attached = true;

    /// r_cut (not squared) given to the neighbor list
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist;

    /// Keep track of number of each type of particle
    std::vector<unsigned int> m_num_particles_by_type;

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;
    void computeForcesTriangle(uint64_t timestep);
    void computeForcesParticle(uint64_t timestep);

    }; // end class PotentialMeshTriangle

template<class evaluator>
PotentialMeshTriangle<evaluator>::PotentialMeshTriangle(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef), m_nlist(nlist), m_shift_mode(no_shift)
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialMeshTriangle<" << evaluator::getName() << ">"
                                << std::endl;
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_pdata->getNTypes(), m_exec_conf);
    m_rcutsq.swap(rcutsq);

    GPUArray<Scalar> nlistrcutsq(m_pdata->getNTypes(), m_exec_conf);
    m_nlistrcutsq.swap(nlistrcutsq);

    // allocate the parameters
    GPUArray<param_type> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    //m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(m_pdata->getNTypes(), m_exec_conf);

    Index2D typpair_idx(m_pdata->getNTypes());
    m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

    m_mesh_data->createMeshTriangleList();

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        // m_params is _always_ in unified memory, so memadvise and prefetch
        cudaMemAdvise(m_params.data(),
                      m_params.size() * sizeof(param_type),
                      cudaMemAdviseSetReadMostly,
                      0);
        cudaMemPrefetchAsync(m_params.data(),
                             sizeof(param_type) * m_params.size(),
                             m_exec_conf->getGPUId());
        }
#endif

    // get number of each type of particle, needed for energy and pressure correction
    m_num_particles_by_type.resize(m_pdata->getNTypes());
    std::fill(m_num_particles_by_type.begin(), m_num_particles_by_type.end(), 0);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typeid_i = __scalar_as_int(h_postype.data[i].w);
        m_num_particles_by_type[typeid_i] += 1;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce number of each type of particle on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      m_num_particles_by_type.data(),
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

template<class evaluator> PotentialMeshTriangle<evaluator>::~PotentialMeshTriangle()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialMeshTriangle<" << evaluator::getName() << ">"
                                << std::endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param type Type of the bond to set parameters for
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
void PotentialMeshTriangle<evaluator>::validateType(unsigned int type, std::string action)
    {
    // make sure the type is valid
    if (type >= m_pdata->getNTypes())
        {
        throw std::runtime_error("error in" + action + " for triangle pair potential. invalid type");
        }
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setParams(unsigned int type, const param_type& param)
    {
    // make sure the type is valid
    validateType(type, "setting params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = param;
    }

/*! \param types Type of the bond to set parameters for using string
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
void PotentialMeshTriangle<evaluator>::setParamsPython(std::string type, pybind11::dict param)
    {
    auto itype = m_pdata->getTypeByName(type);
    auto struct_param = param_type(param);
    setParams(itype, struct_param);
    }

/*! \param types Type of the bond to set parameters for using string
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
pybind11::dict PotentialMeshTriangle<evaluator>::getParams(std::string type)
    {
    auto itype = m_pdata->getTypeByName(type);
    validateType(itype, "getting params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[itype].asDict();
    }


template<class evaluator>
void PotentialMeshTriangle<evaluator>::setRcut(unsigned int type, Scalar rcut)
    {
    validateType(type, "setting r_cut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[type] = rcut * rcut;
        }
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setRCutPython(std::string type, Scalar r_cut)
    {
    auto typ = m_pdata->getTypeByName(type);
    setRcut(typ, r_cut);
    }

template<class evaluator> Scalar PotentialMeshTriangle<evaluator>::getRCut(std::string type)
    {
    auto typ = m_pdata->getTypeByName(type);
    validateType(typ, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[typ]);
    }


template<class evaluator>
void PotentialMeshTriangle<evaluator>::setNListRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateType(typ1, "setting r_nlistcut");
    validateType(typ2, "setting r_nlistcut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_nlistrcutsq(m_nlistrcutsq, access_location::host, access_mode::readwrite);
    	Index2D typpair_idx(m_pdata->getNTypes());
	h_nlistrcutsq.data[typpair_idx(typ1, typ2)] = rcut * rcut;
        h_nlistrcutsq.data[typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[typpair_idx(typ1, typ2)] = rcut;
        h_r_cut_nlist.data[typpair_idx(typ2, typ1)] = rcut;
        }
    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setNListRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setNListRcut(typ1, typ2, r_cut);
    }

template<class evaluator> Scalar PotentialMeshTriangle<evaluator>::getNListRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateType(typ1, "getting r_cut.");
    validateType(typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_nlistrcutsq(m_nlistrcutsq, access_location::host, access_mode::read);
    Index2D typpair_idx(m_pdata->getNTypes());
    return sqrt(h_nlistrcutsq.data[typpair_idx(typ1, typ2)]);
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
template<class evaluator>
void PotentialMeshTriangle<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    computeForcesTriangle(timestep);
    computeForcesParticle(timestep);
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::computeForcesTriangle(uint64_t timestep)
    {
    
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
    				    access_location::host,
    				    access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
    				  access_location::host,
    				  access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
    				access_location::host,
    				access_mode::read);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);
    size_t virial_pitch = m_virial.getPitch();

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    // Zero data for force calculation
    m_force.zeroFill();
    m_virial.zeroFill();

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);


    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename Angle::members_t& triangle = h_triangles.data[i];

        unsigned int ttag_a = triangle.tag[0];
        assert(ttag_a < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_b = triangle.tag[1];
        assert(ttag_b < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_c = triangle.tag[2];
        assert(ttag_c < m_pdata->getMaximumTag() + 1);

	std::vector<unsigned int> idces(3);

        idces[0] = h_rtag.data[ttag_a];
        idces[1] = h_rtag.data[ttag_b];
        idces[2] = h_rtag.data[ttag_c];

        assert(idces[0] < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idces[1] < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idces[2] < m_pdata->getN() + m_pdata->getNGhosts());

	std::vector<unsigned int> combined_nlist;

        for (unsigned int v_idx = 0; v_idx < 3; v_idx++)
	{
		unsigned int vv_idx = idces[v_idx];
		for (unsigned int idx_i = 0; idx_i < h_n_neigh.data[vv_idx]; idx_i++)
		{
			unsigned int j = h_nlist.data[h_head_list.data[vv_idx] + idx_i];
			bool not_yet_in = true;
			for(unsigned int cn_idx = 0; cn_idx < combined_nlist.size() && not_yet_in; cn_idx++)
				if( combined_nlist[cn_idx] == j)
					not_yet_in = false;
			if (not_yet_in)
				 combined_nlist.push_back(j);
		}
	}

	if( combined_nlist.size() == 0) 
		continue;

        vec3<Scalar> pos_a(h_pos.data[idces[0]].x, h_pos.data[idces[0]].y, h_pos.data[idces[0]].z);
        vec3<Scalar> pos_b(h_pos.data[idces[1]].x, h_pos.data[idces[1]].y, h_pos.data[idces[1]].z);
        vec3<Scalar> pos_c(h_pos.data[idces[2]].x, h_pos.data[idces[2]].y, h_pos.data[idces[2]].z);

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



        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fa = make_scalar3(0, 0, 0);
        Scalar pea = 0.0;

        Scalar3 fb = make_scalar3(0, 0, 0);
        Scalar peb = 0.0;

        Scalar3 fc = make_scalar3(0, 0, 0);
        Scalar pec = 0.0;

        Scalar viriala[6];
        Scalar virialb[6];
        Scalar virialc[6];
        for (unsigned int k = 0; k < 6; k++)
	{
            viriala[k] = Scalar(0.0);
            virialb[k] = Scalar(0.0);
            virialc[k] = Scalar(0.0);
	}


         for (unsigned int k = 0; k < combined_nlist.size(); k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = combined_nlist[k];

		unsigned int typej = __scalar_as_int(h_pos.data[j].w);
		assert(typej < m_pdata->getNTypes());


                const param_type& param = h_params.data[typej];
                Scalar rcutsq = h_rcutsq.data[typej];

		if( rcutsq == 0) continue;


                // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
        	vec3<Scalar> pj(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        	Scalar3 daj;
        	daj.x = pos_a.x - pj.x;
        	daj.y = pos_a.y - pj.y;
        	daj.z = pos_a.z - pj.z;
                daj = box.minImage(daj);

		Scalar daj_norm = dot(daj,normal_dir);

		Scalar rsq = daj_norm*daj_norm;

		if( rcutsq < rsq) continue;

		Scalar3 dx = normal_dir*daj_norm;

		Scalar Area_a, Area_b, Area_c;

        	Scalar3 dbj;
        	dbj.x = pos_b.x - pj.x;
        	dbj.y = pos_b.y - pj.y;
        	dbj.z = pos_b.z - pj.z;
                dbj = box.minImage(dbj);

		Scalar3 dajbj; 
		dajbj.x = daj.y*dbj.z - daj.z*dbj.y;
		dajbj.y = daj.z*dbj.x - daj.x*dbj.z;
		dajbj.z = daj.x*dbj.y - daj.y*dbj.x;

		Area_c = dot(dajbj,normal_dir);

		Scalar3 dcj;
		dcj.x = pos_c.x - pj.x;
		dcj.y = pos_c.y - pj.y;
		dcj.z = pos_c.z - pj.z;
		dcj = box.minImage(dcj);


		Scalar3 dcjaj; 
		dcjaj.x = dcj.y*daj.z - dcj.z*daj.y;
		dcjaj.y = dcj.z*daj.x - dcj.x*daj.z;
		dcjaj.z = dcj.x*daj.y - dcj.y*daj.x;
		Area_b = dot(dcjaj,normal_dir);

		Scalar3 dbjcj; 
		dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
		dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
		dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;

		Area_a = dot(dbjcj,normal_dir);


		if(Area_c <0)
		{
			if(Area_b <0)
			{
				dx = daj;
				Area_a = 1;
				Area_b = 0;
				Area_c = 0;
			}
			else if(Area_a < 0)
			{
				dx = dbj;
				Area_a = 0;
				Area_b = 1;
				Area_c = 0;
			}
			else
			{
				Scalar length_ab = dot(daj,nab);
				Scalar ratio_ab = length_ab*normal_ab;

				if(ratio_ab < 1 && ratio_ab > 0 )
				{
					dx = daj - length_ab*nab;
					Area_a = ratio_ab;
					Area_b = 1-ratio_ab;
					Area_c = 0;
				}
				else continue;
			}
		}
		else{
			Area_c *= normal_norm;


			if(Area_b <0)
			{
				if(Area_a < 0)
				{
					dx = dcj;
					Area_a = 0;
					Area_b = 0;
					Area_c = 1;
				}
				else
				{
					Scalar length_ac = dot(daj,nac);
					Scalar ratio_ac = length_ac*normal_ac;

					if(ratio_ac < 1 && ratio_ac > 0 )
					{
						dx = daj - length_ac*nac;
						Area_a = ratio_ac;
						Area_c = 1-ratio_ac;
						Area_b = 0;
					}
					else continue;
				}
			}
			else
			{
				Area_b *= normal_norm;

				if(Area_a <0)
				{
					Scalar length_bc = dot(dbj,nbc);
					Scalar ratio_bc = length_bc*normal_bc;

					if(ratio_bc < 1 && ratio_bc > 0 )
					{
						dx = dbj - length_bc*nbc;
						Area_b = ratio_bc;
						Area_c = 1-ratio_bc;
						Area_a = 0;
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

                bool energy_shift = false;
                if (m_shift_mode == shift)
                    energy_shift = true;

                // compute the force and potential energy
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);
                evaluator eval(rsq, rcutsq, param);
                bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);
                if (evaluated)
                    {
                    fa += dx * force_divr* Area_a;
                    fb += dx * force_divr* Area_b;
                    fc += dx * force_divr* Area_c;
                    pea += pair_eng * Scalar(0.5)* Area_a;
                    peb += pair_eng * Scalar(0.5)* Area_b;
                    pec += pair_eng * Scalar(0.5)* Area_c;
                    if (compute_virial)
                        {
                        viriala[0] += force_divr * Area_a * pos_a.x * dx.x;
                        viriala[1] += force_divr * Area_a * pos_a.x * dx.y;
                        viriala[2] += force_divr * Area_a * pos_a.x * dx.z;
                        viriala[3] += force_divr * Area_a * pos_a.y * dx.y;
                        viriala[4] += force_divr * Area_a * pos_a.y * dx.z;
                        viriala[5] += force_divr * Area_a * pos_a.z * dx.z;

                        virialb[0] += force_divr * Area_b * pos_b.x * dx.x;
                        virialb[1] += force_divr * Area_b * pos_b.x * dx.y;
                        virialb[2] += force_divr * Area_b * pos_b.x * dx.z;
                        virialb[3] += force_divr * Area_b * pos_b.y * dx.y;
                        virialb[4] += force_divr * Area_b * pos_b.y * dx.z;
                        virialb[5] += force_divr * Area_b * pos_b.z * dx.z;

                        virialc[0] += force_divr * Area_c * pos_c.x * dx.x;
                        virialc[1] += force_divr * Area_c * pos_c.x * dx.y;
                        virialc[2] += force_divr * Area_c * pos_c.x * dx.z;
                        virialc[3] += force_divr * Area_c * pos_c.y * dx.y;
                        virialc[4] += force_divr * Area_c * pos_c.y * dx.z;
                        virialc[5] += force_divr * Area_c * pos_c.z * dx.z;
                        }
                    }
		}

            if (idces[0] < m_pdata->getN())
		    {
		    h_force.data[idces[0]].x += fa.x;
		    h_force.data[idces[0]].y += fa.y;
		    h_force.data[idces[0]].z += fa.z;
		    h_force.data[idces[0]].w += pea;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[0]] += viriala[jj];
		    }
            if (idces[1] < m_pdata->getN())
		    {
		    h_force.data[idces[1]].x += fb.x;
		    h_force.data[idces[1]].y += fb.y;
		    h_force.data[idces[1]].z += fb.z;
		    h_force.data[idces[1]].w += peb;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[1]] += virialb[jj];
		    }
            if (idces[2] < m_pdata->getN())
		    {
		    h_force.data[idces[2]].x += fc.x;
		    h_force.data[idces[2]].y += fc.y;
		    h_force.data[idces[2]].z += fc.z;
		    h_force.data[idces[2]].w += pec;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[2]] += virialc[jj];
		    }
        }
}
    

template<class evaluator>
void PotentialMeshTriangle<evaluator>::computeForcesParticle(uint64_t timestep)
    {
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
    				    access_location::host,
    				    access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
    				  access_location::host,
    				  access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
    				access_location::host,
    				access_mode::read);
    

    ArrayHandle<unsigned int> h_n_triang(m_mesh_data->getNNeighArray(),
    				    access_location::host,
    				    access_mode::read);
    ArrayHandle<unsigned int> h_trianglist(m_mesh_data->getTriangleList(),
    				  access_location::host,
    				  access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<unsigned int> h_head_triang(m_mesh_data->getHeadList(),
    				access_location::host,
    				access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);
    size_t virial_pitch = m_virial.getPitch();

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);


    for (int i = 0; i < (int)m_pdata->getN(); i++)
    {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);

        const param_type& param = h_params.data[typei];
        Scalar rcutsq = h_rcutsq.data[typei];

	if( rcutsq == 0) continue;

        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const size_t myHead = h_head_list.data[i];
        unsigned int size = (unsigned int)h_n_neigh.data[i];

	std::vector<uint2> reduced_nlist;


	uint2 nk;

	for (unsigned int k = 0; k < size; k++)
	{
		nk.x = h_nlist.data[myHead + k];
                assert(nk.x < m_pdata->getN() + m_pdata->getNGhosts());
		nk.y = h_tag.data[nk.x];
		if (h_n_triang.data[nk.y] > 0)
			reduced_nlist.push_back(nk);
	}

	std::vector<uint3> combined_nlist;

	uint3 triangles;

	size = (unsigned int)reduced_nlist.size();

	if(size < 2)
		continue;

        for (unsigned int k = 0; k < size-2; k++)
        {
            triangles.x = reduced_nlist[k].x;
	    unsigned int trianglesx = reduced_nlist[k].y;

	    unsigned int Nj_tri = h_n_triang.data[trianglesx];
	    unsigned int headj = h_head_triang.data[trianglesx];

            for (unsigned int kk = k+1; kk < size-1; kk++)
            {
                triangles.y = reduced_nlist[kk].x;
	    	unsigned int trianglesy = reduced_nlist[kk].y;

	    	unsigned int Njj_tri = h_n_triang.data[trianglesy];
		unsigned int headjj = h_head_triang.data[trianglesy];

		for(unsigned int j_tri=0; j_tri < Nj_tri; j_tri++)
		{
			Scalar tri_idx = h_trianglist.data[headj+j_tri];
		        for(unsigned int jj_tri=0; jj_tri < Njj_tri; jj_tri++)
		        {
		            if( tri_idx == h_trianglist.data[headjj+jj_tri] )
			    {
			        for (unsigned int kkk = kk+1; kkk < size; kkk++)
			        {
			           triangles.z =  reduced_nlist[kkk].x;
	    			   unsigned int trianglesz = reduced_nlist[kkk].y;
	    			   unsigned int Njjj_tri = h_n_triang.data[trianglesz];
				   unsigned int headjjj = h_head_triang.data[trianglesz];

				   for(unsigned int jjj_tri=0; jjj_tri < Njjj_tri; jjj_tri++)
				   {
					   if(tri_idx == h_trianglist.data[headjjj+jjj_tri])
						   combined_nlist.push_back(triangles);
				   }
				}
			    }

			}
	   	}
	   }
	}

         for (unsigned int k = 0; k < combined_nlist.size(); k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
		//
                unsigned int aj = combined_nlist[k].x;

                vec3<Scalar> pos_a(h_pos.data[aj].x, h_pos.data[aj].y, h_pos.data[aj].z);

        	Scalar3 daj;
        	daj.x = pos_a.x - pi.x;
        	daj.y = pos_a.y - pi.y;
        	daj.z = pos_a.z - pi.z;
                daj = box.minImage(daj);

                unsigned int bj = combined_nlist[k].y;
                unsigned int cj = combined_nlist[k].z;

                vec3<Scalar> pos_b(h_pos.data[bj].x, h_pos.data[bj].y, h_pos.data[bj].z);
                vec3<Scalar> pos_c(h_pos.data[cj].x, h_pos.data[cj].y, h_pos.data[cj].z);

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

		Scalar daj_norm = dot(daj,normal_dir);

		Scalar rsq = daj_norm*daj_norm;

		if( rcutsq < rsq) continue;

		Scalar3 dx = normal_dir*daj_norm;

        	Scalar3 dbj;
        	dbj.x = pos_b.x - pi.x;
        	dbj.y = pos_b.y - pi.y;
        	dbj.z = pos_b.z - pi.z;
                dbj = box.minImage(dbj);

		Scalar3 dajbj; 
		dajbj.x = daj.y*dbj.z - daj.z*dbj.y;
		dajbj.y = daj.z*dbj.x - daj.x*dbj.z;
		dajbj.z = daj.x*dbj.y - daj.y*dbj.x;

		Scalar Area_c = dot(dajbj,normal_dir);

		Scalar3 dcj;
		dcj.x = pos_c.x - pi.x;
		dcj.y = pos_c.y - pi.y;
		dcj.z = pos_c.z - pi.z;
		dcj = box.minImage(dcj);

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
					Scalar length_ab = dot(daj,nab);
					Scalar ratio_ab = length_ab*normal_ab;

					if(ratio_ab < 1 && ratio_ab > 0 )
						dx = daj - length_ab*nab;
					else continue;
				}
			}
		}
		else{
			Scalar3 dbjcj; 
			dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
			dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
			dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;
			Scalar Area_a = dot(dbjcj,normal_dir);
			if(Area_b <0)
			{
				if(Area_a < 0)
					dx = dcj;
				else
				{
					Scalar length_ac = dot(daj,nac);
					Scalar ratio_ac = length_ac*normal_ac;

					if(ratio_ac < 1 && ratio_ac > 0 )
						dx = daj - length_ac*nac;
					else continue;
				}
			}
			else
			{
				if(Area_a <0)
				{
					Scalar length_bc = dot(dbj,nbc);
					Scalar ratio_bc = length_bc*normal_bc;

					if(ratio_bc < 1 && ratio_bc > 0 )
						dx = dbj - length_bc*nbc;
					else continue;
				}
			}
		}

		rsq = dot(dx,dx);

                bool energy_shift = false;
                if (m_shift_mode == shift)
                    energy_shift = true;

                // compute the force and potential energy
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);
                evaluator eval(rsq, rcutsq, param);
                bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);
                if (evaluated)
                    {
                    fi -= dx * force_divr;
                    pei += pair_eng * Scalar(0.5);
                    if (compute_virial)
                        {
	                virialxxi -= force_divr * pi.x * dx.x;
	                virialxyi -= force_divr * pi.x * dx.y;
	                virialxzi -= force_divr * pi.x * dx.z;
	                virialyyi -= force_divr * pi.y * dx.y;
	                virialyzi -= force_divr * pi.y * dx.z;
	                virialzzi -= force_divr * pi.z * dx.z;
	                }
                    }
		}

	    h_force.data[i].x += fi.x;
	    h_force.data[i].y += fi.y;
	    h_force.data[i].z += fi.z;
	    h_force.data[i].w += pei;
            if (compute_virial)
                {
                h_virial.data[0 * virial_pitch + i] += virialxxi;
                h_virial.data[1 * virial_pitch + i] += virialxyi;
                h_virial.data[2 * virial_pitch + i] += virialxzi;
                h_virial.data[3 * virial_pitch + i] += virialyyi;
                h_virial.data[4 * virial_pitch + i] += virialyzi;
                h_virial.data[5 * virial_pitch + i] += virialzzi;
                }
        }
}

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialMeshTriangle<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags[comm_flag::tag] = 1;

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    flags |= MeshForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {
//! Exports the PotentialMeshTriangle class to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialMeshTriangle(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialMeshTriangle<T>,
                     MeshForceCompute,
                     std::shared_ptr<PotentialMeshTriangle<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<NeighborList>,std::shared_ptr<MeshDefinition>>())
        .def("setParams", &PotentialMeshTriangle<T>::setParamsPython)
        .def("getParams", &PotentialMeshTriangle<T>::getParams)
        .def("setRCut", &PotentialMeshTriangle<T>::setRCutPython)
        .def("getRCut", &PotentialMeshTriangle<T>::getRCut)
        .def("setNlistRCut", &PotentialMeshTriangle<T>::setNListRCutPython)
        .def("getNlistRCut", &PotentialMeshTriangle<T>::getNListRCut)
        .def_property("mode",
                      &PotentialMeshTriangle<T>::getShiftMode,
                      &PotentialMeshTriangle<T>::setShiftModePython);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
