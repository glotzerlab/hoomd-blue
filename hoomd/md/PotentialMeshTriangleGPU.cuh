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
    //if (enable_shared_cache)
    //    {
    //    hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, compute_virial, true>),
    //                       grid,
    //                       threads,
    //                       shared_bytes,
    //                       0,
    //                       meshtriangle_args.d_force,
    //                       meshtriangle_args.d_virial,
    //                       meshtriangle_args.virial_pitch,
    //                       meshtriangle_args.N,
    //                       meshtriangle_args.d_pos,
    //                       meshtriangle_args.box,
    //                       meshtriangle_args.d_n_neigh,
    //                       meshtriangle_args.d_nlist,
    //                       meshtriangle_args.d_head_list,
    //                       meshtriangle_args.d_rcutsq,
    //                       meshtriangle_args.d_gpu_meshtrianglelist,
    //                       meshtriangle_args.gpu_table_indexer,
    //                       meshtriangle_args.d_gpu_meshtriangle_pos,
    //                       meshtriangle_args.d_gpu_n_meshtriangles,
    //                       meshtriangle_args.n_meshtriangle_types,
    //                       d_params,
    //                       d_flags);
    //    }
    //else
    //    {
    //    hipLaunchKernelGGL((gpu_compute_mesh_triangle_triangles_force_kernel<evaluator, compute_virial, false>),
    //                       grid,
    //                       threads,
    //                       shared_bytes,
    //    		   0,
    //                       meshtriangle_args.d_force,
    //                       meshtriangle_args.d_virial,
    //                       meshtriangle_args.virial_pitch,
    //                       meshtriangle_args.N,
    //                       meshtriangle_args.d_pos,
    //                       meshtriangle_args.box,
    //                       meshtriangle_args.d_n_neigh,
    //                       meshtriangle_args.d_nlist,
    //                       meshtriangle_args.d_head_list,
    //                       meshtriangle_args.d_rcutsq,
    //                       meshtriangle_args.d_gpu_meshtrianglelist,
    //                       meshtriangle_args.gpu_table_indexer,
    //                       meshtriangle_args.d_gpu_meshtriangle_pos,
    //                       meshtriangle_args.d_gpu_n_meshtriangles,
    //                       meshtriangle_args.n_meshtriangle_types,
    //                       d_params,
    //                       d_flags);
    //    }

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
   //                        d_params,
   //                        d_flags);
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
   //                        d_params,
   //                        d_flags);
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
