// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/TextureTools.h"

#include "hoomd/BondedGroupData.cuh"

#include <assert.h>

/*! \file PotentialBondGPU.cuh
    \brief Defines templated GPU kernel code for calculating the bond forces.
*/

#ifndef __POTENTIAL_BOND_GPU_CUH__
#define __POTENTIAL_BOND_GPU_CUH__

//! Wraps arguments to gpu_cgbf
struct bond_args_t
    {
    //! Construct a bond_args_t
    bond_args_t(Scalar4 *_d_force,
              Scalar *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const unsigned int _n_max,
              const Scalar4 *_d_pos,
              const Scalar *_d_charge,
              const Scalar *_d_diameter,
              const BoxDim& _box,
              const group_storage<2> *_d_gpu_bondlist,
              const Index2D & _gpu_table_indexer,
              const unsigned int *_d_gpu_n_bonds,
              const unsigned int _n_bond_types,
              const unsigned int _block_size)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  N(_N),
                  n_max(_n_max),
                  d_pos(_d_pos),
                  d_charge(_d_charge),
                  d_diameter(_d_diameter),
                  box(_box),
                  d_gpu_bondlist(_d_gpu_bondlist),
                  gpu_table_indexer(_gpu_table_indexer),
                  d_gpu_n_bonds(_d_gpu_n_bonds),
                  n_bond_types(_n_bond_types),
                  block_size(_block_size)
        {
        };

    Scalar4 *d_force;                   //!< Force to write out
    Scalar *d_virial;                   //!< Virial to write out
    const unsigned int virial_pitch;   //!< pitch of 2D array of virial matrix elements
    unsigned int N;                    //!< number of particles
    unsigned int n_max;                //!< Size of local pdata arrays
    const Scalar4 *d_pos;              //!< particle positions
    const Scalar *d_charge;            //!< particle charges
    const Scalar *d_diameter;          //!< particle diameters
    const BoxDim& box;            //!< Simulation box in GPU format
    const group_storage<2> *d_gpu_bondlist;       //!< List of bonds stored on the GPU
    const Index2D& gpu_table_indexer;  //!< Indexer of 2D bond list
    const unsigned int *d_gpu_n_bonds; //!< List of number of bonds stored on the GPU
    const unsigned int n_bond_types;   //!< Number of bond types in the simulation
    const unsigned int block_size;     //!< Block size to execute
    };

#ifdef NVCC

//! Kernel for calculating bond forces
/*! This kernel is called to calculate the bond forces on all N particles. Actual evaluation of the potentials and
    forces for each bond is handled via the template class \a evaluator.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N Number of particles in the system
    \param d_pos particle positions on the GPU
    \param d_charge particle charges
    \param d_diameter particle diameters
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param pitch Pitch of 2D bond list
    \param n_bonds_list List of numbers of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_params Parameters for the potential, stored per bond type
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be evaluated


    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorBond class to evaluate V(r) and -delta V(r)/r

*/
template< class evaluator >
__global__ void gpu_compute_bond_forces_kernel(Scalar4 *d_force,
                                               Scalar *d_virial,
                                               const unsigned int virial_pitch,
                                               const unsigned int N,
                                               const Scalar4 *d_pos,
                                               const Scalar *d_charge,
                                               const Scalar *d_diameter,
                                               const BoxDim box,
                                               const group_storage<2> *blist,
                                               const Index2D blist_idx,
                                               const unsigned int *n_bonds_list,
                                               const unsigned int n_bond_type,
                                               const typename evaluator::param_type *d_params,
                                               unsigned int *d_flags)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared array for per bond type parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params =
        (typename evaluator::param_type *)(&s_data[0]);

    // load in per bond type parameters
    for (unsigned int cur_offset = 0; cur_offset < n_bond_type; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_bond_type)
            {
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }

    __syncthreads();

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds =n_bonds_list[idx];

    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // read in the diameter of our particle if needed
    Scalar diam(0);
    if (evaluator::needsDiameter())
        {
        diam = __ldg(d_diameter + idx);
        }
    else
        diam += 0; // shut up compiler warning

    Scalar q(0);
    if (evaluator::needsCharge())
        {
        q = __ldg(d_charge + idx);
        }
    else
        q += 0; // shut up compiler warning

    // initialize the force to 0
    Scalar4 force = make_scalar4(0, 0, 0, 0);
    // initialize the virial tensor to 0
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    // loop over neighbors
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        group_storage<2> cur_bond = blist[blist_idx(idx, bond_idx)];

        int cur_bond_idx = cur_bond.idx[0];
        int cur_bond_type = cur_bond.idx[1];

        // get the bonded particle's position (MEM_TRANSFER: 16 bytes)
        Scalar4 neigh_postypej = __ldg(d_pos + cur_bond_idx);
        Scalar3 neigh_pos= make_scalar3(neigh_postypej.x, neigh_postypej.y, neigh_postypej.z);

        // calculate dr (FLOPS: 3)
        Scalar3 dx = pos - neigh_pos;

        // apply periodic boundary conditions (FLOPS: 12)
        dx = box.minImage(dx);

        // get the bond parameters (MEM TRANSFER: 8 bytes)
        typename evaluator::param_type param = s_params[cur_bond_type];

        Scalar rsq = dot(dx, dx);

        // evaluate the potential
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);

        evaluator eval(rsq, param);

        // get the bonded particle's diameter if needed
        if (evaluator::needsDiameter())
            {
            Scalar neigh_diam = __ldg(d_diameter + cur_bond_idx);
            eval.setDiameter(diam, neigh_diam);
            }
        if (evaluator::needsCharge())
            {
            Scalar neigh_q = __ldg(d_charge + cur_bond_idx);
            eval.setCharge(q, neigh_q);
            }

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        if (evaluated)
            {
            // add up the virial (double counting, multiply by 0.5)
            Scalar force_div2r = force_divr/Scalar(2.0);
            virial[0] += dx.x * dx.x * force_div2r; // xx
            virial[1] += dx.x * dx.y * force_div2r; // xy
            virial[2] += dx.x * dx.z * force_div2r; // xz
            virial[3] += dx.y * dx.y * force_div2r; // yy
            virial[4] += dx.y * dx.z * force_div2r; // yz
            virial[5] += dx.z * dx.z * force_div2r; // zz

            // add up the forces
            force.x += dx.x * force_divr;
            force.y += dx.y * force_divr;
            force.z += dx.z * force_divr;
            // energy is double counted: multiply by 0.5
            force.w += bond_eng * Scalar(0.5);
            }
        else
            {
            *d_flags = 1;
            return;
            }
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6 ; i++)
        d_virial[i*virial_pitch + idx] = virial[i];
    }

#include <iostream>
//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param bond_args Other arguments to pass onto the kernel
    \param d_params Parameters for the potential, stored per bond type
    \param d_flags flags on the device - a 1 will be written if evaluation
                   of forces failed for any bond

    This is just a driver function for gpu_compute_bond_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_bond_forces(const bond_args_t& bond_args,
                                    const typename evaluator::param_type *d_params,
                                    unsigned int *d_flags)
    {
    assert(d_params);
    assert(bond_args.n_bond_types > 0);

    // check that block_size is valid
    assert(bond_args.block_size != 0);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_compute_bond_forces_kernel<evaluator>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(bond_args.block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( bond_args.N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    unsigned int shared_bytes = sizeof(typename evaluator::param_type) *
                                bond_args.n_bond_types;

    // run the kernel
    gpu_compute_bond_forces_kernel<evaluator><<<grid, threads, shared_bytes>>>(
        bond_args.d_force, bond_args.d_virial, bond_args.virial_pitch, bond_args.N,
        bond_args.d_pos, bond_args.d_charge, bond_args.d_diameter, bond_args.box, bond_args.d_gpu_bondlist,
        bond_args.gpu_table_indexer, bond_args.d_gpu_n_bonds, bond_args.n_bond_types, d_params, d_flags);

    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_BOND_GPU_CUH__
