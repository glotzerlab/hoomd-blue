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

#include "HOOMDMath.h"
#include "ParticleData.cuh"
#include "BondData.cuh"
#include "Index1D.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file PotentialBondGPU.cuh
    \brief Defines templated GPU kernel code for calculating the bond forces.
*/

#ifndef __POTENTIAL_BOND_GPU_CUH__
#define __POTENTIAL_BOND_GPU_CUH__

//! Wrapps arguments to gpu_cgbf
struct bond_args_t
    {
    //! Construct a bond_args_t
    bond_args_t(float4 *_d_force,
              float *_d_virial,
              const unsigned int _virial_pitch,
              const unsigned int _N,
              const Scalar4 *_d_pos,
              const Scalar *_d_charge,
              const Scalar *_d_diameter,
              const gpu_boxsize &_box,
              const gpu_bondtable_array &_btable,
              const unsigned int _n_bond_types,
              const unsigned int _block_size)
                : d_force(_d_force),
                  d_virial(_d_virial),
                  virial_pitch(_virial_pitch),
                  N(_N),
                  d_pos(_d_pos),
                  d_diameter(_d_diameter),
                  d_charge(_d_charge),
                  box(_box),
                  btable(_btable),
                  n_bond_types(_n_bond_types),
                  block_size(_block_size)
        {
        };

    float4 *d_force;                   //!< Force to write out
    float *d_virial;                   //!< Virial to write out
    const unsigned int virial_pitch;   //!< pitch of 2D array of virial matrix elements
    unsigned int N;                    //!< number of particles
    const Scalar4 d_pos;               //!< particle positions
    const Scalar d_diameter;           //!< particle diameters
    const Scalar d_charge;             //!< particle charges
    const gpu_boxsize &box;            //!< Simulation box in GPU format
    const gpu_bondtable_array &btable; //!< List of bonds stored on the GPU
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
    \param d_pos particle positions on the GPU
    \param d_diameter particle diameters
    \param d_charge particle charges
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_params Parameters for the potential, stored per bond type
    \param d_checkr Flag allocated on the device for use in checking for bonds that are too long


    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorBond class to evualuate V(r) and -delta V(r)/r

*/
template< class evaluator >
__global__ void gpu_compute_bond_forces_kernel(float4 *d_force,
                                               float *d_virial,
                                               const unsigned int virial_pitch,
                                               Scalar4 *d_pos,
                                               Scalar *d_diameter,
                                               Scalar *d_charge,
                                               const gpu_boxsize box,
                                               gpu_bondtable_array blist,
                                               const unsigned int n_bond_type,
                                               const typename evaluator::param_type *d_params,
                                               unsigned int *d_checkr)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds = blist.n_bonds[idx];

    // shared array for per bond type parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params =
        (typename evaluator::param_type *)(&s_data[0]);

    // load in per bond type parameters
    if (threadIdx.x < n_bond_type)
       s_params[threadIdx.x] = d_params[threadIdx.x];
    __syncthreads();

    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
    Scalar4 pos = d_pos[idx];

    // read in the diameter of our particle if needed
    float diam = 0;
    if (evaluator::needsDiameter())
       diam = d_diameter[idx];

    // read in the diameter of our particle if needed
    float q = 0;
    if (evaluator::needsCharge())
       q = d_charge[idx];

    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // initialize the virial tensor to 0
    float virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    // loop over neighbors
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        // MEM TRANSFER: 8 bytes
        // the volatile fails to compile in device emulation mode
#ifdef _DEVICEEMU
        uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + idx];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + idx];
#endif

        int cur_bond_idx = cur_bond.x;
        int cur_bond_type = cur_bond.y;

        // get the bonded particle's position (MEM_TRANSFER: 16 bytes)
        float4 neigh_pos = d_pos[cur_bond_idx];

        // calculate dr (FLOPS: 3)
        float dx = pos.x - neigh_pos.x;
        float dy = pos.y - neigh_pos.y;
        float dz = pos.z - neigh_pos.z;

        // apply periodic boundary conditions (FLOPS: 12)
        dx -= box.Lx * rintf(dx * box.Lxinv);
        dy -= box.Ly * rintf(dy * box.Lyinv);
        dz -= box.Lz * rintf(dz * box.Lzinv);

        // get the bond parameters (MEM TRANSFER: 8 bytes)
        typename evaluator::param_type param = s_params[cur_bond_type];

        float rsq = dx*dx + dy*dy + dz*dz;

        // evaluate the potential
        float force_divr = 0.0f;
        float bond_eng = 0.0f;

        evaluator eval(rsq, param);

        // get the bonded particle's diameter if needed
        if (evaluator::needsDiameter())
            {
            float neigh_diam = d_diameter[cur_bond_idx];
            eval.setDiameter(diam, neigh_diam);
            }
        if (evaluator::needsCharge())
            {
            float neigh_q = d_charge[cur_bond_idx];
            eval.setCharge(q, neigh_q);
            }

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        if (evaluated)
            {
            // add up the virial (double counting, multiply by 0.5)
            float force_div2r = force_divr/2.0f;
            virial[0] += dx * dx * force_div2r; // xx
            virial[1] += dx * dy * force_div2r; // xy
            virial[2] += dx * dz * force_div2r; // xz
            virial[3] += dy * dy * force_div2r; // yy
            virial[4] += dy * dz * force_div2r; // yz
            virial[5] += dz * dz * force_div2r; // zz

            // add up the forces
            force.x += dx * force_divr;
            force.y += dy * force_divr;
            force.z += dz * force_divr;
            // energy is double counted: multiply by 0.5
            force.w += bond_eng * 0.5f;
            }
        else {
            *d_checkr = 1;
             }
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;
    for (unsigned int i = 0; i < 6 ; i++)
        d_virial[i*virial_pitch + idx] = virial[i];
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param bond_args Other arugments to pass onto the kernel
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

    // chck that block_size is valid
    assert(bond_args.block_size != 0);

    // setup the grid to run the kernel
    dim3 grid( bond_args.N / bond_args.block_size + 1, 1, 1);
    dim3 threads(bond_args.block_size, 1, 1);

    unsigned int shared_bytes = sizeof(typename evaluator::param_type) *
                                bond_args.n_bond_types;

    // run the kernel
    gpu_compute_bond_forces_kernel<evaluator><<<grid, threads, shared_bytes>>>(
        bond_args.d_force, bond_args.d_virial, bond_args.virial_pitch, bond_args.N,
        bond_args.pos, bond_args.diameter, bond_args.charge, bond_args.box, bond_args.btable,
        bond_args.n_bond_types, d_params, d_flags);

    return cudaSuccess;
    }
#endif

#endif // __POTENTIAL_BOND_GPU_CUH__
