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

#include "FENEBondForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file FENEBondForceGPU.cu
    \brief Defines GPU kernel code for calculating the FENE bond forces. Used by FENEBondForceComputeGPU.
*/

//! Texture for reading bond parameters
texture<float4, 1, cudaReadModeElementType> bond_params_tex;

//! Kernel for caculating FENE bond forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_diameter device array of particle diameters
    \param box Box dimensions for periodic boundary condition handling
    \param blist Bond data to use in calculating the forces
    \param d_checkr Flag allocated on the device for use in checking for bonds that are too long
*/
extern "C" __global__ 
void gpu_compute_fene_bond_forces_kernel(float4* d_force,
                                         float* d_virial,
                                         const unsigned int virial_pitch,
                                         const unsigned int N,
                                         const Scalar4 *d_pos,
                                         const Scalar *d_diameter,
                                         gpu_boxsize box,
                                         gpu_bondtable_array blist,
                                         unsigned int *d_checkr)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds = blist.n_bonds[idx];
    
    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
    float4 pos = d_pos[idx];
    
    // read in the diameter of our particle.
    float diam = d_diameter[idx];
    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // initialize the virial to 0
    float virialxx = 0.0f;
    float virialxy = 0.0f;
    float virialxz = 0.0f;
    float virialyy = 0.0f;
    float virialyz = 0.0f;
    float virialzz = 0.0f;


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
        
        // get the bonded particle's diameter
        float neigh_diam = d_diameter[cur_bond_idx];
        
        // calculate dr (FLOPS: 3)
        float dx = pos.x - neigh_pos.x;
        float dy = pos.y - neigh_pos.y;
        float dz = pos.z - neigh_pos.z;
        
        // apply periodic boundary conditions (FLOPS: 12)
        dx -= box.Lx * rintf(dx * box.Lxinv);
        dy -= box.Ly * rintf(dy * box.Lyinv);
        dz -= box.Lz * rintf(dz * box.Lzinv);
        
        // get the bond parameters (MEM TRANSFER: 8 bytes)
        float4 params = tex1Dfetch(bond_params_tex, cur_bond_type);
        float K = params.x;
        float r_0 = params.y;
        
        // lj1 is defined as 4*epsilon*sigma^12
        float lj1 = 4.0f * params.w * params.z * params.z * params.z * params.z * params.z * params.z * 
                        params.z * params.z * params.z * params.z * params.z * params.z;
        // lj2 is defined as 4*epsilon*sigma^6
        float lj2 = 4.0f * params.w * params.z * params.z * params.z * params.z * params.z * params.z;
        float epsilon = params.w;
       
        // FLOPS: 5
        float rsq = dx*dx + dy*dy + dz*dz;
        float rmdoverr = 1.0f;

        // if particles have diameters that are not 1.0 need to correct this value by alpha
        float rinv = rsqrtf(rsq);
        float r = 1.0f / rinv;
        float radj =  r - (diam/2.0f + neigh_diam/2.0f - 1.0f);
        rmdoverr = radj * rinv;
        rsq = radj*radj;  // This is now a diameter adjusted potential distance for diameter shifted potentials
        
        float wcaforcemag_divr = 0.0f;
        float pair_eng = 0.0f;
         
        if (rsq < 1.2599210498f && epsilon != 0.0f)  // comparing to the WCA limit
            {
            // calculate 1/r^6 (FLOPS: 2)
            float r2inv = rinv * rinv;
            float r6inv = r2inv*r2inv*r2inv;
            // calculate the force magnitude / r (FLOPS: 6)
            wcaforcemag_divr = r2inv * r6inv * (12.0f * lj1  * r6inv - 6.0f * lj2);
            // calculate the pair energy (FLOPS: 3)
            pair_eng = r6inv * (lj1 * r6inv - lj2) + epsilon;
            }
        if (!isfinite(pair_eng))
            pair_eng = 0.0f;    

        // FLOPS: 7
        float forcemag_divr = -K / (1.0f - rsq/(r_0*r_0))*rmdoverr + wcaforcemag_divr*rmdoverr;
        float bond_eng = -0.5f * K * r_0*r_0*logf(1.0f - rsq/(r_0*r_0));
        
        // detect non-finite results and zero them. This will result in the correct 0 force for r ~= 0. The energy
        // will be incorrect for r > r_0, however. Assuming that r > r_0 because K == 0, this is fine.
        if (!isfinite(forcemag_divr))
            forcemag_divr = 0.0f;
        if (!isfinite(bond_eng))
            bond_eng = 0.0f;
        
        // add up the virial (FLOPS: 3)
        float forcemag_div2r = 0.5f * forcemag_divr;
        virialxx += dx * dx * forcemag_div2r;
        virialxy += dx * dy * forcemag_div2r;
        virialxz += dx * dz * forcemag_div2r;
        virialyy += dy * dy * forcemag_div2r;
        virialyz += dy * dz * forcemag_div2r;
        virialzz += dz * dz * forcemag_div2r;

        // add up the forces (FLOPS: 7)
        force.x += dx * forcemag_divr;
        force.y += dy * forcemag_divr;
        force.z += dz * forcemag_divr;
        force.w += bond_eng + pair_eng;
        
        // Checking to see if bond length restriction is violated.
        if (rsq >= r_0*r_0) *d_checkr = 1;
        }
        
    // energy is double counted: multiply by 0.5
    force.w *= 0.5f;
    
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;
    d_virial[0*virial_pitch+idx] = virialxx;
    d_virial[1*virial_pitch+idx] = virialxy;
    d_virial[2*virial_pitch+idx] = virialxz;
    d_virial[3*virial_pitch+idx] = virialyy;
    d_virial[4*virial_pitch+idx] = virialyz;
    d_virial[5*virial_pitch+idx] = virialzz;
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_diameter device array of particle diameters
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param btable List of bonds stored on the GPU
    \param d_params K, r_0, lj1, and lj2 params packed as float4 variables
    \param n_bond_types Number of bond types in d_params
    \param block_size Block size to use when performing calculations
    \param d_flags flags on the device - a 1 will be written if the r > R0

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float4 element per bond type. The x component contains K the spring constant
    and the y component contains r_0 the equilibrium length, z and w contain lj1 and lj2.
*/
cudaError_t gpu_compute_fene_bond_forces(float4* d_force,
                                         float* d_virial,
                                         const unsigned int virial_pitch,
                                         const unsigned int N,
                                         const Scalar4 *d_pos,
                                         const Scalar *d_diameter,
                                         const gpu_boxsize &box,
                                         const gpu_bondtable_array &btable,
                                         float4 *d_params,
                                         unsigned int n_bond_types,
                                         int block_size,
                                         unsigned int *d_flags)
    {
    assert(d_params);
    // check that block_size is valid
    assert(block_size != 0);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    error = cudaBindTexture(0, bond_params_tex, d_params, sizeof(float4) * n_bond_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_fene_bond_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, d_diameter, box, btable, d_flags);
    
    return cudaSuccess;
    }

