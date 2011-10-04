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
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include "HarmonicBondForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file HarmonicBondForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic bond forces. Used by HarmonicBondForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading bond parameters
texture<float2, 1, cudaReadModeElementType> bond_params_tex;

//! Kernel for caculating harmonic bond forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data arrays to calculate forces on
    \param box Box dimensions for periodic boundary condition handling
    \param blist Bond data to use in calculating the forces
*/
extern "C" __global__
void gpu_compute_harmonic_bond_forces_kernel(float4* d_force,
                                             float* d_virial,
                                             gpu_pdata_arrays pdata,
                                             gpu_boxsize box,
                                             gpu_bondtable_array blist)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= pdata.N)
        return;
        
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds = blist.n_bonds[idx];
    
    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
    float4 pos = tex1Dfetch(pdata_pos_tex, idx);
    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // initialize the virial to 0
    float virial = 0.0f;
    
    // loop over neighbors
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        // the volatile fails to compile in device emulation mode (MEM TRANSFER: 8 bytes)
#ifdef _DEVICEEMU
        uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + idx];
#else
        // the volatile is needed to force the compiler to load the uint2 coalesced
        volatile uint2 cur_bond = blist.bonds[blist.pitch*bond_idx + idx];
#endif
        
        int cur_bond_idx = cur_bond.x;
        int cur_bond_type = cur_bond.y;
        
        // get the bonded particle's position (MEM TRANSFER: 16 bytes)
        float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_bond_idx);
        
        // calculate dr (FLOPS: 3)
        float dx = pos.x - neigh_pos.x;
        float dy = pos.y - neigh_pos.y;
        float dz = pos.z - neigh_pos.z;
        
        // apply periodic boundary conditions (FLOPS: 12)
        dx -= box.Lx * rintf(dx * box.Lxinv);
        dy -= box.Ly * rintf(dy * box.Lyinv);
        dz -= box.Lz * rintf(dz * box.Lzinv);
        
        // get the bond parameters (MEM TRANSFER: 8 bytes)
        float2 params = tex1Dfetch(bond_params_tex, cur_bond_type);
        float K = params.x;
        float r_0 = params.y;
        
        // FLOPS: 16
        float rsq = dx*dx + dy*dy + dz*dz;
        float rinv = rsqrtf(rsq);
        float forcemag_divr = K * (r_0 * rinv - 1.0f);
        if (!isfinite(forcemag_divr))
            forcemag_divr = 0.0f;
        float bond_eng = 0.5f * K * (r_0 - 1.0f / rinv) * (r_0 - 1.0f / rinv);
        
        // add up the virial (FLOPS: 3)
        virial += float(1.0/6.0) * rsq * forcemag_divr;
        
        // add up the forces (FLOPS: 7)
        force.x += dx * forcemag_divr;
        force.y += dy * forcemag_divr;
        force.z += dz * forcemag_divr;
        force.w += bond_eng;
        }
        
    // energy is double counted: multiply by 0.5
    force.w *= 0.5f;
    
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;
    d_virial[idx] = virial;
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param btable List of bonds stored on the GPU
    \param d_params K and r_0 params packed as float2 variables
    \param n_bond_types Number of bond types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one float2 element per bond type. The x component contains K the spring constant
    and the y component contains r_0 the equilibrium length.
*/
cudaError_t gpu_compute_harmonic_bond_forces(float4* d_force,
                                             float* d_virial,
                                             const gpu_pdata_arrays &pdata,
                                             const gpu_boxsize &box,
                                             const gpu_bondtable_array &btable,
                                             float2 *d_params, unsigned int n_bond_types,
                                             int block_size)
    {
    assert(d_params);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, bond_params_tex, d_params, sizeof(float2) * n_bond_types);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_harmonic_bond_forces_kernel<<< grid, threads>>>(d_force, d_virial, pdata, box, btable);
    
    return cudaSuccess;
    }

