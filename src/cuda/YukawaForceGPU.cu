/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#include "YukawaForceGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file YukawaForceGPU.cu
    \brief Defines GPU kernel code for calculating Yukawa pair forces. Used by YukawaForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Kernel for calculating yukawa forces
/*! This kerenel is called to calculate the lennard-jones forces on all N particles

    \param force_data Device memory array to write calculated forces to
    \param pdata Particle data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param nlist Neigbhor list data on the GPU to use to calculate the forces
    \param d_coeffs Coefficients to the lennard jones force.
    \param coeff_width Width of the coefficient matrix
    \param r_cutsq Precalculated r_cut*r_cut, where r_cut is the radius beyond which forces are
        set to 0
    \param kappa Screening Length

    \a coeffs is a pointer to a matrix in memory. \c coeffs[i*coeff_width+j] is epsilon for the type pair \a i, \a j.
    The values in d_coeffs are read into shared memory, so
    \c coeff_width*coeff_width*sizeof(float) bytes of extern shared memory must be allocated for the kernel call.

    Developer information:
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
extern "C" __global__ void gpu_compute_yukawa_forces_kernel(gpu_force_data_arrays force_data, gpu_pdata_arrays pdata, gpu_boxsize box, gpu_nlist_array nlist, float *d_coeffs, int coeff_width, float r_cutsq, float kappa)
    {
    // read in the coefficients
    extern __shared__ float s_coeffs[];
    for (int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < coeff_width*coeff_width)
            s_coeffs[cur_offset + threadIdx.x] = d_coeffs[cur_offset + threadIdx.x];
        }
    __syncthreads();
    
    // start by identifying which particle we are to handle
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx_local >= pdata.local_num)
        return;
        
    int idx_global = idx_local + pdata.local_beg;
    
    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    int n_neigh = nlist.n_neigh[idx_global];
    
    // read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virial = 0.0f;
    
    // loop over neighbors
#ifdef ARCH_SM13
    // sm13 offers warp voting which makes this hardware bug workaround less of a performance penalty
    for (int neigh_idx = 0; __any(neigh_idx < n_neigh); neigh_idx++)
#else
    for (int neigh_idx = 0; neigh_idx < nlist.height; neigh_idx++)
#endif
        {
        if (neigh_idx < n_neigh)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            int cur_neigh = nlist.list[nlist.pitch*neigh_idx + idx_global];
            
            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
            
            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float dx = pos.x - neigh_pos.x;
            float dy = pos.y - neigh_pos.y;
            float dz = pos.z - neigh_pos.z;
            
            // apply periodic boundary conditions: (FLOPS 12)
            dx -= box.Lx * rintf(dx * box.Lxinv);
            dy -= box.Ly * rintf(dy * box.Lyinv);
            dz -= box.Lz * rintf(dz * box.Lzinv);
            
            // calculate r squard (FLOPS: 5)
            float rsq = dx*dx + dy*dy + dz*dz;
            
            // calculate r and rinv (FLOPS: 2)
            float r = sqrtf(rsq);
            
            float rinv;
            if (rsq >= r_cutsq)
                rinv = 0.0f;
            else
                rinv = 1.0f / r;
                
            // calculate 1/r^2 (FLOPS: 1)
            float r2inv = rinv*rinv;
            
            // lookup the coefficients between this combination of particle types
            int typ_pair = __float_as_int(neigh_pos.w) * coeff_width + __float_as_int(pos.w);
            float epsilon = s_coeffs[typ_pair];
            
            // calculate the force magnitude / r (FLOPS: 6)
            float forcemag_divr = epsilon*expf(-kappa*r)*r2inv*(kappa + rinv);
            
            // calculate the virial (FLOPS: 3)
            virial += float(1.0/6.0) * rsq * forcemag_divr;
            // calculate the pair energy (FLOPS: 3)
            float pair_eng = epsilon*expf(-kappa*r)*rinv;
            
            // add up the force vector components (FLOPS: 7)
            force.x += dx * forcemag_divr;
            force.y += dy * forcemag_divr;
            force.z += dz * forcemag_divr;
            force.w += pair_eng;
            }
        }
        
    // potential energy per particle must be halved
    force.w *= 0.5f;
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    force_data.force[idx_local] = force;
    force_data.virial[idx_local] = virial;
    }


/*! \param force_data Force data on GPU to write forces to
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param nlist Neighbor list stored on the gpu
    \param d_coeffs Coefficients to the lennard jones force.
    \param coeff_width Width of the coefficient matrix
    \param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
        force is set to 0
    \param kappa Screening Length
    \param block_size Block size to execute

    This is just a driver for calcYukawaForces_kernel, see it for more details.

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_yukawa_forces(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_nlist_array &nlist, float *d_coeffs, int coeff_width, float r_cutsq, float kappa, int block_size)
    {
    assert(d_coeffs);
    assert(coeff_width > 0);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.local_num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the texture
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_yukawa_forces_kernel<<< grid, threads, sizeof(float)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, r_cutsq, kappa);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

// vim:syntax=cpp
