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

// $Id$
// $URL$
// Maintainer: joaander

#include "gpu_settings.h"
#include "LJForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file LJForceGPU.cu
    \brief Defines GPU kernel code for calculating the Lennard-Jones pair forces. Used by LJForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
texture<unsigned int, 1, cudaReadModeElementType> pdata_body_tex;

//! Texture for reading particle diameters, only used if in slj mode
texture<float, 1, cudaReadModeElementType> pdata_diam_tex;

//! Kernel for calculating lj forces
/*! This kernel is called to calculate the lennard-jones forces on all N particles

    \param force_data Device memory array to write calculated forces to
    \param pdata Particle data on the GPU to calculate forces on
    \param nlist Neigbhor list data on the GPU to use to calculate the forces
    \param d_coeffs Coefficients to the lennard jones force.
    \param coeff_width Width of the coefficient matrix
    \param r_cutsq Precalculated r_cut*r_cut, where r_cut is the radius beyond which forces are
        set to 0
    \param rcut6inv Precalculated 1/r_cut**6
    \param xplor_denom_inv Precalculated 1/xplor denominator
    \param r_on_sq Precalculated r_on*r_on (for xplor)
    \param box Box dimensions used to implement periodic boundary conditions

    \a coeffs is a pointer to a matrix in memory. \c coeffs[i*coeff_width+j].x is \a lj1 for the type pair \a i, \a j.
    Similarly, .y is the \a lj2 parameter. The values in d_coeffs are read into shared memory, so
    \c coeff_width*coeff_width*sizeof(float2) bytes of extern shared memory must be allocated for the kernel call.

    Developer information:
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template<bool ulf_workaround, unsigned int shift_mode, bool slj> __global__ void gpu_compute_lj_forces_kernel(gpu_force_data_arrays force_data, gpu_pdata_arrays pdata, gpu_boxsize box, gpu_nlist_array nlist, float3 *d_coeffs, int coeff_width, float r_cutsq, float rcut6inv, float xplor_denom_inv, float r_on_sq)
    {
    // read in the coefficients
    extern __shared__ float3 s_coeffs[];
    for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < coeff_width*coeff_width)
            s_coeffs[cur_offset + threadIdx.x] = d_coeffs[cur_offset + threadIdx.x];
        }
    __syncthreads();
    
    // start by identifying which particle we are to handle
    unsigned int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx_local >= pdata.local_num)
        return;
        
    unsigned int idx_global = idx_local + pdata.local_beg;
    
    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = nlist.n_neigh[idx_global];
    
    // read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
    unsigned int my_body = tex1Dfetch(pdata_body_tex, idx_global);

    float diam;
    if (slj == true)
        {
        // read in the diameter of our particle.
        // (MEM TRANSFER: 4 bytes)
        diam = tex1Dfetch(pdata_diam_tex, idx_global);
        }
        
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virial = 0.0f;
    
    // prefetch neighbor index
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = nlist.list[idx_global];
    
    // loop over neighbors
    // on pre C1060 hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    int n_loop;
    if (ulf_workaround)
        n_loop = nlist.height;
    else
        n_loop = n_neigh;
        
    for (int neigh_idx = 0; neigh_idx < n_loop; neigh_idx++)
        {
        if (!ulf_workaround || neigh_idx < n_neigh)
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_neigh = next_neigh;
            next_neigh = nlist.list[nlist.pitch*(neigh_idx+1) + idx_global];
            
            // get the neighbor's body
            unsigned int neigh_body = tex1Dfetch(pdata_body_tex, cur_neigh);
            if (neigh_body != 0xffffffff && neigh_body == my_body)
                continue;
            
            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
            
            float neigh_diam;
            if (slj)
                {
                // get the neighbor's diameter (MEM TRANSFER: 4 bytes)
                neigh_diam = tex1Dfetch(pdata_diam_tex, cur_neigh);
                }
                
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
            
            // Shift the distance if the diameter-shifted LJ force is being used
            float r, radj;
            if (slj)
                {
                r = sqrtf(rsq);
                radj =  r - (diam/2.0f + neigh_diam/2.0f - 1.0);
                rsq = radj*radj;  // This is now a diameter adjusted potential distance for shifted LJ pair potentials
                }
                
            // lookup the coefficients between this combination of particle types
            int typ_pair = __float_as_int(neigh_pos.w) * coeff_width + __float_as_int(pos.w);
            float lj1 = s_coeffs[typ_pair].x;
            float lj2 = s_coeffs[typ_pair].y;
            r_cutsq = s_coeffs[typ_pair].z * s_coeffs[typ_pair].z; // r_cut per type
            
            // calculate 1/r^2 (FLOPS: 2)
            float r2inv;
            if (rsq >= r_cutsq)
                r2inv = 0.0f;
            else
                r2inv = 1.0f / rsq;
                
            // calculate 1/r^6 (FLOPS: 2)
            float r6inv = r2inv*r2inv*r2inv;
            // calculate the force magnitude / r (FLOPS: 6)
            float forcemag_divr;
            
            if (slj)
                {
                float radj_inv = 1.0f/radj;
                float r_inv = 1.0f/r;
                forcemag_divr = radj_inv * r_inv * r6inv * (12.0f * lj1  * r6inv - 6.0f * lj2);
                }
            else forcemag_divr = r2inv * r6inv * (12.0f * lj1  * r6inv - 6.0f * lj2);
            
            // calculate the pair energy (FLOPS: 3)
            float pair_eng = r6inv * (lj1 * r6inv - lj2);
            
            if (shift_mode == 1)
                {
                // shifting is enabled: shift the energy (FLOPS: 4)
                if (rsq < r_cutsq)
                    pair_eng -= rcut6inv * (lj1*rcut6inv - lj2);
                }
            else if (shift_mode == 2)
                {
                if (rsq >= r_on_sq)
                    {
                    // Implement XPLOR smoothing (FLOPS: 15)
                    float old_pair_eng = pair_eng;
                    float old_forcemag_divr = forcemag_divr;
                    
                    float rsq_minus_r_cut_sq = rsq - r_cutsq;
                    float s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq * (r_cutsq + 2.0f * rsq - 3.0f * r_on_sq) * xplor_denom_inv;
                    float ds_dr_divr = 12.0f * (rsq - r_on_sq) * rsq_minus_r_cut_sq * xplor_denom_inv;
                    
                    // make modifications to the old pair energy and force
                    if (rsq < r_cutsq)
                        {
                        pair_eng = old_pair_eng * s;
                        // note: I'm not sure why the minus sign needs to be there: my notes have a +. But this is verified correct
                        // I think it might have something to do with the fact that I'm actually calculating \vec{r}_{ji} instead of {ij}
                        forcemag_divr = s * old_forcemag_divr - ds_dr_divr * old_pair_eng;
                        }
                    }
                }
                
            // calculate the virial (FLOPS: 3)
            if (!slj) virial += float(1.0/6.0) * rsq * forcemag_divr;
            else virial += float(1.0/6.0) * r * r * forcemag_divr; //rsq has been "adjusted" for diameter, r has not!
            
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
    \param d_coeffs A \a coeff_width by \a coeff_width matrix of coefficients indexed by type
        pair i,j. The x-component is lj1 and the y-component is lj2.
    \param coeff_width Width of the \a d_coeffs matrix.
    \param opt More execution options bundled up in a strct

    \returns Any error code resulting from the kernel launch

    This is just a driver for calcLJForces_kernel, see the documentation for it for more information.
*/
cudaError_t gpu_compute_lj_forces(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_nlist_array &nlist, float3 *d_coeffs, int coeff_width, const lj_options& opt)
    {
    assert(d_coeffs);
    assert(coeff_width > 0);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.local_num / (double)opt.block_size), 1, 1);
    dim3 threads(opt.block_size, 1, 1);
    
    // bind the texture
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_body_tex, pdata.body, sizeof(unsigned int) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    if (opt.slj == true)
        {
        // bind the texture for the diameter read, even though this is only used if the slj option is turned on.
        pdata_diam_tex.normalized = false;
        pdata_diam_tex.filterMode = cudaFilterModePoint;
        error = cudaBindTexture(0, pdata_diam_tex, pdata.diameter, sizeof(float) * pdata.N);
        if (error != cudaSuccess)
            return error;
        }
        
    // precompue some values   - note these values will be recomputed in the kernel for opt.slj == true
    float rcut2inv = 1.0f / opt.r_cutsq;
    float rcut6inv = rcut2inv * rcut2inv * rcut2inv;
    float r_on_sq = opt.xplor_fraction * opt.xplor_fraction * opt.r_cutsq;
    float xplor_denom_inv = 1.0f / ((opt.r_cutsq - r_on_sq) * (opt.r_cutsq - r_on_sq) * (opt.r_cutsq - r_on_sq));
    
    // run the kernel
    if (opt.ulf_workaround)
        {
        if (opt.shift_mode == 0)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<true, 0, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<true, 0, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else if (opt.shift_mode == 1)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<true, 1, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<true, 1, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else if (opt.shift_mode == 2)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<true, 2, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<true, 2, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else
            return cudaErrorUnknown;
        }
    else
        {
        if (opt.shift_mode == 0)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<false, 0, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<false, 0, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else if (opt.shift_mode == 1)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<false, 1, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<false, 1, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else if (opt.shift_mode == 2)
            if (opt.slj == true)
                gpu_compute_lj_forces_kernel<false, 2, true><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
            else
                gpu_compute_lj_forces_kernel<false, 2, false><<< grid, threads, sizeof(float3)*coeff_width*coeff_width >>>(force_data, pdata, box, nlist, d_coeffs, coeff_width, opt.r_cutsq, rcut6inv, xplor_denom_inv, r_on_sq);
        else
            return cudaErrorUnknown;
        }
        
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

