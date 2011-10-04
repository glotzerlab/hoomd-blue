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

// Maintainer: sbarr

#include "PPPMForceGPU.cuh"
#include <iostream>
    using namespace std;

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define MAX_BLOCK_DIM_SIZE 65535

//! Constant memory for gridpoint weighting
#define CONSTANT_SIZE 2048
__device__ __constant__ float GPU_rho_coeff[CONSTANT_SIZE];


// global variables for thermodynamic output
int g_Nx, g_Ny, g_Nz, g_block_size;
cufftComplex *g_rho_real_space;
float3 *g_vg;
float *g_green_hat;
float2 *o_data, *idat;

/*! \file HarmonicBondForceGPU.cu
  \brief Defines GPU kernel code for calculating the harmonic bond forces. Used by HarmonicBondForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading charge parameters
texture<float, 1, cudaReadModeElementType> pdata_charge_tex;

__device__ inline void atomicFloatAdd(float* address, float value)
    {
#if (__CUDA_ARCH__ < 200)
    float old = value;
    float new_old;
    do
        {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
        }
    while ((old = atomicExch(address, new_old))!=0.0f);
#else
    atomicAdd(address, value);
#endif
    }

__device__ inline void AddToGridpoint(int X, int Y, int Z, cufftComplex* array, float value, int Ny, int Nz)
    {
    atomicFloatAdd(&array[Z + Nz * (Y + Ny * X)].x, value);
    }


extern "C" __global__
void assign_charges_to_grid_kernel(gpu_pdata_arrays pdata, 
                                   gpu_boxsize box, 
                                   cufftComplex *rho_real_space, 
                                   int Nx, 
                                   int Ny, 
                                   int Nz, 
                                   int order,
                                   unsigned int *d_group_members,
                                   unsigned int group_size)
    {
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        //get particle information
        float qi = tex1Dfetch(pdata_charge_tex, idx);
        if(fabs(qi) > 0.0f) {
            float4 posi = tex1Dfetch(pdata_pos_tex, idx);
            //calculate dx, dy, dz for the charge density grid:
            float box_dx = box.Lx / ((float)Nx);
            float box_dy = box.Ly / ((float)Ny);
            float box_dz = box.Lz / ((float)Nz);
    
        
            //normalize position to gridsize:
            posi.x += box.Lx / 2.0f;
            posi.y += box.Ly / 2.0f;
            posi.z += box.Lz / 2.0f;
   
            posi.x /= box_dx;
            posi.y /= box_dy;
            posi.z /= box_dz;
    
    
            float shift, shiftone, x0, y0, z0, dx, dy, dz;
            int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
            nlower = -(order-1)/2;
            nupper = order/2;
    
            if (order % 2) 
                {
                shift =0.5f;
                shiftone = 0.0f;
                }
            else 
                {
                shift = 0.0f;
                shiftone = 0.5f;
                }
        
            nxi = __float2int_rd(posi.x + shift);
            nyi = __float2int_rd(posi.y + shift);
            nzi = __float2int_rd(posi.z + shift);
    
            dx = shiftone+(float)nxi-posi.x;
            dy = shiftone+(float)nyi-posi.y;
            dz = shiftone+(float)nzi-posi.z;
    
            int n,m,l,k;
            float result;
            int mult_fact = 2*order+1;

            x0 = qi / (box_dx*box_dy*box_dz);
            for (n = nlower; n <= nupper; n++) {
                mx = n+nxi;
                if(mx >= Nx) mx -= Nx;
                if(mx < 0)  mx += Nx;
                result = 0.0f;
                for (k = order-1; k >= 0; k--) {
                    result = GPU_rho_coeff[n-nlower + k*mult_fact] + result * dx;
                    }
                y0 = x0*result;
                for (m = nlower; m <= nupper; m++) {
                    my = m+nyi;
                    if(my >= Ny) my -= Ny;
                    if(my < 0)  my += Ny;
                    result = 0.0f;
                    for (k = order-1; k >= 0; k--) {
                        result = GPU_rho_coeff[m-nlower + k*mult_fact] + result * dy;
                        }
                    z0 = y0*result;
                    for (l = nlower; l <= nupper; l++) {
                        mz = l+nzi;
                        if(mz >= Nz) mz -= Nz;
                        if(mz < 0)  mz += Nz;
                        result = 0.0f;
                        for (k = order-1; k >= 0; k--) {
                            result = GPU_rho_coeff[l-nlower + k*mult_fact] + result * dz;
                            }
                        AddToGridpoint(mx, my, mz, rho_real_space, z0*result, Ny, Nz);
                        }
                    }
                }
            }
        }
    }

extern "C" __global__
void combined_green_e_kernel(cufftComplex* E_x, 
                             cufftComplex* E_y, 
                             cufftComplex* E_z, 
                             float3* k_vec, 
                             cufftComplex* rho, 
                             int Nx, 
                             int Ny, 
                             int Nz, 
                             float* green_function)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(idx < Nx * Ny * Nz)
        {
        float3 k_vec_local = k_vec[idx];
        cufftComplex E_x_local, E_y_local, E_z_local;
        float scale_times_green = green_function[idx] / ((float)(Nx*Ny*Nz));
        cufftComplex rho_local = rho[idx];
    
        rho_local.x *= scale_times_green;
        rho_local.y *= scale_times_green;
      
        E_x_local.x = k_vec_local.x * rho_local.y;
        E_x_local.y = -k_vec_local.x * rho_local.x;
    
        E_y_local.x = k_vec_local.y * rho_local.y;
        E_y_local.y = -k_vec_local.y * rho_local.x;
    
        E_z_local.x = k_vec_local.z * rho_local.y;
        E_z_local.y = -k_vec_local.z * rho_local.x;
    
    
        E_x[idx] = E_x_local;
        E_y[idx] = E_y_local;
        E_z[idx] = E_z_local;   
        }
    }


__global__ void set_gpu_field_kernel(cufftComplex* E_x, 
                                     cufftComplex* E_y, 
                                     cufftComplex* E_z, 
                                     float3* Electric_field, 
                                     int Nx, 
                                     int Ny, 
                                     int Nz)
    {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < Nx * Ny * Nz)
        {
        float3 local_field;
        local_field.x = E_x[tid].x;
        local_field.y = E_y[tid].x;
        local_field.z = E_z[tid].x;
      
        Electric_field[tid] = local_field;
        }
    }

__global__
void zero_forces(float4 *d_force, float *d_virial, gpu_pdata_arrays pdata)
    {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pdata.N)
        {
        d_force[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        d_virial[idx] = 0.0f;
        }
    }

extern "C" __global__ 
void calculate_forces_kernel(float4 *d_force,
                             float *d_virial,
                             gpu_pdata_arrays pdata,
                             gpu_boxsize box,
                             float3 *E_field,
                             int Nx,
                             int Ny,
                             int Nz,
                             int order,
                             unsigned int *d_group_members,
                             unsigned int group_size)
    {  
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        //get particle information
        float qi = tex1Dfetch(pdata_charge_tex, idx);
        if(fabs(qi) > 0.0f) {
            float4 posi = tex1Dfetch(pdata_pos_tex, idx);
    
            //calculate dx, dy, dz for the charge density grid:
            float box_dx = box.Lx / ((float)Nx);
            float box_dy = box.Ly / ((float)Ny);
            float box_dz = box.Lz / ((float)Nz);
    
            //normalize position to gridsize:
            posi.x += box.Lx * 0.5f;
            posi.y += box.Ly * 0.5f;
            posi.z += box.Lz * 0.5f;
   
            posi.x /= box_dx;
            posi.y /= box_dy;
            posi.z /= box_dz;
    
            float shift, shiftone, x0, y0, z0, dx, dy, dz;
            int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
            nlower = -(order-1)/2;
            nupper = order/2;
    
            float4 local_force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if(order % 2) 
                {
                shift =0.5f;
                shiftone = 0.0f;
                }
            else 
                {
                shift = 0.0f;
                shiftone = 0.5f;
                }
    
    
            nxi = __float2int_rd(posi.x + shift);
            nyi = __float2int_rd(posi.y + shift);
            nzi = __float2int_rd(posi.z + shift);
    
            dx = shiftone+(float)nxi-posi.x;
            dy = shiftone+(float)nyi-posi.y;
            dz = shiftone+(float)nzi-posi.z;

            int n,m,l,k;
            float result;
            int mult_fact = 2*order+1;
    
            for (n = nlower; n <= nupper; n++) {
                mx = n+nxi;
                if(mx >= Nx) mx -= Nx;
                if(mx < 0)  mx += Nx;
                result = 0.0f;
                for (k = order-1; k >= 0; k--) {
                    result = GPU_rho_coeff[n-nlower + k*mult_fact] + result * dx;
                    }
                x0 = result;
                for (m = nlower; m <= nupper; m++) {
                    my = m+nyi;
                    if(my >= Ny) my -= Ny;
                    if(my < 0)  my += Ny;
                    result = 0.0f;
                    for (k = order-1; k >= 0; k--) {
                        result = GPU_rho_coeff[m-nlower + k*mult_fact] + result * dy;
                        }
                    y0 = x0*result;
                    for (l = nlower; l <= nupper; l++) {
                        mz = l+nzi;
                        if(mz >= Nz) mz -= Nz;
                        if(mz < 0)  mz += Nz;
                        result = 0.0f;
                        for (k = order-1; k >= 0; k--) {
                            result = GPU_rho_coeff[l-nlower + k*mult_fact] + result * dz;
                            }
                        z0 = y0*result;
                        float local_field_x = E_field[mz + Nz * (my + Ny * mx)].x;
                        float local_field_y = E_field[mz + Nz * (my + Ny * mx)].y;
                        float local_field_z = E_field[mz + Nz * (my + Ny * mx)].z;
                        local_force.x += qi*z0*local_field_x;
                        local_force.y += qi*z0*local_field_y;
                        local_force.z += qi*z0*local_field_z;
                        }
                    }
                }
            d_force[idx] = local_force;
            }
        }
    } 


cudaError_t gpu_compute_pppm_forces(float4 *d_force,
                                    float *d_virial,
                                    const gpu_pdata_arrays &pdata,
                                    const gpu_boxsize &box,
                                    int Nx,
                                    int Ny,
                                    int Nz,
                                    int order,
                                    float *CPU_rho_coeff,
                                    cufftComplex *GPU_rho_real_space,
                                    cufftHandle plan,
                                    cufftComplex *GPU_E_x,
                                    cufftComplex *GPU_E_y,
                                    cufftComplex *GPU_E_z,
                                    float3 *GPU_k_vec,
                                    float *GPU_green_hat,
                                    float3 *E_field,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    int block_size)
    {
    
    cudaMemcpyToSymbol(GPU_rho_coeff, &(CPU_rho_coeff[0]), order * (2*order+1) * sizeof(float));

    // setup the grid to run the kernel with one thread per particle in the group
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // setup the grid to run the kernel with one thread per particle in the group
    dim3 P_grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 P_threads(block_size, 1, 1);
    
    // setup the grid to run the kernel with one thread per grid point
    dim3 N_grid( (int)ceil((double)Nx*Ny*Nz / (double)block_size), 1, 1);
    dim3 N_threads(block_size, 1, 1);

    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_charge_tex, pdata.charge, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // set the grid charge to zero
    cudaMemset(GPU_rho_real_space, 0, sizeof(cufftComplex)*Nx*Ny*Nz);

    // zero the force arrays for all particles
    // zero_forces <<< grid, threads >>> (force_data, pdata);
    cudaMemset(d_force, 0, sizeof(float4)*pdata.N);
    cudaMemset(d_virial, 0, sizeof(float)*pdata.N);


    // run the kernels
    // assign charges to the grid points, one thread per particles
    assign_charges_to_grid_kernel <<< P_grid, P_threads >>> (pdata, 
                                                             box, 
                                                             GPU_rho_real_space, 
                                                             Nx, 
                                                             Ny, 
                                                             Nz, 
                                                             order, 
                                                             d_group_members,
                                                             group_size);
    cudaThreadSynchronize();    

    // FFT
    cufftExecC2C(plan, GPU_rho_real_space, GPU_rho_real_space, CUFFT_FORWARD);
    cudaThreadSynchronize();

    // multiply Green's function to get E field, one thread per grid point
    combined_green_e_kernel <<< N_grid, N_threads >>> (GPU_E_x, 
                                                       GPU_E_y, 
                                                       GPU_E_z, 
                                                       GPU_k_vec, 
                                                       GPU_rho_real_space, 
                                                       Nx, 
                                                       Ny, 
                                                       Nz, 
                                                       GPU_green_hat);
    cudaThreadSynchronize();

    // FFT
    cufftExecC2C(plan, GPU_E_x, GPU_E_x, CUFFT_INVERSE);
    cufftExecC2C(plan, GPU_E_y, GPU_E_y, CUFFT_INVERSE);
    cufftExecC2C(plan, GPU_E_z, GPU_E_z, CUFFT_INVERSE);
    cudaThreadSynchronize();

    set_gpu_field_kernel <<< N_grid, N_threads >>> (GPU_E_x, GPU_E_y, GPU_E_z, E_field, Nx, Ny, Nz);
    cudaThreadSynchronize();

    //calculate forces on particles, one thread per particles
    calculate_forces_kernel <<< P_grid, P_threads >>>(d_force,
                                                      d_virial, 
                                                      pdata, 
                                                      box, 
                                                      E_field, 
                                                      Nx, 
                                                      Ny, 
                                                      Nz, 
                                                      order,
                                                      d_group_members,
                                                      group_size);

    return cudaSuccess;
        }

__global__ void calculate_thermo_quantities_kernel(cufftComplex* rho, 
                                                   float* green_function, 
                                                   float2* GPU_virial_energy, 
                                                   float3* vg, 
                                                   int Nx, 
                                                   int Ny, 
                                                   int Nz)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(idx < Nx * Ny * Nz)
        {

        float energy = green_function[idx]*(rho[idx].x*rho[idx].x + rho[idx].y*rho[idx].y);
        float pressure = energy*(vg[idx].x + vg[idx].y + vg[idx].z);
        GPU_virial_energy[idx].x = pressure;
        GPU_virial_energy[idx].y = energy;
        }
    }

bool isPow2(unsigned int x)
    {
    return ((x&(x-1))==0);
    }

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
    }

template<class T>
struct SharedMemory  //!< Used to speed up the sum over grid points, in this case "T" is a placeholder for the data type
    {
        //!< used to get shared memory for data type T*
        __device__ inline operator       T*() 
            {
            extern __shared__ T __smem[];
            return (T*)__smem;
            }

    
        __device__ inline operator const T() const //!< used to get shared memory for data type T
            {
            extern __shared__ T __smem[];
            return (T*)__smem;
            }
    };

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
    {
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int idx = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum;
    mySum.x = 0.0f;
    mySum.y = 0.0f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
        {         
        mySum.x += g_idata[i].x;
        mySum.y += g_idata[i].y;
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum.x += g_idata[i+blockSize].x;  
            mySum.y += g_idata[i+blockSize].y; 
            }
        i += gridSize;

        } 

    // each thread puts its local sum into shared memory 
    sdata[idx].x = mySum.x;
    sdata[idx].y = mySum.y;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (idx < 256) { sdata[idx].x = mySum.x = mySum.x + sdata[idx + 256].x; sdata[idx].y = mySum.y = mySum.y + sdata[idx + 256].y; } __syncthreads(); }
    if (blockSize >= 256) { if (idx < 128) { sdata[idx].x = mySum.x = mySum.x + sdata[idx + 128].x; sdata[idx].y = mySum.y = mySum.y + sdata[idx + 128].y; } __syncthreads(); }
    if (blockSize >= 128) { if (idx <  64) { sdata[idx].x = mySum.x = mySum.x + sdata[idx +  64].x; sdata[idx].y = mySum.y = mySum.y + sdata[idx +  64].y; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (idx < 32)
#endif
        {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[idx].x = mySum.x = mySum.x + smem[idx + 32].x; smem[idx].y = mySum.y = mySum.y + smem[idx + 32].y; EMUSYNC; }
        if (blockSize >=  32) { smem[idx].x = mySum.x = mySum.x + smem[idx + 16].x; smem[idx].y = mySum.y = mySum.y + smem[idx + 16].y; EMUSYNC; }
        if (blockSize >=  16) { smem[idx].x = mySum.x = mySum.x + smem[idx +  8].x; smem[idx].y = mySum.y = mySum.y + smem[idx +  8].y; EMUSYNC; }
        if (blockSize >=   8) { smem[idx].x = mySum.x = mySum.x + smem[idx +  4].x; smem[idx].y = mySum.y = mySum.y + smem[idx +  4].y; EMUSYNC; }
        if (blockSize >=   4) { smem[idx].x = mySum.x = mySum.x + smem[idx +  2].x; smem[idx].y = mySum.y = mySum.y + smem[idx +  2].y; EMUSYNC; }
        if (blockSize >=   2) { smem[idx].x = mySum.x = mySum.x + smem[idx +  1].x; smem[idx].y = mySum.y = mySum.y + smem[idx +  1].y; EMUSYNC; }
        }
    
    // write result for this block to global mem 
    if (idx == 0) {
        g_odata[blockIdx.x].x = sdata[0].x;
        g_odata[blockIdx.x].y = sdata[0].y;
        }
    }


template <class T>
void 
reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
    {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
        {
        switch (threads)
            {
            case 512:
                reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
    else
        {
        switch (threads)
            {
            case 512:
                reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 256:
                reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 128:
                reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 64:
                reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 32:
                reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case 16:
                reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  8:
                reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  4:
                reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  2:
                reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            case  1:
                reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
            }
        }
    }

float2 gpu_compute_pppm_thermo(int Nx,
                               int Ny,
                               int Nz,
                               cufftComplex *GPU_rho_real_space,
                               float3 *GPU_vg,
                               float *GPU_green_hat,
                               float2 *o_data,
                               float2 *i_data,
                               int block_size)

    {

    dim3 N_grid( (int)ceil((double)Nx*Ny*Nz / (double)block_size), 1, 1);
    dim3 N_threads(block_size, 1, 1);

    int n = Nx*Ny*Nz;
    float2 gpu_result = make_float2(0.0f, 0.0f);
    calculate_thermo_quantities_kernel <<< N_grid, N_threads >>> (GPU_rho_real_space, GPU_green_hat, i_data, GPU_vg, Nx, Ny, Nz);
    int threads, blocks, maxBlocks = 64, maxThreads = 256, cpuFinalThreshold = 1;
    bool needReadBack = true;
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    blocks = MIN(maxBlocks, blocks);
    if (blocks == 1) cpuFinalThreshold = 1;

    int maxNumBlocks = MIN( n / maxThreads, MAX_BLOCK_DIM_SIZE);

    reduce<float2>(n, threads, blocks, i_data, o_data);

    // sum partial block sums on GPU
    int s=blocks;
    while(s > cpuFinalThreshold) 
        {
        threads = 0;
        blocks = 0;
        threads = (s < maxThreads*2) ? nextPow2((s + 1)/ 2) : maxThreads;
        blocks = (s + (threads * 2 - 1)) / (threads * 2);
        blocks = MIN(maxBlocks, blocks);
        reduce<float2>(s, threads, blocks, o_data, o_data);
        cudaThreadSynchronize();
        s = (s + (threads*2-1)) / (threads*2);
        }
            
    if (s > 1)
        {
        // copy result from device to host
        float2* h_odata = (float2 *) malloc(maxNumBlocks*sizeof(float2));
        cudaMemcpy( h_odata, o_data, s * sizeof(float2), cudaMemcpyDeviceToHost);


        for(int i=0; i < s; i++) 
            {
            gpu_result.x += h_odata[i].x;
            gpu_result.y += h_odata[i].y;
            }
        needReadBack = false;
        free(h_odata);
        }

    //copy to CPU:
    if (needReadBack) cudaMemcpy( &gpu_result,  o_data, sizeof(float2), cudaMemcpyDeviceToHost);

    return gpu_result;
    }




__global__ void reset_kvec_green_hat_kernel(gpu_boxsize box, 
                                            int Nx, 
                                            int Ny, 
                                            int Nz, 
                                            int order, 
                                            float kappa, 
                                            float3* kvec_array, 
                                            float* green_hat, 
                                            float3* vg, 
                                            int nbx, 
                                            int nby, 
                                            int nbz, 
                                            float* gf_b)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < Nx*Ny*Nz) {

        int N2 = Ny*Nz;

        int xn = idx/N2;
        int yn = (idx - xn*N2)/Nz;
        int zn = (idx - xn*N2 - yn*Nz);

        float invdet = 6.28318531f/(box.Lx*box.Ly*box.Lz);
        float3 inverse_lattice_vector, j;
        float kappa2 = kappa*kappa;

        inverse_lattice_vector.x = invdet*box.Ly*box.Lz;
        inverse_lattice_vector.y = invdet*box.Lx*box.Lz;
        inverse_lattice_vector.z = invdet*box.Lx*box.Ly;

        j.x = xn > Nx/2 ? (float)(xn - Nx) : (float)xn;
        j.y = yn > Ny/2 ? (float)(yn - Ny) : (float)yn;
        j.z = zn > Nz/2 ? (float)(zn - Nz) : (float)zn;
        kvec_array[idx].x = j.x*inverse_lattice_vector.x;
        kvec_array[idx].y = j.y*inverse_lattice_vector.y;
        kvec_array[idx].z = j.z*inverse_lattice_vector.z;

        float sqk =  kvec_array[idx].x*kvec_array[idx].x + kvec_array[idx].y*kvec_array[idx].y + kvec_array[idx].z*kvec_array[idx].z;
        if(sqk == 0.0f) {
            vg[idx].x = 0.0f;
            vg[idx].y = 0.0f;
            vg[idx].z = 0.0f;
            }
        else {
            float vterm = (-2.0f/sqk - 0.5f/kappa2);
            vg[idx].x = 1.0f+vterm*kvec_array[idx].x*kvec_array[idx].x;
            vg[idx].y = 1.0f+vterm*kvec_array[idx].y*kvec_array[idx].y;
            vg[idx].z = 1.0f+vterm*kvec_array[idx].z*kvec_array[idx].z;
            }

        float unitkx = (6.28318531f/box.Lx);
        float unitky = (6.28318531f/box.Ly);
        float unitkz = (6.28318531f/box.Lz);
        int ix, iy, iz, kper, lper, mper;
        float snx, sny, snz, snx2, sny2, snz2;
        float argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
        float sum1, dot1, dot2;
        float numerator, denominator;

        mper = zn - Nz*(2*zn/Nz);
        snz = sinf(0.5f*unitkz*mper*box.Lz/Nz);
        snz2 = snz*snz;

        lper = yn - Ny*(2*yn/Ny);
        sny = sinf(0.5f*unitky*lper*box.Ly/Ny);
        sny2 = sny*sny;

        kper = xn - Nx*(2*xn/Nx);
        snx = sinf(0.5f*unitkx*kper*box.Lx/Nx);
        snx2 = snx*snx;
        sqk = unitkx*kper*unitkx*kper + unitky*lper*unitky*lper + unitkz*mper*unitkz*mper;


        int l;
        sz = sy = sx = 0.0f;
        for (l = order-1; l >= 0; l--) {
            sx = gf_b[l] + sx*snx2;
            sy = gf_b[l] + sy*sny2;
            sz = gf_b[l] + sz*snz2;
            }
        denominator = sx*sy*sz;
        denominator *= denominator;

        float W;
        if (sqk != 0.0f) {
            numerator = 12.5663706f/sqk;
            sum1 = 0.0f;
            for (ix = -nbx; ix <= nbx; ix++) {
                qx = unitkx*(kper+(float)(Nx*ix));
                sx = expf(-.25f*qx*qx/kappa2);
                wx = 1.0f;
                argx = 0.5f*qx*box.Lx/(float)Nx;
                if (argx != 0.0f) wx = powf(sinf(argx)/argx,order);
                for (iy = -nby; iy <= nby; iy++) {
                    qy = unitky*(lper+(float)(Ny*iy));
                    sy = expf(-.25f*qy*qy/kappa2);
                    wy = 1.0f;
                    argy = 0.5f*qy*box.Ly/(float)Ny;
                    if (argy != 0.0f) wy = powf(sinf(argy)/argy,order);
                    for (iz = -nbz; iz <= nbz; iz++) {
                        qz = unitkz*(mper+(float)(Nz*iz));
                        sz = expf(-.25f*qz*qz/kappa2);
                        wz = 1.0f;
                        argz = 0.5f*qz*box.Lz/(float)Nz;
                        if (argz != 0.0f) wz = powf(sinf(argz)/argz,order);

                        dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                        dot2 = qx*qx+qy*qy+qz*qz;
                        W = wx*wy*wz;
                        sum1 += (dot1/dot2) * sx*sy*sz * W*W;
                        }
                    }
                }
            green_hat[idx] = numerator*sum1/denominator;
            } else green_hat[idx] = 0.0f;
        }
    }

cudaError_t reset_kvec_green_hat(const gpu_boxsize &box,
                                 int Nx,
                                 int Ny,
                                 int Nz,
                                 int nbx,
                                 int nby,
                                 int nbz,
                                 int order,
                                 float kappa,
                                 float3 *kvec,
                                 float *green_hat,
                                 float3 *vg,
                                 float *gf_b,
                                 int block_size)
    {
    dim3 grid( (int)ceil((double)Nx*Ny*Nz / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    reset_kvec_green_hat_kernel <<< grid, threads >>> (box, Nx, Ny, Nz, order, kappa, kvec, green_hat, vg, nbx, nby, nbz, gf_b);
    return cudaSuccess;
    }


__global__ void gpu_fix_exclusions_kernel(float4 *d_force,
                                          float *d_virial,
                                          const gpu_pdata_arrays pdata,
                                          const gpu_boxsize box,
                                          const unsigned int *d_n_neigh,
                                          const unsigned int *d_nlist,
                                          const Index2D nli,
                                          float kappa,  
                                          unsigned int *d_group_members,
                                          unsigned int group_size)
    {
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        const float sqrtpi = sqrtf(M_PI);
        unsigned int n_neigh = d_n_neigh[idx];
        float4 posi = tex1Dfetch(pdata_pos_tex, idx);
        float  qi = tex1Dfetch(pdata_charge_tex, idx);
        // initialize the force to 0
        float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float virial = 0.0f;
        unsigned int cur_j = 0;
        // prefetch neighbor index
        unsigned int next_j = d_nlist[nli(idx, 0)];

#if (__CUDA_ARCH__ < 200)
        for (int neigh_idx = 0; neigh_idx < nli.getH(); neigh_idx++)
#else
            for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
#endif
                {
#if (__CUDA_ARCH__ < 200)
                if (neigh_idx < n_neigh)
#endif
                    {
                    // read the current neighbor index (MEM TRANSFER: 4 bytes)
                    // prefetch the next value and set the current one
                    cur_j = next_j;
                    next_j = d_nlist[nli(idx, neigh_idx+1)];
            
                    // get the neighbor's position (MEM TRANSFER: 16 bytes)
                    float4 posj = tex1Dfetch(pdata_pos_tex, cur_j);
            
                    float qj = tex1Dfetch(pdata_charge_tex, cur_j);
                
                    // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                    float dx = posi.x - posj.x;
                    float dy = posi.y - posj.y;
                    float dz = posi.z - posj.z;
            
                    // apply periodic boundary conditions: (FLOPS 12)
                    dx -= box.Lx * rintf(dx * box.Lxinv);
                    dy -= box.Ly * rintf(dy * box.Lyinv);
                    dz -= box.Lz * rintf(dz * box.Lzinv);
            
                    // calculate r squard (FLOPS: 5)
                    float rsq = dx*dx + dy*dy + dz*dz;
                    float r = sqrtf(rsq);
                    float qiqj = qi * qj;
                    float erffac = erf(kappa * r) / r;
                    float force_divr = qiqj * (-2.0f * exp(-rsq * kappa * kappa) * kappa / (sqrtpi * rsq) + erffac / rsq);
                    float pair_eng = qiqj * erffac; 

                    virial += float(1.0/6.0) * rsq * force_divr;
#if (__CUDA_ARCH__ >= 200)
                    force.x += dx * force_divr;
                    force.y += dy * force_divr;
                    force.z += dz * force_divr;
#else
                    // fmad causes momentum drift here, prevent it from being used
                    force.x += __fmul_rn(dx, force_divr);
                    force.y += __fmul_rn(dy, force_divr);
                    force.z += __fmul_rn(dz, force_divr);
#endif
            
                    force.w += pair_eng;
                    }
                }
        force.w *= 0.5f;
        d_force[idx].x -= force.x;
        d_force[idx].y -= force.y;
        d_force[idx].z -= force.z;
        d_force[idx].w -= force.w;
        d_virial[idx] = -virial;
        }
    }


cudaError_t fix_exclusions(float4 *d_force,
                           float *d_virial,
                           const gpu_pdata_arrays &pdata,
                           const gpu_boxsize &box,
                           const unsigned int *d_n_ex,
                           const unsigned int *d_exlist,
                           const Index2D nex,
                           float kappa,
                           unsigned int *d_group_members,
                           unsigned int group_size,
                           int block_size)
    {
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    

    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_charge_tex, pdata.charge, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;

    gpu_fix_exclusions_kernel <<< grid, threads >>>  (d_force,
                                                      d_virial,
                                                      pdata,
                                                      box,
                                                      d_n_ex,
                                                      d_exlist,
                                                      nex,
                                                      kappa, 
                                                      d_group_members,
                                                      group_size);
    return cudaSuccess;
    }

