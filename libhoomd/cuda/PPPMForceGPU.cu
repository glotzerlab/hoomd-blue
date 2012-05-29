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

#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define MAX_BLOCK_DIM_SIZE 65535

// Constant memory for gridpoint weighting
#define CONSTANT_SIZE 2048
//! The developer has chosen not to document this variable
__device__ __constant__ Scalar GPU_rho_coeff[CONSTANT_SIZE];

/*! \file PPPMForceGPU.cu
  \brief Defines GPU kernel code for calculating the Fourier space forces for the Coulomb interaction. Used by PPPMForceComputeGPU.
*/

//! Texture for reading particle positions
texture<Scalar4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Texture for reading charge parameters
texture<Scalar, 1, cudaReadModeElementType> pdata_charge_tex;

//! Implements workaround atomic Scalar addition on sm_1x hardware
__device__ inline void atomicFloatAdd(Scalar* address, Scalar value)
    {
#if (__CUDA_ARCH__ < 200)
    Scalar old = value;
    Scalar new_old;
    do
        {
        new_old = atomicExch(address, Scalar(0.0));
        new_old += old;
        }
    while ((old = atomicExch(address, new_old))!=Scalar(0.0));
#else
    atomicAdd(address, value);
#endif
    }

//! The developer has chosen not to document this function
__device__ inline void AddToGridpoint(int X, int Y, int Z, cufftComplex* array, Scalar value, int Ny, int Nz)
    {
    atomicFloatAdd(&array[Z + Nz * (Y + Ny * X)].x, value);
    }


//! The developer has chosen not to document this function
extern "C" __global__
void assign_charges_to_grid_kernel(const unsigned int N,
                                   const Scalar4 *d_pos,
                                   const Scalar *d_charge,
                                   BoxDim box, 
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
        Scalar qi = tex1Dfetch(pdata_charge_tex, idx);

        if(fabs(qi) > Scalar(0.0)) {
            Scalar4 posi = tex1Dfetch(pdata_pos_tex, idx);
            //calculate dx, dy, dz for the charge density grid:
            Scalar3 L = box.getL();
            Scalar box_dx = L.x / ((Scalar)Nx);
            Scalar box_dy = L.y / ((Scalar)Ny);
            Scalar box_dz = L.z / ((Scalar)Nz);
    
        
            //normalize position to gridsize:
            posi.x += L.x / Scalar(2.0);
            posi.y += L.y / Scalar(2.0);
            posi.z += L.z / Scalar(2.0);
   
            posi.x /= box_dx;
            posi.y /= box_dy;
            posi.z /= box_dz;
    
    
            Scalar shift, shiftone, x0, y0, z0, dx, dy, dz;
            int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
            nlower = -(order-1)/2;
            nupper = order/2;
    
            if (order % 2) 
                {
                shift =Scalar(0.5);
                shiftone = Scalar(0.0);
                }
            else 
                {
                shift = Scalar(0.0);
                shiftone = Scalar(0.5);
                }
        
            nxi = __scalar2int_rd(posi.x + shift);
            nyi = __scalar2int_rd(posi.y + shift);
            nzi = __scalar2int_rd(posi.z + shift);
    
            dx = shiftone+(Scalar)nxi-posi.x;
            dy = shiftone+(Scalar)nyi-posi.y;
            dz = shiftone+(Scalar)nzi-posi.z;
    
            int n,m,l,k;
            Scalar result;
            int mult_fact = 2*order+1;

            x0 = qi / (box_dx*box_dy*box_dz);
            for (n = nlower; n <= nupper; n++) {
                mx = n+nxi;
                if(mx >= Nx) mx -= Nx;
                if(mx < 0)  mx += Nx;
                result = Scalar(0.0);
                for (k = order-1; k >= 0; k--) {
                    result = GPU_rho_coeff[n-nlower + k*mult_fact] + result * dx;
                    }
                y0 = x0*result;
                for (m = nlower; m <= nupper; m++) {
                    my = m+nyi;
                    if(my >= Ny) my -= Ny;
                    if(my < 0)  my += Ny;
                    result = Scalar(0.0);
                    for (k = order-1; k >= 0; k--) {
                        result = GPU_rho_coeff[m-nlower + k*mult_fact] + result * dy;
                        }
                    z0 = y0*result;
                    for (l = nlower; l <= nupper; l++) {
                        mz = l+nzi;
                        if(mz >= Nz) mz -= Nz;
                        if(mz < 0)  mz += Nz;
                        result = Scalar(0.0);
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

//! The developer has chosen not to document this function
extern "C" __global__
void combined_green_e_kernel(cufftComplex* E_x, 
                             cufftComplex* E_y, 
                             cufftComplex* E_z, 
                             Scalar3* k_vec, 
                             cufftComplex* rho, 
                             int Nx, 
                             int Ny, 
                             int Nz, 
                             Scalar* green_function)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(idx < Nx * Ny * Nz)
        {
        Scalar3 k_vec_local = k_vec[idx];
        cufftComplex E_x_local, E_y_local, E_z_local;
        Scalar scale_times_green = green_function[idx] / ((Scalar)(Nx*Ny*Nz));
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


//! The developer has chosen not to document this function
__global__ void set_gpu_field_kernel(cufftComplex* E_x, 
                                     cufftComplex* E_y, 
                                     cufftComplex* E_z, 
                                     Scalar3* Electric_field, 
                                     int Nx, 
                                     int Ny, 
                                     int Nz)
    {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < Nx * Ny * Nz)
        {
        Scalar3 local_field;
        local_field.x = E_x[tid].x;
        local_field.y = E_y[tid].x;
        local_field.z = E_z[tid].x;
      
        Electric_field[tid] = local_field;
        }
    }

//! The developer has chosen not to document this function
__global__
void zero_forces(Scalar4 *d_force, Scalar *d_virial, const unsigned int virial_pitch, const unsigned int N)
    {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        d_force[idx] = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        for (unsigned int i = 0; i < 6; i++)
            d_virial[i*virial_pitch+idx] = Scalar(0.0);
        }
    }

//! The developer has chosen not to document this function
extern "C" __global__ 
void calculate_forces_kernel(Scalar4 *d_force,
                             const unsigned int N,
                             const Scalar4 *d_pos,
                             const Scalar *d_charge,
                             BoxDim box,
                             Scalar3 *E_field,
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
        Scalar qi = tex1Dfetch(pdata_charge_tex, idx);
        if(fabs(qi) > Scalar(0.0)) {
            Scalar4 posi = tex1Dfetch(pdata_pos_tex, idx);
    
            //calculate dx, dy, dz for the charge density grid:
            Scalar3 L = box.getL();
            Scalar box_dx = L.x / ((Scalar)Nx);
            Scalar box_dy = L.y / ((Scalar)Ny);
            Scalar box_dz = L.z / ((Scalar)Nz);
    
            //normalize position to gridsize:
            posi.x += L.x * Scalar(0.5);
            posi.y += L.y * Scalar(0.5);
            posi.z += L.z * Scalar(0.5);
   
            posi.x /= box_dx;
            posi.y /= box_dy;
            posi.z /= box_dz;
    
            Scalar shift, shiftone, x0, y0, z0, dx, dy, dz;
            int nlower, nupper, mx, my, mz, nxi, nyi, nzi; 
    
            nlower = -(order-1)/2;
            nupper = order/2;
    
            Scalar4 local_force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
            if(order % 2) 
                {
                shift =Scalar(0.5);
                shiftone = Scalar(0.0);
                }
            else 
                {
                shift = Scalar(0.0);
                shiftone = Scalar(0.5);
                }
    
    
            nxi = __scalar2int_rd(posi.x + shift);
            nyi = __scalar2int_rd(posi.y + shift);
            nzi = __scalar2int_rd(posi.z + shift);
    
            dx = shiftone+(Scalar)nxi-posi.x;
            dy = shiftone+(Scalar)nyi-posi.y;
            dz = shiftone+(Scalar)nzi-posi.z;

            int n,m,l,k;
            Scalar result;
            int mult_fact = 2*order+1;
    
            for (n = nlower; n <= nupper; n++) {
                mx = n+nxi;
                if(mx >= Nx) mx -= Nx;
                if(mx < 0)  mx += Nx;
                result = Scalar(0.0);
                for (k = order-1; k >= 0; k--) {
                    result = GPU_rho_coeff[n-nlower + k*mult_fact] + result * dx;
                    }
                x0 = result;
                for (m = nlower; m <= nupper; m++) {
                    my = m+nyi;
                    if(my >= Ny) my -= Ny;
                    if(my < 0)  my += Ny;
                    result = Scalar(0.0);
                    for (k = order-1; k >= 0; k--) {
                        result = GPU_rho_coeff[m-nlower + k*mult_fact] + result * dy;
                        }
                    y0 = x0*result;
                    for (l = nlower; l <= nupper; l++) {
                        mz = l+nzi;
                        if(mz >= Nz) mz -= Nz;
                        if(mz < 0)  mz += Nz;
                        result = Scalar(0.0);
                        for (k = order-1; k >= 0; k--) {
                            result = GPU_rho_coeff[l-nlower + k*mult_fact] + result * dz;
                            }
                        z0 = y0*result;
                        Scalar local_field_x = E_field[mz + Nz * (my + Ny * mx)].x;
                        Scalar local_field_y = E_field[mz + Nz * (my + Ny * mx)].y;
                        Scalar local_field_z = E_field[mz + Nz * (my + Ny * mx)].z;
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


cudaError_t gpu_compute_pppm_forces(Scalar4 *d_force,
                                    const unsigned int N,
                                    const Scalar4 *d_pos,
                                    const Scalar *d_charge,
                                    const BoxDim& box,
                                    int Nx,
                                    int Ny,
                                    int Nz,
                                    int order,
                                    Scalar *CPU_rho_coeff,
                                    cufftComplex *GPU_rho_real_space,
                                    cufftHandle plan,
                                    cufftComplex *GPU_E_x,
                                    cufftComplex *GPU_E_y,
                                    cufftComplex *GPU_E_z,
                                    Scalar3 *GPU_k_vec,
                                    Scalar *GPU_green_hat,
                                    Scalar3 *E_field,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    int block_size)
    {
    
    cudaMemcpyToSymbol(GPU_rho_coeff, &(CPU_rho_coeff[0]), order * (2*order+1) * sizeof(Scalar));

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
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, pdata_charge_tex, d_charge, sizeof(Scalar) * N);
    if (error != cudaSuccess)
        return error;


    // set the grid charge to zero
    cudaMemset(GPU_rho_real_space, 0, sizeof(cufftComplex)*Nx*Ny*Nz);

    // run the kernels
    // assign charges to the grid points, one thread per particles
    assign_charges_to_grid_kernel <<< P_grid, P_threads >>> (N,
                                                             d_pos,
                                                             d_charge,
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
                                                      N,
                                                      d_pos,
                                                      d_charge,
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

//! The developer has chosen not to document this function
__global__ void calculate_thermo_quantities_kernel(cufftComplex* rho, 
                                                   Scalar* green_function, 
                                                   Scalar* energy_sum,
                                                   Scalar* v_xx,
                                                   Scalar* v_xy,
                                                   Scalar* v_xz,
                                                   Scalar* v_yy,
                                                   Scalar* v_yz,
                                                   Scalar* v_zz,
                                                   Scalar* vg,
                                                   int Nx, 
                                                   int Ny, 
                                                   int Nz)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(idx < Nx * Ny * Nz)
        {

        Scalar energy = green_function[idx]*(rho[idx].x*rho[idx].x + rho[idx].y*rho[idx].y);
        v_xx[idx] = energy*vg[  6*idx];
        v_xy[idx] = energy*vg[1+6*idx];
        v_xz[idx] = energy*vg[2+6*idx];
        v_yy[idx] = energy*vg[3+6*idx];
        v_yz[idx] = energy*vg[4+6*idx];
        v_zz[idx] = energy*vg[5+6*idx];
        energy_sum[idx] = energy;
        }
    }

//! The developer has chosen not to document this function
bool isPow2(unsigned int x)
    {
    return ((x&(x-1))==0);
    }

//! The developer has chosen not to document this function
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
        //! used to get shared memory for data type T*
        __device__ inline operator       T*() 
            {
            extern __shared__ T __smem[];
            return (T*)__smem;
            }

        //! used to get shared memory for data type T
        __device__ inline operator const T() const 
            {
            extern __shared__ T __smem[];
            return (T*)__smem;
            }
    };

//! The developer has chosen not to document this function
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
    mySum = Scalar(0.0);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
        {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum += g_idata[i+blockSize];  
            }
        i += gridSize;

        } 

    // each thread puts its local sum into shared memory 
    sdata[idx] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (idx < 256) { sdata[idx] = mySum = mySum + sdata[idx + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (idx < 128) { sdata[idx] = mySum = mySum + sdata[idx + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (idx <  64) { sdata[idx] = mySum = mySum + sdata[idx + 64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (idx < 32)
#endif
        {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[idx] = mySum = mySum + smem[idx + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[idx] = mySum = mySum + smem[idx + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[idx] = mySum = mySum + smem[idx + 8]; EMUSYNC; }
        if (blockSize >=   8) { smem[idx] = mySum = mySum + smem[idx + 4]; EMUSYNC; }
        if (blockSize >=   4) { smem[idx] = mySum = mySum + smem[idx + 2]; EMUSYNC; }
        if (blockSize >=   2) { smem[idx] = mySum = mySum + smem[idx + 1]; EMUSYNC; }
        }
    
    // write result for this block to global mem 
    if (idx == 0) {
        g_odata[blockIdx.x] = sdata[0];
        }
    }


//! The developer has chosen not to document this function
template <class T> void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
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



//! The developer has chosen not to document this function
void gpu_compute_pppm_thermo(int Nx,
                             int Ny,
                             int Nz,
                             cufftComplex *GPU_rho_real_space,
                             Scalar *GPU_vg,
                             Scalar *GPU_green_hat,
                             Scalar *o_data,
                             Scalar *energy_sum, 
                             Scalar *v_xx,
                             Scalar *v_xy,
                             Scalar *v_xz,
                             Scalar *v_yy,
                             Scalar *v_yz,
                             Scalar *v_zz,
                             Scalar *pppm_virial_energy,
                             int block_size)

    {

    // setup the grid to run the kernel with one thread per grid point
    dim3 N_grid( (int)ceil((double)Nx*Ny*Nz / (double)block_size), 1, 1);
    dim3 N_threads(block_size, 1, 1);

    calculate_thermo_quantities_kernel <<< N_grid, N_threads >>> (GPU_rho_real_space,
                                                                  GPU_green_hat,
                                                                  energy_sum,
                                                                  v_xx,
                                                                  v_xy,
                                                                  v_xz,
                                                                  v_yy,
                                                                  v_yz,
                                                                  v_zz,
                                                                  GPU_vg, 
                                                                  Nx, 
                                                                  Ny, 
                                                                  Nz);


    cudaThreadSynchronize();

    int n = Nx*Ny*Nz;
    pppm_virial_energy[0] = Scalar_reduce(energy_sum, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[1] = Scalar_reduce(v_xx, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[2] = Scalar_reduce(v_xy, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[3] = Scalar_reduce(v_xz, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[4] = Scalar_reduce(v_yy, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[5] = Scalar_reduce(v_yz, o_data, n);

    cudaMemset(o_data, 0, sizeof(Scalar)*Nx*Ny*Nz);
    pppm_virial_energy[6] = Scalar_reduce(v_zz, o_data, n);

}

//! The developer has chosen not to document this function
Scalar Scalar_reduce(Scalar* i_data, Scalar* o_data, int n) {

    Scalar gpu_result = 0.0;
    int threads, blocks, maxBlocks = 64, maxThreads = 256, cpuFinalThreshold = 1;
    bool needReadBack = true;
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    blocks = MIN(maxBlocks, blocks);
    if (blocks == 1) cpuFinalThreshold = 1;

    int maxNumBlocks = MIN( n / maxThreads, MAX_BLOCK_DIM_SIZE);


    reduce<Scalar>(n, threads, blocks, i_data, o_data);

    int s=blocks;
    while(s > cpuFinalThreshold)
        {
        threads = 0;
        blocks = 0;
        threads = (s < maxThreads*2) ? nextPow2((s + 1)/ 2) : maxThreads;
        blocks = (s + (threads * 2 - 1)) / (threads * 2);
        blocks = MIN(maxBlocks, blocks);
        reduce<Scalar>(s, threads, blocks, o_data, o_data);
        cudaThreadSynchronize();
        s = (s + (threads*2-1)) / (threads*2);
        }

    if (s > 1)
        {
        Scalar* h_odata = (Scalar *) malloc(maxNumBlocks*sizeof(Scalar));
        cudaMemcpy( h_odata, o_data, s * sizeof(Scalar), cudaMemcpyDeviceToHost);


        for(int i=0; i < s; i++)
            {
            gpu_result += h_odata[i];
            }
        needReadBack = false;
        free(h_odata);
        }

    if (needReadBack) cudaMemcpy( &gpu_result,  o_data, sizeof(Scalar), cudaMemcpyDeviceToHost);

    return gpu_result;
}

//! The developer has chosen not to document this function
__global__ void reset_kvec_green_hat_kernel(BoxDim box, 
                                            int Nx, 
                                            int Ny, 
                                            int Nz, 
                                            int order, 
                                            Scalar kappa, 
                                            Scalar3* kvec_array, 
                                            Scalar* green_hat, 
                                            Scalar* vg, 
                                            int nbx, 
                                            int nby, 
                                            int nbz, 
                                            Scalar* gf_b)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < Nx*Ny*Nz) {

        int N2 = Ny*Nz;

        int xn = idx/N2;
        int yn = (idx - xn*N2)/Nz;
        int zn = (idx - xn*N2 - yn*Nz);

        Scalar3 L = box.getL();
        Scalar invdet = Scalar(6.28318531)/(L.x*L.y*L.z);
        Scalar3 inverse_lattice_vector, j;
        Scalar kappa2 = kappa*kappa;

        inverse_lattice_vector.x = invdet*L.y*L.z;
        inverse_lattice_vector.y = invdet*L.x*L.z;
        inverse_lattice_vector.z = invdet*L.x*L.y;

        j.x = xn > Nx/2 ? (Scalar)(xn - Nx) : (Scalar)xn;
        j.y = yn > Ny/2 ? (Scalar)(yn - Ny) : (Scalar)yn;
        j.z = zn > Nz/2 ? (Scalar)(zn - Nz) : (Scalar)zn;
        kvec_array[idx].x = j.x*inverse_lattice_vector.x;
        kvec_array[idx].y = j.y*inverse_lattice_vector.y;
        kvec_array[idx].z = j.z*inverse_lattice_vector.z;

        Scalar sqk =  kvec_array[idx].x*kvec_array[idx].x + kvec_array[idx].y*kvec_array[idx].y + kvec_array[idx].z*kvec_array[idx].z;
        if(sqk == Scalar(0.0)) {
            vg[0+6*idx] = Scalar(0.0);
            vg[1+6*idx] = Scalar(0.0);
            vg[2+6*idx] = Scalar(0.0);
            vg[3+6*idx] = Scalar(0.0);
            vg[4+6*idx] = Scalar(0.0);
            vg[5+6*idx] = Scalar(0.0);
            }
        else {
            Scalar vterm = (-Scalar(2.0)/sqk - Scalar(0.5)/kappa2);
            vg[0+6*idx] = Scalar(1.0)+vterm*kvec_array[idx].x*kvec_array[idx].x;
            vg[1+6*idx] =      vterm*kvec_array[idx].x*kvec_array[idx].y;
            vg[2+6*idx] =      vterm*kvec_array[idx].x*kvec_array[idx].z;
            vg[3+6*idx] = Scalar(1.0)+vterm*kvec_array[idx].y*kvec_array[idx].y;
            vg[4+6*idx] =      vterm*kvec_array[idx].y*kvec_array[idx].z;
            vg[5+6*idx] = Scalar(1.0)+vterm*kvec_array[idx].z*kvec_array[idx].z;
            }

        Scalar unitkx = (Scalar(6.28318531)/L.x);
        Scalar unitky = (Scalar(6.28318531)/L.y);
        Scalar unitkz = (Scalar(6.28318531)/L.z);
        int ix, iy, iz, kper, lper, mper;
        Scalar snx, sny, snz, snx2, sny2, snz2;
        Scalar argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
        Scalar sum1, dot1, dot2;
        Scalar numerator, denominator;

        mper = zn - Nz*(2*zn/Nz);
        snz = sinf(Scalar(0.5)*unitkz*mper*L.z/Nz);
        snz2 = snz*snz;

        lper = yn - Ny*(2*yn/Ny);
        sny = sinf(Scalar(0.5)*unitky*lper*L.y/Ny);
        sny2 = sny*sny;

        kper = xn - Nx*(2*xn/Nx);
        snx = sinf(Scalar(0.5)*unitkx*kper*L.x/Nx);
        snx2 = snx*snx;
        sqk = unitkx*kper*unitkx*kper + unitky*lper*unitky*lper + unitkz*mper*unitkz*mper;


        int l;
        sz = sy = sx = Scalar(0.0);
        for (l = order-1; l >= 0; l--) {
            sx = gf_b[l] + sx*snx2;
            sy = gf_b[l] + sy*sny2;
            sz = gf_b[l] + sz*snz2;
            }
        denominator = sx*sy*sz;
        denominator *= denominator;

        Scalar W;
        if (sqk != Scalar(0.0)) {
            numerator = Scalar(12.5663706)/sqk;
            sum1 = Scalar(0.0);
            for (ix = -nbx; ix <= nbx; ix++) {
                qx = unitkx*(kper+(Scalar)(Nx*ix));
                sx = expf(Scalar(-.25)*qx*qx/kappa2);
                wx = Scalar(1.0);
                argx = Scalar(0.5)*qx*L.x/(Scalar)Nx;
                if (argx != Scalar(0.0)) wx = powf(sinf(argx)/argx,order);
                for (iy = -nby; iy <= nby; iy++) {
                    qy = unitky*(lper+(Scalar)(Ny*iy));
                    sy = expf(Scalar(-.25)*qy*qy/kappa2);
                    wy = Scalar(1.0);
                    argy = Scalar(0.5)*qy*L.y/(Scalar)Ny;
                    if (argy != Scalar(0.0)) wy = powf(sinf(argy)/argy,order);
                    for (iz = -nbz; iz <= nbz; iz++) {
                        qz = unitkz*(mper+(Scalar)(Nz*iz));
                        sz = expf(Scalar(-.25)*qz*qz/kappa2);
                        wz = Scalar(1.0);
                        argz = Scalar(0.5)*qz*L.z/(Scalar)Nz;
                        if (argz != Scalar(0.0)) wz = powf(sinf(argz)/argz,order);

                        dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                        dot2 = qx*qx+qy*qy+qz*qz;
                        W = wx*wy*wz;
                        sum1 += (dot1/dot2) * sx*sy*sz * W*W;
                        }
                    }
                }
            green_hat[idx] = numerator*sum1/denominator;
            } else green_hat[idx] = Scalar(0.0);
        }
    }

//! The developer has chosen not to document this function
cudaError_t reset_kvec_green_hat(const BoxDim& box,
                                 int Nx,
                                 int Ny,
                                 int Nz,
                                 int nbx,
                                 int nby,
                                 int nbz,
                                 int order,
                                 Scalar kappa,
                                 Scalar3 *kvec,
                                 Scalar *green_hat,
                                 Scalar *vg,
                                 Scalar *gf_b,
                                 int block_size)
    {
    dim3 grid( (int)ceil((double)Nx*Ny*Nz / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    reset_kvec_green_hat_kernel <<< grid, threads >>> (box, Nx, Ny, Nz, order, kappa, kvec, green_hat, vg, nbx, nby, nbz, gf_b);
    return cudaSuccess;
    }


//! The developer has chosen not to document this function
__global__ void gpu_fix_exclusions_kernel(Scalar4 *d_force,
                                          Scalar *d_virial,
                                          const unsigned int virial_pitch,
                                          const unsigned int N,
                                          const Scalar4 *d_pos,
                                          const Scalar *d_charge,
                                          const BoxDim box,
                                          const unsigned int *d_n_neigh,
                                          const unsigned int *d_nlist,
                                          const Index2D nli,
                                          Scalar kappa,  
                                          unsigned int *d_group_members,
                                          unsigned int group_size)
    {
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        const Scalar sqrtpi = sqrtf(M_PI);
        unsigned int n_neigh = d_n_neigh[idx];
        Scalar4 postypei =  tex1Dfetch(pdata_pos_tex, idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        Scalar  qi = tex1Dfetch(pdata_charge_tex, idx);
        // initialize the force to 0
        Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        Scalar virial[6];
        for (unsigned int i = 0; i < 6; i++)
            virial[i] = Scalar(0.0);
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
                    Scalar4 postypej = tex1Dfetch(pdata_pos_tex, cur_j);
                    Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                    Scalar qj = tex1Dfetch(pdata_charge_tex, cur_j);

                    // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                    Scalar3 dx = posi - posj;

                    // apply periodic boundary conditions: (FLOPS 12)
                    dx = box.minImage(dx);

                    // calculate r squard (FLOPS: 5)
                    Scalar rsq = dot(dx,dx);
                    Scalar r = sqrtf(rsq);
                    Scalar qiqj = qi * qj;
                    Scalar erffac = erf(kappa * r) / r;
                    Scalar force_divr = qiqj * (-Scalar(2.0) * exp(-rsq * kappa * kappa) * kappa / (sqrtpi * rsq) + erffac / rsq);
                    Scalar pair_eng = qiqj * erffac; 

                    Scalar force_div2r = Scalar(0.5) * force_divr;
                    virial[0] += dx.x * dx.x * force_div2r;
                    virial[1] += dx.x * dx.y * force_div2r;
                    virial[2] += dx.x * dx.z * force_div2r;
                    virial[3] += dx.y * dx.y * force_div2r;
                    virial[4] += dx.y * dx.z * force_div2r;
                    virial[5] += dx.z * dx.z * force_div2r;

#if (__CUDA_ARCH__ >= 200)
                    force.x += dx.x * force_divr;
                    force.y += dx.y * force_divr;
                    force.z += dx.z * force_divr;
#else
                    // fmad causes momentum drift here, prevent it from being used
                    force.x += __fmul_rn(dx.x, force_divr);
                    force.y += __fmul_rn(dx.y, force_divr);
                    force.z += __fmul_rn(dx.z, force_divr);
#endif

                    force.w += pair_eng;
                    }
                }
        force.w *= Scalar(0.5);
        d_force[idx].x -= force.x;
        d_force[idx].y -= force.y;
        d_force[idx].z -= force.z;
        d_force[idx].w -= force.w;
        for (unsigned int i = 0; i < 6; i++)
            d_virial[i*virial_pitch+idx] = - virial[i];
        }
    }


//! The developer has chosen not to document this function
cudaError_t fix_exclusions(Scalar4 *d_force,
                           Scalar *d_virial,
                           const unsigned int virial_pitch,
                           const unsigned int N,
                           const Scalar4 *d_pos,
                           const Scalar *d_charge,
                           const BoxDim& box,
                           const unsigned int *d_n_ex,
                           const unsigned int *d_exlist,
                           const Index2D nex,
                           Scalar kappa,
                           unsigned int *d_group_members,
                           unsigned int group_size,
                           int block_size)
    {
    dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, pdata_charge_tex, d_charge, sizeof(Scalar) * N);
    if (error != cudaSuccess)
        return error;



    // zero the force arrays for all particles
    // zero_forces <<< grid, threads >>> (force_data, N);
    cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    cudaMemset(d_virial, 0, 6*sizeof(Scalar)*virial_pitch);

    gpu_fix_exclusions_kernel <<< grid, threads >>>  (d_force,
                                                      d_virial,
                                                      virial_pitch,
                                                      N,
                                                      d_pos,
                                                      d_charge,
                                                      box,
                                                      d_n_ex,
                                                      d_exlist,
                                                      nex,
                                                      kappa, 
                                                      d_group_members,
                                                      group_size);
    return cudaSuccess;
    }

