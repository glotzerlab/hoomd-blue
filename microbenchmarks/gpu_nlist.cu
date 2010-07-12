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

#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

using namespace std;

// safe call macros
#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)

#define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaThreadSynchronize();                                                \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


//*************** parameters of the benchmark
unsigned int g_N;
float g_Lx;
float g_Ly;
float g_Lz;
float g_rcut;
const unsigned int g_Nmax = 128;    // Maximum number of particles each cell can hold
const float tweak_dist = 0.1f;
const unsigned int g_neigh_max = 256;

//*************** data structures
float4 *gh_pos, *gd_pos;            // particle positions
unsigned int g_Mx;  // X-dimension of the cell grid
unsigned int g_My;  // Y-dimension of the cell grid
unsigned int g_Mz;  // Z-dimension of the cell grid
cudaArray *gd_idxlist_coord_array;  // \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
float4 *gh_idxlist_coord;   // \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
float4 *gd_idxlist_coord;   // \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
cudaArray *gd_idxlist_coord_trans_array;    // transposed \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
float4 *gh_idxlist_coord_trans; // transposed \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
float4 *gd_idxlist_coord_trans; // transposed \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell
unsigned int *gd_bin_size;  // number of particles in each bin
unsigned int *gh_bin_size;  // number of particles in each bin

unsigned int *gh_bin_adj;   // 27 x nbins adjacency array
unsigned int *gd_bin_adj;   // 27 x nbins adjacency array
cudaArray *gd_bin_adj_array;    // 27 x nbins adjancency array
unsigned int *gh_bin_adj_trans; // nbins x 27 adjacency array
unsigned int *gd_bin_adj_trans; // nbins x 27 adjacency array
cudaArray *gd_bin_adj_trans_array;  // nbins x 27 adjancency array

uint4 *gd_bin_coords;   // pre-calculated bin coordinates for each bin
uint4 *gh_bin_coords;   // pre-calculated bin coordinates for each bin

unsigned int g_nlist_pitch; // pitch of nlist array
unsigned int *gh_nlist;     // \a g_nlist_pitch x g_neigh_max array of the neighbor list
unsigned int *gh_n_neigh;   // \a g_N length array listing the number of neighbors

unsigned int *gh_nlist_ref;     // \a g_nlist_pitch x g_neigh_max array of the neighbor list (reference)
unsigned int *gh_n_neigh_ref;   // \a g_N length array listing the number of neighbors (reference)

unsigned int *gd_nlist;     // \a g_nlist_pitch x g_neigh_max array of the neighbor list
unsigned int *gd_n_neigh;   // \a g_N length array listing the number of neighbors

unsigned int g_actual_Nmax; // maximum number of particles in any given bin

//*************** functions for allocating and freeing the data structures
void allocate_data()
    {
    // allocate particle positions
    gh_pos = (float4 *)malloc(sizeof(float4) * g_N);
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_pos, sizeof(float4) * g_N));
    
    // determine grid dimensions
    g_Mx = int((g_Lx) / (g_rcut));
    g_My = int((g_Ly) / (g_rcut));
    g_Mz = int((g_Lz) / (g_rcut));
    
    // allocate bins
    unsigned int Nbins = g_Mx * g_My * g_Mz;
    gh_idxlist_coord = (float4 *)malloc(Nbins * g_Nmax * sizeof(float4));
    gh_idxlist_coord_trans = (float4 *)malloc(Nbins * g_Nmax * sizeof(float4));
    cudaChannelFormatDesc channelDescFloat4 = cudaCreateChannelDesc<float4>();
    CUDA_SAFE_CALL(cudaMallocArray(&gd_idxlist_coord_trans_array, &channelDescFloat4,  Nbins, g_Nmax));
    CUDA_SAFE_CALL(cudaMallocArray(&gd_idxlist_coord_array, &channelDescFloat4, g_Nmax, Nbins));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_idxlist_coord, Nbins * g_Nmax * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_idxlist_coord_trans, Nbins * g_Nmax * sizeof(float4)));
    
    gh_bin_size = (unsigned int *)malloc(Nbins * sizeof(unsigned int));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_size, Nbins * g_Nmax * sizeof(unsigned int)));
    
    // allocate adjancency arrays
    gh_bin_adj = (unsigned int *)malloc(Nbins * 27 * sizeof(unsigned int));
    gh_bin_adj_trans = (unsigned int *)malloc(Nbins * 27 * sizeof(unsigned int));
    cudaChannelFormatDesc channelDescUint = cudaCreateChannelDesc<unsigned int>();
    CUDA_SAFE_CALL(cudaMallocArray(&gd_bin_adj_array, &channelDescUint,  27, Nbins));
    CUDA_SAFE_CALL(cudaMallocArray(&gd_bin_adj_trans_array, &channelDescUint, Nbins, 27));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_adj, Nbins * 27 * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_adj_trans, Nbins * 27 * sizeof(unsigned int)));
    
    gh_bin_coords = (uint4*)malloc(Nbins*sizeof(uint4));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_coords, Nbins*sizeof(uint4)));
    
    // round nlist pitch up to the nearest multiple of 16
    g_nlist_pitch = (g_N + (16 - g_N & 15));
    gh_nlist = (unsigned int *)malloc(g_nlist_pitch * g_neigh_max * sizeof(unsigned int));
    gh_nlist_ref = (unsigned int *)malloc(g_nlist_pitch * g_neigh_max * sizeof(unsigned int));
    gh_n_neigh = (unsigned int *)malloc(g_N * sizeof(unsigned int));
    gh_n_neigh_ref = (unsigned int *)malloc(g_N * sizeof(unsigned int));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_nlist, g_nlist_pitch * g_neigh_max * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&gd_n_neigh, g_N * sizeof(unsigned int)));
    }

void free_data()
    {
    // free host memory
    free(gh_pos);
    free(gh_idxlist_coord);
    free(gh_idxlist_coord_trans);
    free(gh_bin_size);
    free(gh_bin_adj);
    free(gh_bin_adj_trans);
    free(gh_bin_coords);
    free(gh_nlist);
    free(gh_nlist_ref);
    free(gh_n_neigh);
    free(gh_n_neigh_ref);
    
    // free GPU memory
    CUDA_SAFE_CALL(cudaFree(gd_pos));
    CUDA_SAFE_CALL(cudaFree(gd_bin_adj));
    CUDA_SAFE_CALL(cudaFree(gd_bin_adj_trans));
    CUDA_SAFE_CALL(cudaFree(gd_idxlist_coord));
    CUDA_SAFE_CALL(cudaFree(gd_idxlist_coord_trans));
    CUDA_SAFE_CALL(cudaFreeArray(gd_idxlist_coord_array));
    CUDA_SAFE_CALL(cudaFreeArray(gd_idxlist_coord_trans_array));
    CUDA_SAFE_CALL(cudaFree(gd_bin_size));
    CUDA_SAFE_CALL(cudaFreeArray(gd_bin_adj_array));
    CUDA_SAFE_CALL(cudaFreeArray(gd_bin_adj_trans_array));
    CUDA_SAFE_CALL(cudaFree(gd_bin_coords));
    CUDA_SAFE_CALL(cudaFree(gd_nlist));
    CUDA_SAFE_CALL(cudaFree(gd_n_neigh));
    }

float my_int_as_float(int a)
    {
    volatile union
        {
        int a; float b;
        } u;
        
    u.a = a;
    
    return u.b;
    }

int my_float_as_int(float b)
    {
    volatile union
        {
        int a; float b;
        } u;
        
    u.b = b;
    
    return u.a;
    }

void initialize_data()
    {
    // initialize particles randomly
    for (unsigned int i = 0; i < g_N; i++)
        {
        gh_pos[i].x = float((rand())/float(RAND_MAX) - 0.5)*g_Lx;
        gh_pos[i].y = float((rand())/float(RAND_MAX) - 0.5)*g_Ly;
        gh_pos[i].z = float((rand())/float(RAND_MAX) - 0.5)*g_Lz;
        gh_pos[i].w = 0.0f;
        }
        
    // copy particles to the device
    CUDA_SAFE_CALL(cudaMemcpy(gd_pos, gh_pos, sizeof(float4)*g_N, cudaMemcpyHostToDevice));
    
    // zero all other memory
    memset(gh_idxlist_coord, 0xff, sizeof(float4)*g_Mx*g_My*g_Mz*g_Nmax);
    memset(gh_idxlist_coord_trans, 0xff, sizeof(float4)*g_Mx*g_My*g_Mz*g_Nmax);
    memset(gh_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz);
    
    CUDA_SAFE_CALL(cudaMemset(gd_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz));
    
    memset(gh_bin_adj, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27);
    memset(gh_bin_adj_trans, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27);
    
    // initialize the bin coords
    for (unsigned int i = 0; i < g_Mx; i++)
        for (unsigned int j = 0; j < g_My; j++)
            for (unsigned int k = 0; k < g_Mz; k++)
                gh_bin_coords[i*(g_Mz*g_My) + j * g_Mz + k] = make_uint4(i,j,k,0);
                
    CUDA_SAFE_CALL(cudaMemcpy(gd_bin_coords, gh_bin_coords, g_Mx*g_My*g_Mz*sizeof(uint4), cudaMemcpyHostToDevice));
    
    // initialize the adjacency array
    for (unsigned int bin = 0; bin < g_Mx*g_My*g_Mz; bin++)
        {
        int cntr_i = (int)gh_bin_coords[bin].x;
        int cntr_j = (int)gh_bin_coords[bin].y;
        int cntr_k = (int)gh_bin_coords[bin].z;
        
        unsigned int count = 0;
        for (int i = cntr_i-1; i <= cntr_i+1; i++)
            {
            int cur_i = i;
            if (cur_i < 0)
                cur_i += g_Mx;
            if (cur_i >= (int)g_Mx)
                cur_i -= g_Mx;
                
            for (int j = cntr_j-1; j <= cntr_j+1; j++)
                {
                int cur_j = j;
                if (cur_j < 0)
                    cur_j += g_My;
                if (cur_j >= (int)g_My)
                    cur_j -= g_My;
                    
                for (int k = cntr_k-1; k <= cntr_k+1; k++)
                    {
                    int cur_k = k;
                    if (cur_k < 0)
                        cur_k += g_Mz;
                    if (cur_k >= (int)g_Mz)
                        cur_k -= g_Mz;
                        
                    unsigned int neigh_bin = cur_i*(g_Mz*g_My) + cur_j * g_Mz + cur_k;
                    gh_bin_adj[27*bin + count] = neigh_bin;
                    gh_bin_adj_trans[g_Mx*g_My*g_Mz*count + bin] = neigh_bin;
                    count++;
                    }
                }
            }
        }
        
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_bin_adj_array, 0, 0, gh_bin_adj, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_bin_adj_trans_array, 0, 0, gh_bin_adj_trans, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gd_bin_adj, gh_bin_adj, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gd_bin_adj_trans, gh_bin_adj_trans, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));

    memset(gh_nlist, 0, sizeof(unsigned int)*g_nlist_pitch*g_neigh_max);
    memset(gh_n_neigh, 0, sizeof(unsigned int)*g_N);
    memset(gh_nlist_ref, 0, sizeof(unsigned int)*g_nlist_pitch*g_neigh_max);
    memset(gh_n_neigh_ref, 0, sizeof(unsigned int)*g_N);
    CUDA_SAFE_CALL(cudaMemset(gd_nlist, 0, sizeof(unsigned int)*g_nlist_pitch*g_neigh_max));
    CUDA_SAFE_CALL(cudaMemset(gd_n_neigh, 0, sizeof(unsigned int)*g_N));
    }

// moves the particles a "little bit" randomly and copies the new positions to the device
void tweak_data()
    {
    float xhi = g_Lx / 2.0f;
    float xlo = -xhi;
    float yhi = g_Ly / 2.0f;
    float ylo = -yhi;
    float zhi = g_Lz / 2.0f;
    float zlo = -zhi;
    
    for (unsigned int i = 0; i < g_N; i++)
        {
        // OK, so it is a poorly distributed tweak. So what, it serves it's purpose.
        float x = float((rand())/float(RAND_MAX) - 0.5);
        float y = float((rand())/float(RAND_MAX) - 0.5);
        float z = float((rand())/float(RAND_MAX) - 0.5);
        float len = sqrt(x*x + y*y + z*z);
        x = x / len * tweak_dist;
        y = y / len * tweak_dist;
        z = z / len * tweak_dist;
        
        gh_pos[i].x += x;
        gh_pos[i].y += y;
        gh_pos[i].z += z;
        
        // fix up boundary conditions
        if (gh_pos[i].x >= xhi)
            gh_pos[i].x -= g_Lx;
        if (gh_pos[i].x <= xlo)
            gh_pos[i].x += g_Lx;
        if (gh_pos[i].y >= yhi)
            gh_pos[i].y -= g_Ly;
        if (gh_pos[i].y <= ylo)
            gh_pos[i].y += g_Ly;
        if (gh_pos[i].z >= zhi)
            gh_pos[i].z -= g_Lz;
        if (gh_pos[i].z <= zlo)
            gh_pos[i].z += g_Lz;
        }
        
    // update the data on the device
    cudaMemcpy(gd_pos, gh_pos, sizeof(float4)*g_N, cudaMemcpyHostToDevice);
    }

// sorts the data to mimic HOOMD's standard data pattern (sort of)
void sort_data()
    {
    std::vector< std::pair<unsigned int, unsigned int> > bin_list(g_N);
    // make even bin dimensions
    float binx = g_Lx / float(g_Mx);
    float biny = g_Ly / float(g_My);
    float binz = g_Lz / float(g_Mz);
    
    float xlo = -g_Lx/2.0f;
    float ylo = -g_Lx/2.0f;
    float zlo = -g_Lx/2.0f;
    
    // precompute scale factors to eliminate division in inner loop
    float scalex = 1.0f / binx;
    float scaley = 1.0f / biny;
    float scalez = 1.0f / binz;
    
    for (unsigned int i = 0; i < g_N; i++)
        {
        // find the bin each particle belongs in
        unsigned int ib = (unsigned int)((gh_pos[i].x-xlo)*scalex);
        unsigned int jb = (unsigned int)((gh_pos[i].y-ylo)*scaley);
        unsigned int kb = (unsigned int)((gh_pos[i].z-zlo)*scalez);
        
        // need to handle the case where the particle is exactly at the box hi
        if (ib == g_Mx)
            ib = 0;
        if (jb == g_My)
            jb = 0;
        if (kb == g_Mz)
            kb = 0;
            
        // update the bin
        unsigned int bin = ib*(g_Mz*g_My) + jb * g_Mz + kb;
        bin_list[i] = std::pair<unsigned int, unsigned int>(bin, i);
        }
    
    std::sort(bin_list.begin(), bin_list.end());
    float4 *tmp_pos = (float4*)malloc(sizeof(float4)*g_N);
    memcpy(tmp_pos, gh_pos, sizeof(float4)*g_N);
    
    for (unsigned int i = 0; i < g_N; i++)
        {
        unsigned int j = bin_list[i].second;
        gh_pos[i] = tmp_pos[j];
        }
    
    free(tmp_pos);
    // update the data on the device
    cudaMemcpy(gd_pos, gh_pos, sizeof(float4)*g_N, cudaMemcpyHostToDevice);
    }

void sort_bin_adj()
    {
    /*// print out some bins
    unsigned int start_bin = 200;
    unsigned int end_bin = start_bin+32;
    for (int i = 0; i < 27; i++)
        {
        for (int cur_bin = start_bin; cur_bin < end_bin; cur_bin++)
            cout << setw(4) << gh_bin_adj[27*cur_bin + i] << " ";
        cout << endl;
        }
    cout << endl;*/
    
    // sort the bin_adj lists
    for (unsigned int cur_bin = 0; cur_bin < g_Mx*g_My*g_Mz; cur_bin++)
        {
        bool swapped = false;
        do
            {
            swapped = false;
            for (unsigned int i = 0; i < 27-1; i++)
                {
                if (gh_bin_adj[27*cur_bin + i] > gh_bin_adj[27*cur_bin + i+1])
                    {
                    unsigned int tmp = gh_bin_adj[27*cur_bin + i+1];
                    gh_bin_adj[27*cur_bin + i+1] = gh_bin_adj[27*cur_bin + i];
                    gh_bin_adj[27*cur_bin + i] = tmp;
                    swapped = true;
                    }
                }
            }
        while (swapped);
        
        }
        
    /*for (int i = 0; i < 27; i++)
        {
        for (int cur_bin = start_bin; cur_bin < end_bin; cur_bin++)
            cout << setw(4) << gh_bin_adj[27*cur_bin + i] << " ";
        cout << endl;
        }*/
        
    unsigned int nbins = g_Mx*g_My*g_Mz;
    for (unsigned int cur_bin = 0; cur_bin < g_Mx*g_My*g_Mz; cur_bin++)
        {
        bool swapped = false;
        do
            {
            swapped = false;
            for (unsigned int i = 0; i < 27-1; i++)
                {
                if (gh_bin_adj_trans[nbins*i + cur_bin] > gh_bin_adj_trans[nbins*(i+1) + cur_bin])
                    {
                    unsigned int tmp = gh_bin_adj_trans[nbins*(i+1) + cur_bin];
                    gh_bin_adj_trans[nbins*(i+1) + cur_bin] = gh_bin_adj_trans[nbins*i + cur_bin];
                    gh_bin_adj_trans[nbins*i + cur_bin] = tmp;
                    swapped = true;
                    }
                }
            }
        while (swapped);
        
        }
        
        
        
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_bin_adj_array, 0, 0, gh_bin_adj, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_bin_adj_trans_array, 0, 0, gh_bin_adj_trans, sizeof(unsigned int)*g_Mx*g_My*g_Mz*27, cudaMemcpyHostToDevice));
    }

__global__ void fast_memclear_kernal(unsigned int *d_data, unsigned int N)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_data[idx] = 0;
    }

void rebin_particles_host(float4 *idxlist_coord, float4*idxlist_coord_trans, unsigned int *bin_size, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax);


//****************** verify gh_nlist and gh_n_neigh vs the reference ones
bool verify()
    {
    // just check that the number of neighbors is OK
    for (unsigned int i = 0; i < g_N; i++)
        {
        if (gh_n_neigh[i] != gh_n_neigh_ref[i])
            {
            printf("n_neigh[%d] doesn't match %d != %d\n", i, gh_n_neigh[i], gh_n_neigh_ref[i]);
            return false;
            }
        }
        
    return true;
    }

//****************** bins the data on the host from scratch
void rebin_particles_host(float4 *idxlist_coord, float4*idxlist_coord_trans, unsigned int *bin_size, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
    {
    // clear the bin sizes
    for (unsigned int i = 0; i < Mx*My*Mz; i++)
        bin_size[i] = 0;
        
    // make even bin dimensions
    float binx = Lx / float(Mx);
    float biny = Ly / float(My);
    float binz = Lz / float(Mz);
    
    float xlo = -Lx/2.0f;
    float ylo = -Lx/2.0f;
    float zlo = -Lx/2.0f;
    
    // precompute scale factors to eliminate division in inner loop
    float scalex = 1.0f / binx;
    float scaley = 1.0f / biny;
    float scalez = 1.0f / binz;
    
    g_actual_Nmax = 0;
    // bin each particle
    for (unsigned int i = 0; i < N; i++)
        {
        // find the bin each particle belongs in
        unsigned int ib = (unsigned int)((pos[i].x-xlo)*scalex);
        unsigned int jb = (unsigned int)((pos[i].y-ylo)*scaley);
        unsigned int kb = (unsigned int)((pos[i].z-zlo)*scalez);
        
        // need to handle the case where the particle is exactly at the box hi
        if (ib == Mx)
            ib = 0;
        if (jb == My)
            jb = 0;
        if (kb == Mz)
            kb = 0;
            
        // update the bin
        unsigned int bin = ib*(Mz*My) + jb * Mz + kb;
        unsigned int size = bin_size[bin];
        if (size < Nmax)
            {
            idxlist_coord[bin*Nmax + size] = make_float4(pos[i].x, pos[i].y, pos[i].z, my_int_as_float(i));
            idxlist_coord_trans[size*Mx*My*Mz + bin] = make_float4(pos[i].x, pos[i].y, pos[i].z, my_int_as_float(i));
            }
        else
            {
            printf("Error, bins overflowed!\n");
            exit(1);
            }
        bin_size[bin]++;
        
        if (bin_size[bin] > g_actual_Nmax)
            g_actual_Nmax = bin_size[bin];
        }
    }

//****************** builds the neighbor list on the host
template<bool transpose_idxlist_coord, bool transpose_bin_adj> void neighbor_particles_host(unsigned int *nlist, unsigned int *n_neigh, unsigned int nlist_pitch, float r_cut_sq, unsigned int neigh_max, float4 *idxlist_coord, unsigned int *bin_size, uint4 *bin_coords, unsigned int *bin_adj, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
    {
    float binx = Lx / float(Mx);
    float biny = Ly / float(My);
    float binz = Lz / float(Mz);
    
    float xlo = -Lx/2.0f;
    float ylo = -Lx/2.0f;
    float zlo = -Lx/2.0f;
    
    // precompute scale factors to eliminate division in inner loop
    float scalex = 1.0f / binx;
    float scaley = 1.0f / biny;
    float scalez = 1.0f / binz;
    
    // for each particle
    for (unsigned int i = 0; i < N; i++)
        {
        unsigned int cur_n_neigh = 0;
        
        // find the bin each particle belongs in
        unsigned int ib = (unsigned int)((pos[i].x-xlo)*scalex);
        unsigned int jb = (unsigned int)((pos[i].y-ylo)*scaley);
        unsigned int kb = (unsigned int)((pos[i].z-zlo)*scalez);
        
        // need to handle the case where the particle is exactly at the box hi
        if (ib == Mx)
            ib = 0;
        if (jb == My)
            jb = 0;
        if (kb == Mz)
            kb = 0;
            
        // identify the bin
        unsigned int bin = ib*(Mz*My) + jb * Mz + kb;
        
        // loop through all neighboring bins
        for (unsigned int cur_bin_idx = 0; cur_bin_idx < 27; cur_bin_idx++)
            {
            unsigned int neigh_bin;
            if (transpose_bin_adj)
                neigh_bin = bin_adj[Mx*My*Mz*cur_bin_idx + bin];
            else
                neigh_bin = bin_adj[27*bin + cur_bin_idx];
                
            // check against all the particles in that neighboring bin to see if it is a neighbor
            unsigned int cur_bin_size = bin_size[neigh_bin];
            for (unsigned int slot = 0; slot < cur_bin_size; slot++)
                {
                float4 neigh;
                if (transpose_idxlist_coord)
                    neigh = idxlist_coord[slot*Mx*My*Mz + neigh_bin];
                else
                    neigh = idxlist_coord[neigh_bin*Nmax + slot];
                    
                float dx = pos[i].x - neigh.x;
                if (dx >= Lx/2.0)
                    dx -= Lx;
                if (dx <= -Lx/2.0)
                    dx += Lx;
                    
                float dy = pos[i].y - neigh.y;
                if (dy >= Ly/2.0)
                    dy -= Ly;
                if (dy <= -Ly/2.0)
                    dy += Ly;
                    
                float dz = pos[i].z - neigh.z;
                if (dz >= Lz/2.0)
                    dz -= Lz;
                if (dz <= -Lz/2.0)
                    dz += Lz;
                    
                float dr_sq = dx*dx + dy*dy + dz*dz;
                
                if (dr_sq <= r_cut_sq && i != my_float_as_int(neigh.w))
                    {
                    if (cur_n_neigh == neigh_max)
                        {
                        printf("Error, nlist overflowed!\n");
                        exit(1);
                        }
                    else
                        {
                        nlist[nlist_pitch * cur_n_neigh + i] = my_float_as_int(neigh.w);
                        cur_n_neigh++;
                        }
                    }
                }
            }
            
        n_neigh[i] = cur_n_neigh;
        }
    }


// benchmark the host neighborlist
template<bool transpose_idxlist_coord, bool transpose_bin_adj> void bmark_host_nlist()
    {
    float4 *idxlist_coord;
    if (transpose_idxlist_coord)
        idxlist_coord = gh_idxlist_coord_trans;
    else
        idxlist_coord = gh_idxlist_coord;
        
    unsigned int *bin_adj;
    if (transpose_bin_adj)
        bin_adj = gh_bin_adj_trans;
    else
        bin_adj = gh_bin_adj;
        
    // warm up
    neighbor_particles_host<transpose_idxlist_coord, transpose_bin_adj>(gh_nlist, gh_n_neigh, g_nlist_pitch, g_rcut*g_rcut, g_neigh_max, idxlist_coord, gh_bin_size, gh_bin_coords, bin_adj, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
    
    // verify results
    if (!verify())
        {
        printf("Invalid results in host bmark!\n");
        return;
        }
        
    // benchmarks
    timeval start;
    gettimeofday(&start, NULL);
    
    unsigned int iters = 10;
    for (unsigned int i = 0; i < iters; i++)
        {
        neighbor_particles_host<transpose_idxlist_coord, transpose_bin_adj>(gh_nlist, gh_n_neigh, g_nlist_pitch, g_rcut*g_rcut, g_neigh_max, idxlist_coord, gh_bin_size, gh_bin_coords, bin_adj, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
        }
        
    timeval end;
    gettimeofday(&end, NULL);
    float t = (end.tv_sec - start.tv_sec)*1000.0f + (end.tv_usec - start.tv_usec)/1000.0f;
    float avg_t = t/float(iters);
    
    printf("Host<%1d,%1d>            : ", transpose_idxlist_coord, transpose_bin_adj);
    printf("%f ms\n", avg_t);
    }


//////************************************** GPU Nlist
//! Texture for reading coord_idxlist from the binned particle data
texture<float4, 2, cudaReadModeElementType> nlist_coord_idxlist_tex;
//! Texture for reading the bins adjacent to a given bin
texture<unsigned int, 2, cudaReadModeElementType> bin_adj_tex;

texture<unsigned int, 1, cudaReadModeElementType> bin_size_tex;

#define EMPTY_BIN 0xffffffff

template<bool transpose_idxlist_coord, bool transpose_bin_adj> __global__ void gpu_compute_nlist_binned_kernel(unsigned int *d_nlist, unsigned int *d_n_neigh, unsigned int nlist_pitch, unsigned int neigh_max, float4 *d_pos, float4 *d_idxlist_coord, unsigned int Nmax, unsigned int local_num, float Lx, float Ly, float Lz, float Lxinv, float Lyinv, float Lzinv, float r_maxsq, unsigned int actual_Nmax, float scalex, float scaley, float scalez, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int *d_bin_size, unsigned int *d_bin_adj)
    {
    // each thread is going to compute the neighbor list for a single particle
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int my_pidx = idx;
    
    // quit early if we are past the end of the array
    if (idx >= local_num)
        return;
   
    unsigned int idxlist_coord_width = 0;
    if (transpose_idxlist_coord)
        idxlist_coord_width = Mx*My*Mz;
    else
        idxlist_coord_width = Nmax;

    unsigned int bin_adj_width = 0;
    if (transpose_bin_adj)
        bin_adj_width = Mx*My*Mz;
    else
        bin_adj_width = 27;

    // first, determine which bin this particle belongs to
    // MEM TRANSFER: 32 bytes
    float4 my_pos = d_pos[my_pidx];
    
    // FLOPS: 9
    unsigned int ib = (unsigned int)((my_pos.x+Lx/2.0f)*scalex);
    unsigned int jb = (unsigned int)((my_pos.y+Ly/2.0f)*scaley);
    unsigned int kb = (unsigned int)((my_pos.z+Lz/2.0f)*scalez);
    
    // need to handle the case where the particle is exactly at the box hi
    if (ib == Mx)
        ib = 0;
    if (jb == My)
        jb = 0;
    if (kb == Mz)
        kb = 0;
        
    // MEM TRANSFER: 4 bytes
    int my_bin = ib*(Mz*My) + jb * Mz + kb;
    
    // each thread will determine the neighborlist of a single particle
    int n_neigh = 0;    // count number of neighbors found so far
    
    // loop over all adjacent bins
    for (unsigned int cur_adj = 0; cur_adj < 27; cur_adj++)
        {
        // MEM TRANSFER: 4 bytes
        int neigh_bin;
        /*if (transpose_bin_adj)
            neigh_bin = tex2D(bin_adj_tex, my_bin, cur_adj);
        else
            neigh_bin = tex2D(bin_adj_tex, cur_adj, my_bin);*/
        if (transpose_bin_adj)
            neigh_bin = d_bin_adj[my_bin + cur_adj*bin_adj_width];
        else
            neigh_bin = d_bin_adj[cur_adj + my_bin*bin_adj_width];
            
        //unsigned int size = tex1Dfetch(bin_size_tex, neigh_bin);
        unsigned int size = d_bin_size[neigh_bin];
        
        // now, we are set to loop through the array
        for (int cur_offset = 0; cur_offset < size; cur_offset++)
            {
            // MEM TRANSFER: 16 bytes
            float4 cur_neigh_blob;
            /*if (transpose_idxlist_coord)
                cur_neigh_blob = tex2D(nlist_coord_idxlist_tex, neigh_bin, cur_offset);
            else
                cur_neigh_blob = tex2D(nlist_coord_idxlist_tex, cur_offset, neigh_bin);*/
            if (transpose_idxlist_coord)
                cur_neigh_blob = d_idxlist_coord[cur_offset * idxlist_coord_width + neigh_bin];
            else
                cur_neigh_blob = d_idxlist_coord[neigh_bin * idxlist_coord_width + cur_offset];
                
            float3 neigh_pos;
            neigh_pos.x = cur_neigh_blob.x;
            neigh_pos.y = cur_neigh_blob.y;
            neigh_pos.z = cur_neigh_blob.z;
            int cur_neigh = __float_as_int(cur_neigh_blob.w);
            
            // FLOPS: 15
            float dx = my_pos.x - neigh_pos.x;
            dx = dx - Lx * rintf(dx * Lxinv);
            
            float dy = my_pos.y - neigh_pos.y;
            dy = dy - Ly * rintf(dy * Lyinv);
            
            float dz = my_pos.z - neigh_pos.z;
            dz = dz - Lz * rintf(dz * Lzinv);
            
            // FLOPS: 5
            float dr = dx*dx + dy*dy + dz*dz;
            
            // FLOPS: 1 / MEM TRANSFER total = N * estimated number of neighbors * 4
            if (dr <= r_maxsq && my_pidx != cur_neigh)
                {
                if (n_neigh < neigh_max)
                    {
                    d_nlist[my_pidx + n_neigh*nlist_pitch] = cur_neigh;
                    n_neigh++;
                    }
                }
            }
        }
        
    d_n_neigh[my_pidx] = n_neigh;
    }

template<bool transpose_idxlist_coord, bool transpose_bin_adj> void gpu_compute_nlist_binned(unsigned int *nlist, unsigned int *n_neigh, unsigned int nlist_pitch, float r_cut_sq, unsigned int neigh_max, cudaArray *idxlist_coord, unsigned int *bin_size, uint4 *bin_coords, cudaArray *bin_adj, float4 *pos, float4 *d_idxlist_coord, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax, unsigned int *d_bin_adj)
    {
    const int block_size = 64;
    
    // setup the grid to run the kernel
    int nblocks = (int)ceil((double)N/ (double)block_size);
    
    dim3 grid(nblocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    nlist_coord_idxlist_tex.normalized = false;
    nlist_coord_idxlist_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTextureToArray(nlist_coord_idxlist_tex, idxlist_coord);
    CUT_CHECK_ERROR("error binding texture");
    
    bin_adj_tex.normalized = false;
    bin_adj_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTextureToArray(bin_adj_tex, bin_adj);
    CUT_CHECK_ERROR("error binding texture");
    
    cudaBindTexture(0, bin_size_tex, bin_size, sizeof(unsigned int)*Mx*My*Mz);
    
    // make even bin dimensions
    float binx = (Lx) / float(Mx);
    float biny = (Ly) / float(My);
    float binz = (Lz) / float(Mz);
    
    // precompute scale factors to eliminate division in inner loop
    float scalex = 1.0f / binx;
    float scaley = 1.0f / biny;
    float scalez = 1.0f / binz;
    
    // run the kernel
    gpu_compute_nlist_binned_kernel<transpose_idxlist_coord, transpose_bin_adj><<<grid,threads>>>(nlist, n_neigh, nlist_pitch, neigh_max, pos, d_idxlist_coord, Nmax, N, Lx, Ly, Lz, 1.0f/Lx, 1.0f/Ly, 1.0f/Lz, r_cut_sq, g_actual_Nmax, scalex, scaley, scalez, Mx, My, Mz, bin_size, d_bin_adj);
    }

// benchmark the device neighborlist
template<bool transpose_idxlist_coord, bool transpose_bin_adj> void bmark_device_nlist()
    {
    cudaArray *idxlist_coord_array;
    float4 *d_idxlist_coord;
    if (transpose_idxlist_coord)
        {
        idxlist_coord_array = gd_idxlist_coord_trans_array;
        d_idxlist_coord = gd_idxlist_coord_trans;
        }
    else
        {
        idxlist_coord_array = gd_idxlist_coord_array;
        d_idxlist_coord = gd_idxlist_coord;
        }
        
    cudaArray *bin_adj_array;
    unsigned int *d_bin_adj;
    if (transpose_bin_adj)
        {
        bin_adj_array = gd_bin_adj_trans_array;
        d_bin_adj = gd_bin_adj_trans;
        }
    else
        {
        bin_adj_array = gd_bin_adj_array;
        d_bin_adj = gd_bin_adj;
        }
        
    // warm up
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_compute_nlist_binned_kernel<transpose_idxlist_coord, transpose_bin_adj>, cudaFuncCachePreferL1));
    gpu_compute_nlist_binned<transpose_idxlist_coord, transpose_bin_adj>(gd_nlist, gd_n_neigh, g_nlist_pitch, g_rcut*g_rcut, g_neigh_max, idxlist_coord_array, gd_bin_size, gd_bin_coords, bin_adj_array, gd_pos, d_idxlist_coord, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax, d_bin_adj);
    CUT_CHECK_ERROR("kernel failed");
    
    // copy results back
    CUDA_SAFE_CALL(cudaMemcpy(gh_n_neigh, gd_n_neigh, g_N*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // verify results
    if (!verify())
        {
        printf("Invalid results in device bmark!\n");
        return;
        }
        
    // benchmarks
    timeval start;
    cudaThreadSynchronize();
    gettimeofday(&start, NULL);
    
    unsigned int iters = 100;
    for (unsigned int i = 0; i < iters; i++)
        {
        gpu_compute_nlist_binned<transpose_idxlist_coord, transpose_bin_adj>(gd_nlist, gd_n_neigh, g_nlist_pitch, g_rcut*g_rcut, g_neigh_max, idxlist_coord_array, gd_bin_size, gd_bin_coords, bin_adj_array, gd_pos, d_idxlist_coord, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax, d_bin_adj);
        }
        
    cudaThreadSynchronize();
    timeval end;
    gettimeofday(&end, NULL);
    float t = (end.tv_sec - start.tv_sec)*1000.0f + (end.tv_usec - start.tv_usec)/1000.0f;
    float avg_t = t/float(iters);
    
    printf("Device<%1d,%1d>          : ", transpose_idxlist_coord, transpose_bin_adj);
    printf("%f ms\n", avg_t);
    }


int main(int argc, char **argv)
    {
    // choose defaults if no args specified
    if (argc == 1)
        {
        g_N = 64000;
        g_rcut = 3.8f;
        }
    if (argc == 2)
        {
        g_N = atoi(argv[1]);
        g_rcut = 3.8f;
        }
    if (argc == 3)
        {
        g_N = atoi(argv[1]);
        g_rcut = atof(argv[2]);
        }
        
    float L = pow(float(M_PI/6.0)*float(g_N) / 0.20f, 1.0f/3.0f);
    g_Lx = g_Ly = g_Lz = L;
    
    // setup
    printf("Running gpu_nlist microbenchmark: %d %f\n", g_N, g_rcut);
    allocate_data();
    initialize_data();
    sort_data();
    
    // normally, data in HOOMD is not perfectly sorted:
    //for (unsigned int i = 0; i < 100; i++)
    //  tweak_data();
    
    // prepare the binned data
    rebin_particles_host(gh_idxlist_coord, gh_idxlist_coord_trans, gh_bin_size, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
    // copy it to the device
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_idxlist_coord_array, 0, 0, gh_idxlist_coord, g_Mx*g_My*g_Mz*g_Nmax*sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToArray(gd_idxlist_coord_trans_array, 0, 0, gh_idxlist_coord_trans, g_Mx*g_My*g_Mz*g_Nmax*sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gd_idxlist_coord, gh_idxlist_coord, g_Mx*g_My*g_Mz*g_Nmax*sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gd_idxlist_coord_trans, gh_idxlist_coord_trans, g_Mx*g_My*g_Mz*g_Nmax*sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(gd_bin_size, gh_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // generate the reference data
    neighbor_particles_host<false, false>(gh_nlist_ref, gh_n_neigh_ref, g_nlist_pitch, g_rcut*g_rcut, g_neigh_max, gh_idxlist_coord, gh_bin_size, gh_bin_coords, gh_bin_adj, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
    
    // run the benchmarks
    /*bmark_host_nlist<false, false>();
    bmark_host_nlist<false, true>();
    bmark_host_nlist<true, false>();
    bmark_host_nlist<true, true>();*/
    
    bmark_device_nlist<false, false>();
    bmark_device_nlist<false, true>();
    bmark_device_nlist<true, false>();
    bmark_device_nlist<true, true>();
    
    printf("sorting bin_adj:\n");
    sort_bin_adj();
    
    /*bmark_host_nlist<false, false>();
    bmark_host_nlist<false, true>();
    bmark_host_nlist<true, false>();
    bmark_host_nlist<true, true>();*/
    
    bmark_device_nlist<false, false>();
    bmark_device_nlist<false, true>();
    bmark_device_nlist<true, false>();
    bmark_device_nlist<true, true>();
    
    free_data();
    
    return 0;
    }

