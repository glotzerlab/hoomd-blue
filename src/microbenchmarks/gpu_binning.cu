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

#include <stdio.h>
#include <sys/time.h>

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
const unsigned int g_N = 64000;
const float g_Lx = 55.1294f;
const float g_Ly = 55.1294f;
const float g_Lz = 55.1294f;
const float g_rcut = 3.8f;
const unsigned int g_Nmax = 128;	// Maximum number of particles each cell can hold
const float tweak_dist = 0.1f;

//*************** data structures
float4 *gh_pos, *gd_pos;			// particle positions
unsigned int g_Mx;	// X-dimension of the cell grid
unsigned int g_My;	// Y-dimension of the cell grid
unsigned int g_Mz;	// Z-dimension of the cell grid
unsigned int *gd_idxlist;	// \a Mx x \a My x \a Mz x \a Nmax 4D array holding the indices of the particles in each cell
unsigned int *gh_idxlist;	// \a Mx x \a My x \a Mz x \a Nmax 4D array holding the indices of the particles in each cell
unsigned int *gd_bin_size;	// number of particles in each bin
unsigned int *gh_bin_size;	// number of particles in each bin

unsigned int *gd_old_idxlist;	// old data for the above array (swapped on each update call)
unsigned int *gd_old_bin_size;	// old data for the above array (swapped on each update call)

uint4 *gd_bin_coords;	// pre-calculated bin coordinates for each bin
uint4 *gh_bin_coords;	// pre-calculated bin coordinates for each bin

unsigned int *g_ref_idxlist;	// reference idxlist for correctness comparison
unsigned int *g_ref_bin_size;	// reference bin_size for correctness comparison

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
	gh_idxlist = (unsigned int *)malloc(Nbins * g_Nmax * sizeof(unsigned int));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_idxlist, Nbins * g_Nmax * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_old_idxlist, Nbins * g_Nmax * sizeof(unsigned int)));
	gh_bin_size = (unsigned int *)malloc(Nbins * sizeof(unsigned int));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_size, Nbins * g_Nmax * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_old_bin_size, Nbins * g_Nmax * sizeof(unsigned int)));
	
	g_ref_idxlist = (unsigned int *)malloc(Nbins * g_Nmax * sizeof(unsigned int));
	g_ref_bin_size = (unsigned int *)malloc(Nbins * sizeof(unsigned int));
	
	gh_bin_coords = (uint4*)malloc(Nbins*sizeof(uint4));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_bin_coords, Nbins*sizeof(uint4)));
	}
	
void free_data()
	{
	// free host memory
	free(gh_pos);
	free(gh_idxlist);
	free(gh_bin_size);
	free(g_ref_idxlist);
	free(g_ref_bin_size);
	free(gh_bin_coords);
	
	// free GPU memory
	CUDA_SAFE_CALL(cudaFree(gd_pos));
	CUDA_SAFE_CALL(cudaFree(gd_idxlist));
	CUDA_SAFE_CALL(cudaFree(gd_old_idxlist));
	CUDA_SAFE_CALL(cudaFree(gd_bin_size));
	CUDA_SAFE_CALL(cudaFree(gd_old_bin_size));
	CUDA_SAFE_CALL(cudaFree(gd_bin_coords));
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
	memset(gh_idxlist, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*g_Nmax);
	memset(g_ref_idxlist, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*g_Nmax);
	memset(gh_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz);
	memset(g_ref_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz);
	
	CUDA_SAFE_CALL(cudaMemset(gd_idxlist, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*g_Nmax));
	CUDA_SAFE_CALL(cudaMemset(gd_old_idxlist, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz*g_Nmax));
	CUDA_SAFE_CALL(cudaMemset(gd_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz));
	CUDA_SAFE_CALL(cudaMemset(gd_old_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz));
	
	// initialize the bin coords
	for (unsigned int i = 0; i < g_Mx; i++)
		for (unsigned int j = 0; j < g_My; j++)
			for (unsigned int k = 0; k < g_Mz; k++)
				gh_bin_coords[i*(g_Mz*g_My) + j * g_Mz + k] = make_uint4(i,j,k,0);
	
	CUDA_SAFE_CALL(cudaMemcpy(gd_bin_coords, gh_bin_coords, g_Mx*g_My*g_Mz*sizeof(uint4), cudaMemcpyHostToDevice));
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
	

void rebin_particles_host(unsigned int *idxlist, unsigned int *bin_size, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax);
	

//****************** verify gh_idxlist and gh_bin_size vs the reference ones
bool verify()
	{
	// generate the reference data
	rebin_particles_host(g_ref_idxlist, g_ref_bin_size, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);	
	
	for (unsigned int bin = 0; bin < g_Mx*g_My*g_Mz; bin++)
		{
		// check bin sizes first
		if (gh_bin_size[bin] != g_ref_bin_size[bin])
			{
			printf("bin sizes differ for bin %d : %d != %d\n", bin, gh_bin_size[bin], g_ref_bin_size[bin]);
			return false;
			}
			
		// now check every single particle in the bins
		unsigned int size = gh_bin_size[bin];
		for (unsigned int ref_i = 0; ref_i < size; ref_i++)
			{
			unsigned int particle_i = g_ref_idxlist[bin*g_Nmax + ref_i];
			
			bool found = false;
			for (unsigned int j = 0; j < size; j++)
				{
				if (particle_i == gh_idxlist[bin*g_Nmax + j])
					{
					found = true;
					break;
					}
				}
			if (!found)
				{
				printf("particle %d not present in bin %d\n", particle_i, bin);
				return false;
				}
			}
		}
	return true;
	}

//****************** bins the data on the host from scratch
void rebin_particles_host(unsigned int *idxlist, unsigned int *bin_size, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
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
			idxlist[bin*Nmax + size] = i;
		else
			{
			printf("Error, bins overflowed!\n");
			exit(1);
			}
		bin_size[bin]++;
		}
	}
	
// benchmark the host rebinning
void bmark_host_rebinning(bool include_memcpy)
	{
	// warm up
	rebin_particles_host(gh_idxlist, gh_bin_size, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
	
	// verify results
	if (!verify())
		{
		printf("Invalid results in host bmark!\n");
		return;
		}
	
	// benchmarks
	timeval start;
	gettimeofday(&start, NULL);
	
	unsigned int iters = 1000;
	for (unsigned int i = 0; i < iters; i++)
		{
		if (include_memcpy)
			cudaMemcpy(gh_pos, gd_pos, g_N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
					
		rebin_particles_host(gh_idxlist, gh_bin_size, gh_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
		
		if (include_memcpy)
			{
			cudaMemcpy(gd_idxlist, gh_idxlist, g_Mx*g_My*g_Mz*g_Nmax*sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMemcpy(gd_bin_size, gh_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyHostToDevice);
			}
		}
	
	timeval end;
	gettimeofday(&end, NULL);
	float t = (end.tv_sec - start.tv_sec)*1000.0f + (end.tv_usec - start.tv_usec)/1000.0f;
	float avg_t = t/float(iters);
	
	if (include_memcpy)
		printf("Host w/device memcpy: ");
	else
		printf("Host                : ");
	printf("%f ms\n", avg_t);
	}

//*************************** simple method of binning on the GPU
// Run one thread per particle
// determine the bin that particle belongs in
// atomicInc the bin size
// write the particle into the bin
// done.
__global__ void rebin_simple_kernel(unsigned int *d_idxlist, unsigned int *d_bin_size, float4 *d_pos, unsigned int N, float xlo, float ylo, float zlo, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax, float scalex, float scaley, float scalez)
	{
	// read in the particle that belongs to this thread
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;
		
	float4 pos = d_pos[idx];
	
	// determine which bin it belongs in
	unsigned int ib = (unsigned int)((pos.x-xlo)*scalex);
	unsigned int jb = (unsigned int)((pos.y-ylo)*scaley);
	unsigned int kb = (unsigned int)((pos.z-zlo)*scalez);
	
	// need to handle the case where the particle is exactly at the box hi
	if (ib == Mx)
		ib = 0;
	if (jb == My)
		jb = 0;
	if (kb == Mz)
		kb = 0;
		
	unsigned int bin = ib*(Mz*My) + jb * Mz + kb;
	unsigned int size = atomicInc(&d_bin_size[bin], 0xffffffff);
	if (size < Nmax)
		d_idxlist[bin*Nmax + size] = idx;
	}
	
void rebin_particles_simple(unsigned int *idxlist, unsigned int *bin_size, float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
	{
	// run one particle per thread
	int block_size = 128;
	int n_blocks = (int)ceil(float(N)/(float)block_size);

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
	
	// call the kernel
	cudaMemset(gd_bin_size, 0, sizeof(unsigned int)*g_Mx*g_My*g_Mz);
	rebin_simple_kernel<<<n_blocks, block_size>>>(idxlist, bin_size, pos, N, xlo, ylo, zlo, Mx, My, Mz, Nmax, scalex, scaley, scalez);
	}
	
// benchmark the device rebinning
void bmark_simple_rebinning()
	{
	// warm up
	rebin_particles_simple(gd_idxlist, gd_bin_size, gd_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
	CUT_CHECK_ERROR("kernel failed");
	// copy back from device
	CUDA_SAFE_CALL(cudaMemcpy(gh_idxlist, gd_idxlist, g_Mx*g_My*g_Mz*g_Nmax*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(gh_bin_size, gd_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// verify results
	if (!verify())
		{
		printf("Invalid results in GPU/simple bmark!\n");
		return;
		}
	
	// benchmarks
	float total_time = 0.0f;
	cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
	
	
	unsigned int iters = 1000;
	for (unsigned int i = 0; i < iters; i++)
		{
		cudaEventRecord(start, 0);
		rebin_particles_simple(gd_idxlist, gd_bin_size, gd_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
		cudaEventRecord(end, 0);
		
		float tmp;
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&tmp, start, end);
		total_time += tmp;
		}
	
	float avg_t = total_time/float(iters);
	
	// copy back from device
	CUDA_SAFE_CALL(cudaMemcpy(gh_idxlist, gd_idxlist, g_Mx*g_My*g_Mz*g_Nmax*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(gh_bin_size, gd_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// verify results again to be sure
	if (!verify())
		{
		printf("Invalid results at end of GPU/simple bmark!\n");
		return;
		}	
	
	printf("GPU/simple          : ");
	printf("%f ms\n", avg_t);
	}
	
//*************************** simple update method of binning on the GPU
// Run one thread per bin
// loop through all particles in neighboring bins
// determine which of those particles belong in this bin, and write them there
// done.
texture<float4, 1, cudaReadModeElementType> pos_tex;
texture<unsigned int, 1, cudaReadModeElementType> in_bin_size_tex;
texture<unsigned int, 1, cudaReadModeElementType> in_idxlist_tex;

__global__ void update_simple_kernel(unsigned int *d_out_idxlist, unsigned int *d_out_bin_size, uint4* d_bin_coords, float xlo, float ylo, float zlo, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax, float scalex, float scaley, float scalez)
	{
	// find the coordinates of our bin
	unsigned int bin = blockIdx.x * blockDim.x + threadIdx.x;
	if (bin >= Mx*My*Mz)
		return;
	uint4 coords = d_bin_coords[bin];
	int bin_i = coords.x;
	int bin_j = coords.y;
	int bin_k = coords.z;
	
	// intialize the new particles in this bin to 0
	unsigned int bin_size = 0;
	
	// loop through all the neighboring bins and find particles that now belong in this bin
	for (int cur_i = bin_i - 1; cur_i <= bin_i+1; cur_i++)
		{
		int neigh_i = cur_i;
		if (neigh_i == -1)
			neigh_i = Mx-1;
		if (neigh_i == Mx)
			neigh_i = 0;
			
		for (int cur_j = bin_j - 1; cur_j <= bin_j+1; cur_j++)
			{
			int neigh_j = cur_j;
			if (neigh_j == -1)
				neigh_j = My-1;
			if (neigh_j == My)
				neigh_j = 0;
				
			for (int cur_k = bin_k - 1; cur_k <= bin_k+1; cur_k++)
				{
				int neigh_k = cur_k;
				if (neigh_k == -1)
					neigh_k = Mz-1;
				if (neigh_k == Mz)
					neigh_k = 0;
					
				// determine the index of the neighboring bin
				unsigned int neigh_bin = neigh_i*(Mz*My) + neigh_j * Mz + neigh_k;
				
				// loop through all particles in that neighboring bin
				unsigned int neigh_bin_size = tex1Dfetch(in_bin_size_tex, neigh_bin);
				
				for (unsigned int cur_particle = 0; cur_particle < neigh_bin_size; cur_particle++)
					{
					// read in the current particle in the neighboring bin
					unsigned int pidx = tex1Dfetch(in_idxlist_tex, neigh_bin*Nmax + cur_particle);
					float4 pos = tex1Dfetch(pos_tex, pidx);
					
					// determine which bin the particle should be in
					unsigned int ib = (unsigned int)((pos.x-xlo)*scalex);
					unsigned int jb = (unsigned int)((pos.y-ylo)*scaley);
					unsigned int kb = (unsigned int)((pos.z-zlo)*scalez);
					
					// need to handle the case where the particle is exactly at the box hi
					if (ib == Mx)
						ib = 0;
					if (jb == My)
						jb = 0;
					if (kb == Mz)
						kb = 0;
						
					unsigned int cur_bin = ib*(Mz*My) + jb * Mz + kb;
					
					// if that is this bin, add it to the bin
					if (bin == cur_bin && bin_size < Nmax)
						{
						d_out_idxlist[bin*Nmax + bin_size] = pidx;
						bin_size++;
						}
					}
				}
			}
		}
		
	d_out_bin_size[bin] = bin_size;
	}


void update_particles_simple(float4 *pos, unsigned int N, float Lx, float Ly, float Lz, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
	{
	// run one bin per thread
	int block_size = 128;
	int n_blocks = (int)ceil(float(Mx*My*Mz)/(float)block_size);

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
	
	// swap the pointers
	unsigned int * tmp;
	tmp = gd_idxlist;
	gd_idxlist = gd_old_idxlist;
	gd_old_idxlist = tmp;
	
	tmp = gd_bin_size;
	gd_bin_size = gd_old_bin_size;
	gd_old_bin_size = tmp;
	
	// bind the textures
	cudaBindTexture(0, pos_tex, pos, sizeof(float4) * N);
	cudaBindTexture(0, in_bin_size_tex, gd_old_bin_size, sizeof(unsigned int) * Mx*My*Mz);
	cudaBindTexture(0, in_idxlist_tex, gd_old_idxlist, sizeof(unsigned int) * Mx*My*Mz*Nmax);
	
	// call the kernel
	update_simple_kernel<<<n_blocks, block_size>>>(gd_idxlist, gd_bin_size, gd_bin_coords, xlo, ylo, zlo, Mx, My, Mz, Nmax, scalex, scaley, scalez);
	}

// benchmark the device rebinning
void bmark_simple_updating()
	{
	// warm up
	rebin_particles_simple(gd_idxlist, gd_bin_size, gd_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
	CUT_CHECK_ERROR("kernel failed");
	update_particles_simple(gd_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
		
	// copy back from device
	CUDA_SAFE_CALL(cudaMemcpy(gh_idxlist, gd_idxlist, g_Mx*g_My*g_Mz*g_Nmax*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(gh_bin_size, gd_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// verify results
	if (!verify())
		{
		printf("Invalid results in GPU/update bmark!\n");
		return;
		}
	
	// benchmarks
	float total_time = 0.0f;
	cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
	
	unsigned int iters = 1000;
	for (unsigned int i = 0; i < iters; i++)
		{
		tweak_data();
		
		cudaThreadSynchronize();
		
		cudaEventRecord(start, 0);
		update_particles_simple(gd_pos, g_N, g_Lx, g_Ly, g_Lz, g_Mx, g_My, g_Mz, g_Nmax);
		cudaEventRecord(end, 0);
		
		float tmp;
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&tmp, start, end);
		total_time += tmp;
		}
	
	float avg_t = total_time/float(iters);
	
	// copy back from device
	CUDA_SAFE_CALL(cudaMemcpy(gh_idxlist, gd_idxlist, g_Mx*g_My*g_Mz*g_Nmax*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(gh_bin_size, gd_bin_size, g_Mx*g_My*g_Mz*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// verify results again to be sure
	if (!verify())
		{
		printf("Invalid results at end of GPU/update bmark!\n");
		return;
		}	
	
	printf("GPU/update          : ");
	printf("%f ms\n", avg_t);
	}


int main(void)
	{
	#ifdef ENABLE_CAC_GPU_ID
	if (!getenv("CAC_GPU_ID"))
		printf("Error! Compiled with CAC_GPU_ID support, but no $CAC_GPU_ID specified\n");
	else
		cudaSetDevice(atoi(getenv("CAC_GPU_ID")));
	#endif
	
	// setup
	allocate_data();
	initialize_data();
	
	// run the various benchmarks
	bmark_host_rebinning(false);
	bmark_host_rebinning(true);
	bmark_simple_rebinning();
	bmark_simple_updating();
	
	free_data();
	
	return 0;
	}
