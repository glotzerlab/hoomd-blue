// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegratorHPMCMonoGPU.cuh"

#include "hoomd/TextureTools.h"

#include <stdio.h>

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoGPU.cu
    \brief Definition of CUDA kernels and drivers for IntegratorHPMCMono
*/

//! Kernel to generate expanded cells
/*! \param d_excell_idx Output array to list the particle indices in the expanded cells
    \param d_excell_size Output array to list the number of particles in each expanded cell
    \param excli Indexer for the expanded cells
    \param d_cell_idx Particle indices in the normal cells
    \param d_cell_size Number of particles in each cell
    \param d_cell_adj Cell adjacency list
    \param ci Cell indexer
    \param cli Cell list indexer
    \param cadji Cell adjacency indexer

    gpu_hpmc_excell_kernel executes one thread per cell. It gathers the particle indices from all neighboring cells
    into the output expanded cell.
*/
__global__ void gpu_hpmc_excell_kernel(unsigned int *d_excell_idx,
                                       unsigned int *d_excell_size,
                                       const Index2D excli,
                                       const unsigned int *d_cell_idx,
                                       const unsigned int *d_cell_size,
                                       const unsigned int *d_cell_adj,
                                       const Index3D ci,
                                       const Index2D cli,
                                       const Index2D cadji)
    {
    // compute the output cell
    unsigned int my_cell = 0;
    my_cell = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_cell >= ci.getNumElements())
        return;

    unsigned int my_cell_size = 0;

    // loop over neighboring cells and build up the expanded cell list
    for (unsigned int offset = 0; offset < cadji.getW(); offset++)
        {
        unsigned int neigh_cell = d_cell_adj[cadji(offset, my_cell)];
        unsigned int neigh_cell_size = d_cell_size[neigh_cell];

        for (unsigned int k = 0; k < neigh_cell_size; k++)
            {
            // read in the index of the new particle to add to our cell
            unsigned int new_idx = __ldg(d_cell_idx + cli(k, neigh_cell));
            d_excell_idx[excli(my_cell_size, my_cell)] = new_idx;
            my_cell_size++;
            }
        }

    // write out the final size
    d_excell_size[my_cell] = my_cell_size;
    }

//! Kernel driver for gpu_hpmc_excell_kernel()
cudaError_t gpu_hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size)
    {
    assert(d_excell_idx);
    assert(d_excell_size);
    assert(d_cell_idx);
    assert(d_cell_size);
    assert(d_cell_adj);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    if (max_block_size == -1)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_hpmc_excell_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // setup the grid to run the kernel
    dim3 threads(min(block_size, (unsigned int)max_block_size), 1, 1);
    dim3 grid(ci.getNumElements() / block_size + 1, 1, 1);

    gpu_hpmc_excell_kernel<<<grid, threads>>>(d_excell_idx,
                                              d_excell_size,
                                              excli,
                                              d_cell_idx,
                                              d_cell_size,
                                              d_cell_adj,
                                              ci,
                                              cli,
                                              cadji);

    return cudaSuccess;
    }


//! Kernel for grid shift
/*! \param d_postype postype of each particle
    \param d_image Image flags for each particle
    \param N number of particles
    \param box Simulation box
    \param shift Vector by which to translate the particles

    Shift all the particles by a given vector.

    \ingroup hpmc_kernels
*/
__global__ void gpu_hpmc_shift_kernel(Scalar4 *d_postype,
                                      int3 *d_image,
                                      const unsigned int N,
                                      const BoxDim box,
                                      const Scalar3 shift)
    {
    // identify the active cell that this thread handles
    unsigned int my_pidx = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread is inactive if it indexes past the end of the particle list
    if (my_pidx >= N)
        return;

    // pull in the current position
    Scalar4 postype = d_postype[my_pidx];

    // shift the position
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    pos += shift;

    // wrap the particle back into the box
    int3 image = d_image[my_pidx];
    box.wrap(pos, image);

    // write out the new position and orientation
    d_postype[my_pidx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
    d_image[my_pidx] = image;
    }

//! Kernel driver for gpu_hpmc_shift_kernel()
cudaError_t gpu_hpmc_shift(Scalar4 *d_postype,
                           int3 *d_image,
                           const unsigned int N,
                           const BoxDim& box,
                           const Scalar3 shift,
                           const unsigned int block_size)
    {
    assert(d_postype);
    assert(d_image);

    // setup the grid to run the kernel
    dim3 threads_shift(block_size, 1, 1);
    dim3 grid_shift(N / block_size + 1, 1, 1);

    gpu_hpmc_shift_kernel<<<grid_shift, threads_shift>>>(d_postype,
                                                         d_image,
                                                         N,
                                                         box,
                                                         shift);

    // after this kernel we return control of cuda managed memory to the host
    cudaDeviceSynchronize();

    return cudaSuccess;
    }

}; // end namespace detail

} // end namespace hpmc
