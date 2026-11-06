// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/LeesEdwardsBoxDeformerGPU.cu
    \brief Definition of CUDA kernels for md::LeesEdwardsBoxDeformerGPU
*/

#include "LeesEdwardsBoxDeformerGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
/*! Particle remap procedure
    First update the x image index using the transformation below:
        i1' = i1 - i2*(xy' - xy)*Ly/Lx
    This preserves the particle's physical periodic image under the new lattice basis.

    Convert the particle position to fractional coordinates with respect to the flipped box.
    Wrap the fractional coordinates into the interval [0,1), updating the image flags for any
    periodic crossings.

    Convert the wrapped fractional coordinates back to Cartesian coordinates in the flipped box.
*/
__global__ void gpu_lees_edwards_remap_kernel(const unsigned int N,
                                              Scalar4* d_pos,
                                              int3* d_image,
                                              const BoxDim flipped_box,
                                              const Scalar xy)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar3 L = flipped_box.getL();
    Scalar xy_flip = flipped_box.getTiltFactorXY();

    if (idx < N)
        {
        // Transformation of x-images
        d_image[idx].x -= d_image[idx].y * static_cast<int>(std::round((xy_flip - xy) * L.y / L.x));

        // Convert position to fractional coordinates
        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);
        Scalar3 fpos = flipped_box.makeFraction(pos);

        int ix = static_cast<int>(std::floor(fpos.x));
        int iy = static_cast<int>(std::floor(fpos.y));
        int iz = static_cast<int>(std::floor(fpos.z));

        // Wrap into flipped box and update image flags
        fpos.x -= ix;
        fpos.y -= iy;
        fpos.z -= iz;

        d_image[idx].x += ix;
        d_image[idx].y += iy;
        d_image[idx].z += iz;

        // Convert back to Cartesian coordinates
        pos = flipped_box.makeCoordinates(fpos);

        d_pos[idx].x = pos.x;
        d_pos[idx].y = pos.y;
        d_pos[idx].z = pos.z;
        }
    }

hipError_t gpu_lees_edwards_remap(const unsigned int N,
                                  Scalar4* d_pos,
                                  int3* d_image,
                                  const BoxDim& flipped_box,
                                  const Scalar xy,
                                  unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_lees_edwards_remap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_lees_edwards_remap_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       d_pos,
                       d_image,
                       flipped_box,
                       xy);

    return hipSuccess;
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd
