
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallDistanceForceComputeGPU.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file WallDistanceForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating shear forces forces on the GPU. Used by
   WallDistanceForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for setting shear force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_force particle force on device
    \param orientationLink check if particle orientation is linked to shear force vector
*/
__global__ void gpu_compute_wall_distance_force_set_forces_kernel(const unsigned int group_size,
                                                             unsigned int* d_index_array,
                                                             Scalar4* d_force,
                                                             const Scalar4* d_pos,
                                                             const Scalar k,
                                                             const Scalar R,
                                                             const unsigned int N)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);

    Scalar norm = fast::sqrt(posidx.x*posidx.x+posidx.y*posidx.y);

    Scalar dist = R - norm;

    vec3<Scalar> fi(0, 0, 0);
    fi.x = k*dist*posidx.x/norm;
    fi.y = k*dist*posidx.y/norm;
    d_force[idx] = vec_to_scalar4(fi, 0);
    }

hipError_t gpu_compute_wall_distance_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 const Scalar4* d_pos,
                                                 const Scalar k,
                                                 const Scalar R,
                                                 const unsigned int N,
                                                 unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_wall_distance_force_set_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_index_array,
                       d_force,
                       d_pos,
                       k,
                       R,
                       N);
    return hipSuccess;
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
