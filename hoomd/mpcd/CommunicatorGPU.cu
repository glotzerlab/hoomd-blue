// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CommunicatorGPU.cu
 * \brief Implementation of communication algorithms on the GPU
 */

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"

#include "CommunicatorUtilities.h"
#include "ReductionOperators.h"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "hoomd/extern/cub/cub/device/device_reduce.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Select a particle for migration
/*!
 * \param d_comm_flag Communication flags to write out
 * \param d_pos Device array of particle positions
 * \param N Number of local particles
 * \param box Local box
 *
 * Checks for particles being out of bounds, and aggregates send flags.
 */
__global__ void stage_particles(unsigned int *d_comm_flag,
                                const Scalar4 *d_pos,
                                unsigned int N,
                                const BoxDim box)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();

    unsigned int flags = 0;
    if (pos.x >= hi.x) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::east);
    else if (pos.x < lo.x) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::west);
    if (pos.y >= hi.y) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::north);
    else if (pos.y < lo.y) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::south);
    if (pos.z >= hi.z) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::up);
    else if (pos.z < lo.z) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::down);

    d_comm_flag[idx] = flags;
    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd

/*!
 * \param d_comm_flag Communication flags to write out
 * \param d_pos Device array of particle positions
 * \param N Number of local particles
 * \param box Local box
 *
 * \returns Accumulated communication flags of all particles
 */
cudaError_t mpcd::gpu::stage_particles(unsigned int *d_comm_flag,
                                        const Scalar4 *d_pos,
                                        const unsigned int N,
                                        const BoxDim& box,
                                        const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_particles);
        max_block_size = attr.maxThreadsPerBlock;
        }
    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::stage_particles<<<grid, run_block_size>>>(d_comm_flag,
                                                                 d_pos,
                                                                 N,
                                                                 box);

    return cudaSuccess;
    }

/*!
 * \param d_req_flags Reduced requested communication flags (output)
 * \param d_tmp Temporary storage for reduction
 * \param tmp_bytes Number of temporary storage bytes requested
 * \param d_comm_flags Communication flags to reduce
 * \param N Number of local particles
 *
 * Bitwise OR reduction is performed on the communication flags to determine
 * requested migration direction.
 *
 * \note This function must be called \b twice. The first call sizes the temporary
 * arrays. The caller must then allocate the necessary temporary storage, and then
 * call again to perform the reduction.
 */
void mpcd::gpu::reduce_comm_flags(unsigned int *d_req_flags,
                                  void *d_tmp,
                                  size_t& tmp_bytes,
                                  const unsigned int *d_comm_flags,
                                  const unsigned int N)
    {
    mpcd::ops::BitwiseOr bit_or;
    cub::DeviceReduce::Reduce(d_tmp, tmp_bytes, d_comm_flags, d_req_flags, N, bit_or, (unsigned int)0);
    }

namespace mpcd
{
namespace gpu
{
//! Wrap a particle in a pdata_element
struct wrap_particle_op : public thrust::unary_function<const mpcd::detail::pdata_element, mpcd::detail::pdata_element>
    {
    const BoxDim box; //!< The box for which we are applying boundary conditions

    //! Constructor
    /*!
     * \param _box Shifted simulation box for wrapping
     */
    wrap_particle_op(const BoxDim _box)
        : box(_box)
        {
        }

    //! Wrap position information inside particle data element
    /*!
     * \param p Particle data element
     * \returns The particle data element with wrapped coordinates
     */
    __device__ mpcd::detail::pdata_element operator()(const mpcd::detail::pdata_element p)
        {
        mpcd::detail::pdata_element ret = p;
        int3 image = make_int3(0,0,0);
        box.wrap(ret.pos, image);
        return ret;
        }
     };
} // end namespace gpu
} // end namespace mpcd

/*!
 * \param n_recv Number of particles in buffer
 * \param d_in Buffer of particle data elements
 * \param box Box for which to apply boundary conditions
 */
void mpcd::gpu::wrap_particles(const unsigned int n_recv,
                               mpcd::detail::pdata_element *d_in,
                               const BoxDim& box)
    {
    // Wrap device ptr
    thrust::device_ptr<mpcd::detail::pdata_element> in_ptr(d_in);

    // Apply box wrap to input buffer
    thrust::transform(in_ptr, in_ptr + n_recv, in_ptr, mpcd::gpu::wrap_particle_op(box));
    }
#endif // ENABLE_MPI
