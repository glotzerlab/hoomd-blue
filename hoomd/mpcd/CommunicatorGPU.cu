// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CommunicatorGPU.cu
 * \brief Implementation of communication algorithms on the GPU
 */

#ifdef ENABLE_MPI
#include "CommunicatorGPU.cuh"

#include "CommunicatorUtilities.h"
#include "ReductionOperators.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#pragma GCC diagnostic pop

#include <cub/device/device_reduce.cuh>

namespace hoomd
    {
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
__global__ void
stage_particles(unsigned int* d_comm_flag, const Scalar4* d_pos, unsigned int N, const BoxDim box)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar3 f = box.makeFraction(pos);

    unsigned int flags = 0;
    if (f.x >= Scalar(1.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::east);
    else if (f.x < Scalar(0.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::west);

    if (f.y >= Scalar(1.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::north);
    else if (f.y < Scalar(0.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::south);

    if (f.z >= Scalar(1.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::up);
    else if (f.z < Scalar(0.0))
        flags |= static_cast<unsigned int>(mpcd::detail::send_mask::down);

    d_comm_flag[idx] = flags;
    }
    } // end namespace kernel

//! Functor to select a particle for migration
struct get_migrate_key : public thrust::unary_function<const unsigned int, unsigned int>
    {
    const uint3 my_pos;             //!< My domain decomposition position
    const Index3D di;               //!< Domain indexer
    const unsigned int mask;        //!< Mask of allowed directions
    const unsigned int* cart_ranks; //!< Rank lookup table

    //! Constructor
    /*!
     * \param _my_pos Domain decomposition position
     * \param _di Domain indexer
     * \param _mask Mask of allowed directions
     * \param _cart_ranks Rank lookup table
     */
    get_migrate_key(const uint3 _my_pos,
                    const Index3D _di,
                    const unsigned int _mask,
                    const unsigned int* _cart_ranks)
        : my_pos(_my_pos), di(_di), mask(_mask), cart_ranks(_cart_ranks)
        {
        }

    //! Generate key for a sent particle
    /*!
     * \param element Particle data being sent
     */
    __device__ __forceinline__ unsigned int operator()(const mpcd::detail::pdata_element& element)
        {
        const unsigned int flags = element.comm_flag;
        int ix, iy, iz;
        ix = iy = iz = 0;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::east))
            && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::east)))
            ix = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::west))
                 && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::west)))
            ix = -1;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::north))
            && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::north)))
            iy = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::south))
                 && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::south)))
            iy = -1;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::up))
            && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::up)))
            iz = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::down))
                 && (mask & static_cast<unsigned int>(mpcd::detail::send_mask::down)))
            iz = -1;

        int i = my_pos.x;
        int j = my_pos.y;
        int k = my_pos.z;

        i += ix;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        j += iy;
        if (j == (int)di.getH())
            j = 0;
        else if (j < 0)
            j += di.getH();

        k += iz;
        if (k == (int)di.getD())
            k = 0;
        else if (k < 0)
            k += di.getD();

        return cart_ranks[di(i, j, k)];
        }
    };

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
cudaError_t mpcd::gpu::stage_particles(unsigned int* d_comm_flag,
                                       const Scalar4* d_pos,
                                       const unsigned int N,
                                       const BoxDim& box,
                                       const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stage_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::stage_particles<<<grid, run_block_size>>>(d_comm_flag, d_pos, N, box);

    return cudaSuccess;
    }

/*!
 * \param d_sendbuf Particle data buffer to sort
 * \param d_neigh_send Neighbor ranks that particles are being sent to (output)
 * \param d_num_send Number of particles being sent to each neighbor
 * \param d_tmp_keys Temporary array (size \a Nsend) used for sorting
 * \param grid_pos Grid position of the rank
 * \param di Domain decomposition indexer
 * \param mask Sending mask for the current stage
 * \param d_cart_ranks Cartesian array of domains
 * \param Nsend Number of particles in send buffer
 *
 * \returns The number of unique neighbor ranks to send to
 *
 * The communication flags in \a d_sendbuf are first transformed into a destination
 * rank (see mpcd::gpu::get_migrate_key). The send buffer is then sorted using
 * the destination rank as the key. Run-length encoding is then performed to
 * determine the number of particles going to each destination rank, and how
 * many ranks will be sent to.
 */
size_t mpcd::gpu::sort_comm_send_buffer(mpcd::detail::pdata_element* d_sendbuf,
                                        unsigned int* d_neigh_send,
                                        unsigned int* d_num_send,
                                        unsigned int* d_tmp_keys,
                                        const uint3 grid_pos,
                                        const Index3D& di,
                                        const unsigned int mask,
                                        const unsigned int* d_cart_ranks,
                                        const unsigned int Nsend)
    {
    // transform extracted communication flags into destination rank
    thrust::device_ptr<mpcd::detail::pdata_element> sendbuf(d_sendbuf);
    thrust::device_ptr<unsigned int> keys(d_tmp_keys);
    thrust::transform(sendbuf,
                      sendbuf + Nsend,
                      keys,
                      mpcd::gpu::get_migrate_key(grid_pos, di, mask, d_cart_ranks));

    // sort the destination ranks
    thrust::sort_by_key(keys, keys + Nsend, sendbuf);

    // run length encode to get the number going to each rank
    thrust::device_ptr<unsigned int> neigh_send(d_neigh_send);
    thrust::device_ptr<unsigned int> num_send(d_num_send);
    size_t num_neigh = thrust::reduce_by_key(keys,
                                             keys + Nsend,
                                             thrust::constant_iterator<int>(1),
                                             neigh_send,
                                             num_send)
                           .first
                       - neigh_send;

    return num_neigh;
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
void mpcd::gpu::reduce_comm_flags(unsigned int* d_req_flags,
                                  void* d_tmp,
                                  size_t& tmp_bytes,
                                  const unsigned int* d_comm_flags,
                                  const unsigned int N)
    {
    mpcd::ops::BitwiseOr bit_or;
    cub::DeviceReduce::Reduce(d_tmp,
                              tmp_bytes,
                              d_comm_flags,
                              d_req_flags,
                              N,
                              bit_or,
                              (unsigned int)0);
    }

namespace mpcd
    {
namespace gpu
    {
//! Wrap a particle in a pdata_element
struct wrap_particle_op
    : public thrust::unary_function<const mpcd::detail::pdata_element, mpcd::detail::pdata_element>
    {
    const BoxDim box; //!< The box for which we are applying boundary conditions

    //! Constructor
    /*!
     * \param _box Shifted simulation box for wrapping
     */
    wrap_particle_op(const BoxDim _box) : box(_box) { }

    //! Wrap position information inside particle data element
    /*!
     * \param p Particle data element
     * \returns The particle data element with wrapped coordinates
     */
    __device__ mpcd::detail::pdata_element operator()(const mpcd::detail::pdata_element p)
        {
        mpcd::detail::pdata_element ret = p;
        int3 image = make_int3(0, 0, 0);
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
                               mpcd::detail::pdata_element* d_in,
                               const BoxDim& box)
    {
    // Wrap device ptr
    thrust::device_ptr<mpcd::detail::pdata_element> in_ptr(d_in);

    // Apply box wrap to input buffer
    thrust::transform(in_ptr, in_ptr + n_recv, in_ptr, mpcd::gpu::wrap_particle_op(box));
    }
    } // end namespace hoomd

#endif // ENABLE_MPI
