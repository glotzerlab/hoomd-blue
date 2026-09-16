// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SorterGPU.cc
 * \brief Defines the mpcd::SorterGPU
 */

#include "SorterGPU.h"
#include "SorterGPU.cuh"

namespace hoomd
    {
/*!
 * \param sysdef System definition
 */
mpcd::SorterGPU::SorterGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Trigger> trigger)
    : mpcd::Sorter(sysdef, trigger), m_cell_id(m_exec_conf)
    {
    m_compute_order_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_sort_sentinel"));
    m_apply_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "mpcd_sort_apply"));
    m_autotuners.insert(m_autotuners.end(), {m_compute_order_tuner, m_apply_tuner});
    }

/*!
 * \param timestep Current timestep
 *
 * Performs stream compaction on the GPU of the computed cell list into the order
 * particles appear. This will put the particles into a cell-list order, which
 * should be more friendly for other MPCD cell-based operations.
 */
void mpcd::SorterGPU::computeOrder(uint64_t timestep)
    {
    // ensure auxiliary array is correct size
    const unsigned int mpcd_N = m_mpcd_pdata->getN();
    if (mpcd_N > m_cell_id.getNumElements())
        {
        GPUArray<unsigned int> cell_id(mpcd_N, m_exec_conf);
        m_cell_id.swap(cell_id);
        }

        // compute the cell list at current timestep, guarantees owned particles are on rank
        // create an order to sort the particle indices by their cell
        {
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_id(m_cell_id,
                                            access_location::device,
                                            access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        m_compute_order_tuner->begin();
        mpcd::gpu::set_order(d_order.data,
                             d_cell_id.data,
                             d_vel.data,
                             mpcd_N,
                             m_compute_order_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_compute_order_tuner->end();

        // perform sorting
        mpcd::gpu::compute_order(d_order.data, d_cell_id.data, mpcd_N);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

/*!
 * The sorted order is applied by swapping out the alternate per-particle data
 * arrays. The communication flags are \b not sorted in MPI because by design,
 * the caller is responsible for clearing out any old flags before using them.
 */
void mpcd::SorterGPU::applyOrder() const
    {
        // apply the sorted order
        {
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(),
                                        access_location::device,
                                        access_mode::read);

        ArrayHandle<Scalar4> d_pos_alt(m_mpcd_pdata->getAltPositions(),
                                       access_location::device,
                                       access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_mpcd_pdata->getAltVelocities(),
                                       access_location::device,
                                       access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_mpcd_pdata->getAltTags(),
                                            access_location::device,
                                            access_mode::overwrite);

        m_apply_tuner->begin();
        mpcd::gpu::sort_apply(d_pos_alt.data,
                              d_vel_alt.data,
                              d_tag_alt.data,
                              d_pos.data,
                              d_vel.data,
                              d_tag.data,
                              d_order.data,
                              m_mpcd_pdata->getN(),
                              m_apply_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_apply_tuner->end();

        // copy virtual particle data if it exists
        if (m_mpcd_pdata->getNVirtual() > 0)
            {
            const unsigned int N = m_mpcd_pdata->getN();
            const unsigned int Nvirtual = m_mpcd_pdata->getNVirtual();
            cudaMemcpyAsync(d_pos_alt.data + N,
                            d_pos.data + N,
                            Nvirtual * sizeof(Scalar4),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(d_vel_alt.data + N,
                            d_vel.data + N,
                            Nvirtual * sizeof(Scalar4),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(d_tag_alt.data + N,
                            d_tag.data + N,
                            Nvirtual * sizeof(unsigned int),
                            cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            }
        }

    // swap out sorted data
    m_mpcd_pdata->swapPositions();
    m_mpcd_pdata->swapVelocities();
    m_mpcd_pdata->swapTags();
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_SorterGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::SorterGPU, mpcd::Sorter, std::shared_ptr<mpcd::SorterGPU>>(m,
                                                                                      "SorterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
