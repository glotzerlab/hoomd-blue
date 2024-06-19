// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
    : mpcd::Sorter(sysdef, trigger)
    {
    m_sentinel_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_sort_sentinel"));
    m_reverse_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                           m_exec_conf,
                                           "mpcd_sort_reverse"));
    m_apply_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "mpcd_sort_apply"));
    m_autotuners.insert(m_autotuners.end(), {m_sentinel_tuner, m_reverse_tuner, m_apply_tuner});
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
    // compute the cell list at current timestep, guarantees owned particles are on rank
    m_cl->compute(timestep);

        // fill the empty cell list entries with a sentinel larger than number of MPCD particles
        {
        ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(),
                                              access_location::device,
                                              access_mode::readwrite);
        ArrayHandle<unsigned int> d_cell_np(m_cl->getCellSizeArray(),
                                            access_location::device,
                                            access_mode::read);

        m_sentinel_tuner->begin();
        mpcd::gpu::sort_set_sentinel(d_cell_list.data,
                                     d_cell_np.data,
                                     m_cl->getCellListIndexer(),
                                     0xffffffff,
                                     m_sentinel_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_sentinel_tuner->end();
        }

        // use thrust to select out the indexes of MPCD particles
        {
        ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(),
                                              access_location::device,
                                              access_mode::read);
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::overwrite);
        const unsigned int num_select
            = mpcd::gpu::sort_cell_compact(d_order.data,
                                           d_cell_list.data,
                                           m_cl->getCellListIndexer().getNumElements(),
                                           m_mpcd_pdata->getN());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        if (num_select != m_mpcd_pdata->getN())
            {
            m_exec_conf->msg->error()
                << "Error compacting cell list for sorting, lost particles." << std::endl;
            }
        }

        // fill out the reverse ordering map
        {
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rorder(m_rorder,
                                           access_location::device,
                                           access_mode::overwrite);

        m_reverse_tuner->begin();
        mpcd::gpu::sort_gen_reverse(d_rorder.data,
                                    d_order.data,
                                    m_mpcd_pdata->getN(),
                                    m_reverse_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_reverse_tuner->end();
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
