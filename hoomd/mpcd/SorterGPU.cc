// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SorterGPU.cc
 * \brief Defines the mpcd::SorterGPU
 */

#include "SorterGPU.h"
#include "SorterGPU.cuh"

/*!
 * \param sysdata MPCD system data
 */
mpcd::SorterGPU::SorterGPU(std::shared_ptr<mpcd::SystemData> sysdata)
    : mpcd::Sorter(sysdata), m_tmp_storage(m_exec_conf), m_compact_flag(m_exec_conf)
    {
    m_sentinel_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_sort_sentinel", m_exec_conf));
    m_reverse_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_sort_reverse", m_exec_conf));
    m_apply_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_sort_apply", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 *
 * Performs stream compaction on the GPU of the computed cell list into the order
 * particles appear. This will put the particles into a cell-list order, which
 * should be more friendly for other MPCD cell-based operations.
 */
void mpcd::SorterGPU::computeOrder(unsigned int timestep)
    {
    if (m_prof) m_prof->pop(m_exec_conf);
    // compute the cell list at current timestep, guarantees owned particles are on rank
    m_cl->compute(timestep);
    if (m_prof) m_prof->push(m_exec_conf,"MPCD sort");

    // fill the empty cell list entries with a sentinel larger than number of MPCD particles
        {
        ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_cell_np(m_cl->getCellSizeArray(), access_location::device, access_mode::read);

        m_sentinel_tuner->begin();
        mpcd::gpu::sort_set_sentinel(d_cell_list.data,
                                     d_cell_np.data,
                                     m_cl->getCellListIndexer(),
                                     0xffffffff,
                                     m_sentinel_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_sentinel_tuner->end();
        }

    // use CUB to select out the indexes of MPCD particles
        {
        // size the required temporary storage for compaction
        ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::overwrite);
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        mpcd::gpu::sort_cell_compact(d_order.data,
                                     m_compact_flag.getDeviceFlags(),
                                     d_tmp_storage,
                                     tmp_storage_bytes,
                                     d_cell_list.data,
                                     m_cl->getCellListIndexer().getNumElements(),
                                     m_mpcd_pdata->getN());

        // resize temporary storage as requested
        m_tmp_storage.resize(tmp_storage_bytes);

        // perform the compaction
        ArrayHandle<unsigned char> tmp_storage_handle(m_tmp_storage, access_location::device, access_mode::overwrite);
        d_tmp_storage = static_cast<void*>(tmp_storage_handle.data);
        mpcd::gpu::sort_cell_compact(d_order.data,
                                     m_compact_flag.getDeviceFlags(),
                                     d_tmp_storage,
                                     tmp_storage_bytes,
                                     d_cell_list.data,
                                     m_cl->getCellListIndexer().getNumElements(),
                                     m_mpcd_pdata->getN());

        // in debug mode, the number of elements in the compaction should be exactly the number of MPCD particles
        assert(m_compact_flag.readFlags() == m_mpcd_pdata->getN());
        }

    // fill out the reverse ordering map
        {
        ArrayHandle<unsigned int> d_order(m_order, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rorder(m_rorder, access_location::device, access_mode::overwrite);

        m_reverse_tuner->begin();
        mpcd::gpu::sort_gen_reverse(d_rorder.data,
                                    d_order.data,
                                    m_mpcd_pdata->getN(),
                                    m_reverse_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
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

        ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(), access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_pos_alt(m_mpcd_pdata->getAltPositions(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_mpcd_pdata->getAltVelocities(), access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_mpcd_pdata->getAltTags(), access_location::device, access_mode::overwrite);

        m_apply_tuner->begin();
        mpcd::gpu::sort_apply(d_pos_alt.data,
                              d_vel_alt.data,
                              d_tag_alt.data,
                              d_pos.data,
                              d_vel.data,
                              d_tag.data,
                              d_order.data,
                              m_mpcd_pdata->getN(),
                              m_apply_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_apply_tuner->end();
        }

    // swap out sorted data
    m_mpcd_pdata->swapPositions();
    m_mpcd_pdata->swapVelocities();
    m_mpcd_pdata->swapTags();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SorterGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::SorterGPU, std::shared_ptr<mpcd::SorterGPU> >(m, "SorterGPU", py::base<mpcd::Sorter>())
        .def(py::init< std::shared_ptr<mpcd::SystemData> >())
        ;
    }
