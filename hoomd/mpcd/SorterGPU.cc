// Copyright (c) 2009-2017 The Regents of the University of Michigan
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
    : mpcd::Sorter(sysdata)
    {
    m_apply_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_sort_apply", m_exec_conf));
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
