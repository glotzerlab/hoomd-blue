// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file CellListGPU.cc
    \brief Defines CellListGPU
*/

#include "CellListGPU.h"
#include "CellListGPU.cuh"

namespace py = pybind11;

using namespace std;

/*! \param sysdef system to compute the cell list of
*/
CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef)
    : CellList(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a CellListGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing CellListGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "cell_list", this->m_exec_conf));
    }

void CellListGPU::computeCellList()
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getBox();

    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cell_size, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tdb(m_tdb, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_cell_orientation(m_orientation, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_cell_idx(m_idx, access_location::device, access_mode::overwrite);


    // autotune block sizes
    m_tuner->begin();
    gpu_compute_cell_list(d_cell_size.data,
                          d_xyzf.data,
                          d_tdb.data,
                          d_cell_orientation.data,
                          d_cell_idx.data,
                          m_conditions.getDeviceFlags(),
                          d_pos.data,
                          d_orientation.data,
                          d_charge.data,
                          d_diameter.data,
                          d_body.data,
                          m_pdata->getN(),
                          m_pdata->getNGhosts(),
                          m_Nmax,
                          m_flag_charge,
                          m_flag_type,
                          box,
                          m_cell_indexer,
                          m_cell_list_indexer,
                          getGhostWidth(),
                          m_tuner->getParam());
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_sort_cell_list)
        {
        ScopedAllocation<uint2> d_sort_idx(m_exec_conf->getCachedAllocator(), m_cell_list_indexer.getNumElements());
        ScopedAllocation<unsigned int> d_sort_permutation(m_exec_conf->getCachedAllocator(), m_cell_list_indexer.getNumElements());
        ScopedAllocation<unsigned int> d_cell_idx_new(m_exec_conf->getCachedAllocator(), m_idx.getNumElements());
        ScopedAllocation<Scalar4> d_xyzf_new(m_exec_conf->getCachedAllocator(), m_xyzf.getNumElements());
        ScopedAllocation<Scalar4> d_cell_orientation_new(m_exec_conf->getCachedAllocator(), m_orientation.getNumElements());
        ScopedAllocation<Scalar4> d_tdb_new(m_exec_conf->getCachedAllocator(), m_tdb.getNumElements());

        gpu_sort_cell_list(d_cell_size.data,
                           d_xyzf.data,
                           d_xyzf_new.data,
                           d_tdb.data,
                           d_tdb_new.data,
                           d_cell_orientation.data,
                           d_cell_orientation_new.data,
                           d_cell_idx.data,
                           d_cell_idx_new.data,
                           d_sort_idx.data,
                           d_sort_permutation.data,
                           m_cell_indexer,
                           m_cell_list_indexer);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_CellListGPU(py::module& m)
    {
    py::class_<CellListGPU, std::shared_ptr<CellListGPU> >(m,"CellListGPU",py::base<CellList>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
        ;
    }
