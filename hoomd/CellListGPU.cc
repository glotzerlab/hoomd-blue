// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CellListGPU.cc
    \brief Defines CellListGPU
*/

#include "CellListGPU.h"
#include "CellListGPU.cuh"

using namespace std;

namespace hoomd
    {
/*! \param sysdef system to compute the cell list of
 */
CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef) : CellList(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a CellListGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing CellListGPU");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   this->m_exec_conf,
                                   "cell_list"));

    m_autotuners.insert(m_autotuners.end(), {m_tuner});
    }

void CellListGPU::computeCellList()
    {
    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);

    BoxDim box = m_pdata->getBox();

        {
        // access the cell list data arrays
        ArrayHandle<unsigned int> d_cell_size(m_cell_size,
                                              access_location::device,
                                              access_mode::overwrite);
        ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
        ArrayHandle<uint2> d_type_body(m_type_body,
                                       access_location::device,
                                       access_mode::overwrite);
        ArrayHandle<Scalar4> d_cell_orientation(m_orientation,
                                                access_location::device,
                                                access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_idx(m_idx,
                                             access_location::device,
                                             access_mode::overwrite);

        // error conditions
        ArrayHandle<uint3> d_conditions(m_conditions,
                                        access_location::device,
                                        access_mode::overwrite);

        // reset cell list contents
        m_cell_size.zeroFill();
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_exec_conf->setDevice();

        // autotune block sizes
        m_tuner->begin();

        // compute cell list, and write to temporary arrays with multi-GPU
        gpu_compute_cell_list(d_cell_size.data,
                              d_xyzf.data,
                              d_type_body.data,
                              d_cell_orientation.data,
                              d_cell_idx.data,
                              d_conditions.data,
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
                              m_tuner->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner->end();

        if (m_sort_cell_list)
            {
            ArrayHandle<unsigned int> d_cell_size(m_cell_size,
                                                  access_location::device,
                                                  access_mode::overwrite);
            ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
            ArrayHandle<uint2> d_type_body(m_type_body,
                                           access_location::device,
                                           access_mode::overwrite);
            ArrayHandle<Scalar4> d_cell_orientation(m_orientation,
                                                    access_location::device,
                                                    access_mode::overwrite);
            ArrayHandle<unsigned int> d_cell_idx(m_idx,
                                                 access_location::device,
                                                 access_mode::overwrite);

            ScopedAllocation<uint2> d_sort_idx(m_exec_conf->getCachedAllocator(),
                                               m_cell_list_indexer.getNumElements());
            ScopedAllocation<unsigned int> d_sort_permutation(m_exec_conf->getCachedAllocator(),
                                                              m_cell_list_indexer.getNumElements());
            ScopedAllocation<unsigned int> d_cell_idx_new(m_exec_conf->getCachedAllocator(),
                                                          m_idx.getNumElements());
            ScopedAllocation<Scalar4> d_xyzf_new(m_exec_conf->getCachedAllocator(),
                                                 m_xyzf.getNumElements());
            ScopedAllocation<Scalar4> d_cell_orientation_new(m_exec_conf->getCachedAllocator(),
                                                             m_orientation.getNumElements());
            ScopedAllocation<uint2> d_type_body_new(m_exec_conf->getCachedAllocator(),
                                                    m_type_body.getNumElements());

            gpu_sort_cell_list(d_cell_size.data,
                               d_xyzf.data,
                               d_xyzf_new.data,
                               d_type_body.data,
                               d_type_body_new.data,
                               d_cell_orientation.data,
                               d_cell_orientation_new.data,
                               d_cell_idx.data,
                               d_cell_idx_new.data,
                               d_sort_idx.data,
                               d_sort_permutation.data,
                               m_cell_indexer,
                               m_cell_list_indexer);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }
    }

namespace detail
    {
void export_CellListGPU(pybind11::module& m)
    {
    pybind11::class_<CellListGPU, CellList, std::shared_ptr<CellListGPU>>(m, "CellListGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail

    } // end namespace hoomd
