// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "TablePotentialGPU.h"

namespace py = pybind11;
#include <stdexcept>

/*! \file TablePotentialGPU.cc
    \brief Defines the TablePotentialGPU class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TablePotentialGPU::TablePotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<NeighborList> nlist,
                                     unsigned int table_width,
                                     const std::string& log_suffix)
    : TablePotential(sysdef, nlist, table_width, log_suffix)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TableForceComputeGPUwith no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing TableForceComputeGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "pair_table", this->m_exec_conf));
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation

Calls gpu_compute_table_forces to do the leg work
*/
void TablePotentialGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "Table pair");

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error() << "TablePotentialGPU cannot handle a half neighborlist" << endl;
        throw runtime_error("Error computing forces in TablePotentialGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> d_tables(m_tables, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    this->m_exec_conf->beginMultiGPU();

    // run the kernel on all GPUs in parallel
    m_tuner->begin();
    gpu_compute_table_forces(d_force.data,
                             d_virial.data,
                             m_virial.getPitch(),
                             m_pdata->getN(),
                             m_pdata->getNGhosts(),
                             d_pos.data,
                             box,
                             d_n_neigh.data,
                             d_nlist.data,
                             d_head_list.data,
                             d_tables.data,
                             d_params.data,
                             this->m_nlist->getNListArray().getPitch(),
                             m_ntypes,
                             m_table_width,
                             m_tuner->getParam(),
                             m_pdata->getGPUPartition());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    this->m_exec_conf->endMultiGPU();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_TablePotentialGPU(py::module& m)
    {
    py::class_<TablePotentialGPU, std::shared_ptr<TablePotentialGPU> >(m, "TablePotentialGPU", py::base<TablePotential>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                                std::shared_ptr<NeighborList>,
                                unsigned int,
                                const std::string& >())
                                ;
    }
