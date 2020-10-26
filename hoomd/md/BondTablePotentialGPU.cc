// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "BondTablePotentialGPU.h"

namespace py = pybind11;
#include <stdexcept>

/*! \file BondTablePotentialGPU.cc
    \brief Defines the BondTablePotentialGPU class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
BondTablePotentialGPU::BondTablePotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     unsigned int table_width,
                                     const std::string& log_suffix)
    : BondTablePotential(sysdef, table_width, log_suffix)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondTablePotentialGPU" << endl;

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a BondTableForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing BondTableForceComputeGPU");
        }

     // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "table_bond", this->m_exec_conf));
    }

BondTablePotentialGPU::~BondTablePotentialGPU()
        {
        m_exec_conf->msg->notice(5) << "Destroying BondTablePotentialGPU" << endl;
        }

/*! \post The table based forces are computed for the given timestep.

\param timestep specifies the current time step of the simulation

Calls gpu_compute_bondtable_forces to do the leg work
*/
void BondTablePotentialGPU::computeForces(unsigned int timestep)
    {

    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "Bond Table");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> d_tables(m_tables, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

        {
        // Access the bond table for reading
        ArrayHandle<BondData::members_t> d_gpu_bondlist(this->m_bond_data->getGPUTable(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int > d_gpu_n_bonds(this->m_bond_data->getNGroupsArray(), access_location::device, access_mode::read);
        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::overwrite);


        // run the kernel on all GPUs in parallel
        m_tuner->begin();
        gpu_compute_bondtable_forces(d_force.data,
                             d_virial.data,
                             m_virial.getPitch(),
                             m_pdata->getN(),
                             d_pos.data,
                             box,
                             d_gpu_bondlist.data,
                             m_bond_data->getGPUTableIndexer().getW(),
                             d_gpu_n_bonds.data,
                             m_bond_data->getNTypes(),
                             d_tables.data,
                             d_params.data,
                             m_table_width,
                             m_table_value,
                             d_flags.data,
                             m_tuner->getParam());
        }


    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0])
            {
            m_exec_conf->msg->errorAllRanks() << endl << "Table bond out of bounds" << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_BondTablePotentialGPU(py::module& m)
    {
    py::class_<BondTablePotentialGPU, std::shared_ptr<BondTablePotentialGPU> >(m, "BondTablePotentialGPU", py::base<BondTablePotential>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                            unsigned int,
                            const std::string& >())
                            ;
    }
