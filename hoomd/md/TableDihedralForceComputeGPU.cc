// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TableDihedralForceComputeGPU.h"

#include <stdexcept>

/*! \file TableDihedralForceComputeGPU.cc
    \brief Defines the TableDihedralForceComputeGPU class
*/

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param table_width Width the tables will be in memory
*/
TableDihedralForceComputeGPU::TableDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                           unsigned int table_width)
    : TableDihedralForceCompute(sysdef, table_width)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a BondTableForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing BondTableForceComputeGPU");
        }

    // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "table_dihedral"));
    m_autotuners.push_back(m_tuner);
    }

/*! \post The table based forces are computed for the given timestep.

\param timestep specifies the current time step of the simulation

Calls gpu_compute_bondtable_forces to do the leg work
*/
void TableDihedralForceComputeGPU::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    // access the table data
    ArrayHandle<Scalar2> d_tables(m_tables, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

        {
        // Access the dihedral data for reading
        ArrayHandle<group_storage<4>> d_gpu_dihedrallist(m_dihedral_data->getGPUTable(),
                                                         access_location::device,
                                                         access_mode::read);
        ArrayHandle<unsigned int> d_gpu_n_dihedrals(m_dihedral_data->getNGroupsArray(),
                                                    access_location::device,
                                                    access_mode::read);
        ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
                                                   access_location::device,
                                                   access_mode::read);

        // run the kernel on all GPUs in parallel
        m_tuner->begin();
        kernel::gpu_compute_table_dihedral_forces(d_force.data,
                                                  d_virial.data,
                                                  m_virial.getPitch(),
                                                  m_pdata->getN(),
                                                  d_pos.data,
                                                  box,
                                                  d_gpu_dihedrallist.data,
                                                  d_dihedrals_ABCD.data,
                                                  m_dihedral_data->getGPUTableIndexer().getW(),
                                                  d_gpu_n_dihedrals.data,
                                                  d_tables.data,
                                                  m_table_width,
                                                  m_table_value,
                                                  m_tuner->getParam()[0]);
        }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    m_tuner->end();
    }

namespace detail
    {
void export_TableDihedralForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TableDihedralForceComputeGPU,
                     TableDihedralForceCompute,
                     std::shared_ptr<TableDihedralForceComputeGPU>>(m,
                                                                    "TableDihedralForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
