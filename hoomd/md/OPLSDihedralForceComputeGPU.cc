// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ksil

/*! \file OPLSDihedralForceComputeGPU.cc
    \brief Defines OPLSDihedralForceComputeGPU
*/



#include "OPLSDihedralForceComputeGPU.h"

namespace py = pybind11;

using namespace std;

/*! \param sysdef System to compute bond forces on
*/
OPLSDihedralForceComputeGPU::OPLSDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : OPLSDihedralForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating an OPLSDihedralForceComputeGPU with no GPU in execution configuration" << endl;
        throw std::runtime_error("Error initializing OPLSDihedralForceComputeGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "opls_dihedral", this->m_exec_conf));
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_opls_dihedral_forces to do the dirty work.
*/
void OPLSDihedralForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "OPLS Dihedral");

    ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(), access_location::device, access_mode::read);

    // the dihedral table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel in parallel on all GPUs
    this->m_tuner->begin();
    gpu_compute_opls_dihedral_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         box,
                                         d_gpu_dihedral_list.data,
                                         d_dihedrals_ABCD.data,
                                         m_dihedral_data->getGPUTableIndexer().getW(),
                                         d_n_dihedrals.data,
                                         d_params.data,
                                         m_dihedral_data->getNTypes(),
                                         this->m_tuner->getParam());
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_OPLSDihedralForceComputeGPU(py::module& m)
    {
    py::class_<OPLSDihedralForceComputeGPU, std::shared_ptr<OPLSDihedralForceComputeGPU> >(m, "OPLSDihedralForceComputeGPU", py::base<OPLSDihedralForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
