// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

/*! \file HarmonicImproperForceComputeGPU.cc
    \brief Defines HarmonicImproperForceComputeGPU
*/



#include "HarmonicImproperForceComputeGPU.h"

namespace py = pybind11;
using namespace std;

/*! \param sysdef System to compute improper forces on
*/
HarmonicImproperForceComputeGPU::HarmonicImproperForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : HarmonicImproperForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ImproperForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ImproperForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params(m_improper_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "harmonic_improper", this->m_exec_conf));
    }

HarmonicImproperForceComputeGPU::~HarmonicImproperForceComputeGPU()
    {
    }

/*! \param type Type of the improper to set parameters for
    \param K Stiffness parameter for the force computation.
        \param chi Equilibrium value of the dihedral angle.

    Sets parameters for the potential of a particular improper type and updates the
    parameters on the GPU.
*/
void HarmonicImproperForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar chi)
    {
    HarmonicImproperForceCompute::setParams(type, K, chi);

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(Scalar(K), Scalar(chi));
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_improper_forces to do the dirty work.
*/
void HarmonicImproperForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "Harmonic Improper");

    ArrayHandle<ImproperData::members_t> d_gpu_dihedral_list(m_improper_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_improper_data->getNGroupsArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_improper_data->getGPUPosTable(), access_location::device, access_mode::read);

    // the improper table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel in parallel on all GPUs
    m_tuner->begin();
    gpu_compute_harmonic_improper_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         box,
                                         d_gpu_dihedral_list.data,
                                         d_dihedrals_ABCD.data,
                                         m_improper_data->getGPUTableIndexer().getW(),
                                         d_n_dihedrals.data,
                                         d_params.data,
                                         m_improper_data->getNTypes(),
                                         m_tuner->getParam());
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_HarmonicImproperForceComputeGPU(py::module& m)
    {
    py::class_<HarmonicImproperForceComputeGPU, std::shared_ptr<HarmonicImproperForceComputeGPU> >(m, "HarmonicImproperForceComputeGPU", py::base<HarmonicImproperForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
