// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

/*! \file CGCMMAngleForceComputeGPU.cc
    \brief Defines CGCMMAngleForceComputeGPU
*/



#include "CGCMMAngleForceComputeGPU.h"

namespace py = pybind11;

using namespace std;

/*! \param sysdef System to compute angle forces on
*/
CGCMMAngleForceComputeGPU::CGCMMAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : CGCMMAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a CGCMMAngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing CGCMMAngleForceComputeGPU");
        }

    prefact[0] = Scalar(0.0);
    prefact[1] = Scalar(6.75);
    prefact[2] = Scalar(2.59807621135332);
    prefact[3] = Scalar(4.0);

    cgPow1[0]  = Scalar(0.0);
    cgPow1[1]  = Scalar(9.0);
    cgPow1[2]  = Scalar(12.0);
    cgPow1[3]  = Scalar(12.0);

    cgPow2[0]  = Scalar(0.0);
    cgPow2[1]  = Scalar(6.0);
    cgPow2[2]  = Scalar(4.0);
    cgPow2[3]  = Scalar(6.0);

    // allocate and zero device memory
    GPUArray<Scalar2> params (m_CGCMMAngle_data->getNTypes(),m_exec_conf);
    m_params.swap(params);
    GPUArray<Scalar2> CGCMMsr(m_CGCMMAngle_data->getNTypes(),m_exec_conf);
    m_CGCMMsr.swap(CGCMMsr);
    GPUArray<Scalar4> CGCMMepow(m_CGCMMAngle_data->getNTypes(),m_exec_conf);
    m_CGCMMepow.swap(CGCMMepow);

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "cgcmm_angle", this->m_exec_conf));
    }

CGCMMAngleForceComputeGPU::~CGCMMAngleForceComputeGPU()
    {
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation
    \param cg_type the type of course grained angle we are using
    \param eps the well depth
    \param sigma the particle radius

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void CGCMMAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma)
    {
    CGCMMAngleForceCompute::setParams(type, K, t_0, cg_type, eps, sigma);

    const Scalar myPow1 = cgPow1[cg_type];
    const Scalar myPow2 = cgPow2[cg_type];
    const Scalar myPref = prefact[cg_type];

    Scalar my_rcut = sigma*exp(Scalar(1.0)/(myPow1-myPow2)*log(myPow1/myPow2));

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_CGCMMsr(m_CGCMMsr, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_CGCMMepow(m_CGCMMepow, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, t_0);
    h_CGCMMsr.data[type] = make_scalar2(sigma, my_rcut);
    h_CGCMMepow.data[type] = make_scalar4(eps, myPow1, myPow2, myPref);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_CGCMM_angle_forces to do the dirty work.
*/
void CGCMMAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "CGCMM Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    //Not necessary - force and virial are zeroed in the kernel
    //m_force.memclear();
    //m_virial.memclear();
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_CGCMMsr(m_CGCMMsr, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_CGCMMepow(m_CGCMMepow, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_CGCMMAngle_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_CGCMMAngle_data->getGPUPosTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_CGCMMAngle_data->getNGroupsArray(), access_location::device, access_mode::read);

    // run the kernel
    m_tuner->begin();
    gpu_compute_CGCMM_angle_forces(d_force.data,
                                   d_virial.data,
                                   m_virial.getPitch(),
                                   m_pdata->getN(),
                                   d_pos.data,
                                   box,
                                   d_gpu_anglelist.data,
                                   d_gpu_angle_pos_list.data,
                                   m_CGCMMAngle_data->getGPUTableIndexer().getW(),
                                   d_gpu_n_angles.data,
                                   d_params.data,
                                   d_CGCMMsr.data,
                                   d_CGCMMepow.data,
                                   m_CGCMMAngle_data->getNTypes(),
                                   m_tuner->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_CGCMMAngleForceComputeGPU(py::module& m)
    {
    py::class_<CGCMMAngleForceComputeGPU, std::shared_ptr<CGCMMAngleForceComputeGPU> >(m, "CGCMMAngleForceComputeGPU", py::base<CGCMMAngleForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
