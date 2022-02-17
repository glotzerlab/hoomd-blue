// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HarmonicAngleForceComputeGPU.cc
    \brief Defines HarmonicAngleForceComputeGPU
*/

#include "HarmonicAngleForceComputeGPU.h"

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute angle forces on
 */
HarmonicAngleForceComputeGPU::HarmonicAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : HarmonicAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a AngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing AngleForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params(m_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(
        new Autotuner(warp_size, 1024, warp_size, 5, 100000, "harmonic_angle", this->m_exec_conf));
    }

HarmonicAngleForceComputeGPU::~HarmonicAngleForceComputeGPU() { }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void HarmonicAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    HarmonicAngleForceCompute::setParams(type, K, t_0);

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, t_0);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_angle_forces to do the dirty work.
*/
void HarmonicAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(),
                                                      access_location::device,
                                                      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(),
                                                   access_location::device,
                                                   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(),
                                             access_location::device,
                                             access_mode::read);

    // run the kernel on the GPU
    m_tuner->begin();
    kernel::gpu_compute_harmonic_angle_forces(d_force.data,
                                              d_virial.data,
                                              m_virial.getPitch(),
                                              m_pdata->getN(),
                                              d_pos.data,
                                              box,
                                              d_gpu_anglelist.data,
                                              d_gpu_angle_pos_list.data,
                                              m_angle_data->getGPUTableIndexer().getW(),
                                              d_gpu_n_angles.data,
                                              d_params.data,
                                              m_angle_data->getNTypes(),
                                              m_tuner->getParam());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_HarmonicAngleForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<HarmonicAngleForceComputeGPU,
                     HarmonicAngleForceCompute,
                     std::shared_ptr<HarmonicAngleForceComputeGPU>>(m,
                                                                    "HarmonicAngleForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
