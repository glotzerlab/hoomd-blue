// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproperForceComputeGPU.h"

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute bond forces on
 */
PeriodicImproperForceComputeGPU::PeriodicImproperForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef)
    : PeriodicImproperForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("ImproperForceComputeGPU requires a GPU device.");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "periodic_improper"));
    m_autotuners.push_back(m_tuner);
    }

PeriodicImproperForceComputeGPU::~PeriodicImproperForceComputeGPU() { }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

*/
void PeriodicImproperForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<ImproperData::members_t> d_gpu_improper_list(m_improper_data->getGPUTable(),
                                                             access_location::device,
                                                             access_mode::read);
    ArrayHandle<unsigned int> d_n_impropers(m_improper_data->getNGroupsArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_impropers_ABCD(m_improper_data->getGPUPosTable(),
                                               access_location::device,
                                               access_mode::read);

    // the improper table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<periodic_improper_params> d_params(m_params,
                                                   access_location::device,
                                                   access_mode::read);

    // run the kernel in parallel on all GPUs
    this->m_tuner->begin();
    kernel::gpu_compute_periodic_improper_forces(d_force.data,
                                                 d_virial.data,
                                                 m_virial.getPitch(),
                                                 m_pdata->getN(),
                                                 d_pos.data,
                                                 box,
                                                 d_gpu_improper_list.data,
                                                 d_impropers_ABCD.data,
                                                 m_improper_data->getGPUTableIndexer().getW(),
                                                 d_n_impropers.data,
                                                 d_params.data,
                                                 m_improper_data->getNTypes(),
                                                 this->m_tuner->getParam()[0],
                                                 this->m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();
    }

namespace detail
    {
void export_PeriodicImproperForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<PeriodicImproperForceComputeGPU,
                     PeriodicImproperForceCompute,
                     std::shared_ptr<PeriodicImproperForceComputeGPU>>(
        m,
        "PeriodicImproperForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
