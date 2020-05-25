// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifdef ENABLE_HIP

#include "UpdaterGridShiftGPU.h"
#include "hoomd/RNGIdentifiers.h"

#include "IntegratorHPMCMonoGPUTypes.cuh"

namespace py = pybind11;

/*! \file UpdaterGridShiftGPU.cc
    \brief Definition of UpdaterGridShiftGPU
*/

namespace hpmc
{

UpdaterGridShiftGPU::UpdaterGridShiftGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMC> mc,
                             const unsigned int seed,
                             std::shared_ptr<RandomTrigger> trigger)
        : UpdaterGridShift(sysdef, mc, seed, trigger)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterGridShiftGPU" << std::endl;
    }

UpdaterGridShiftGPU::~UpdaterGridShiftGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterGridShiftGPU" << std::endl;
    }

/*! Perform grid shift
    \param timestep Current time step of the simulation
*/
void UpdaterGridShiftGPU::update(unsigned int timestep)
    {
    if (!m_trigger->isEligibleForExecution(timestep, *this))
        return;

    if (m_prof) m_prof->push(this->m_exec_conf, "UpdaterGridShiftGPU");

    // RNG for grid shift
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);

    // shift particles
    Scalar nominal_width = m_mc->getNominalWidth();
    Scalar3 shift = make_scalar3(0,0,0);
    hoomd::UniformDistribution<Scalar> uniform(-nominal_width/Scalar(2.0),nominal_width/Scalar(2.0));
    shift.x = uniform(rng);
    shift.y = uniform(rng);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        shift.z = uniform(rng);
        }

    if (this->m_pdata->getN() > 0)
        {
        BoxDim box = this->m_pdata->getBox();

        // access the particle data
        ArrayHandle<Scalar4> d_postype(this->m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(this->m_pdata->getImages(), access_location::device, access_mode::readwrite);

        gpu::hpmc_shift(d_postype.data,
           d_image.data,
           this->m_pdata->getN(),
           box,
           shift,
           128);
        }

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // update the particle data origin
    this->m_pdata->translateOrigin(shift);

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    this->m_mc->communicate(true);
    }

void export_UpdaterGridShiftGPU(py::module& m)
    {
   py::class_< UpdaterGridShiftGPU, UpdaterGridShift, std::shared_ptr< UpdaterGridShiftGPU > >(m, "UpdaterGridShiftGPU")
    .def(py::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr<IntegratorHPMC>,
                         const unsigned int,
                         std::shared_ptr<RandomTrigger> >())
    ;
    }

} // end namespace hpmc
#endif
