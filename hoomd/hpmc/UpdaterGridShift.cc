// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "UpdaterGridShift.h"
#include "hoomd/RNGIdentifiers.h"

namespace py = pybind11;

/*! \file UpdaterGridShift.cc
    \brief Definition of UpdaterGridShift
*/

namespace hpmc
{

UpdaterGridShift::UpdaterGridShift(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMC> mc,
                             const unsigned int seed,
                             std::shared_ptr<RandomTrigger> trigger)
        : Updater(sysdef),
          m_mc(mc),
          m_seed(seed),
          m_trigger(trigger)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterGridShift" << std::endl;
    }

UpdaterGridShift::~UpdaterGridShift()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterGridShift" << std::endl;
    }

/*! Perform grid shift
    \param timestep Current time step of the simulation
*/
void UpdaterGridShift::update(unsigned int timestep)
    {
    if (!m_trigger->isEligibleForExecution(timestep, *this))
        return;

    if (m_prof) m_prof->push("UpdaterGridShift");

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    Scalar nominal_width = m_mc->getNominalWidth();

    // precalculate the grid shift
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoShift, this->m_seed, timestep);
    Scalar3 shift = make_scalar3(0,0,0);
    hoomd::UniformDistribution<Scalar> uniform(-nominal_width/Scalar(2.0),nominal_width/Scalar(2.0));
    shift.x = uniform(rng);
    shift.y = uniform(rng);
    if (this->m_sysdef->getNDimensions() == 3)
        {
        shift.z = uniform(rng);
        }
    auto box = this->m_pdata->getBox();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift);
        h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
        box.wrap(h_postype.data[i], h_image.data[i]);
        }
    this->m_pdata->translateOrigin(shift);

    if (m_prof) m_prof->pop();

    // migrate and exchange particles
    this->m_mc->communicate(true);
    }

void export_UpdaterGridShift(py::module& m)
    {
   py::class_< UpdaterGridShift, Updater, std::shared_ptr< UpdaterGridShift > >(m, "UpdaterGridShift")
    .def(py::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr<IntegratorHPMC>,
                         const unsigned int,
                         std::shared_ptr<RandomTrigger> >())
    ;
    }

} // end namespace hpmc
