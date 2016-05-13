// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander


#include "Updater.h"

#include <boost/python.hpp>
using namespace boost::python;

/*! \file Updater.cc
    \brief Defines a base class for all updaters
*/

/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data and a NULL profiler.
*/
Updater::Updater(boost::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

/*! It is useful for the user to know where computation time is spent, so all Updaters
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Updaters will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (boost::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void Updater::setProfiler(boost::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

//! Wrapper class to expose pure virtual method to python
class UpdaterWrap: public Updater, public wrapper<Updater>
    {
    public:
        //! Forwards construction on to the base class
        /*! \param sysdef parameter to forward to the base class constructor
        */
        UpdaterWrap(boost::shared_ptr<SystemDefinition> sysdef) : Updater(sysdef) { }

        //! Hanldes pure virtual Updater::update()
        /*! \param timestep parameter to forward to Updater::update()
        */
        void update(unsigned int timestep)
            {
            this->get_override("update")(timestep);
            }
    };

void export_Updater()
    {
    class_<UpdaterWrap, boost::shared_ptr<UpdaterWrap>, boost::noncopyable>("Updater", init< boost::shared_ptr<SystemDefinition> >())
    .def("update", pure_virtual(&Updater::update))
    .def("setProfiler", &Updater::setProfiler)
    ;
    }
