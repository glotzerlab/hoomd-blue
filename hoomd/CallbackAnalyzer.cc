// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: csadorf,samnola

/*! \file CallbackAnalyzer.cc
    \brief Defines the CallbackAnalyzer class
*/



#include "CallbackAnalyzer.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze
    \param callback A python functor object to be used as callback
*/
CallbackAnalyzer::CallbackAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                         py::object callback)
    : Analyzer(sysdef), callback(callback)
    {
    m_exec_conf->msg->notice(5) << "Constructing CallbackAnalyzer" << endl;
    }

CallbackAnalyzer::~CallbackAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying CallbackAnalyzer" << endl;
    }

/*!\param timestep Current time step of the simulation

    analyze() will call the callback
*/
void CallbackAnalyzer::analyze(unsigned int timestep)
    {
      callback(timestep);
    }

void export_CallbackAnalyzer(py::module& m)
    {
    py::class_<CallbackAnalyzer, std::shared_ptr<CallbackAnalyzer> >(m,"CallbackAnalyzer",py::base<Analyzer>())
    .def(py::init< std::shared_ptr<SystemDefinition>, py::object>())
    ;
    }
