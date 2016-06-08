// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: csadorf,samnola

/*! \file CallbackAnalyzer.h
    \brief Declares the CallbackAnalyzer class
*/

#ifndef __CALLBACK_ANALYZER_H__
#define __CALLBACK_ANALYZER_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Analyzer.h"
#include "ParticleGroup.h"

#include <string>
#include <fstream>
#include <memory>

//! Calls a python functor object
/*! On construction, CallbackAnalyzer stores a python object to be called every analyzer period.
    The functor is expected to take the current timestep as single argument.

    \ingroup analyzers
*/
class CallbackAnalyzer : public Analyzer
    {
    public:
        //! Construct the callback analyzer
        CallbackAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                    boost::python::object callback);

        //! Destructor
        ~CallbackAnalyzer();

        //! Call the analyzer callback
        void analyze(unsigned int timestep);

    private:

        ////! The callback function to be called at each analyzer period.
        boost::python::object callback;
    };

//! Exports the CallbackAnalyzer class to python
void export_CallbackAnalyzer();

#endif
