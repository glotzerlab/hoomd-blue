/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file Analyzer.h
    \brief Declares a base class for all analyers
*/

#ifndef __ANALYZER_H__
#define __ANALYZER_H__

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#include "Profiler.h"
#include "SystemDefinition.h"

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup analyzers Analyzers
    \brief All classes that implement the Analyzer concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Base class for analysis of particle data
/*! An Analyzer is a concept that encapsulates some process that is performed during
    the simulation with the sole purpose of outputting data to the user in some fashion.
    The results of an Analyzer can not modify the simulation in any way, that is what
    the Updater classes are for. In general, analyzers are likely to only be called every 1,000
    time steps or much less often (this value entirely at the user's discrestion).
    The System class will handle this. An Analyzer just needs to perform its calculations
    and make its output every time analyze() is called.

    By design Analyzers can reference any number of Computes while performing their
    analysis. The base class provides no methods for doing this, derived classes must
    implement the tracking of the attached Compute classes (via shared pointers)
    themselves. (it is recomenned to pass a shared pointer to the Compute
    into the constructor of the derived class).

    See \ref page_dev_info for more information

    \ingroup analyzers
*/
class Analyzer : boost::noncopyable
    {
    public:
        //! Constructs the analyzer and associates it with the ParticleData
        Analyzer(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~Analyzer() {};
        
        //! Abstract method that performs the analysis
        /*! Derived classes will implement this method to calculate their results
            \param timestep Current time step of the simulation
            */
        virtual void analyze(unsigned int timestep) = 0;
        
        //! Sets the profiler for the analyzer to use
        void setProfiler(boost::shared_ptr<Profiler> prof);
        
        //! Print some basic stats to stdout
        /*! Derived classes can optionally implement this function. A System will
            call all of the Analyzers' printStats functions at the end of a run
            so the user can see useful information 
        */
        virtual void printStats()
            {
            }
        
        //! Reset stat counters
        /*! If derived classes implement printStats, they should also implement resetStats() to clear any running 
            counters printed by printStats. System will reset the stats before any run() so that stats printed
            at the end of the run only apply to that run() alone.
        */
        virtual void resetStats()
            {
            }
        
    protected:
        const boost::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this analyzer is associated with
        const boost::shared_ptr<ParticleData> m_pdata;      //!< The particle data this analyzer is associated with
        boost::shared_ptr<Profiler> m_prof;                 //!< The profiler this analyzer is to use
    };

//! Export the Analyzer class to python
void export_Analyzer();

#endif

