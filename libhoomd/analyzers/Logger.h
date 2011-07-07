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

/*! \file Logger.h
    \brief Declares the Logger class
*/

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include <boost/shared_ptr.hpp>

#include "ClockSource.h"
#include "Analyzer.h"
#include "Compute.h"
#include "Updater.h"

#ifndef __LOGGER_H__
#define __LOGGER_H__

//! Logs registered quantities to a delimited file
/*! \note design notes: Computes and Updaters have getProvidedLogQuantities and getLogValue. The first lists
    all quantities that the compute/updater provides (a list of strings). And getLogValue takes a string
    as an argument and returns a scalar.

    Logger will open and overwrite its log file on construction. Any number of computes and updaters
    can be registered with the Logger. It will track which quantities are provided. If any particular
    quantity is registered twice, a warning is printed and the most recent registered source will take
    effect. setLoggedQuantities will specify a list of quantities to log. When it is called, a header
    is written to the file. Every call to analyze() will result in the computes for the logged quantities
    being called and getLogValue called for each value to produce a line in the file. If a logged quantity
    is not registered, a 0 is printed to the file and a warning to stdout.

    The removeAll method can be used to clear all registered computes and updaters. hoomd_script will
    removeAll() and re-register all active computes and updaters before every run()

    \ingroup analyzers
*/
class Logger : public Analyzer
    {
    public:
        //! Constructs a logger and opens the file
        Logger(boost::shared_ptr<SystemDefinition> sysdef,
               const std::string& fname,
               const std::string& header_prefix="",
               bool overwrite=false);
        
        //! Registers a compute
        void registerCompute(boost::shared_ptr<Compute> compute);
        
        //! Registers an updater
        void registerUpdater(boost::shared_ptr<Updater> updater);
        
        //! Clears all registered computes and updaters
        void removeAll();
        
        //! Selects which quantities to log
        void setLoggedQuantities(const std::vector< std::string >& quantities);
        
        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);
        
        //! Query the last logged value for a given quantity
        Scalar getCachedQuantity(const std::string& quantity="timestep");
        
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);
        
        //! Get needed pdata flags
        /*! Logger may potentially log any of the optional quantities, enable all of the bits.
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            flags[pdata_flag::isotropic_virial] = 1;
            flags[pdata_flag::potential_energy] = 1;
            return flags;
            }
    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;
        //! The prefix written at the beginning of the header line
        std::string m_header_prefix;
        //! Flag indicating this file is being appended to
        bool m_appending;
        //! The file we write out to
        std::ofstream m_file;
        //! A map of computes indexed by logged quantity that they provide
        std::map< std::string, boost::shared_ptr<Compute> > m_compute_quantities;
        //! A map of updaters indexed by logged quantity that they provide
        std::map< std::string, boost::shared_ptr<Updater> > m_updater_quantities;
        //! List of quantities to log
        std::vector< std::string > m_logged_quantities;
        //! Clock for the time log quantity
        ClockSource m_clk;
        //! The number of the last timestep when the logger was run.
        int cached_timestep;
        //! The values of the logged quantities at the last logger update.
        std::vector< Scalar > cached_quantities;
        
        //! Helper function to get a value for a given quantity
        Scalar getValue(const std::string &quantity, int timestep);
    };

//! exports the Logger class to python
void export_Logger();

#endif

