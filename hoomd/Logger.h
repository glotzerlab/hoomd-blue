// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Logger.h
    \brief Declares the Logger class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Analyzer.h"
#include "ClockSource.h"
#include "Compute.h"
#include "Updater.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#ifndef __LOGGER_H__
#define __LOGGER_H__

//! Logs registered quantities and offers an interface for other classes to obtain these values.
/*! \note design notes: Computes and Updaters have getProvidedLogQuantities and getLogValue. The first lists
    all quantities that the compute/updater provides (a list of strings). And getLogValue takes a string
    as an argument and returns a scalar.

    Any number of computes and updaters can be registered with the
    Logger. It will track which quantities are provided. If any
    particular quantity is registered twice, a warning is printed and
    the most recent registered source will take
    effect. setLoggedQuantities will specify a list of quantities to
    log. Every call to analyze() will result in the computes for the
    logged quantities being called.

    The removeAll method can be used to clear all registered computes and updaters. hoomd will
    removeAll() and re-register all active computes and updaters before every run()

    \ingroup analyzers
*/
class __attribute__((visibility("default"))) Logger : public Analyzer
    {
    public:
        //! Constructs a logger
        Logger(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~Logger();

        //! Registers a compute
        virtual void registerCompute(std::shared_ptr<Compute> compute);

        //! Registers an updater
        virtual void registerUpdater(std::shared_ptr<Updater> updater);

        //! Register a callback
        virtual void registerCallback(std::string name, pybind11::handle callback);

        //! Clears all registered computes and updaters
        virtual void removeAll();

        //! Selects which quantities to log
        virtual void setLoggedQuantities(const std::vector< std::string >& quantities);

        //! Returns the currently logged quantities
        std::vector<std::string> getLoggedQuantities(void)const{return m_logged_quantities;}

        //! Query the current value for a given quantity
        virtual Scalar getQuantity(const std::string& quantity, unsigned int timestep, bool use_cache);

        //! Write out the data for the current timestep
        virtual void analyze(unsigned int timestep);

        //! Get needed pdata flags
        /*! Logger may potentially log any of the optional quantities, enable all of the bits.
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            flags[pdata_flag::isotropic_virial] = 1;
            flags[pdata_flag::potential_energy] = 1;
            flags[pdata_flag::pressure_tensor] = 1;
            flags[pdata_flag::rotational_kinetic_energy] = 1;
            return flags;
            }

    protected:
        //! A map of computes indexed by logged quantity that they provide
        std::map< std::string, std::shared_ptr<Compute> > m_compute_quantities;
        //! A map of updaters indexed by logged quantity that they provide
        std::map< std::string, std::shared_ptr<Updater> > m_updater_quantities;
        //! List of callbacks
        std::map< std::string, PyObject * > m_callback_quantities;
        //! List of quantities to log
        std::vector< std::string > m_logged_quantities;
        //! Clock for the time log quantity
        ClockSource m_clk;
        //! The number of the last timestep when quantities were computed.
        unsigned int m_cached_timestep;
        //! The values of the logged quantities at the last logger update.
        std::vector< Scalar > m_cached_quantities;

    private:
        //! Helper function to get a value for a given quantity
        Scalar getValue(const std::string &quantity, int timestep);
    };

//! exports the Logger class to python
void export_Logger(pybind11::module& m);

#endif
