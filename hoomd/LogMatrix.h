// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogMatrix.h
    \brief Declares the LogMatrix class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Logger.h"
#include "Analyzer.h"
#include "Compute.h"
#include "Updater.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#ifndef __LOGMATRIX_H__
#define __LOGMATRIX_H__

//! Works like the base class Logger, but handles matrix quantities in addition to scalar once.
/*!
    \ingroup analyzers
*/
class LogMatrix : public Logger
    {
    public:
        //! Constructs a logMatrix
        LogMatrix(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~LogMatrix(void);

        //! Registers a compute
        virtual void registerCompute(std::shared_ptr<Compute> compute);

        //! Registers an updater
        virtual void registerUpdater(std::shared_ptr<Updater> updater);

        //! Register a callback for matrix quantities
        virtual void registerMatrixCallback(std::string name, pybind11::object callback);

        //! Clears all registered computes and updaters
        virtual void removeAll(void);

        //! Selects which matrix quantities to log
        virtual void setLoggedMatrixQuantities(const std::vector< std::string >& quantities);

        //! return all currently logged matrix quantities.
        std::vector<std::string> getLoggedMatrixQuantities(void)const{return m_logged_matrix_quantities;}

        //! Query the cached matrix for a given quantity.
        virtual pybind11::array getMatrixQuantity(const std::string& quantity, unsigned int timestep);

        //! Cache the data for the current timestep
        virtual void analyze(unsigned int timestep);

    protected:
        //! A map of computes indexed by logged matrix quantity that they provide
        std::map< std::string, std::shared_ptr<Compute> > m_compute_matrix_quantities;
        //! A map of updaters indexed by logged matrix quantity that they provide
        std::map< std::string, std::shared_ptr<Updater> > m_updater_matrix_quantities;
        //! List of callbacks
        std::map< std::string, pybind11::object > m_callback_matrix_quantities;
        //! List of matrix quantities to log
        std::vector< std::string > m_logged_matrix_quantities;
        //! Cached matrix quantities
        std::vector< pybind11::array > m_cached_matrix_quantities;
    private:
        //! Obtain the matrix quantities for caching.
        virtual pybind11::array getMatrix(const std::string& quantity, unsigned int timestep);
    };

//! exports the LogMatrix class to python
void export_LogMatrix(pybind11::module& m);

#endif
