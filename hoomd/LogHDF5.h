// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogHDF5.h
    \brief Declares the LogHDF5 class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "LogMatrix.h"

#ifndef __LOGHDF5_H__
#define __LOGHDF5_H__

//! Logs registered quantities to an hdf5 file. At least it prepares data, such that the python class can write the data.
/*!
    \ingroup analyzers
*/
class LogHDF5 : public LogMatrix
    {
    public:
        //! Constructs a logger and opens the file
        LogHDF5(std::shared_ptr<SystemDefinition> sysdef,
                pybind11::function python_analyze);

        //! Destructor
        ~LogHDF5(void);

        //! Selects which quantities to log
        virtual void setLoggedQuantities(const std::vector< std::string >& quantities);

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        pybind11::array& getSingleValueArray(void){return m_single_value_array;}

    private:
        pybind11::function m_python_analyze;
        pybind11::array m_single_value_array;
        std::vector<Scalar> m_holder_array;
    };

//! exports the Logger class to python
void export_LogHDF5(pybind11::module& m);

#endif
