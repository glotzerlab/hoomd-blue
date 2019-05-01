// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogHDF5.h
    \brief Declares the LogHDF5 class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "LogMatrix.h"
#include <hoomd/extern/pybind/include/pybind11/numpy.h>
#ifndef __LOGHDF5_H__
#define __LOGHDF5_H__

//! Logs registered quantities to an hdf5 file. At least it prepares data, such that the python class can write the data.
/*!
  Because it inherits from LogMatrix, which inherits from
  Logger. This class offers access to single value variables and
  matrix quantities.

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

        //! Get numpy array containing all logged non-matrix quantities.
        pybind11::array getQuantitiesArray(void){return m_quantities_array;}

    private:
        //! python function, which is called to write the data to disk.
        pybind11::function m_python_analyze;
        //! python numpy array for all non-matrix quantities.
        pybind11::array m_quantities_array;
        //! memory space of the numpy array m_quantities_array
        std::vector<Scalar> m_holder_array;
    };

//! exports the LogHDF5 class to python
void export_LogHDF5(pybind11::module& m);

#endif
