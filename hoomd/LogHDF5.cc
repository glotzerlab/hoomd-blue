
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogHDF5.cc
  \brief Defines the LogHDF5 class
*/

#include "LogHDF5.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;

using namespace std;

/*! \param sysdef Specified for LogMatrix, but not used directly by Logger
  \param python_analyze Python object, which gets called at the end of LogHDF5::analyze(). Accepts the timestep as argument and returns the timestep.
  Should obtain the prepared data and writes it via h5py to the file.
*/
LogHDF5::LogHDF5(std::shared_ptr<SystemDefinition> sysdef,
                 pybind11::function python_analyze)
    : LogMatrix(sysdef),
      m_python_analyze(python_analyze)
    {
    m_exec_conf->msg->notice(5) << "Constructing LogHDF5: "  << endl;
    }

LogHDF5::~LogHDF5(void)
    {
    m_exec_conf->msg->notice(5) << "Destroying LogHDF5" << endl;
    }

/*! \param timestep Time step
 */
void LogHDF5::analyze(unsigned int timestep)
    {
    //Call the base class to cache all values.
    LogMatrix::analyze(timestep);

    if (m_prof) m_prof->push("LogHDF5");

    //Prepare the non-matrix data
    auto numpy_array_buf = m_quantities_array.request();
    assert( numpy_array_buf.shape[0] == m_logged_quantities.size());
    assert( numpy_array_buf.itemsize == sizeof(Scalar));
    Scalar*const numpy_array_data = static_cast<Scalar*>(numpy_array_buf.ptr);
    //Prepare non-matrix data in a single array.
    for(unsigned int i=0; i < m_logged_quantities.size(); i++)
        {
        numpy_array_data[i] = this->getQuantity(m_logged_quantities[i],timestep,true);
        }

    //Call the python function, which manages the prepared data and writes it to disk.
    m_python_analyze(timestep);

    if (m_prof) m_prof->pop();
    }

/*! \param quantities A list of quantities to log
*/
void LogHDF5::setLoggedQuantities(const std::vector< std::string >& quantities)
    {
    Logger::setLoggedQuantities(quantities);

    m_holder_array.resize(quantities.size());
    //Create a new numpy array of the correct size.
    m_quantities_array = py::array(quantities.size(),m_holder_array.data());
    }

void export_LogHDF5(py::module& m)
    {
    py::class_<LogHDF5, std::shared_ptr<LogHDF5> >(m,"LogHDF5", py::base<LogMatrix>())
        .def(py::init< std::shared_ptr<SystemDefinition>, pybind11::function >())
        .def("get_quantity_array",&LogHDF5::getQuantitiesArray,py::return_value_policy::copy)
        ;
    }
