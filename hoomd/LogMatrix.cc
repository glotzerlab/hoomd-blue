// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogMatrix.cc
    \brief Defines the LogMatrix class
*/

#include "LogMatrix.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

namespace py = pybind11;

using namespace std;

/*! \param sysdef Specified for Logger, but not used directly by LogMatrix
*/
LogMatrix::LogMatrix(std::shared_ptr<SystemDefinition> sysdef)
    : Logger(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing LogMatrix: " << endl;
    }

LogMatrix::~LogMatrix(void)
    {
    m_exec_conf->msg->notice(5) << "Destroying LogMatrix" << endl;
    }

/*! \param compute The Compute to register

    After the compute is registered, all of the compute's provided log
    matrix quantities are available for logging.
*/
void LogMatrix::registerCompute(std::shared_ptr<Compute> compute)
    {
    //Call base class for single value support.
    Logger::registerCompute(compute);
    vector< string > provided_matrix_quantities = compute->getProvidedLogMatrixQuantities();

    // loop over all log matrix quantities
    for (unsigned int i = 0; i < provided_matrix_quantities.size(); i++)
        {
        // first check if this quantity is already set, printing a warning if so
        if (   m_compute_matrix_quantities.count(provided_matrix_quantities[i])
            || m_updater_matrix_quantities.count(provided_matrix_quantities[i])
            || m_callback_matrix_quantities.count(provided_matrix_quantities[i])
            )
            m_exec_conf->msg->warning() << "analyze.log: The log matrix quantity " << provided_matrix_quantities[i] <<
                 " has been registered more than once. Only the most recent registration takes effect" << endl;
        m_compute_matrix_quantities[provided_matrix_quantities[i]] = compute;
        m_exec_conf->msg->notice(6) << "analyze.log: Registering log matrix quantity " << provided_matrix_quantities[i] << endl;
        }
    }

/*! \param updater The Updater to register

    After the updater is registered, all of the updater's provided log
    matrix quantities are available for logging.
*/
void LogMatrix::registerUpdater(std::shared_ptr<Updater> updater)
    {
    //Call base class for single value support.
    Logger::registerUpdater(updater);
    vector< string > provided_matrix_quantities = updater->getProvidedLogMatrixQuantities();

    // loop over all log quantities
    for (unsigned int i = 0; i < provided_matrix_quantities.size(); i++)
        {
        // first check if this quantity is already set, printing a warning if so
        if (   m_compute_matrix_quantities.count(provided_matrix_quantities[i])
            || m_updater_matrix_quantities.count(provided_matrix_quantities[i])
            || m_callback_matrix_quantities.count(provided_matrix_quantities[i])
            )
            m_exec_conf->msg->warning() << "analyze.log: The log matrix quantity " << provided_matrix_quantities[i] <<
                " has been registered more than once. Only the most recent registration takes effect" << endl;
        m_updater_matrix_quantities[provided_matrix_quantities[i]] = updater;
        }
    }

/*! \param name Name of the matrix quantity
    \param callback Python callback that produces the quantity

    After the callback is registered \a name is available as a logger
    matrix quantity. The callback must return a
    pybind11::array_t<Scalar> (numpy array) and accept
    the time step as an argument.
*/
//A separate registerMatrixCallback function is needed, because every
//callback can either produce a single value or a matrix, but not
//both.
void LogMatrix::registerMatrixCallback(std::string name, py::object callback)
    {
    // first check if this quantity is already set, printing a warning if so
    if (   m_compute_matrix_quantities.count(name)
        || m_updater_matrix_quantities.count(name)
        || m_callback_matrix_quantities.count(name)
        )
        m_exec_conf->msg->warning() << "analyze.log: The log matrix quantity " << name <<
            " has been registered more than once. Only the most recent registration takes effect" << endl;
    m_callback_matrix_quantities[name] = callback;
    }

/*! After calling removeAll(), no quantities (matrix and single value) are registered for logging
*/
void LogMatrix::removeAll(void)
    {
    Logger::removeAll();
    m_compute_matrix_quantities.clear();
    m_updater_matrix_quantities.clear();
    //As in Logger the python callbacks are intentionally not cleared.
    }

/*! \param quantities A list of quantities to log

    When analyze() is called, each matrix quantity in the list will, in order, be requested
    from the matching registered compute or updater.
*/
void LogMatrix::setLoggedMatrixQuantities(const std::vector< std::string >& quantities)
    {
    m_logged_matrix_quantities = quantities;
    m_cached_matrix_quantities.resize(quantities.size());

    //Init with empty arrays.
    for(unsigned int i=0; i < m_cached_matrix_quantities.size(); i++)
        {
        unsigned char tmp[] = {0};
        m_cached_matrix_quantities[i] = py::array(0,tmp);
        }
    }

/*! \param timestep Time step
*/
void LogMatrix::analyze(unsigned int timestep)
    {
    Logger::analyze(timestep);

    //Cache all matrices
    for(unsigned int i=0; i < m_logged_matrix_quantities.size(); i++)
        m_cached_matrix_quantities[i] = this->getMatrix( m_logged_matrix_quantities[i],timestep );

    }


/*! \param quantity Matrix to get
    \param timestep Time step to compute value for (needed for Compute classes)
*/
py::array LogMatrix::getMatrix(const std::string &quantity, unsigned int timestep)
    {
    // check to see if the quantity exists in the compute list
    if (m_compute_matrix_quantities.count(quantity))
        {
        // update the compute
        m_compute_matrix_quantities[quantity]->compute(timestep);
        // get the log value
        py::array tmp = m_compute_matrix_quantities[quantity]->getLogMatrix(quantity, timestep);
        if(!tmp)
            {
            m_exec_conf->msg->warning() << "analyze.log: Log matrix callback "
                                        << quantity << " no matrix obtainable from Compute." << endl;
            }
        return tmp;
        }
    // check to see if the quantity exists in the updaters list
    else if (m_updater_matrix_quantities.count(quantity))
        {
        // get the log value
        py::array tmp = m_updater_matrix_quantities[quantity]->getLogMatrix(quantity, timestep);
        if(!tmp)
            {
            m_exec_conf->msg->warning() << "analyze.log: Log matrix callback "
                                        << quantity << " no matrix obtainable from Updater." << endl;
            }
        return tmp;
        }
    else if (m_callback_matrix_quantities.count(quantity))
        {
        // get a quantity from a callback
        try
            {
            py::object rv = m_callback_matrix_quantities[quantity](timestep);
            py::object extracted_rv = rv.cast<py::object >();
            return extracted_rv;
            }
        catch (const py::cast_error&)
            {
            m_exec_conf->msg->warning() << "analyze.log: Log matrix callback "
                                        << quantity << " no matrix obtainable from callback." << endl;
            return py::object();
            }
        }
    else
        {
        m_exec_conf->msg->warning() << "analyze.log: Log matrix callback "
                                    << quantity << " no matrix obtainable." << endl;
        return py::object();
        }
    }

/*! \param quantity Matrix to get
    \param timestep Time step to compute value for (needed for Compute classes)
*/
py::array LogMatrix::getMatrixQuantity(const std::string &quantity, unsigned int timestep)
    {
    for(unsigned int i=0; i< m_logged_matrix_quantities.size(); i++)
        if( m_logged_matrix_quantities[i] == quantity )
            return m_cached_matrix_quantities[i];
    m_exec_conf->msg->error() << "Matrix quantity " << quantity << " unknow to analyzer Unable to return."<<endl;
    unsigned char tmp[] = {0};
    return py::array(0,tmp);
    }


void export_LogMatrix(py::module& m)
    {
    py::class_<LogMatrix, std::shared_ptr<LogMatrix> >(m,"LogMatrix", py::base<Logger>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    .def("setLoggedMatrixQuantities", &LogMatrix::setLoggedMatrixQuantities)
    .def("getLoggedMatrixQuantities", &LogMatrix::getLoggedMatrixQuantities)
    .def("registerMatrixCallback", &LogMatrix::registerMatrixCallback)
    .def("getMatrixQuantity", &LogMatrix::getMatrixQuantity, py::return_value_policy::take_ownership)
    ;
    }
