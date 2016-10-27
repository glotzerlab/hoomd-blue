// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Logger.cc
    \brief Defines the Logger class
*/



#include "Logger.h"
#include "Filesystem.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;


#include <stdexcept>
#include <iomanip>
using namespace std;

/*! \param sysdef Specified for Analyzer, but not used directly by Logger
    \param fname File name to write the log to
    \param header_prefix String to write before the header
    \param overwrite Will overwrite an exiting file if true (default is to append)

    Constructing a logger will open the file \a fname, overwriting it when overwrite is True, and appending if
    overwrite is false.

    If \a fname is an empty string, no file is output.
*/
Logger::Logger(std::shared_ptr<SystemDefinition> sysdef,
               const std::string& fname,
               const std::string& header_prefix,
               bool overwrite)
    : Analyzer(sysdef), m_delimiter("\t"), m_filename(fname), m_header_prefix(header_prefix), m_appending(!overwrite),
                        m_is_initialized(false), m_file_output(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing Logger: " << fname << " " << header_prefix << " " << overwrite << endl;

    if (m_filename == string(""))
        m_file_output=false;
    }

void Logger::openOutputFiles()
    {
    // do nothing if we are not writing a file
    if (!m_file_output)
        return;

#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_comm)
        if (! m_exec_conf->isRoot())
            return;
#endif
    // open the file
    if (filesystem::exists(m_filename) && m_appending)
        {
        m_exec_conf->msg->notice(3) << "analyze.log: Appending log to existing file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::in | ios_base::out | ios_base::ate);
        }
    else
        {
        m_exec_conf->msg->notice(3) << "analyze.log: Creating new log in file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::out);
        m_appending = false;
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.log: Error opening log file " << m_filename << endl;
        throw runtime_error("Error initializing Logger");
        }
    }

Logger::~Logger()
    {
    m_exec_conf->msg->notice(5) << "Destroying Logger" << endl;
    }

/*! \param compute The Compute to register

    After the compute is registered, all of the compute's provided log quantities are available for
    logging.
*/
void Logger::registerCompute(std::shared_ptr<Compute> compute)
    {
    vector< string > provided_quantities = compute->getProvidedLogQuantities();

    // loop over all log quantities
    for (unsigned int i = 0; i < provided_quantities.size(); i++)
        {
        // first check if this quantity is already set, printing a warning if so
        if (   m_compute_quantities.count(provided_quantities[i])
            || m_updater_quantities.count(provided_quantities[i])
            || m_callback_quantities.count(provided_quantities[i])
            )
            m_exec_conf->msg->warning() << "analyze.log: The log quantity " << provided_quantities[i] <<
                 " has been registered more than once. Only the most recent registration takes effect" << endl;
        m_compute_quantities[provided_quantities[i]] = compute;
        m_exec_conf->msg->notice(6) << "analyze.log: Registering log quantity " << provided_quantities[i] << endl;
        }
    }

/*! \param updater The Updater to register

    After the updater is registered, all of the updater's provided log quantities are available for
    logging.
*/
void Logger::registerUpdater(std::shared_ptr<Updater> updater)
    {
    vector< string > provided_quantities = updater->getProvidedLogQuantities();

    // loop over all log quantities
    for (unsigned int i = 0; i < provided_quantities.size(); i++)
        {
        // first check if this quantity is already set, printing a warning if so
        if (   m_compute_quantities.count(provided_quantities[i])
            || m_updater_quantities.count(provided_quantities[i])
            || m_callback_quantities.count(provided_quantities[i])
            )
            m_exec_conf->msg->warning() << "analyze.log: The log quantity " << provided_quantities[i] <<
                 " has been registered more than once. Only the most recent registration takes effect" << endl;
        m_updater_quantities[provided_quantities[i]] = updater;
        }
    }

/*! \param name Name of the quantity
    \param callback Python callback that produces the quantity

    After the callback is registered \a name is available as a logger quantity. The callback must return a scalar
    value and accept the time step as an argument.
*/
void Logger::registerCallback(std::string name, py::object callback)
    {
    // first check if this quantity is already set, printing a warning if so
    if (   m_compute_quantities.count(name)
        || m_updater_quantities.count(name)
        || m_callback_quantities.count(name)
        )
    m_exec_conf->msg->warning() << "analyze.log: The log quantity " << name <<
                         " has been registered more than once. Only the most recent registration takes effect" << endl;
    m_callback_quantities[name] = callback;
    }

/*! After calling removeAll(), no quantities are registered for logging
*/
void Logger::removeAll()
    {
    m_compute_quantities.clear();
    m_updater_quantities.clear();
    //The callbacks are intentionally not cleared, because before each
    //run all compute and updaters should be cleared, but the python
    //callbacks should not be cleared for this.
    }

/*! \param quantities A list of quantities to log

    When analyze() is called, each quantitiy in the list will, in order, be requested
    from the matching registered compute or updtaer and written to the file separated
    by delimiters. After all quantities are written to the file a newline is written.

    Each time setLoggedQuantities is called, a header listing the column names is also written.
*/
void Logger::setLoggedQuantities(const std::vector< std::string >& quantities)
    {
    m_logged_quantities = quantities;

    // prepare or adjust storage for caching the logger properties.
    m_cached_timestep = -1;
    m_cached_quantities.resize(quantities.size());

#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot())
            return;
#endif

    // open output files for writing
    if (! m_is_initialized)
        openOutputFiles();

    m_is_initialized = true;

    // only write the header if this is a new file
    if (!m_appending && m_file_output)
        {
        // write out the header prefix
        m_file << m_header_prefix;

        // timestep is always output
        m_file << "timestep";
        }

    if (quantities.size() == 0)
        {
        m_exec_conf->msg->warning() << "analyze.log: No quantities specified for logging" << endl;
        return;
        }

    // only write the header if this is a new file
    if (!m_appending && m_file_output)
        {
        // only print the delimiter after the timestep if there are more quantities logged
        m_file << m_delimiter;

        // write all but the last of the quantities separated by the delimiter
        for (unsigned int i = 0; i < quantities.size()-1; i++)
            m_file << quantities[i] << m_delimiter;
        // write the last one with no delimiter after it
        m_file << quantities[quantities.size()-1] << endl;
        m_file.flush();
        }
    }


/*! \param delimiter Delimiter to place between columns in the output file
*/
void Logger::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param timestep Time step to write out data for

    Writes a single line of output to the log file with each specified quantity separated by
    the delimiter;
*/
void Logger::analyze(unsigned int timestep)
    {
    // do nothing if we do not output to a file
    if (!m_file_output)
        return;

    if (m_prof) m_prof->push("Log");

    // update info in cache for later use and for immediate output.
    for (unsigned int i = 0; i < m_logged_quantities.size(); i++)
        m_cached_quantities[i] = getValue(m_logged_quantities[i], timestep);

    m_cached_timestep = timestep;

#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_comm)
        if (! m_exec_conf->isRoot())
            {
            if (m_prof) m_prof->pop();
            return;
            }
#endif

    // The timestep is always output
    m_file << setprecision(10) << timestep;

    // quit now if there is nothing to log
    if (m_logged_quantities.size() == 0)
        {
        return;
        }

    // only print the delimiter after the timestep if there are more quantities logged
    m_file << m_delimiter;

    // write all but the last of the quantities separated by the delimiter
    for (unsigned int i = 0; i < m_logged_quantities.size()-1; i++)
        m_file << setprecision(10) << m_cached_quantities[i] << m_delimiter;
    // write the last one with no delimiter after it
    m_file << setprecision(10) << m_cached_quantities[m_logged_quantities.size()-1] << endl;
    m_file.flush();

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.log: I/O error while writing log file" << endl;
        throw runtime_error("Error writting log file");
        }

    if (m_prof) m_prof->pop();
    }

/*! \param quantity Quantity to get
*/
Scalar Logger::getQuantity(const std::string &quantity, unsigned int timestep, bool use_cache)
    {
    // update info in cache for later use
    if (!use_cache && timestep != m_cached_timestep)
        {
        for (unsigned int i = 0; i < m_logged_quantities.size(); i++)
            m_cached_quantities[i] = getValue(m_logged_quantities[i], timestep);
        m_cached_timestep = timestep;
        }

    // first see if it is the timestep number
    if (quantity == "timestep")
        {
        return Scalar(m_cached_timestep);
        }

    // check to see if the quantity exists in the compute list
    for (unsigned int i = 0; i < m_logged_quantities.size(); i++)
        if (m_logged_quantities[i] == quantity)
            return m_cached_quantities[i];

    m_exec_conf->msg->warning() << "analyze.log: Log quantity " << quantity << " is not registered, returning a value of 0" << endl;
    return Scalar(0.0);
    }

/*! \param quantity Quantity to get
    \param timestep Time step to compute value for (needed for Compute classes)
*/
Scalar Logger::getValue(const std::string &quantity, int timestep)
    {
    // first see if it is the built-in time quantity
    if (quantity == "time")
        {
        return Scalar(double(m_clk.getTime())/1e9);
        }
    // check to see if the quantity exists in the compute list
    else if (m_compute_quantities.count(quantity))
        {
        // update the compute
        m_compute_quantities[quantity]->compute(timestep);
        // get the log value
        return m_compute_quantities[quantity]->getLogValue(quantity, timestep);
        }
    // check to see if the quantity exists in the updaters list
    else if (m_updater_quantities.count(quantity))
        {
        // get the log value
        return m_updater_quantities[quantity]->getLogValue(quantity, timestep);
        }
    else if (m_callback_quantities.count(quantity))
        {
        // get a quantity from a callback
        try
            {
            py::object rv = m_callback_quantities[quantity](timestep);
            Scalar extracted_rv = rv.cast<Scalar>();
            return extracted_rv;
            }
        catch (py::cast_error)
            {
                m_exec_conf->msg->warning() << "analyze.log: Log callback " << quantity << " returned invalid value, logging 0." << endl;
                return Scalar(0.0);
            }
        }
    else
        {
        m_exec_conf->msg->warning() << "analyze.log: Log quantity " << quantity << " is not registered, logging a value of 0" << endl;
        return Scalar(0.0);
        }
    }


void export_Logger(py::module& m)
    {
    py::class_<Logger, std::shared_ptr<Logger> >(m,"Logger", py::base<Analyzer>())
    .def(py::init< std::shared_ptr<SystemDefinition>, const std::string&, const std::string&, bool >())
    .def("registerCompute", &Logger::registerCompute)
    .def("registerUpdater", &Logger::registerUpdater)
    .def("registerCallback", &Logger::registerCallback)
    .def("removeAll", &Logger::removeAll)
    .def("setLoggedQuantities", &Logger::setLoggedQuantities)
    .def("setDelimiter", &Logger::setDelimiter)
    .def("getQuantity", &Logger::getQuantity)
    ;
    }
