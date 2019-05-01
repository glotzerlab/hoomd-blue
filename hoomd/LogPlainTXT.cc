// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogPlainTXT.cc
    \brief Defines the LogPlainTXT class
*/

#include "LogPlainTXT.h"
#include "Filesystem.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;

#include <stdexcept>
#include <iomanip>
using namespace std;

/*! \param sysdef Specified for Logger, but not used directly by Logger
    \param fname File name to write the log to
    \param header_prefix String to write before the header
    \param overwrite Will overwrite an exiting file if true (default is to append)

    Constructing a logger will open the file \a fname, overwriting it when overwrite is True, and appending if
    overwrite is false.

    If \a fname is an empty string, no file is output.
*/
LogPlainTXT::LogPlainTXT(std::shared_ptr<SystemDefinition> sysdef,
                         const std::string& fname,
                         const std::string& header_prefix,
                         bool overwrite)
    : Logger(sysdef), m_delimiter("\t"), m_filename(fname), m_header_prefix(header_prefix), m_appending(!overwrite),
                        m_is_initialized(false), m_file_output(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing LogPlainTXT: " << fname << " " << header_prefix << " " << overwrite << endl;

    if (m_filename == string(""))
        m_file_output=false;
    }

void LogPlainTXT::openOutputFiles()
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

LogPlainTXT::~LogPlainTXT()
    {
    m_exec_conf->msg->notice(5) << "Destroying LogPlainTXT" << endl;
    }

/*! \param delimiter Delimiter to place between columns in the output file
*/
void LogPlainTXT::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param timestep Time step to write out data for

    Writes a single line of output to the log file with each specified quantity separated by
    the delimiter;
*/
void LogPlainTXT::analyze(unsigned int timestep)
    {
    // do nothing if we do not output to a file
    if (!m_file_output)
        return;

    //Call the base class to cache all values.
    Logger::analyze(timestep);

    if (m_prof) m_prof->push("LogPlainTXT");

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
        throw runtime_error("Error writing log file");
        }

    if (m_prof) m_prof->pop();
    }

/*! \param quantities A list of quantities to log

    When analyze() is called, each quantity in the list will, in order, be requested
    from the matching registered compute or updater and written to the file separated
    by delimiters. After all quantities are written to the file a newline is written.

    Each time setLoggedQuantities is called, a header listing the column names is also written.
*/
void LogPlainTXT::setLoggedQuantities(const std::vector< std::string >& quantities)
    {
    Logger::setLoggedQuantities(quantities);

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

void export_LogPlainTXT(py::module& m)
    {
    py::class_<LogPlainTXT, std::shared_ptr<LogPlainTXT> >(m,"LogPlainTXT", py::base<Logger>())
    .def(py::init< std::shared_ptr<SystemDefinition>, const std::string&, const std::string&, bool >())
    .def("setDelimiter", &LogPlainTXT::setDelimiter)
    ;
    }
