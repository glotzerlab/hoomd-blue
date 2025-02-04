// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Messenger.cc
    \brief Defines the Messenger class
*/

#include "Messenger.h"
#include "ClockSource.h"
#include "ExecutionConfiguration.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <stdlib.h>
#include <vector>

#include <assert.h>
#include <pybind11/iostream.h>
#include <sstream>
using namespace std;

namespace hoomd
    {
namespace detail
    {
#ifdef ENABLE_MPI
//! Class that supports writing to a shared log file using MPI-IO
class mpi_io : public std::streambuf
    {
    public:
    //! Constructor
    mpi_io(const MPI_Comm& mpi_comm, const std::string& filename);
    virtual ~mpi_io()
        {
        close();
        };

    //! Close the log file
    void close();

    //! \return true if file is open
    bool is_open()
        {
        return m_file_open;
        }

    //! Write a character
    virtual int overflow(int ch);

    private:
    MPI_Comm m_mpi_comm; //!< The MPI communicator
    MPI_File m_file;     //!< The file handle
    bool m_file_open;    //!< Whether the file is open
    };
#endif

    } // end namespace detail

/*! \post Warning and error streams are set to cerr
    \post The notice stream is set to cout
    \post The notice level is set to 2
    \post prefixes are "error!!!!" , "warning!!" and "notice"
*/
Messenger::Messenger(std::shared_ptr<MPIConfiguration> mpi_config) : m_mpi_config(mpi_config)
    {
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;

    m_nullstream = std::shared_ptr<detail::nullstream>(new detail::nullstream());
    m_notice_level = 2;
    m_err_prefix = "**ERROR**";
    m_warning_prefix = "*Warning*";
    m_notice_prefix = "notice";

    if (!m_mpi_config)
        {
        // create one on the fly
        m_mpi_config = std::shared_ptr<MPIConfiguration>(new MPIConfiguration());
        }

    // silence output on non-zero ranks
    assert(m_mpi_config);
    if (m_mpi_config->getRank() != 0)
        m_notice_level = 0;
    }

Messenger::Messenger(const Messenger& msg)
    {
    m_err_stream = msg.m_err_stream;
    m_warning_stream = msg.m_warning_stream;
    m_notice_stream = msg.m_notice_stream;
    m_streambuf_out = msg.m_streambuf_out;
    m_streambuf_err = msg.m_streambuf_err;
    m_nullstream = msg.m_nullstream;
    m_file_out = msg.m_file_out;
    m_file_err = msg.m_file_err;
    m_err_prefix = msg.m_err_prefix;
    m_warning_prefix = msg.m_warning_prefix;
    m_notice_prefix = msg.m_notice_prefix;
    m_notice_level = msg.m_notice_level;

    m_mpi_config = msg.m_mpi_config;
    }

Messenger& Messenger::operator=(Messenger& msg)
    {
    m_err_stream = msg.m_err_stream;
    m_warning_stream = msg.m_warning_stream;
    m_notice_stream = msg.m_notice_stream;
    m_streambuf_out = msg.m_streambuf_out;
    m_streambuf_err = msg.m_streambuf_err;
    m_nullstream = msg.m_nullstream;
    m_file_out = msg.m_file_out;
    m_file_err = msg.m_file_err;
    m_err_prefix = msg.m_err_prefix;
    m_warning_prefix = msg.m_warning_prefix;
    m_notice_prefix = msg.m_notice_prefix;
    m_notice_level = msg.m_notice_level;

    m_mpi_config = msg.m_mpi_config;

    return *this;
    }

Messenger::~Messenger()
    {
    // set pointers to NULL
    m_err_stream = NULL;
    m_warning_stream = NULL;
    m_notice_stream = NULL;
    }

/*! \returns The error stream for use in printing error messages
    \post If the error prefix is not the empty string, the message is preceded with
    "${error_prefix}: ".

    error() is intended to be used as a collective method where all MPI ranks report the same error.
   Only rank 0 will print the error.
*/
std::ostream& Messenger::error()
    {
    assert(m_err_stream);
#ifdef ENABLE_MPI
    if (m_mpi_config->getNRanks() > 1)
        {
        return *m_nullstream;
        }
#endif
    reopenPythonIfNeeded();
    if (m_err_prefix != string(""))
        *m_err_stream << m_err_prefix << ": ";
    if (m_mpi_config->getNRanks() > 1)
        *m_err_stream << " (Rank " << m_mpi_config->getRank() << "): ";
    return *m_err_stream;
    }

/*! \returns The error stream for use in printing error messages
    \post If the error prefix is not the empty string, the message is preceded with
    "${error_prefix}: (Rank N):".

    errorAllRanks() is intended to be used when only one or a small number of ranks report an error.
   All ranks that call will errorAllRanks() will write output. Callers should flush the stream.
*/
std::ostream& Messenger::errorAllRanks()
    {
    assert(m_err_stream);

    reopenPythonIfNeeded();

    // Delay so that multiple ranks calling this have a good chance of writing non-overlapping
    // messages
    Sleep(m_mpi_config->getRank() * 10);

    if (m_err_prefix != string(""))
        *m_err_stream << m_err_prefix << ": ";
    if (m_mpi_config->getNRanks() > 1)
        *m_err_stream << " (Rank " << m_mpi_config->getRank() << "): ";

    return *m_err_stream;
    }

/*! \param msg Message to print
    \sa error()
*/
void Messenger::errorStr(const std::string& msg)
    {
    error() << msg << std::flush;
    }

/*! \returns The warning stream for use in printing warning messages
    \post If the warning prefix is not the empty string, the message is preceded with
    "${warning_prefix}: ".
*/
std::ostream& Messenger::warning()
    {
    if (m_mpi_config->getRank() != 0)
        return *m_nullstream;

    assert(m_warning_stream);
    reopenPythonIfNeeded();
    if (m_warning_prefix != string(""))
        *m_warning_stream << m_warning_prefix << ": ";
    return *m_warning_stream;
    }

/*! \param msg Message to print
    \sa warning()
*/
void Messenger::warningStr(const std::string& msg)
    {
    warning() << msg << std::flush;
    ;
    }

/*! \returns The notice stream for use in printing notice messages
    \post If the notice prefix is not the empty string and the level is greater than 1 the message
   is preceded with
    "${notice_prefix}(n): ".

    If level is greater than the notice level, a null stream is returned so that the output is not
   printed.
*/
std::ostream& Messenger::notice(unsigned int level)
    {
    assert(m_notice_stream);
    if (level <= m_notice_level)
        {
        reopenPythonIfNeeded();
        if (m_notice_prefix != string("") && level > 1)
            *m_notice_stream << m_notice_prefix << "(" << level << "): ";
        return *m_notice_stream;
        }
    else
        {
        return *m_nullstream;
        }
    }

/*! Outputs the the collective notice string on the processor with rank zero, in rank order.

 \param level The notice level
 \param msg Content of the notice
 */
void Messenger::collectiveNoticeStr(unsigned int level, const std::string& msg)
    {
    std::vector<std::string> rank_notices;

#ifdef ENABLE_MPI
    gather_v(msg, rank_notices, 0, m_mpi_config->getCommunicator());
#else
    rank_notices.push_back(msg);
#endif

#ifdef ENABLE_MPI
    if (m_mpi_config->getRank() == 0)
        {
        if (rank_notices.size() > 1)
            {
            // Output notices in rank order, combining similar ones
            std::vector<std::string>::iterator notice_it;
            std::string last_msg = rank_notices[0];
            int last_output_rank = -1;
            for (notice_it = rank_notices.begin(); notice_it != rank_notices.end() + 1; notice_it++)
                {
                if (notice_it == rank_notices.end() || *notice_it != last_msg)
                    {
                    int rank = int(notice_it - rank_notices.begin());
                    // output message for accumulated ranks
                    if (last_output_rank + 1 == rank - 1)
                        notice(level)
                            << "Rank " << last_output_rank + 1 << ": " << last_msg << std::flush;
                    else
                        notice(level) << "Ranks " << last_output_rank + 1 << "-" << rank - 1 << ": "
                                      << last_msg << std::flush;

                    if (notice_it != rank_notices.end())
                        {
                        last_msg = *notice_it;
                        last_output_rank = rank - 1;
                        }
                    }
                }
            }
        else
#endif
            {
            // output without prefix
            notice(level) << rank_notices[0] << std::flush;
            }
#ifdef ENABLE_MPI
        }
#endif
    }

/*! \param level Notice level
    \param msg Message to print
    \sa notice()
*/
void Messenger::noticeStr(unsigned int level, const std::string& msg)
    {
    notice(level) << msg << std::flush;
    }

/*! \param fname File name
    The file is overwritten if it exists. If there is an error opening the file, all level's streams
   are left as is and an error() is issued.
*/
void Messenger::openFile(const std::string& fname)
    {
#ifdef ENABLE_MPI
    if (m_mpi_config->getNRanks() > 1)
        {
        // open the shared file
        std::string broadcast_fname = fname;
        bcast(broadcast_fname, 0, m_mpi_config->getCommunicator());

        m_streambuf_out
            = std::make_shared<detail::mpi_io>(m_mpi_config->getCommunicator(), broadcast_fname);
        m_file_out = std::make_shared<std::ostream>(m_streambuf_out.get());
        }
    else
        {
        // open the file
        m_file_out = std::make_shared<std::ofstream>(fname.c_str());
        }
#else
    m_file_out = std::make_shared<std::ofstream>(fname.c_str());
#endif

    // update the error, warning, and notice streams
    m_file_err = std::shared_ptr<std::ostream>();
    m_err_stream = m_file_out.get();
    m_warning_stream = m_file_out.get();
    m_notice_stream = m_file_out.get();
    }

/*! Sets all messenger output streams to ones that use PySys_WriteStd* functions so that messenger
   output is sent through sys.std*. This is useful in jupyter notebooks so that the output is
   displayed in the notebook properly. It is also useful if the user decides to remap sys.stdout to
   something.

    The iostream writes acquire the GIL, so they are safe to call from methods that have released
   the GIL.
*/
void Messenger::openPython()
    {
    // only import sys on first load
    if (!m_python_open)
        m_sys = pybind11::module::import("sys");

    m_pystdout = m_sys.attr("stdout");
    m_pystderr = m_sys.attr("stderr");

    m_streambuf_out = std::shared_ptr<std::streambuf>(new pybind11::detail::pythonbuf(m_pystdout));
    m_streambuf_err = std::shared_ptr<std::streambuf>(new pybind11::detail::pythonbuf(m_pystderr));

    // now update the error, warning, and notice streams
    m_file_out = std::shared_ptr<std::ostream>(new std::ostream(m_streambuf_out.get()));
    m_file_err = std::shared_ptr<std::ostream>(new std::ostream(m_streambuf_err.get()));

    m_err_stream = m_file_err.get();
    m_warning_stream = m_file_err.get();
    m_notice_stream = m_file_out.get();
    m_python_open = true;
    }

/*! Some notebook operations swap out sys.stdout and sys.stderr. Check if these have been swapped
   and reopen the output streams as necessary.
*/
void Messenger::reopenPythonIfNeeded()
    {
    // only attempt to reopen python streams if we previously opened them
    // and python is initialized
    if (m_python_open && Py_IsInitialized())
        {
        // flush and reopen the streams if sys.stdout or sys.stderr change
        pybind11::object new_pystdout = m_sys.attr("stdout");
        pybind11::object new_pystderr = m_sys.attr("stderr");
        if (!new_pystdout.is(m_pystdout) || !new_pystderr.is(m_pystderr))
            {
            m_file_out->flush();
            m_file_err->flush();
            openPython();
            }
        }
    }

/*! Any open file is closed. stdout is opened again for notices and stderr for warnings and errors.
 */
void Messenger::openStd()
    {
    m_file_out = std::shared_ptr<std::ostream>();
    m_file_err = std::shared_ptr<std::ostream>();
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;
    }

#ifdef ENABLE_MPI
/*! \param filename The output filename
    \param mpi_comm The MPI communicator to use for MPI file IO
 */
detail::mpi_io::mpi_io(const MPI_Comm& mpi_comm, const std::string& filename)
    : m_mpi_comm(mpi_comm), m_file_open(false)
    {
    assert(m_mpi_comm);

    // overwrite old file
    MPI_File_delete((char*)filename.c_str(), MPI_INFO_NULL);

    // open the log file
    int ret = MPI_File_open(m_mpi_comm,
                            (char*)filename.c_str(),
                            MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN,
                            MPI_INFO_NULL,
                            &m_file);

    if (ret == 0)
        m_file_open = true;
    }

int detail::mpi_io::overflow(int ch)
    {
    assert(m_file_open);

    char out_data = char(ch);

    // write value to log file using MPI-IO
    MPI_Status status;
    MPI_File_write_shared(m_file, &out_data, 1, MPI_CHAR, &status);
    return 0;
    }

void detail::mpi_io::close()
    {
    if (m_file_open)
        MPI_File_close(&m_file);

    m_file_open = false;
    }

#endif

namespace detail
    {
void export_Messenger(pybind11::module& m)
    {
    pybind11::class_<Messenger, std::shared_ptr<Messenger>>(m, "Messenger")
        .def(pybind11::init<std::shared_ptr<MPIConfiguration>>())
        .def("error", &Messenger::errorStr)
        .def("warning", &Messenger::warningStr)
        .def("notice", &Messenger::noticeStr)
        .def("getNoticeLevel", &Messenger::getNoticeLevel)
        .def("setNoticeLevel", &Messenger::setNoticeLevel)
        .def("getErrorPrefix",
             &Messenger::getErrorPrefix,
             pybind11::return_value_policy::reference_internal)
        .def("setErrorPrefix", &Messenger::setErrorPrefix)
        .def("getWarningPrefix",
             &Messenger::getWarningPrefix,
             pybind11::return_value_policy::reference_internal)
        .def("setWarningPrefix", &Messenger::setWarningPrefix)
        .def("getNoticePrefix",
             &Messenger::getNoticePrefix,
             pybind11::return_value_policy::reference_internal)
        .def("setWarningPrefix", &Messenger::setWarningPrefix)
        .def("openFile", &Messenger::openFile)
        .def("openPython", &Messenger::openPython)
        .def("openStd", &Messenger::openStd);
    }

    } // end namespace detail

    } // end namespace hoomd
