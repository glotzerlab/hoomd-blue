// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Messenger.h
    \brief Declares the Messenger class
*/

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "HOOMDMPI.h"
#include "MPIConfiguration.h"

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __MESSENGER_H__
#define __MESSENGER_H__

namespace hoomd
    {
namespace detail
    {
//! A null stream that doesn't write anything sent to it
/*! From: http://bytes.com/topic/c/answers/127843-null-output-stream#post444998
 */
struct PYBIND11_EXPORT nullstream : std::ostream
    {
    //! Construct a null stream
    nullstream() : std::ios(0), std::ostream(0) { }
    };

    } // end namespace detail

//! Utility class for controlling message printing
/*! Large code projects need something more intelligent than just cout's for warning and
    notices and cerr for errors. To aid in user debugging, multiple levels of notice messages are
   required. Not all notice levels need to be printed in every run. A notice level can be set to
   control how much information is printed. Furthermore, in MPI runs not all processes need to print
   messages or one may wan to log the output of every rank to a different file.

    The Messenger class solves these issues. It provides a set of message levels that can be linked
   to chosen streams. The streams are returned by function calls so that users of this class can do
   the following: \code msg.error() << "Bad input" << endl; msg.warning() << "I hope you know what
   you are doing" << endl; msg.notice(1) << "Some useful info" << endl; msg.notice(5) << "Info that
   nobody cares about, unless they are debugging" << endl; \endcode Calls to notice(N) with N > the
   notice level will return a null stream so that the output is not printed.

    Furthermore, a chosen header may be added to messages of each type. So that the screen output
   from the previous could be: \code error!!!!: Bad input warning!!: I hope you know what you are
   doing Some useful info notice(5): Info that nobody cares about, unless they are debugging
    \endcode

    Messenger is copyable. This enables use cases where one global Messenger (possibly even having
   an open file) is copied into a local class and local settings changes applied.

    \b Implementation

     - Errors and warnings are always printed.
     - Notice messages are printed when n <= the notice level.
     - 1 Should be the minimum notice level actually used so that all noticed messages may be
   silenced by setting 0.
     - Streams for each level are stored separately as pointers and can be set.
     - Streams default to cerr for errors and warnings and cout for notice messages.
     - Arbitrary streams may be set - however, since they are stored by pointer the caller is
   responsible for deleting them when set in this manner.
     - An alternate interface openFile opens a file for overwrite for all output levels, owned by
   the Messenger.

    \b HOOMD specific

    One global Messenger will be initialized in python and passed to the ExecutionConfiguration.
   From there, all C++ classes can print messages via commands like m_exec_conf->msg->notice(1) <<
   "blah". Maybe a macro NOTICE() to simplify the typing??? Need to debate that.

    The following notice levels will be used:
    - Error: Any condition that is erroneous and will prevent the run from continuing
        - Generally followed by a thrown exception
    - Warning: Out of bounds parameters, settings that will use a lot of memory, etc... Things that
   won't prevent continued execution, but that may lead to incorrect behavior.
    - 1: typical status messages
        - python command echos
        - TPS reports
        - which hardware is active
    - 2: notifications of general interest that most people want to see
        - notices that certain hardware is unavailable
        - status information printed at the end of a run
    - 3,4: Additional details on top of 2.
    - 5-10: Varying debug messages, number chosen arbitrarily based on how often the message is
   likely to print
        - Some examples for consistency
        - 5 construction/destruction messages from every major class
        - 6 memory allocation/reallocation notices from every major class
        - 7 memory allocation/reallocation notices from GPUArray
    - 10: Trace messages that may print many times per time step.
*/
class PYBIND11_EXPORT Messenger
    {
    public:
    //! Construct a messenger
    Messenger(std::shared_ptr<MPIConfiguration> mpi_conf = std::shared_ptr<MPIConfiguration>());

    //! Copy constructor
    Messenger(const Messenger& msg);

    //! Assignment operator
    Messenger& operator=(Messenger& msg);

    //! Destructor
    virtual ~Messenger();

    //! Get the error stream
    std::ostream& error();

    //! Get the error stream on all ranks
    std::ostream& errorAllRanks();

    //! Alternate method to print error strings
    void errorStr(const std::string& msg);

    //! Get the warning stream
    std::ostream& warning();

    //! Alternate method to print warning strings
    void warningStr(const std::string& msg);

    //! Get a notice stream
    std::ostream& notice(unsigned int level);

    //! Print a notice message in rank-order
    void collectiveNoticeStr(unsigned int level, const std::string& msg);

    //! Alternate method to print notice strings
    void noticeStr(unsigned int level, const std::string& msg);

    //! Get the notice level
    /*! \returns Current notice level
     */
    unsigned int getNoticeLevel() const
        {
        unsigned int level = m_notice_level;

#ifdef ENABLE_MPI
        bcast(level, 0, m_mpi_config->getCommunicator());
#endif

        return level;
        }

    //! Set the notice level
    /*! \param level Notice level to set
     */
    void setNoticeLevel(unsigned int level)
        {
        m_notice_level = (m_mpi_config->getRank() == 0) ? level : 0;
        }

    //! Set the error stream
    /*! If not a built-in stream, the caller is responsible for deleting it
     */
    void setErrorStream(std::ostream& stream)
        {
        m_err_stream = &stream;
        }

    //! Set the warning stream
    /*! If not a built-in stream, the caller is responsible for deleting it
     */
    void setWarningStream(std::ostream& stream)
        {
        m_warning_stream = &stream;
        }

    //! Set the notice stream
    /*! If not a built-in stream, the caller is responsible for deleting it
     */
    void setNoticeStream(std::ostream& stream)
        {
        m_notice_stream = &stream;
        }

    //! Get the null stream
    /*! Use this with set*Stream: i.e. msg.setNoticeStream(msg.getNullStream()).
        Since this is passing an internal reference back in, there are no dangling reference
       problems. And there is no need for the caller to manage the creation and deletion of a null
       stream.
    */
    std::ostream& getNullStream() const
        {
        return *m_nullstream;
        }

    //! Get the error prefix
    /*! \returns Current prefix applied to error messages
     */
    const std::string& getErrorPrefix() const
        {
        return m_err_prefix;
        }

    //! Set the error prefix
    /*! \param prefix Prefix to apply to error messages
        \note ": " is appended to the end of the prefix
    */
    void setErrorPrefix(const std::string& prefix)
        {
        m_err_prefix = prefix;
        }

    //! Get the warning prefix
    /*! \returns Current prefix applied to warning messages
     */
    const std::string& getWarningPrefix() const
        {
        return m_warning_prefix;
        }

    //! Set the warning prefix
    /*! \param prefix Prefix to apply to warning messages
        \note ": " is appended to the end of the prefix
    */
    void setWarningPrefix(const std::string& prefix)
        {
        m_warning_prefix = prefix;
        }

    //! Get the notice prefix
    /*! \returns Current prefix applied to notice messages
     */
    const std::string& getNoticePrefix() const
        {
        return m_notice_prefix;
        }

    //! Set the notice prefix
    /*! \param prefix Prefix to apply to notice messages
        \note "(level): " is appended to the end of the prefix when level > 1
    */
    void setNoticePrefix(const std::string& prefix)
        {
        m_notice_prefix = prefix;
        }

    //! Open a file for error, warning, and notice streams
    void openFile(const std::string& fname);

    //! "Open" python sys.stdout and sys.stderr
    void openPython();

    //! Reopen the python streams if sys.stdout/err changes
    void reopenPythonIfNeeded();

    //! Open stdout and stderr again, closing any open file
    void openStd();

    private:
    std::shared_ptr<MPIConfiguration> m_mpi_config; //!< The MPI configuration

    std::ostream* m_err_stream;     //!< error stream
    std::ostream* m_warning_stream; //!< warning stream
    std::ostream* m_notice_stream;  //!< notice stream

    std::shared_ptr<std::streambuf> m_streambuf_out;  //!< streambuf (stdout)
    std::shared_ptr<std::streambuf> m_streambuf_err;  //!< streambuf (if err different from out)
    std::shared_ptr<detail::nullstream> m_nullstream; //!< null stream
    std::shared_ptr<std::ostream> m_file_out;         //!< File stream (stdout)
    std::shared_ptr<std::ostream> m_file_err;         //!< File stream (stderr)

    std::string m_err_prefix;     //!< Prefix for error messages
    std::string m_warning_prefix; //!< Prefix for warning messages
    std::string m_notice_prefix;  //!< Prefix for notice messages

    unsigned int m_notice_level; //!< Notice level

    bool m_python_open = false;  //!< True when the python output stream is open
    pybind11::module m_sys;      //!< sys module
    pybind11::object m_pystdout; //!< Currently bound python sys.stdout
    pybind11::object m_pystderr; //!< Currently bound python sys.stderr
    };

namespace detail
    {
//! Exports Messenger to python
void export_Messenger(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif // #ifndef __MESSENGER_H__
