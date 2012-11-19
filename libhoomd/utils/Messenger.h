/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file Messenger.h
    \brief Declares the Messenger class
*/

#include <iostream>
#include <fstream>
#include <string>
#include <boost/shared_ptr.hpp>
#include <sstream>

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/categories.hpp>
#endif 

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __MESSENGER_H__
#define __MESSENGER_H__

//! A null stream that doesn't write anything sent to it
/*! From: http://bytes.com/topic/c/answers/127843-null-output-stream#post444998
*/
struct nullstream: std::ostream
    {
    //! Construct a null stream
    nullstream(): std::ios(0), std::ostream(0) {}
    };

#ifdef ENABLE_MPI
//! Class that supports writing to a shared log file using MPI-IO
class mpi_io
    {
    public:
        //! Defintions used by boost::iostreams
        typedef char         char_type;
        typedef boost::iostreams::sink_tag category;

        //! Constructor
        mpi_io(const MPI_Comm& mpi_comm, const std::string& filename);
        virtual ~mpi_io() { };

        //! Close the log file
        void close(); 

        //! \return true if file is open
        bool is_open()
            {
            return m_file_open;
            }

        //! Write a sequence of characters
        std::streamsize write ( const char * s, std::streamsize n );

    private:
        MPI_Comm m_mpi_comm;        //!< The MPI communciator
        MPI_File m_file;            //!< The file handle
        bool m_file_open;           //!< Whether the file is open
    };
#endif

//! Utility class for controlling message printing
/*! Large code projects need something more inteligent than just cout's for warning and
    notices and cerr for errors. To aid in user debugging, multiple levels of notice messages are required. Not all
    notice levels need to be printed in every run. A notice level can be set to control how much information is printed.
    Furthermore, in MPI runs not all processes need to print messages or one may wan to log the output of every rank to
    a different file.

    The Messenger class solves these issues. It provides a set of message levels that can be linked to chosen streams.
    The streams are returned by function calls so that users of this class can do the following:
    \code
    msg.error() << "Bad input" << endl;
    msg.warning() << "I hope you know what you are doing" << endl;
    msg.notice(1) << "Some useful info" << endl;
    msg.notice(5) << "Info that nobody cares about, unless they are debugging" << endl;
    \endcode
    Calls to notice(N) with N > the notice level will return a null stream so that the output is not printed.

    Furthermore, a chosen header may be added to messages of each type. So that the screen output from the previous
    could be:
    \code
    error!!!!: Bad input
    warning!!: I hope you know what you are doing
    Some useful info
    notice(5): Info that nobody cares about, unless they are debugging
    \endcode

    Messenger is copyable. This enables use cases where one global Messegner (possibly even having an open file)
    is copied into a local class and local settings changes applied.

    \b Implemntation

     - Errors and warnings are always printed.
     - Notice messages are printed when n <= the notice level.
     - 1 Should be the minimum notice level actually used so that all noticed messages may be silenced by setting 0.
     - Streams for each level are stored separately as pointers and can be set.
     - Streams default to cerr for errors and warnings and cout for notice messages.
     - Arbitrary streams may be set - however, since they are stored by pointer the caller is responsible for
       deleting them when set in this manner.
     - An alternate interface openFile opens a file for overwrite for all output levels, owned by the Messenger.

    \b HOOMD specific

    One global Messenger will be initialized in python and passed to the ExecutionConfiguration. From there, all C++
    classes can print messages via commands like m_exec_conf->msg->notice(1) << "blah". Maybe a macro NOTICE() to 
    simplify the typing??? Need to debate that.

    The following notice levels will be used:
    - Error: Any condtition that is erroneous and will prevent the run from continuing
        - Generally followed by a thrown exception
    - Warning: Out of bounds parameters, settings that will use a lot of memory, etc... Things that won't prevent
      continued execution, but that may lead to incorrect behavior.
    - 1: typical status messages
        - python command echos
        - TPS reports
        - which hardware is active
    - 2: notifications of general interest that most people want to see
        - notices that certain hardware is unavailable
        - status information printed at the end of a run
    - 3,4: Additional details on top of 2.
    - 5-10: Varying debug messages, number chosen arbitrarily based on how often the message is likely to print
        - Some examples for consistency
        - 5 construction/desctruction messages from every major class
        - 6 memory allocation/reallocation notices from every major class
        - 7 memory allocation/reallocation notices from GPUArray
    - 10: Trace messages that may print many times per time step.
*/
class Messenger
    {
    public:
        //! Construct a messenger
        Messenger();

        //! Destructor
        ~Messenger();

        //! Get the error stream
        std::ostream& error() const;

        //! Alternate method to print error strings
        void errorStr(const std::string& msg) const;

        //! Get the warning stream
        std::ostream& warning() const;

        //! Alternate method to print warning strings
        void warningStr(const std::string& msg) const;

        //! Get a notice stream
        std::ostream& notice(unsigned int level) const;

        //! Get a notice stream for serialized output
        std::ostream& collectiveNotice();

        //! Output the contents of a collective note in rank order
        void flushCollectiveNotice(unsigned int level);

        //! Alternate method to print notice strings
        void noticeStr(unsigned int level, const std::string& msg) const;

        //! Set processor rank
        /*! Error and warning messages are prefixed with rank information.

            Notice messages are only output on processor with rank 0.

            \param rank This processor's rank

         */ 
        void setRank(unsigned int rank, unsigned int partition)
            {
            // prefix all messages with rank information
            std::ostringstream oss;
            oss << "(" << rank << ")";
            m_rank_prefix = oss.str();
            m_rank = rank;
            m_notice_level = (m_rank == 0) ? m_default_notice_level :0;
            m_partition = partition;
            }


#ifdef ENABLE_MPI
        //! Set MPI communicator
        /*! \param mpi_comm The MPI communicator to use
         */
        void setMPICommunicator(const MPI_Comm mpi_comm)
            {
            m_mpi_comm = mpi_comm;
            m_has_mpi_comm = true;

            // open shared log file if necessary
            if (m_shared_filename != "")
                openSharedFile();
            }
#endif

        //! Get the notice level
        /*! \returns Current notice level
        */
        unsigned int getNoticeLevel() const
            {
            return m_notice_level;
            }

        //! Set the notice level
        /*! \param level Notice level to set
        */
        void setNoticeLevel(unsigned int level)
            {
            m_notice_level = level;
            m_default_notice_level = level;
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
            Since this is passing an internal reference back in, there are no dangling reference problems. And there is
            no need for the caller to manage the creation and deletion of a null stream.
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
            \note ": " is appened to the end of the prefix
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
            \note ": " is appened to the end of the prefix
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
            \note "(level): " is appened to the end of the prefix when level > 1
        */
        void setNoticePrefix(const std::string& prefix)
            {
            m_notice_prefix = prefix;
            }

        //! Open a file for error, warning, and notice streams
        void openFile(const std::string& fname);

#ifdef ENABLE_MPI
        //! Request logging of notices, warning and errors into shared log file
        /*! \param fname The filenam
         */
        void setSharedFile(const std::string& fname)
            {
            m_shared_filename = fname;

            if (m_has_mpi_comm)
                openSharedFile();
            }
#endif

        //! Open stdout and stderr again, closing any open file
        void openStd();
    private:
        std::ostream *m_err_stream;     //!< error stream
        std::ostream *m_warning_stream; //!< warning stream
        std::ostream *m_notice_stream;  //!< notice stream

        std::ostringstream m_collective_notice_stream;  //!< String for serialized notices

        boost::shared_ptr<nullstream>    m_nullstream;   //!< null stream
        boost::shared_ptr<std::ostream>  m_file;           //!< File stream

        std::string m_err_prefix;       //!< Prefix for error messages
        std::string m_warning_prefix;   //!< Prefix for warning messages
        std::string m_notice_prefix;    //!< Prefix for notice messages

        unsigned int m_notice_level;    //!< Notice level
        unsigned int m_default_notice_level; //!< Initial notice level

        std::string m_rank_prefix;      //!< Prefix indicating processor rank
        unsigned int m_rank;            //!< The MPI rank (default 0)
        unsigned int m_partition;       //!< The MPI partition

#ifdef ENABLE_MPI
        std::string m_shared_filename;  //!< Filename of shared log file
        MPI_Comm m_mpi_comm;            //!< The MPI communicator
        bool m_has_mpi_comm;            //!< True if MPI communicator has been set

        //! Open a shared file for error, warning, and notice streams
        void openSharedFile();
#endif
    };

//! Exports Messenger to python
void export_Messenger();

#endif // #ifndef __MESSENGER_H__
