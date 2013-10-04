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

/*! \file Messenger.cc
    \brief Defines the Messenger class
*/

#include "Messenger.h"
#include "ExecutionConfiguration.h"

#include <assert.h>
using namespace std;

#include <boost/python.hpp>

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

using namespace boost::python;

/*! \post Warning and error streams are set to cerr
    \post The notice stream is set to cout
    \post The notice level is set to 1
    \post prefixes are "error!!!!" , "warning!!" and "notice"
*/
Messenger::Messenger()
    : m_default_notice_level(2)
    {
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;
    m_nullstream = boost::shared_ptr<nullstream>(new nullstream());
    m_notice_level = m_default_notice_level;
    m_err_prefix     = "**ERROR**";
    m_warning_prefix = "*Warning*";
    m_notice_prefix  = "notice";

#ifdef ENABLE_MPI
    m_shared_filename = "";
    m_has_mpi_comm = false;
#endif

    // preliminarily initialize rank and partiton
    #ifdef ENABLE_MPI
    setRank(ExecutionConfiguration::guessRank(),0);
    #else
    setRank(0,0);
    #endif
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
*/
std::ostream& Messenger::error() const
    {
    assert(m_err_stream);
    if (m_err_prefix != string(""))
        *m_err_stream << m_err_prefix << " RANK " << m_rank << ": ";
    return *m_err_stream;
    }

/*! \param msg Message to print
    \sa error()
*/
void Messenger::errorStr(const std::string& msg) const
    {
    error() << msg;
    }

/*! \returns The warning stream for use in printing warning messages
    \post If the warning prefix is not the empty string, the message is preceded with
    "${warning_prefix}: ".
*/
std::ostream& Messenger::warning() const
    {
    assert(m_warning_stream);
    if (m_warning_prefix != string(""))
        *m_warning_stream << m_warning_prefix << " RANK " << m_rank << ": ";
    return *m_warning_stream;
    }

/*! \param msg Message to print
    \sa warning()
*/
void Messenger::warningStr(const std::string& msg) const
    {
    warning() << msg;
    }

/*! \returns The notice stream for use in printing notice messages
    \post If the notice prefix is not the empty string and the level is greater than 1 the message is preceded with
    "${notice_prefix}(n): ".

    If level is greater than the notice level, a null stream is returned so that the output is not printed.
*/
std::ostream& Messenger::notice(unsigned int level) const
    {
    assert(m_notice_stream);
    if (level <= m_notice_level)
        {
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
void Messenger::collectiveNoticeStr(unsigned int level, const std::string& msg) const
    {
    std::vector<std::string> rank_notices;

#ifdef ENABLE_MPI
    if (m_has_mpi_comm)
        {
        gather_v(msg, rank_notices, 0, m_mpi_comm);
        }
    else
#endif
        {
        rank_notices.push_back(msg);
        }

#ifdef ENABLE_MPI
    if (!m_has_mpi_comm || m_rank == 0)
#endif
        {
        // Output notices in rank order
        std::vector<std::string>::iterator notice_it;
        for (notice_it = rank_notices.begin(); notice_it != rank_notices.end(); notice_it++)
            {
            notice(level) << *notice_it;
            }
        }
    }

/*! \param level Notice level
    \param msg Message to print
    \sa notice()
*/
void Messenger::noticeStr(unsigned int level, const std::string& msg) const
    {
    notice(level) << msg;
    }

/*! \param fname File name
    The file is ovewritten if it exists. If there is an error opening the file, all level's streams are left
    as is and an error() is issued.
*/
void Messenger::openFile(const std::string& fname)
    {
    m_file = boost::shared_ptr<std::ostream>(new ofstream(fname.c_str()));
    m_err_stream = m_file.get();
    m_warning_stream = m_file.get();
    m_notice_stream = m_file.get();
    }

#ifdef ENABLE_MPI
/*! Open a shared file for error, warning, and notice streams

    A suffix .rank (where rank is the partition number)
    is appended to the filename
*/
void Messenger::openSharedFile()
    {
    assert(m_has_mpi_comm);

    std::ostringstream oss;
    oss << m_shared_filename << "." << m_partition;
    boost::iostreams::stream<mpi_io> *mpi_ios = new boost::iostreams::stream<mpi_io>((const MPI_Comm&) m_mpi_comm, oss.str());

    // now update the error, warning, and notice streams
    m_file = boost::shared_ptr<std::ostream>(mpi_ios);
    m_err_stream = m_file.get();
    m_warning_stream = m_file.get();
    m_notice_stream = m_file.get();
    }
#endif

/*! Any open file is closed. stdout is opened again for notices and stderr for warnings and errors.
*/
void Messenger::openStd()
    {
    m_file = boost::shared_ptr<std::ostream>();
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;
    }

#ifdef ENABLE_MPI
/*! \param filename The output filename
    \param mpi_comm The MPI communicator to use for MPI file IO
 */
mpi_io::mpi_io(const MPI_Comm& mpi_comm, const std::string& filename)
    : m_mpi_comm(mpi_comm),  m_file_open(false)
    {
    assert(m_mpi_comm);

    unsigned int len = filename.size();
    char cfilename[len+1];
    filename.copy(cfilename,len);
    cfilename[len] = '\0';

    // overwrite old file
    MPI_File_delete(cfilename, MPI_INFO_NULL);

    // open the log file
    int ret = MPI_File_open(m_mpi_comm, cfilename,  MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &m_file);

    if (ret == 0)
        m_file_open = true;
    }

std::streamsize mpi_io::write(const char *s,  std::streamsize n)
    {
    assert(m_file_open);

    char out_data[n];
    strncpy(out_data, s, n);

    // write value to log file using MPI-IO
    MPI_Status status;
    MPI_File_write_shared(m_file, out_data, n, MPI_CHAR, &status);
    return n;
    }

void mpi_io::close()
    {
    if (m_file_open)
        MPI_File_close(&m_file);

    m_file_open = false;
    }
#endif

void export_Messenger()
    {
    class_<Messenger, boost::shared_ptr<Messenger>, boost::noncopyable >
         ("Messenger", init< >())
         .def("error", &Messenger::errorStr)
         .def("warning", &Messenger::warningStr)
         .def("notice", &Messenger::noticeStr)
         .def("getNoticeLevel", &Messenger::getNoticeLevel)
         .def("setNoticeLevel", &Messenger::setNoticeLevel)
         .def("getErrorPrefix", &Messenger::getErrorPrefix, return_value_policy<copy_const_reference>())
         .def("setErrorPrefix", &Messenger::setErrorPrefix)
         .def("getWarningPrefix", &Messenger::getWarningPrefix, return_value_policy<copy_const_reference>())
         .def("setWarningPrefix", &Messenger::setWarningPrefix)
         .def("getNoticePrefix", &Messenger::getNoticePrefix, return_value_policy<copy_const_reference>())
         .def("setWarningPrefix", &Messenger::setWarningPrefix)
         .def("openFile", &Messenger::openFile)
#ifdef ENABLE_MPI
         .def("setSharedFile", &Messenger::setSharedFile)
#endif
         .def("openStd", &Messenger::openStd)
         ;
    }
